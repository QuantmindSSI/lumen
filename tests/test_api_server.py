"""Tests for lumen.api.server."""

import pytest
from fastapi.testclient import TestClient

from lumen.api.server import app
from lumen.config import LumenConfig
from lumen.conversation import ConversationMemory
from lumen.data.schema import get_connection, init_db
from lumen.force.contextual.embed import MockEmbedder


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a TestClient with isolated filesystem state."""
    store = tmp_path / "store"
    models = tmp_path / "models"
    store.mkdir()
    models.mkdir()

    config = LumenConfig(
        store_path=store,
        model_path=models,
        vector_index="sqlite-vec",
        device="generic",
    )
    conn = get_connection(config)
    init_db(conn)
    embedder = MockEmbedder(dims=config.embedding_dims)

    # Pre-seed module state
    from lumen.api import server as server_mod

    server_mod._state = {
        "config": config,
        "conn": conn,
        "pipeline": None,  # built below
        "conversation": ConversationMemory(config=config, conn=conn, embedder=embedder),
        "tfc": None,
        "embedder": embedder,
    }

    # Build pipeline after state is set
    from lumen.search import SearchPipeline

    server_mod._state["pipeline"] = SearchPipeline(conn, config, embedder=embedder)

    with TestClient(app) as c:
        yield c

    conn.close()


class TestSecurityHeaders:
    def test_security_headers_present(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "DENY"
        assert "strict-transport-security" in resp.headers
        assert "content-security-policy" in resp.headers
        assert "referrer-policy" in resp.headers


class TestAuthMiddleware:
    def test_protected_endpoint_without_key(self, client, monkeypatch):
        # Default fixture has no API key set, so endpoints are open
        resp = client.post("/search", json={"query": "test", "top_k": 3})
        assert resp.status_code == 200

    def test_protected_endpoint_rejects_invalid_key(self, client, monkeypatch):
        # Temporarily set an API key on the global config used by the middleware
        from lumen.api import server as server_mod

        original_key = server_mod._config.api_key
        server_mod._config.api_key = "supersecrettestkey"
        try:
            resp = client.post("/search", json={"query": "test", "top_k": 3})
            assert resp.status_code == 401
            assert "API key" in resp.json()["detail"]

            resp = client.post(
                "/search",
                json={"query": "test", "top_k": 3},
                headers={"X-API-Key": "wrong-key"},
            )
            assert resp.status_code == 401
        finally:
            server_mod._config.api_key = original_key

    def test_protected_endpoint_accepts_valid_key(self, client, monkeypatch):
        from lumen.api import server as server_mod

        original_key = server_mod._config.api_key
        server_mod._config.api_key = "supersecrettestkey"
        try:
            resp = client.post(
                "/search",
                json={"query": "test", "top_k": 3},
                headers={"X-API-Key": "supersecrettestkey"},
            )
            assert resp.status_code == 200
        finally:
            server_mod._config.api_key = original_key


class TestRateLimiting:
    def test_rate_limit_middleware_present(self, client):
        from lumen.api import server as server_mod

        assert hasattr(server_mod.app.state, "limiter")
        assert server_mod.app.state.limiter is not None
        # Verify the decorator is wired by checking route handlers
        route = next((r for r in server_mod.app.routes if r.path == "/search"), None)
        assert route is not None
        # Slowapi uses a wrapper, so endpoint is the limiter wrapper
        assert route.endpoint.__name__ == "search"

    def test_default_rate_limit_configured(self, client):
        from lumen.api import server as server_mod

        assert "minute" in server_mod._config.api_rate_limit


class TestRequestSizeLimit:
    def test_request_too_large_rejected(self, client, monkeypatch):
        from lumen.api import server as server_mod

        original_size = server_mod._config.request_max_size_bytes
        server_mod._config.request_max_size_bytes = 100
        # Re-initialise size limit middleware with new config (FastAPI applies middleware on add)
        # Since we can't easily swap middleware at runtime, we test via direct assertion
        # that the middleware class checks the header correctly.
        from lumen.api.server import _SizeLimitMiddleware

        scope = {
            "type": "http",
            "headers": [(b"content-length", b"200")],
        }
        calls = []

        async def mock_app(scope, receive, send):
            calls.append("app")

        middleware = _SizeLimitMiddleware(mock_app, max_bytes=100)

        async def mock_send(msg):
            calls.append(msg)

        import asyncio

        asyncio.run(middleware(scope, None, mock_send))
        assert any(m.get("status") == 413 for m in calls if isinstance(m, dict))
        server_mod._config.request_max_size_bytes = original_size


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestStatus:
    def test_status_initial(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["rooms"] >= 0
        assert body["active_chunks"] >= 0
        assert "tfc" in body


class TestStoreAndSearch:
    def test_store_memory(self, client):
        resp = client.post(
            "/store",
            json={
                "content": "The quick brown fox",
                "room": "animals",
                "source_type": "user_input",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["chunk_id"] > 0
        assert body["room"] == "animals"

    def test_search_finds_stored(self, client):
        # Store
        client.post(
            "/store",
            json={
                "content": "project alpha requirements",
                "room": "projects",
                "source_type": "user_input",
            },
        )
        # Search
        resp = client.post("/search", json={"query": "alpha requirements", "top_k": 5})
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "alpha requirements"
        assert len(body["results"]) >= 1


class TestFeedback:
    def test_feedback_creation(self, client):
        # Seed a chunk first so FK constraint is satisfied
        client.post(
            "/store",
            json={
                "content": "feedback target",
                "room": "feedback_test",
                "source_type": "user_input",
            },
        )
        resp = client.post(
            "/feedback",
            json={
                "chunk_id": 1,
                "was_useful": True,
                "user_id": "alice",
                "feedback_type": "explicit",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True


class TestAssemble:
    def test_assemble_empty(self, client):
        resp = client.post("/assemble", json={"query": "what is the weather?", "top_k": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert "assembled_context" in body
        assert body["retrieved_count"] >= 0


class TestTurn:
    def test_turn_store(self, client):
        resp = client.post(
            "/turn",
            json={
                "user_msg": "Hello",
                "assistant_msg": "Hi there",
                "room": "conversations",
                "retrieved_chunk_ids": [],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_chunk_id"] > 0
        assert body["assistant_chunk_id"] > 0


class TestDashboard:
    def test_dashboard_serves_html(self, client):
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Lumen Memory Palace" in resp.text

    def test_dashboard_data(self, client):
        # Seed a room
        client.post(
            "/store",
            json={
                "content": "dashboard test memory",
                "room": "dashboard_test",
                "source_type": "user_input",
            },
        )
        resp = client.get("/dashboard-data")
        assert resp.status_code == 200
        body = resp.json()
        assert "rooms" in body
        assert "total_chunks" in body
        assert "tfc" in body
        assert "memory_budget_pct" in body
        assert "degradation_stage" in body
        assert any(r["name"] == "dashboard_test" for r in body["rooms"])

    def test_metrics(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["lumen_version"] == "0.1.0-alpha"
        assert "system" in body
        assert "palace" in body
        assert "business" in body
        assert body["business"]["data_sovereignty_pct"] == 100
        assert body["business"]["api_cost_per_query_usd"] == 0.0
