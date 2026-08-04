"""Tests for lumen.api.server."""

import pytest
from fastapi.testclient import TestClient

from lumen.api.server import app
from lumen.config import LumenConfig
from lumen.data.schema import get_connection, init_db
from lumen.force.contextual.embed import MockEmbedder
from lumen.conversation import ConversationMemory


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
