"""Tests for field-level encryption in SQLite storage."""

import pytest

from lumen.config import LumenConfig
from lumen.data.schema import get_connection, init_db
from lumen.force.contextual.embed import MockEmbedder
from lumen.force.mnemonic.store import store_memory
from lumen.search import SearchPipeline


@pytest.fixture
def encrypted_client(tmp_path, monkeypatch):
    """Create a palace with encryption enabled."""
    store = tmp_path / "store"
    models = tmp_path / "models"
    store.mkdir()
    models.mkdir()

    config = LumenConfig(
        store_path=store,
        model_path=models,
        vector_index="sqlite-vec",
        device="generic",
        encryption_key="test-secret-key-12345",
    )
    conn = get_connection(config)
    init_db(conn)
    embedder = MockEmbedder(dims=config.embedding_dims)
    pipeline = SearchPipeline(conn, config, embedder=embedder)
    yield {"conn": conn, "config": config, "pipeline": pipeline, "embedder": embedder}
    conn.close()


class TestFieldLevelEncryption:
    def test_content_is_encrypted_in_db(self, encrypted_client):
        conn = encrypted_client["conn"]
        config = encrypted_client["config"]
        chunk_id = store_memory(
            conn,
            content="My secret memory",
            room_name="secrets",
            config=config,
            embedding=encrypted_client["embedder"].encode_single("My secret memory"),
        )
        row = conn.execute(
            "SELECT content, encrypted FROM chunk WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        assert row["encrypted"] == 1
        assert "My secret memory" not in row["content"]
        # Content should be a Fernet token (URL-safe base64)
        assert row["content"].startswith("gAAAA")

    def test_retrieval_decrypts_content(self, encrypted_client):
        conn = encrypted_client["conn"]
        config = encrypted_client["config"]
        store_memory(
            conn,
            content="Project alpha requirements",
            room_name="projects",
            config=config,
            embedding=encrypted_client["embedder"].encode_single("Project alpha requirements"),
        )
        results = encrypted_client["pipeline"].execute("alpha requirements", k=5)
        assert len(results) >= 1
        assert results[0].content == "Project alpha requirements"

    def test_dedup_works_with_encryption(self, encrypted_client):
        conn = encrypted_client["conn"]
        config = encrypted_client["config"]
        id1 = store_memory(
            conn,
            content="Duplicate content",
            room_name="test",
            config=config,
        )
        id2 = store_memory(
            conn,
            content="Duplicate content",
            room_name="test",
            config=config,
        )
        assert id1 == id2
