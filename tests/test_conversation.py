"""Tests for lumen.lumen.conversation."""

import pytest

from lumen.config import LumenConfig
from lumen.conversation import ConversationMemory, TurnResult
from lumen.data.schema import get_connection
from lumen.force.contextual.embed import MockEmbedder


@pytest.fixture
def fresh_config(tmp_path):
    cfg = LumenConfig(
        store_path=tmp_path / "store",
        model_path=tmp_path / "models",
        vector_index="sqlite-vec",
        device="generic",
    )
    cfg.store_path.mkdir(parents=True, exist_ok=True)
    cfg.model_path.mkdir(parents=True, exist_ok=True)
    return cfg


@pytest.fixture
def memory_db(fresh_config):
    conn = get_connection(fresh_config)
    yield conn
    conn.close()


class TestConversationMemory:
    def test_retrieve_and_assemble_empty_db(self, fresh_config, memory_db):
        embedder = MockEmbedder(dims=fresh_config.embedding_dims)
        cm = ConversationMemory(config=fresh_config, conn=memory_db, embedder=embedder)
        turn = cm.retrieve_and_assemble("hello")
        assert isinstance(turn, TurnResult)
        assert turn.retrieved_chunks == []
        assert "hello" in turn.assembled_context

    def test_store_turn(self, fresh_config, memory_db):
        embedder = MockEmbedder(dims=fresh_config.embedding_dims)
        cm = ConversationMemory(config=fresh_config, conn=memory_db, embedder=embedder)
        uid, aid = cm.store_turn("user says hi", "agent replies hello")
        assert uid is not None
        assert aid is not None

        # Verify rows exist
        row = memory_db.execute("SELECT content FROM chunk WHERE chunk_id = ?", (uid,)).fetchone()
        assert row["content"] == "user says hi"
        row = memory_db.execute("SELECT content FROM chunk WHERE chunk_id = ?", (aid,)).fetchone()
        assert row["content"] == "agent replies hello"

    def test_store_turn_with_implicit_feedback(self, fresh_config, memory_db):
        embedder = MockEmbedder(dims=fresh_config.embedding_dims)
        cm = ConversationMemory(config=fresh_config, conn=memory_db, embedder=embedder)

        # Seed a memory
        from lumen.force.mnemonic.store import store_memory

        emb = embedder.encode_single("seed memory")
        sid = store_memory(
            memory_db, "seed memory", room_name="test", embedding=emb, config=fresh_config
        )

        # Stub a RetrievedChunk
        from lumen.fusion import RetrievedChunk

        stub = RetrievedChunk(
            chunk_id=sid,
            room_name="test",
            locus_name="none",
            content="seed memory",
            provenance_id=None,
            rrf_score=1.0,
            vm_score=0.5,
            frqad_score=0.5,
            recency_hours=0.0,
            final_score=1.0,
        )

        cm.store_turn("q", "a", retrieved_chunks=[stub])

        # Feedback should be logged
        rows = memory_db.execute("SELECT * FROM feedback_log WHERE chunk_id = ?", (sid,)).fetchall()
        assert len(rows) == 1
        assert rows[0]["positive"] == 1
        assert rows[0]["feedback_type"] == "implicit"

    def test_explicit_feedback(self, fresh_config, memory_db):
        embedder = MockEmbedder(dims=fresh_config.embedding_dims)
        cm = ConversationMemory(config=fresh_config, conn=memory_db, embedder=embedder)
        # Seed a chunk so FK constraint is satisfied
        from lumen.force.mnemonic.store import store_memory

        emb = embedder.encode_single("feedback target")
        cid = store_memory(
            memory_db, "feedback target", room_name="test", embedding=emb, config=fresh_config
        )
        cm.log_explicit_feedback(cid, True, user_id="alice", feedback_type="explicit")
        row = memory_db.execute("SELECT * FROM feedback_log WHERE chunk_id = ?", (cid,)).fetchone()
        assert row["positive"] == 1
        assert row["user_id"] == "alice"
        assert row["feedback_type"] == "explicit"

    def test_close_only_closes_owned_connection(self, fresh_config):
        embedder = MockEmbedder(dims=fresh_config.embedding_dims)
        conn = get_connection(fresh_config)
        cm = ConversationMemory(config=fresh_config, conn=conn, embedder=embedder)
        cm.close()  # should not raise
        # Connection was passed in, so it should still be usable
        conn.execute("SELECT 1")
        conn.close()
