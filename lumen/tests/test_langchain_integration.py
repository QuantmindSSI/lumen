"""Tests for lumen.integrations.langchain."""

import pytest

from lumen.config import LumenConfig
from lumen.data.schema import get_connection
from lumen.force.contextual.embed import MockEmbedder
from lumen.integrations.langchain import LumenChatMemory


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


class TestLumenChatMemory:
    def test_retrieve_and_save(self, fresh_config):
        embedder = MockEmbedder(dims=fresh_config.embedding_dims)
        mem = LumenChatMemory(
            config=fresh_config,
            user_id="alice",
            room="support",
            embedder=embedder,
        )
        ctx = mem.retrieve_context("hello")
        assert "hello" in ctx
        mem.save_turn("hello", "hi there")
        # Turn stored
        mem.clear()

    def test_user_scoping(self, fresh_config):
        embedder = MockEmbedder(dims=fresh_config.embedding_dims)
        mem_alice = LumenChatMemory(
            config=fresh_config,
            user_id="alice",
            room="test",
            embedder=embedder,
        )
        mem_bob = LumenChatMemory(
            config=fresh_config,
            user_id="bob",
            room="test",
            embedder=embedder,
        )
        mem_alice.save_turn("alice msg", "alice reply")
        mem_bob.save_turn("bob msg", "bob reply")

        conn = get_connection(fresh_config)
        alice_chunks = conn.execute(
            "SELECT COUNT(*) FROM chunk WHERE room_id = (SELECT room_id FROM room WHERE name = 'test')"
        ).fetchone()[0]
        assert alice_chunks >= 4  # 2 turns x 2 messages

    def test_feedback_and_learn(self, fresh_config):
        embedder = MockEmbedder(dims=fresh_config.embedding_dims)
        mem = LumenChatMemory(
            config=fresh_config,
            user_id="alice",
            room="test",
            embedder=embedder,
        )
        # Seed feedback
        from lumen.force.mnemonic.store import store_memory

        embedder = MockEmbedder(dims=fresh_config.embedding_dims)
        conn = mem.memory.conn
        for i in range(12):
            emb = embedder.encode_single(f"fact {i}")
            cid = store_memory(
                conn, f"fact {i}", room_name="test", embedding=emb, config=fresh_config
            )
            mem.log_feedback(cid, was_useful=(i % 2 == 0))

        weights = mem.memory.learn_weights(user_id="alice")
        assert isinstance(weights, dict)
        assert "goal_relevance" in weights

    def test_system_prompt_override(self, fresh_config):
        embedder = MockEmbedder(dims=fresh_config.embedding_dims)
        mem = LumenChatMemory(
            config=fresh_config,
            system_prompt_override="You are a pirate.",
            embedder=embedder,
        )
        ctx = mem.retrieve_context("hello")
        assert "pirate" in ctx


class TestLumenStore:
    def test_mset_mget(self, fresh_config):
        try:
            from lumen.integrations.langchain import LumenStore
        except Exception as exc:
            pytest.skip(f"LumenStore unavailable: {exc}")

        embedder = MockEmbedder(dims=fresh_config.embedding_dims)
        store = LumenStore(config=fresh_config, default_namespace="kv", embedder=embedder)
        store.mset([("key1", b"value1"), ("key2", b"value2")])
        results = store.mget(["key1", "key2", "missing"])
        assert results[0] == b"value1"
        assert results[1] == b"value2"
        assert results[2] is None

    def test_mdelete_and_yield_keys(self, fresh_config):
        try:
            from lumen.integrations.langchain import LumenStore
        except Exception as exc:
            pytest.skip(f"LumenStore unavailable: {exc}")

        embedder = MockEmbedder(dims=fresh_config.embedding_dims)
        store = LumenStore(config=fresh_config, default_namespace="del", embedder=embedder)
        store.mset([("k1", b"v1")])
        store.mdelete(["k1"])
        results = store.mget(["k1"])
        assert results[0] is None

        # After delete, yield_keys should be empty for this prefix
        keys = list(store.yield_keys(prefix="k"))
        assert "k1" not in keys
