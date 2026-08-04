"""Tests for lumen.integrations.langgraph."""

import pytest

from lumen.config import LumenConfig
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


class TestLumenCheckpointSaver:
    def test_put_and_get(self, fresh_config):
        from lumen.integrations.langgraph import LumenCheckpointSaver

        saver = LumenCheckpointSaver(config=fresh_config)
        config = {"configurable": {"thread_id": "t1"}}

        checkpoint = {
            "v": 1,
            "id": "cp-1",
            "ts": "2024-01-01T00:00:00Z",
            "channel_values": {"input": "hello"},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata = {"source": "put", "step": 0, "writes": {}, "parents": {}}

        new_config = saver.put(config, checkpoint, metadata, {})
        assert new_config["configurable"]["checkpoint_id"] == "cp-1"

        tup = saver.get_tuple(new_config)
        assert tup is not None
        assert tup.checkpoint["id"] == "cp-1"
        assert tup.metadata["step"] == 0

    def test_list_checkpoints(self, fresh_config):
        from lumen.integrations.langgraph import LumenCheckpointSaver

        saver = LumenCheckpointSaver(config=fresh_config)
        config = {"configurable": {"thread_id": "t2"}}

        for i in range(3):
            cp = {
                "v": 1,
                "id": f"cp-{i}",
                "ts": f"2024-01-0{i+1}T00:00:00Z",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }
            meta = {"source": "put", "step": i, "writes": {}, "parents": {}}
            saver.put(config, cp, meta, {})

        cps = list(saver.list(config, limit=10))
        assert len(cps) == 3
        # Newest first
        assert cps[0].checkpoint["id"] == "cp-2"
        assert cps[1].checkpoint["id"] == "cp-1"
        assert cps[2].checkpoint["id"] == "cp-0"

    def test_thread_isolation(self, fresh_config):
        from lumen.integrations.langgraph import LumenCheckpointSaver

        saver = LumenCheckpointSaver(config=fresh_config)

        for tid in ("tA", "tB"):
            cfg = {"configurable": {"thread_id": tid}}
            cp = {
                "v": 1,
                "id": f"cp-{tid}",
                "ts": "2024-01-01T00:00:00Z",
                "channel_values": {},
                "channel_versions": {},
                "versions_seen": {},
                "pending_sends": [],
            }
            meta = {"source": "put", "step": 0, "writes": {}, "parents": {}}
            saver.put(cfg, cp, meta, {})

        tup_a = saver.get_tuple({"configurable": {"thread_id": "tA"}})
        tup_b = saver.get_tuple({"configurable": {"thread_id": "tB"}})
        assert tup_a.checkpoint["id"] == "cp-tA"
        assert tup_b.checkpoint["id"] == "cp-tB"

    def test_put_writes(self, fresh_config):
        from lumen.integrations.langgraph import LumenCheckpointSaver

        saver = LumenCheckpointSaver(config=fresh_config)
        config = {"configurable": {"thread_id": "t3"}}
        saver.put_writes([("channel1", "value1"), ("channel2", "value2")], config)

        conn = saver.conn
        rows = conn.execute(
            "SELECT COUNT(*) FROM chunk WHERE content_hash = ?",
            ("lg_write",),
        ).fetchone()
        assert rows[0] == 2

    def test_get_tuple_missing(self, fresh_config):
        from lumen.integrations.langgraph import LumenCheckpointSaver

        saver = LumenCheckpointSaver(config=fresh_config)
        result = saver.get_tuple({"configurable": {"thread_id": "missing"}})
        assert result is None


class TestLumenGraphStore:
    def test_put_and_get(self, fresh_config):
        from lumen.integrations.langgraph import LumenGraphStore

        store = LumenGraphStore(config=fresh_config)
        store.put(("users", "alice"), "goal", {"text": "learn french"})

        item = store.get(("users", "alice"), "goal")
        assert item is not None
        assert item["key"] == "goal"
        assert "learn french" in item["value"]

    def test_delete(self, fresh_config):
        from lumen.integrations.langgraph import LumenGraphStore

        store = LumenGraphStore(config=fresh_config)
        store.put(("tmp",), "k1", "v1")
        store.delete(("tmp",), "k1")

        item = store.get(("tmp",), "k1")
        assert item is None

    def test_search(self, fresh_config):
        embedder = MockEmbedder(dims=fresh_config.embedding_dims)
        from lumen.integrations.langgraph import LumenGraphStore

        store = LumenGraphStore(config=fresh_config, embedder=embedder)
        store.put(("docs",), "doc1", "machine learning is fascinating")
        store.put(("docs",), "doc2", "cloud computing scales well")

        results = store.search(("docs",), "machine learning", limit=5)
        assert len(results) >= 1
        # At least one result should be doc1 (exact semantic match depends on embedder)
        keys = {r["key"] for r in results}
        assert "doc1" in keys

    def test_list_namespaces(self, fresh_config):
        from lumen.integrations.langgraph import LumenGraphStore

        store = LumenGraphStore(config=fresh_config, default_namespace="ns")
        store.put(("ns", "a"), "k1", "v1")
        store.put(("ns", "b"), "k2", "v2")

        namespaces = store.list_namespaces(prefix=("ns",))
        assert ("ns", "a") in namespaces
        assert ("ns", "b") in namespaces
