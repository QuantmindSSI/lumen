"""Tests for chunk-count-based L3 budget eviction."""

import sqlite3

import pytest

from lumen.config import LumenConfig
from lumen.data.schema import get_connection, init_db
from lumen.force.contextual.embed import MockEmbedder
from lumen.force.mnemonic.store import store_memory
from lumen.force.mnemonic.forgetting_l3_budget import budget_curated_eviction


@pytest.fixture
def l3_config(tmp_path):
    return LumenConfig(
        store_path=tmp_path / ".lumen" / "store",
        model_path=tmp_path / ".lumen" / "models",
        vector_index="sqlite-vec",
        device="generic",
        memory_limit_mb=16,  # Very low to force eviction with few chunks
    )


@pytest.fixture
def l3_conn(l3_config):
    conn = get_connection(l3_config)
    init_db(conn)
    yield conn
    conn.close()


def _ingest_n_chunks(conn, config, n):
    embedder = MockEmbedder(dims=config.embedding_dims)
    for i in range(n):
        store_memory(
            conn,
            content=f"chunk_{i:04d} " + "word " * 10,
            room_name="test",
            embedding=embedder.encode_single(f"chunk_{i:04d}"),
            config=config,
        )
    conn.commit()


class TestL3Eviction:
    def test_no_eviction_below_threshold(self, l3_conn, l3_config):
        _ingest_n_chunks(l3_conn, l3_config, 5)
        evicted = budget_curated_eviction(l3_conn, l3_config)
        assert evicted == 0

    def test_eviction_triggered_above_threshold(self, l3_conn, l3_config):
        # memory_limit_mb=16 → target_ram_mb=13.6
        # Each chunk ≈ 1800 bytes → 13.6 MB ≈ 7936 chunks
        _ingest_n_chunks(l3_conn, l3_config, 10_000)
        evicted = budget_curated_eviction(l3_conn, l3_config)
        assert evicted > 0

    def test_eviction_degrades_resolution(self, l3_conn, l3_config):
        _ingest_n_chunks(l3_conn, l3_config, 10_000)
        budget_curated_eviction(l3_conn, l3_config)

        degraded = l3_conn.execute(
            "SELECT COUNT(*) FROM chunk WHERE resolution != 'FP32' AND valid_to IS NULL"
        ).fetchone()[0]
        assert degraded > 0

    def test_eviction_respects_target_ram(self, l3_conn, l3_config):
        _ingest_n_chunks(l3_conn, l3_config, 10_000)
        # Manually set a very low target to force many evictions
        evicted = budget_curated_eviction(l3_conn, l3_config, target_ram_mb=1.0)
        assert evicted > 0
        # After eviction, many chunks should be degraded (not FP32) or released
        degraded = l3_conn.execute(
            "SELECT COUNT(*) FROM chunk WHERE resolution != 'FP32' OR valid_to IS NOT NULL"
        ).fetchone()[0]
        assert degraded > 0

    def test_lowest_vm_evicted_first(self, l3_conn, l3_config):
        embedder = MockEmbedder(dims=l3_config.embedding_dims)
        # Store a very low-value chunk
        low_id = store_memory(
            l3_conn,
            content="low value trash",
            room_name="test",
            embedding=embedder.encode_single("low value trash"),
            config=l3_config,
            vm_weights={"goal_alignment": 0.0, "recency": 0.0, "source_credibility": 0.0},
        )
        # Store many more to trigger eviction
        _ingest_n_chunks(l3_conn, l3_config, 10_000)
        budget_curated_eviction(l3_conn, l3_config)

        row = l3_conn.execute(
            "SELECT resolution FROM chunk WHERE chunk_id = ?", (low_id,)
        ).fetchone()
        assert row is not None
        assert row[0] != "FP32"
