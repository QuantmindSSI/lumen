"""Integration tests for WearAwareBatcher."""

import sqlite3

import pytest

from lumen.data.schema import init_db
from lumen.force.mnemonic.consolidation import run_consolidation_pass
from lumen.force.mnemonic.forgetting_l1_decay import ebbinghaus_decay
from lumen.force.mnemonic.forgetting_l3_budget import budget_curated_eviction
from lumen.sovereign.wear import WearAwareBatcher


@pytest.fixture
def memory_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()


class TestWearAwareBatcher:
    def test_flush_sync_commits_batch(self, memory_db):
        batcher = WearAwareBatcher(memory_db)
        batcher.queue.append(("INSERT INTO room(name, room_type) VALUES (?, ?)", ("alpha", "domain")))
        batcher.queue.append(("INSERT INTO room(name, room_type) VALUES (?, ?)", ("beta", "domain")))
        assert batcher.queue
        batcher.flush_sync(list(batcher.queue))
        batcher.queue.clear()
        rows = memory_db.execute("SELECT name FROM room ORDER BY name").fetchall()
        assert [r["name"] for r in rows] == ["alpha", "beta"]


class TestEbbinghausDecayBatcher:
    def test_stages_updates_without_immediate_flush(self, memory_db):
        memory_db.execute("INSERT INTO room(name, room_type) VALUES (?, ?)", ("room1", "domain"))
        room_id = memory_db.execute("SELECT room_id FROM room WHERE name = ?", ("room1",)).fetchone()[0]
        memory_db.execute("INSERT INTO locus(room_id, name) VALUES (?, ?)", (room_id, "l1"))
        locus_id = memory_db.execute("SELECT locus_id FROM locus WHERE name = ?", ("l1",)).fetchone()[0]
        for i in range(5):
            memory_db.execute(
                "INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score, resolution, created_at) VALUES (?,?,?,?,?,?,unixepoch() - 86400)",
                (locus_id, room_id, f"content {i}", f"hash{i}", 1.0, "FP32"),
            )
        memory_db.commit()
        batcher = WearAwareBatcher(memory_db)
        ebbinghaus_decay(memory_db, user_half_life_days=0.1, batcher=batcher)
        # Before flush, vm_score should still be original
        row = memory_db.execute("SELECT vm_score FROM chunk WHERE content_hash = ?", ("hash0",)).fetchone()
        assert row["vm_score"] == 1.0
        assert len(batcher.queue) == 5
        # Flush
        batcher.flush_sync(list(batcher.queue))
        batcher.queue.clear()
        row = memory_db.execute("SELECT vm_score FROM chunk WHERE content_hash = ?", ("hash0",)).fetchone()
        assert row["vm_score"] == 0.0


class TestBudgetEvictionBatcher:
    def test_stages_resolution_updates(self, memory_db):
        memory_db.execute("INSERT INTO room(name, room_type) VALUES (?, ?)", ("room1", "domain"))
        room_id = memory_db.execute("SELECT room_id FROM room WHERE name = ?", ("room1",)).fetchone()[0]
        memory_db.execute("INSERT INTO locus(room_id, name) VALUES (?, ?)", (room_id, "l1"))
        locus_id = memory_db.execute("SELECT locus_id FROM locus WHERE name = ?", ("l1",)).fetchone()[0]
        memory_db.execute(
            "INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score, resolution, optical_level) VALUES (?,?,?,?,?,?,?)",
            (locus_id, room_id, "content", "hash1", 0.01, "FP32", 0),
        )
        memory_db.commit()
        batcher = WearAwareBatcher(memory_db)

        class FakeConfig:
            memory_limit_mb = 1

        budget_curated_eviction(memory_db, FakeConfig(), target_ram_mb=0, batcher=batcher)
        # Before flush
        row = memory_db.execute("SELECT resolution, optical_level FROM chunk WHERE chunk_id = 1").fetchone()
        assert row["resolution"] == "FP32"
        assert row["optical_level"] == 0
        assert len(batcher.queue) > 0
        batcher.flush_sync(list(batcher.queue))
        batcher.queue.clear()
        row = memory_db.execute("SELECT resolution, optical_level FROM chunk WHERE chunk_id = 1").fetchone()
        assert row["resolution"] == "FP16"
        assert row["optical_level"] == 1


class TestConsolidationBatcher:
    def test_stages_dedup_updates(self, memory_db, test_config):
        # Insert duplicate chunks
        memory_db.execute("INSERT INTO room(name, room_type) VALUES (?, ?)", ("room1", "domain"))
        room_id = memory_db.execute("SELECT room_id FROM room WHERE name = ?", ("room1",)).fetchone()[0]
        memory_db.execute("INSERT INTO locus(room_id, name) VALUES (?, ?)", (room_id, "l1"))
        locus_id = memory_db.execute("SELECT locus_id FROM locus WHERE name = ?", ("l1",)).fetchone()[0]
        for i in range(3):
            memory_db.execute(
                "INSERT INTO chunk(locus_id, room_id, content, content_hash, vm_score, resolution) VALUES (?,?,?,?,?,?)",
                (locus_id, room_id, "dup", "samehash", 0.5 if i == 0 else 0.3, "FP32"),
            )
        memory_db.commit()
        batcher = WearAwareBatcher(memory_db)
        ops = run_consolidation_pass(test_config, batcher=batcher)
        # run_consolidation_pass flushes dedup batch itself, so values should be updated
        assert ops > 0
        winner = memory_db.execute(
            "SELECT chunk_id FROM chunk WHERE content_hash = ? AND valid_to IS NULL", ("samehash",)
        ).fetchone()
        assert winner is not None
        losers = memory_db.execute(
            "SELECT chunk_id FROM chunk WHERE content_hash = ? AND valid_to IS NOT NULL", ("samehash",)
        ).fetchall()
        assert len(losers) == 2
