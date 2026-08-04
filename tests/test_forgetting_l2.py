import sqlite3

import numpy as np
import pytest

from lumen.config import LumenConfig
from lumen.data.schema import init_db
from lumen.force.mnemonic.retrieval_dense import SqliteVecBackend
from lumen.force.mnemonic.forgetting_l2_interference import (
    check_locus_interference,
    INTERFERENCE_THRESHOLD,
)


@pytest.fixture
def db_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    conn.execute("INSERT INTO room(room_id, name, room_type) VALUES (1, 'interf', 'domain')")
    conn.execute("INSERT INTO locus(locus_id, room_id, name) VALUES (1, 1, 'l1')")
    conn.commit()
    backend = SqliteVecBackend(conn, 384)
    yield conn
    conn.close()


def _seed_chunk(conn, chunk_id, locus_id, room_id, content, vm_score=0.8):
    conn.execute(
        """INSERT INTO chunk (chunk_id, locus_id, room_id, content, content_hash, vm_score)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (chunk_id, locus_id, room_id, content, f"hash_{chunk_id}", vm_score),
    )
    conn.commit()


def _seed_embedding(conn, chunk_id, vector):
    blob = vector.astype(np.float32).tobytes()
    conn.execute(
        "INSERT OR REPLACE INTO vec_fallback(chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, blob),
    )
    conn.commit()


def test_check_locus_interference_no_residents(db_conn):
    _seed_chunk(db_conn, 1, 1, 1, "test", 0.8)
    emb = np.random.randn(384).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    weakened = check_locus_interference(db_conn, room_id=1, locus_id=1,
                                         new_chunk_id=1, new_embedding=emb)
    assert weakened == 0


def test_check_locus_interference_below_threshold_no_weakening(db_conn):
    _seed_chunk(db_conn, 10, 1, 1, "old", 0.8)
    old_emb = np.random.randn(384).astype(np.float32)
    old_emb = old_emb / np.linalg.norm(old_emb)
    _seed_embedding(db_conn, 10, old_emb)

    new_emb = -old_emb
    new_emb = new_emb / np.linalg.norm(new_emb)

    weakened = check_locus_interference(db_conn, room_id=1, locus_id=1,
                                         new_chunk_id=20, new_embedding=new_emb)
    assert weakened == 0
    row = db_conn.execute("SELECT vm_score FROM chunk WHERE chunk_id = ?",
                          (10,)).fetchone()
    assert row[0] == 0.8


def test_check_locus_interference_above_threshold_weakens(db_conn):
    _seed_chunk(db_conn, 100, 1, 1, "old", 0.8)
    old_emb = np.random.randn(384).astype(np.float32)
    old_emb = old_emb / np.linalg.norm(old_emb)
    _seed_embedding(db_conn, 100, old_emb)

    new_emb = old_emb + np.random.randn(384).astype(np.float32) * 0.01
    new_emb = new_emb / np.linalg.norm(new_emb)

    weakened = check_locus_interference(db_conn, room_id=1, locus_id=1,
                                         new_chunk_id=200, new_embedding=new_emb)
    actual_sim = float(np.dot(new_emb, old_emb) /
                       (np.linalg.norm(new_emb) * np.linalg.norm(old_emb) + 1e-8))
    if actual_sim <= INTERFERENCE_THRESHOLD:
        assert weakened == 0
    else:
        assert weakened == 1
        row = db_conn.execute("SELECT vm_score FROM chunk WHERE chunk_id = ?",
                              (100,)).fetchone()
        assert row[0] < 0.8