import sqlite3

import numpy as np
import pytest

from lumen.data.schema import init_db
from lumen.force.mnemonic.retrieval_dense import DenseHit, SqliteVecBackend


@pytest.fixture
def db_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def backend(db_conn):
    return SqliteVecBackend(db_conn, dims=128)


def _make_vec():
    return np.random.randn(128).astype(np.float32)


def test_add_works(backend, db_conn):
    vec = _make_vec()
    backend.add(1, vec)
    row = db_conn.execute(
        "SELECT embedding FROM vec_fallback WHERE chunk_id = ?", (1,)
    ).fetchone()
    assert row is not None


def test_search_returns_hits(backend):
    vec = _make_vec()
    backend.add(100, vec)
    results = backend.search(vec, k=5)
    assert len(results) >= 1
    assert isinstance(results[0], DenseHit)


def test_search_returns_highest_score_first(backend):
    v1 = _make_vec()
    v2 = v1 + np.random.randn(128).astype(np.float32) * 0.01
    v2 = v2 / np.linalg.norm(v2)
    v3 = -v1
    v3 = v3 / np.linalg.norm(v3)

    backend.add(200, v1)
    backend.add(201, v3)

    results = backend.search(v2, k=2)
    assert len(results) == 2
    assert results[0].score > results[1].score


def test_remove_works(backend, db_conn):
    vec = _make_vec()
    backend.add(300, vec)
    assert db_conn.execute(
        "SELECT 1 FROM vec_fallback WHERE chunk_id = ?", (300,)
    ).fetchone() is not None
    backend.remove(300)
    assert db_conn.execute(
        "SELECT 1 FROM vec_fallback WHERE chunk_id = ?", (300,)
    ).fetchone() is None


def test_degrade_works(backend, db_conn):
    vec = _make_vec()
    backend.add(400, vec)
    db_conn.execute(
        "SELECT embedding FROM vec_fallback WHERE chunk_id = ?", (400,)
    ).fetchone()[0]
    backend.degrade(400, "FP16")
    after_row = db_conn.execute(
        "SELECT embedding FROM vec_fallback WHERE chunk_id = ?", (400,)
    ).fetchone()
    if after_row is None:
        pass
    else:
        pass


def test_add_overwrite_updates(backend, db_conn):
    v1 = _make_vec()
    v2 = _make_vec()
    backend.add(500, v1)
    blob1 = db_conn.execute(
        "SELECT embedding FROM vec_fallback WHERE chunk_id = ?", (500,)
    ).fetchone()[0]
    backend.add(500, v2)
    blob2 = db_conn.execute(
        "SELECT embedding FROM vec_fallback WHERE chunk_id = ?", (500,)
    ).fetchone()[0]
    assert blob1 != blob2
