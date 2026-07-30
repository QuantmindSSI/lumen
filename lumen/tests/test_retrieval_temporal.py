"""Tests for D7: Temporal Search & Bi-Temporal Query Engine."""

import hashlib
import sqlite3
from datetime import datetime, timezone

import pytest

from lumen.data.schema import init_db
from lumen.force.mnemonic.retrieval_temporal import (
    temporal_point_query,
    walk_supersession_chain,
)


@pytest.fixture
def memory_db():
    """Yield an in-memory SQLite connection with the Lumen schema applied."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()


def _insert_room(conn: sqlite3.Connection, name: str = "test-room") -> int:
    cur = conn.execute("INSERT INTO room (name) VALUES (?)", (name,))
    return cur.lastrowid


def _insert_chunk(
    conn: sqlite3.Connection,
    room_id: int,
    content: str,
    valid_from: int,
    valid_to: int | None = None,
    superseded_by: int | None = None,
) -> int:
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    cur = conn.execute(
        """
        INSERT INTO chunk (room_id, content, content_hash, valid_from, valid_to, superseded_by)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (room_id, content, content_hash, valid_from, valid_to, superseded_by),
    )
    return cur.lastrowid


def test_temporal_point_query_as_of_unix(memory_db):
    """Point-in-time queries must respect valid_from / valid_to windows."""
    room_id = _insert_room(memory_db)

    t_mon = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp())
    t_tue = int(datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp())
    t_wed = int(datetime(2024, 1, 3, tzinfo=timezone.utc).timestamp())

    # Chunk valid Monday through Tuesday
    cid_a = _insert_chunk(memory_db, room_id, "Project alpha started", t_mon, t_tue)
    # Chunk valid Tuesday onwards (still valid)
    cid_b = _insert_chunk(memory_db, room_id, "Project alpha delayed", t_tue, None)
    # Unrelated chunk, always valid
    _insert_chunk(memory_db, room_id, "Project beta is fine", t_mon, None)

    memory_db.commit()

    # Monday evening -> only A is valid
    hits = temporal_point_query(memory_db, ["alpha"], as_of_unix=t_mon + 100)
    assert len(hits) == 1
    assert hits[0].chunk_id == cid_a

    # Wednesday -> only B is valid
    hits = temporal_point_query(memory_db, ["alpha"], as_of_unix=t_wed)
    assert len(hits) == 1
    assert hits[0].chunk_id == cid_b

    # No as_of_unix -> only currently valid (B)
    hits = temporal_point_query(memory_db, ["alpha"])
    assert len(hits) == 1
    assert hits[0].chunk_id == cid_b


def test_temporal_point_query_empty_keywords(memory_db):
    """Empty keyword list must short-circuit to an empty result."""
    room_id = _insert_room(memory_db)
    _insert_chunk(memory_db, room_id, "something", 0)
    memory_db.commit()
    assert temporal_point_query(memory_db, []) == []


def test_temporal_point_query_multiple_keywords(memory_db):
    """Multiple keywords should be ORed together."""
    room_id = _insert_room(memory_db)
    t0 = 1000
    cid = _insert_chunk(memory_db, room_id, "The quick brown fox", t0)
    memory_db.commit()

    hits = temporal_point_query(memory_db, ["quick", "fox"], as_of_unix=t0)
    assert len(hits) == 1
    assert hits[0].chunk_id == cid


def test_temporal_point_query_include_superseded(memory_db):
    """When include_superseded=True the full chain must be reconstructed."""
    room_id = _insert_room(memory_db)
    t0 = 1000
    t1 = 2000

    cid_v1 = _insert_chunk(memory_db, room_id, "Decision: use Postgres", t0, t1)
    cid_v2 = _insert_chunk(memory_db, room_id, "Decision: use SQLite", t1, None)
    memory_db.execute(
        "UPDATE chunk SET superseded_by = ? WHERE chunk_id = ?",
        (cid_v2, cid_v1),
    )
    memory_db.commit()

    hits = temporal_point_query(memory_db, ["Postgres"], as_of_unix=t0, include_superseded=True)
    assert len(hits) == 2
    assert hits[0].chunk_id == cid_v1
    assert hits[1].chunk_id == cid_v2


def test_walk_supersession_chain_linear(memory_db):
    """A linear chain of three versions must be traversed in order."""
    room_id = _insert_room(memory_db)
    t0 = 1000
    t1 = 2000
    t2 = 3000

    cid_v1 = _insert_chunk(memory_db, room_id, "Belief v1", t0, t1)
    cid_v2 = _insert_chunk(memory_db, room_id, "Belief v2", t1, t2)
    cid_v3 = _insert_chunk(memory_db, room_id, "Belief v3", t2, None)

    memory_db.execute(
        "UPDATE chunk SET superseded_by = ? WHERE chunk_id = ?",
        (cid_v2, cid_v1),
    )
    memory_db.execute(
        "UPDATE chunk SET superseded_by = ? WHERE chunk_id = ?",
        (cid_v3, cid_v2),
    )
    memory_db.commit()

    chain = walk_supersession_chain(memory_db, cid_v1)
    assert len(chain) == 3
    assert [c.chunk_id for c in chain] == [cid_v1, cid_v2, cid_v3]
    assert chain[0].content == "Belief v1"
    assert chain[1].content == "Belief v2"
    assert chain[2].content == "Belief v3"


def test_walk_supersession_chain_no_supersession(memory_db):
    """A chunk with no supersession must return a singleton list."""
    room_id = _insert_room(memory_db)
    cid = _insert_chunk(memory_db, room_id, "Standalone", 0, None)
    memory_db.commit()

    chain = walk_supersession_chain(memory_db, cid)
    assert len(chain) == 1
    assert chain[0].chunk_id == cid
    assert chain[0].superseded_by is None


def test_walk_supersession_cycle_guard(memory_db):
    """A malformed cycle must not cause an infinite loop."""
    room_id = _insert_room(memory_db)
    cid_a = _insert_chunk(memory_db, room_id, "A", 0, None)
    cid_b = _insert_chunk(memory_db, room_id, "B", 1, None)

    memory_db.execute(
        "UPDATE chunk SET superseded_by = ? WHERE chunk_id = ?",
        (cid_b, cid_a),
    )
    memory_db.execute(
        "UPDATE chunk SET superseded_by = ? WHERE chunk_id = ?",
        (cid_a, cid_b),
    )
    memory_db.commit()

    chain = walk_supersession_chain(memory_db, cid_a)
    assert len(chain) == 2
    assert chain[0].chunk_id == cid_a
    assert chain[1].chunk_id == cid_b
