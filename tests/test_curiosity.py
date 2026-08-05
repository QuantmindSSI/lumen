"""Tests for D13: Curiosity-Driven Exploration Scheduler."""

import sqlite3

import pytest

from lumen.curiosity import curiosity_probe, curiosity_score
from lumen.data.schema import init_db


@pytest.fixture
def memory_db():
    """In-memory SQLite connection with Lumen schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()


def _insert_room(conn: sqlite3.Connection, name: str) -> int:
    cursor = conn.execute("INSERT INTO room(name) VALUES (?)", (name,))
    return cursor.lastrowid


def _insert_chunk(
    conn: sqlite3.Connection,
    room_id: int,
    content: str,
    content_hash: str,
    vm_score: float = 0.5,
    access_count: int = 0,
    last_access_at: int | None = None,
    created_at: int = 1,
    valid_to: int | None = None,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO chunk(
            room_id, content, content_hash, vm_score, access_count,
            last_access_at, created_at, valid_to
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            room_id,
            content,
            content_hash,
            vm_score,
            access_count,
            last_access_at,
            created_at,
            valid_to,
        ),
    )
    return cursor.lastrowid


def test_probe_empty_db(memory_db):
    """An empty database should return an empty list."""
    assert curiosity_probe(memory_db) == []


def test_probe_returns_oldest_and_least_accessed(memory_db):
    """Signal 1 (old) and Signal 3 (unseen) should surface correct chunks."""
    rid = _insert_room(memory_db, "test_room")

    cid1 = _insert_chunk(
        memory_db,
        rid,
        "old memory",
        "h1",
        vm_score=0.5,
        access_count=0,
        last_access_at=100,
        created_at=1,
    )
    cid2 = _insert_chunk(
        memory_db,
        rid,
        "older memory",
        "h2",
        vm_score=0.5,
        access_count=0,
        last_access_at=50,
        created_at=2,
    )
    cid3 = _insert_chunk(
        memory_db,
        rid,
        "recent memory",
        "h3",
        vm_score=0.5,
        access_count=5,
        last_access_at=10000,
        created_at=3,
    )

    result = curiosity_probe(memory_db, limit=5)
    assert cid2 in result  # oldest access
    assert cid1 in result  # second oldest
    assert (
        cid3 in result
    )  # least unseen in terms of access_count? Actually cid3 has access_count=5, while cid1/cid2 have 0
    # cid3 might be in result due to signal variance, but it's less likely to be first
    # With limit=5, all 3 should appear because there are only 3 valid chunks
    assert set(result) == {cid1, cid2, cid3}


def test_probe_limits_results(memory_db):
    """Respect the limit parameter."""
    rid = _insert_room(memory_db, "limit_room")
    for i in range(10):
        _insert_chunk(
            memory_db,
            rid,
            f"chunk {i}",
            f"h{i}",
            vm_score=0.5,
            access_count=0,
            last_access_at=i * 10,
            created_at=i,
        )

    result = curiosity_probe(memory_db, limit=3)
    assert len(result) == 3


def test_probe_skips_invalid_chunks(memory_db):
    """Superseded/invalid chunks (valid_to IS NOT NULL) must not appear."""
    rid = _insert_room(memory_db, "valid_room")

    valid = _insert_chunk(memory_db, rid, "valid", "hv", valid_to=None, last_access_at=10)
    _insert_chunk(memory_db, rid, "invalid", "hi", valid_to=999, last_access_at=5)

    result = curiosity_probe(memory_db, limit=5)
    assert result == [valid]


def test_probe_deduplicates(memory_db):
    """The same chunk_id should never appear twice in the output."""
    rid = _insert_room(memory_db, "dedup_room")

    cid = _insert_chunk(
        memory_db,
        rid,
        "lonely",
        "h1",
        vm_score=0.5,
        access_count=0,
        last_access_at=None,
        created_at=1,
    )

    result = curiosity_probe(memory_db, limit=5)
    assert result == [cid]
    assert len(result) == len(set(result))


def test_probe_variance_signal(memory_db):
    """Signal 2 should surface chunks from high-variance rooms."""
    rid = _insert_room(memory_db, "variance_room")

    # Far outliers to create a high spread
    cid_low = _insert_chunk(
        memory_db,
        rid,
        "low vm",
        "hl",
        vm_score=0.1,
        access_count=10,
        last_access_at=1000,
        created_at=1,
    )
    cid_high = _insert_chunk(
        memory_db,
        rid,
        "high vm",
        "hh",
        vm_score=0.9,
        access_count=10,
        last_access_at=1000,
        created_at=2,
    )
    # Middle value — not an outlier
    _insert_chunk(
        memory_db,
        rid,
        "mid vm",
        "hm",
        vm_score=0.5,
        access_count=10,
        last_access_at=1000,
        created_at=3,
    )

    result = curiosity_probe(memory_db, limit=5)
    # Outliers should appear because they drive room variance
    assert cid_low in result
    assert cid_high in result


def test_score_invalid_chunk(memory_db):
    """Invalid or missing chunks score 0.0."""
    rid = _insert_room(memory_db, "score_room")
    invalid = _insert_chunk(memory_db, rid, "old", "h1", valid_to=999)

    assert curiosity_score(memory_db, invalid) == 0.0
    assert curiosity_score(memory_db, 99999) == 0.0


def test_score_valid_chunk_is_positive(memory_db):
    """A valid, never-accessed chunk should have a positive curiosity score."""
    rid = _insert_room(memory_db, "score_room")
    cid = _insert_chunk(
        memory_db,
        rid,
        "untouched",
        "h1",
        vm_score=0.5,
        access_count=0,
        last_access_at=None,
        created_at=1,
    )

    score = curiosity_score(memory_db, cid)
    assert 0.0 < score <= 1.0


def test_score_decreases_with_access(memory_db):
    """Higher access_count should lower the curiosity score."""
    rid = _insert_room(memory_db, "score_room")

    cid_unseen = _insert_chunk(
        memory_db,
        rid,
        "unseen",
        "h1",
        vm_score=0.5,
        access_count=0,
        last_access_at=1000,
        created_at=1,
    )
    cid_popular = _insert_chunk(
        memory_db,
        rid,
        "popular",
        "h2",
        vm_score=0.5,
        access_count=100,
        last_access_at=1000,
        created_at=2,
    )

    assert curiosity_score(memory_db, cid_unseen) > curiosity_score(memory_db, cid_popular)


def test_score_older_access_is_higher(memory_db):
    """An older last_access_at should yield a higher curiosity score."""
    import time

    now = int(time.time())
    rid = _insert_room(memory_db, "score_room")

    cid_old = _insert_chunk(
        memory_db,
        rid,
        "old",
        "h1",
        vm_score=0.5,
        access_count=0,
        last_access_at=now - 2000,
        created_at=1,
    )
    cid_recent = _insert_chunk(
        memory_db,
        rid,
        "recent",
        "h2",
        vm_score=0.5,
        access_count=0,
        last_access_at=now - 1000,
        created_at=2,
    )

    assert curiosity_score(memory_db, cid_old) > curiosity_score(memory_db, cid_recent)
