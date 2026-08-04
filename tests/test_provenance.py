import sqlite3
import time

import pytest

from lumen.data.schema import init_db
from lumen.force.mnemonic.provenance import (
    create_provenance,
    get_effective_fact,
    supersede_chunk,
    ProvenanceRecord,
)


@pytest.fixture
def db_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    _setup_test_data(conn)
    yield conn
    conn.close()


def _setup_test_data(conn):
    conn.execute("INSERT INTO room(room_id, name, room_type) VALUES (1, 'prov_room', 'domain')")
    conn.execute("INSERT INTO locus(locus_id, room_id, name) VALUES (1, 1, 'prov_locus')")
    conn.commit()


def test_create_provenance_returns_int(db_conn):
    chunk_id = 1
    db_conn.execute(
        "INSERT INTO chunk(chunk_id, room_id, locus_id, content, content_hash) "
        "VALUES (?, 1, 1, 'test', 'hash1')", (chunk_id,)
    )
    db_conn.commit()
    prov_id = create_provenance(db_conn, chunk_id, "user_input", source_ref="test/1")
    assert isinstance(prov_id, int)
    assert prov_id > 0


def test_create_provenance_stores_record(db_conn):
    chunk_id = 2
    db_conn.execute(
        "INSERT INTO chunk(chunk_id, room_id, locus_id, content, content_hash) "
        "VALUES (?, 1, 1, 'test2', 'hash2')", (chunk_id,)
    )
    db_conn.commit()
    prov_id = create_provenance(
        db_conn, chunk_id, "agent_reasoning",
        source_ref="ag/1", confidence=0.8,
        extraction_method="rule"
    )
    row = db_conn.execute(
        "SELECT * FROM provenance WHERE provenance_id = ?", (prov_id,)
    ).fetchone()
    assert row["chunk_id"] == chunk_id
    assert row["source_type"] == "agent_reasoning"
    assert row["confidence"] == 0.8


def test_supersede_chunk_sets_valid_to(db_conn):
    now = int(time.time())
    db_conn.execute(
        "INSERT INTO chunk(chunk_id, room_id, locus_id, content, content_hash, valid_to) "
        "VALUES (10, 1, 1, 'old', 'h10', NULL)"
    )
    db_conn.execute(
        "INSERT INTO chunk(chunk_id, room_id, locus_id, content, content_hash, valid_to) "
        "VALUES (11, 1, 1, 'new', 'h11', NULL)"
    )
    db_conn.commit()
    supersede_chunk(db_conn, old_chunk_id=10, new_chunk_id=11)
    row = db_conn.execute(
        "SELECT valid_to, superseded_by FROM chunk WHERE chunk_id = ?", (10,)
    ).fetchone()
    assert row["valid_to"] is not None
    assert row["superseded_by"] == 11


def test_get_effective_fact_returns_none_for_missing(db_conn):
    result = get_effective_fact(db_conn, "nonexistent_hash")
    assert result is None


def test_get_effective_fact_returns_valid_chunk(db_conn):
    db_conn.execute(
        "INSERT INTO chunk(chunk_id, room_id, locus_id, content, content_hash, valid_from, valid_to) "
        "VALUES (20, 1, 1, 'valid content', 'ab12', unixepoch(), NULL)"
    )
    db_conn.commit()
    create_provenance(db_conn, 20, "user_input")
    result = get_effective_fact(db_conn, "ab12")
    assert result is not None
    assert result["chunk_id"] == 20


def test_get_effective_fact_excludes_superseded_chunk(db_conn):
    db_conn.execute(
        "INSERT INTO chunk(chunk_id, room_id, locus_id, content, content_hash, valid_from, valid_to) "
        "VALUES (30, 1, 1, 'old', 'cd34', unixepoch(), NULL)"
    )
    db_conn.execute(
        "INSERT INTO chunk(chunk_id, room_id, locus_id, content, content_hash, valid_from, valid_to) "
        "VALUES (31, 1, 1, 'new', 'cd34', unixepoch(), NULL)"
    )
    db_conn.commit()
    create_provenance(db_conn, 30, "user_input")
    create_provenance(db_conn, 31, "user_input")
    supersede_chunk(db_conn, old_chunk_id=30, new_chunk_id=31)
    result = get_effective_fact(db_conn, "cd34")
    assert result is not None
    assert result["chunk_id"] == 31