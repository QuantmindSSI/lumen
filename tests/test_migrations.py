import sqlite3

import pytest

from lumen.data.migrate import CURRENT_SCHEMA_VERSION, get_current_version, migrate
from lumen.data.schema import init_db


@pytest.fixture
def fresh_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS room (room_id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, created_at INTEGER DEFAULT (unixepoch()), last_entry_at INTEGER, locus_count INTEGER DEFAULT 0, room_type TEXT, topological_order REAL DEFAULT 0.0)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS locus (locus_id INTEGER PRIMARY KEY, room_id INTEGER NOT NULL REFERENCES room(room_id) ON DELETE CASCADE, name TEXT NOT NULL, description TEXT, vector_mean BLOB, created_at INTEGER DEFAULT (unixepoch()), access_count INTEGER DEFAULT 0, last_access_at INTEGER, UNIQUE(room_id, name))"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunk (chunk_id INTEGER PRIMARY KEY, locus_id INTEGER REFERENCES locus(locus_id) ON DELETE SET NULL, room_id INTEGER NOT NULL REFERENCES room(room_id) ON DELETE CASCADE, content TEXT NOT NULL, content_hash TEXT NOT NULL, created_at INTEGER DEFAULT (unixepoch()), valid_from INTEGER DEFAULT (unixepoch()), valid_to INTEGER, superseded_by INTEGER REFERENCES chunk(chunk_id), resolution TEXT DEFAULT 'FP32', vm_score REAL DEFAULT 0.5, vm_factors BLOB, access_count INTEGER DEFAULT 0, last_access_at INTEGER, optical_level INTEGER DEFAULT 0, provenance_root INTEGER REFERENCES provenance(provenance_id))"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS provenance (provenance_id INTEGER PRIMARY KEY, chunk_id INTEGER NOT NULL REFERENCES chunk(chunk_id) ON DELETE CASCADE, source_type TEXT, source_ref TEXT, confidence REAL DEFAULT 1.0, extraction_method TEXT, parent_provenance INTEGER REFERENCES provenance(provenance_id))"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS feedback_log (feedback_id INTEGER PRIMARY KEY, chunk_id INTEGER NOT NULL REFERENCES chunk(chunk_id), user_id TEXT DEFAULT 'default', positive INTEGER NOT NULL CHECK(positive IN (0,1)), feedback_type TEXT DEFAULT 'implicit', session_id TEXT, turn_id INTEGER, created_at INTEGER DEFAULT (unixepoch()))"
    )
    conn.execute("PRAGMA user_version = 0")
    yield conn
    conn.close()


@pytest.fixture
def db_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()


def test_get_current_version_returns_int(fresh_conn):
    version = get_current_version(fresh_conn)
    assert isinstance(version, int)
    assert version == 0


def test_migrate_advances_from_version_one_to_two(fresh_conn):
    fresh_conn.execute("PRAGMA user_version = 1")
    assert get_current_version(fresh_conn) == 1
    migrate(fresh_conn, target_version=CURRENT_SCHEMA_VERSION)
    assert get_current_version(fresh_conn) == CURRENT_SCHEMA_VERSION


def test_migrate_is_noop_when_at_target(db_conn):
    version_before = get_current_version(db_conn)
    migrate(db_conn, target_version=version_before)
    assert get_current_version(db_conn) == version_before
    migrate(db_conn)
    assert get_current_version(db_conn) == version_before


def test_migrate_target_zero_is_noop(fresh_conn):
    migrate(fresh_conn, target_version=0)
    assert get_current_version(fresh_conn) == 0
