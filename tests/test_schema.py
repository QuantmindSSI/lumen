import sqlite3

import pytest

from lumen.config import LumenConfig
from lumen.data.schema import ensure_schema, get_connection, init_db


@pytest.fixture
def temp_config(tmp_path):
    return LumenConfig(
        device="generic",
        vector_index="sqlite-vec",
        store_path=str(tmp_path / ".lumen"),
    )


def test_get_connection_returns_sqlite_connection(temp_config):
    conn = get_connection(temp_config)
    assert isinstance(conn, sqlite3.Connection)
    conn.close()


def test_get_connection_sets_row_factory(temp_config):
    conn = get_connection(temp_config)
    row = conn.execute("SELECT 1 AS val").fetchone()
    assert row["val"] == 1
    conn.close()


def test_get_connection_db_file_exists(temp_config):
    conn = get_connection(temp_config)
    db_path = temp_config.store_path / "lumen.db"
    assert db_path.exists()
    conn.close()


def test_ensure_schema_creates_tables(temp_config):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    expected = {"room", "locus", "chunk", "provenance", "feedback_log",
                "user_profile", "event_buffer_meta", "goals", "epistemic_state"}
    assert expected <= tables
    conn.close()


def test_ensure_schema_is_idempotent(temp_config):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    version1 = conn.execute("PRAGMA user_version").fetchone()[0]
    ensure_schema(conn)
    version2 = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version1 == version2
    conn.close()


def test_init_db_sets_user_version(monkeypatch, tmp_path):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version >= 1
    conn.close()
