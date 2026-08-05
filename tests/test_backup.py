import sqlite3
from pathlib import Path

import pytest

from lumen.config import LumenConfig
from lumen.data.backup import backup_database
from lumen.data.schema import get_connection


@pytest.fixture
def db_conn(tmp_path):
    config = LumenConfig(
        device="generic",
        vector_index="sqlite-vec",
        store_path=str(tmp_path / ".lumen"),
    )
    conn = get_connection(config)
    conn.execute(
        "INSERT INTO room(name, room_type) VALUES ('backup_test', 'domain')"
    )
    conn.commit()
    yield conn
    conn.close()


def test_backup_database_creates_file(db_conn, tmp_path):
    db_path = Path(db_conn.execute("PRAGMA database_list").fetchone()[2])
    result = backup_database(db_path)
    assert result.exists()
    assert result.suffix == ".db"


def test_backup_database_file_has_data(db_conn, tmp_path):
    db_path = Path(db_conn.execute("PRAGMA database_list").fetchone()[2])
    result = backup_database(db_path)
    backup_conn = sqlite3.connect(str(result))
    row = backup_conn.execute(
        "SELECT name FROM room WHERE name = 'backup_test'"
    ).fetchone()
    assert row is not None
    assert row[0] == "backup_test"
    backup_conn.close()


def test_backup_database_custom_bakcup_dir(db_conn, tmp_path):
    db_path = Path(db_conn.execute("PRAGMA database_list").fetchone()[2])
    custom_dir = tmp_path / "custom_backups"
    result = backup_database(db_path, backup_dir=custom_dir)
    assert result.parent == custom_dir
    assert result.exists()
