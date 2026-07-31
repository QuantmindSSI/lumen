"""Schema initialisation helpers."""

import sqlite3
from pathlib import Path

from lumen.data.migrate import migrate

_SQL_PATH = Path(__file__).with_suffix(".sql")


def init_db(conn: sqlite3.Connection) -> None:
    """Execute the canonical schema SQL against an open connection."""
    sql = _SQL_PATH.read_text(encoding="utf-8")
    conn.executescript(sql)
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current == 0:
        conn.execute("PRAGMA user_version = 1")
    migrate(conn)


def get_connection(config) -> sqlite3.Connection:
    """Return a SQLite connection initialised with the Lumen schema and optimised pragmas."""
    from lumen.config import LumenConfig

    cfg: LumenConfig = config
    cfg.store_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cfg.store_path / "lumen.db"), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA mmap_size = 268435456")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Check PRAGMA user_version and initialise schema if database is empty."""
    user_version = conn.execute("PRAGMA user_version").fetchone()[0]
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    if user_version == 0 and not tables:
        init_db(conn)
        # init_db may have already advanced user_version via migrate
        if conn.execute("PRAGMA user_version").fetchone()[0] == 0:
            conn.execute("PRAGMA user_version = 1")
            conn.commit()
    migrate(conn)
