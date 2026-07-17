"""Schema initialisation helpers."""

import sqlite3
from pathlib import Path

_SQL_PATH = Path(__file__).with_suffix(".sql")


def init_db(conn: sqlite3.Connection) -> None:
    """Execute the canonical schema SQL against an open connection."""
    sql = _SQL_PATH.read_text(encoding="utf-8")
    conn.executescript(sql)


def get_connection(config) -> sqlite3.Connection:
    """Return a SQLite connection initialised with the Lumen schema."""
    from lumen.config import LumenConfig

    cfg: LumenConfig = config
    cfg.store_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cfg.store_path / "lumen.db"))
    conn.execute("PRAGMA foreign_keys = ON")
    init_db(conn)
    return conn
