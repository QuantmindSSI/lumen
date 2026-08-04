"""Schema initialisation helpers."""

import os
import sqlite3
import stat
from pathlib import Path

from lumen.data.migrate import migrate
from lumen.logging import get_console_logger

logger = get_console_logger(__name__)
_SQL_PATH = Path(__file__).with_suffix(".sql")


def _enforce_permissions(store_path: Path) -> None:
    """Ensure ~/.lumen is 700 and database files are 600."""
    try:
        os.chmod(store_path, stat.S_IRWXU)  # 0o700
        for child in store_path.iterdir():
            if child.suffix in (".db", ".db-wal", ".db-shm", ".toml", ".jsonl"):
                os.chmod(child, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    except OSError as exc:
        logger.warning("permission_enforcement_failed", path=str(store_path), error=str(exc))


def init_db(conn: sqlite3.Connection) -> None:
    """Execute the canonical schema SQL against an open connection."""
    sql = _SQL_PATH.read_text(encoding="utf-8")
    conn.executescript(sql)
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current == 0:
        conn.execute("PRAGMA user_version = 1")
    migrate(conn)


def get_connection(config) -> sqlite3.Connection:
    """Return a SQLite connection initialised with the Lumen schema and optimised pragmas.

    NOTE: Encryption-at-rest is not yet implemented. Use OS-level disk encryption
    (FileVault, BitLocker, LUKS) as a stopgap for sensitive deployments.
    """
    from lumen.config import LumenConfig

    cfg: LumenConfig = config
    cfg.store_path.mkdir(parents=True, exist_ok=True)
    db_path = cfg.store_path / "lumen.db"

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    logger.warning(
        "encryption_at_rest_disabled",
        recommendation="Use OS-level disk encryption (FileVault, BitLocker, LUKS) for production",
    )

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA mmap_size = 268435456")
    ensure_schema(conn)
    _enforce_permissions(cfg.store_path)
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
