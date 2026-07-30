"""D4: Lightweight Schema Migration System.

Track schema version via PRAGMA user_version.
Run numbered .sql files in migrations/.
Current version = 1.
"""

import sqlite3
from pathlib import Path

from lumen.logging import get_console_logger

logger = get_console_logger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
CURRENT_SCHEMA_VERSION = 1


def migrate(conn: sqlite3.Connection, target_version: int = CURRENT_SCHEMA_VERSION) -> None:
    """Apply pending migrations from current PRAGMA user_version to target_version."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current >= target_version:
        if logger:
            logger.debug("migrate_already_at_target", current=current, target=target_version)
        return

    for v in range(current + 1, target_version + 1):
        path = MIGRATIONS_DIR / f"{v:04d}.sql"
        if path.exists():
            conn.executescript(path.read_text(encoding="utf-8"))
            conn.execute(f"PRAGMA user_version = {v}")
            if logger:
                logger.info("migration_applied", from_version=current, to_version=v)
        else:
            if logger:
                logger.warning("migration_missing", version=v, path=str(path))
            break


def get_current_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]
