"""D4: Lightweight Schema Migration System.

Input wire: SQLite, versioned SQL scripts in lumen/data/migrations/
Output wire: A1 (schema at target version)
Secret sauce: Single-file manifest tracking; zero external dependencies
"""

import sqlite3
from pathlib import Path

import structlog

logger = structlog.get_logger()

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
CURRENT_SCHEMA_VERSION = 1


def migrate(conn: sqlite3.Connection, target_version: int = CURRENT_SCHEMA_VERSION) -> None:
    """Apply pending migrations from current PRAGMA user_version to target_version."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current >= target_version:
        logger.debug("migrate_already_at_target", current=current, target=target_version)
        return

    for v in range(current + 1, target_version + 1):
        path = MIGRATIONS_DIR / f"{v:04d}.sql"
        if path.exists():
            conn.executescript(path.read_text(encoding="utf-8"))
            conn.execute(f"PRAGMA user_version = {v}")
            logger.info("migration_applied", from_version=current, to_version=v)
        else:
            logger.warning("migration_missing", version=v, path=str(path))
            break
