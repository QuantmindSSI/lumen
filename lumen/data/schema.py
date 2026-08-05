"""Schema initialisation helpers."""

import os
import sqlite3
import stat
from pathlib import Path

from lumen.data.migrate import migrate
from lumen.logging import get_console_logger

logger = get_console_logger(__name__)
_SQL_PATH = Path(__file__).with_suffix(".sql")

# Graceful SQLCipher import
try:
    import sqlcipher3

    _HAS_SQLCIPHER = True
except Exception:
    sqlcipher3 = None  # type: ignore[assignment]
    _HAS_SQLCIPHER = False


def _enforce_permissions(root_path: Path) -> None:
    """Ensure ~/.lumen tree is 700 (dirs) and 600 (files). Fail-fast on error."""
    if not root_path.exists():
        return
    try:
        os.chmod(root_path, stat.S_IRWXU)  # 0o700
        for child in root_path.rglob("*"):
            if child.is_dir():
                os.chmod(child, stat.S_IRWXU)  # 0o700
            else:
                os.chmod(child, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    except OSError as exc:
        raise RuntimeError(
            f"Cannot enforce file permissions on {root_path}. "
            f"Run as the file owner or check mount options."
        ) from exc


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

    Supports three encryption modes:
      - sqlcipher : Transparent page-level AES-256 encryption via SQLCipher.
      - fernet    : Field-level encryption for chunk.content only (fallback).
      - none      : Plaintext SQLite (not recommended for production).
    """
    from lumen.config import LumenConfig

    cfg: LumenConfig = config
    cfg.store_path.mkdir(parents=True, exist_ok=True)
    db_path = cfg.store_path / "lumen.db"

    mode = getattr(cfg, "database_encryption_mode", "none")

    if mode == "sqlcipher":
        if not _HAS_SQLCIPHER:
            raise RuntimeError(
                "database_encryption_mode is 'sqlcipher' but sqlcipher3 is not installed. "
                "Install it with: pip install 'sqlcipher3>=0.6'"
            )
        conn = sqlcipher3.connect(str(db_path), check_same_thread=False)  # type: ignore[arg-type]
        passphrase = cfg.encryption_key or ""
        if not passphrase:
            raise RuntimeError("database_encryption_mode='sqlcipher' requires LUMEN_ENCRYPTION_KEY to be set.")
        # Derive a high-entropy hex passphrase from the user's key
        from lumen.security.crypto import derive_sqlcipher_passphrase

        derived = derive_sqlcipher_passphrase(passphrase, cfg.store_path)
        conn.execute(f"PRAGMA key = '{derived}'")
        cipher_ver = conn.execute("PRAGMA cipher_version").fetchone()
        if not cipher_ver or not cipher_ver[0]:
            raise RuntimeError("SQLCipher initialisation failed: cipher_version is empty.")
        logger.info("sqlcipher_enabled", version=cipher_ver[0])
    else:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        if mode == "none":
            logger.debug("encryption_at_rest_disabled")

    # sqlcipher3 cursor is not compatible with sqlite3.Row; use a typed tuple fallback
    if mode == "sqlcipher":
        # sqlcipher3 already returns Row-like objects by default in most builds,
        # but sqlite3.Row explodes with sqlcipher3 cursors. We default to None.
        pass  # keep driver default row factory
    else:
        conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA mmap_size = 268435456")
    ensure_schema(conn)
    _enforce_permissions(cfg.store_path.parent)
    return conn


def get_encryption(config):
    """Return a FernetEncryption instance if field-level encryption is required.

    When database_encryption_mode is 'sqlcipher', page-level encryption already
    protects all data, so field-level Fernet is disabled to avoid double-encryption
    and to preserve FTS5 / BM25 search performance.

    Backward compatibility: if mode is 'none' but an encryption_key is set,
    we fall back to Fernet field-level encryption (previous default behaviour).
    """
    from lumen.security.crypto import FernetEncryption

    mode = getattr(config, "database_encryption_mode", "none")
    has_key = getattr(config, "encryption_key", None)
    if mode == "sqlcipher":
        return FernetEncryption(None)
    if has_key:
        return FernetEncryption(config.encryption_key)
    return FernetEncryption(None)


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
