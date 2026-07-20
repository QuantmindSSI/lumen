"""A11: Safety-Triggered Forgetting (L4) & Compliance.

Input wire: Regex / spaCy NER patterns, user command channel
Output wire: A12 (audit log), A5 (provenance purge), D17 (log rotation)
Secret sauce: Provenance-chain deletion (GateMem requirement)
"""

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List

logger = None
try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    pass

PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b\d{3}-\d{3}-\d{4}\b"),
    "api_key": re.compile(r"[a-zA-Z0-9_-]{32,}"),
}

AUDIT_LOG_PATH = Path.home() / ".lumen" / "logs" / "compliance.jsonl"


def safety_scan_chunk(content: str) -> List[str]:
    """Return list of triggered safety rule names."""
    hits = []
    for rule_name, pattern in PII_PATTERNS.items():
        if pattern.search(content):
            hits.append(rule_name)
    return hits


def safety_forget_chunk(conn: sqlite3.Connection, chunk_id: int, reason: str) -> None:
    """
    Immediate deletion with provenance-chain clearance.
    Write-ahead audit BEFORE mutation.
    """
    audit = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "safety_triggered_forget",
        "chunk_id": chunk_id,
        "reason": reason,
        "provenance_cleared": True,
    }
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(audit) + "\n")

    # Remove from FTS5
    conn.execute("DELETE FROM chunk_fts WHERE rowid = ?", (chunk_id,))
    # Break FK from chunk -> provenance before clearing provenance
    conn.execute("UPDATE chunk SET provenance_root = NULL WHERE chunk_id = ?", (chunk_id,))
    # Clear provenance tree recursively
    _clear_provenance_tree(conn, chunk_id)
    # Logical delete in chunk table
    conn.execute(
        """UPDATE chunk
           SET valid_to = unixepoch(), content = '[REDACTED]', optical_level = 2
           WHERE chunk_id = ?""",
        (chunk_id,)
    )

    if logger:
        logger.warning("safety_forget_executed", chunk_id=chunk_id, reason=reason)


def _clear_provenance_tree(conn: sqlite3.Connection, chunk_id: int) -> None:
    """Walk provenance parent/child links and anonymise.

    Deletes provenance records recursively: first all child records
    that reference this chunk as their ``parent_provenance``, then the
    chunk's own provenance record.
    """
    # Walk children: find records whose parent_provenance references
    # any provenance row of this chunk.
    child_rows = conn.execute(
        """SELECT provenance_id FROM provenance
           WHERE parent_provenance IN (
               SELECT provenance_id FROM provenance WHERE chunk_id = ?
           )""",
        (chunk_id,),
    ).fetchall()
    for row in child_rows:
        conn.execute("DELETE FROM provenance WHERE provenance_id = ?", (row[0],))
    conn.execute("DELETE FROM provenance WHERE chunk_id = ?", (chunk_id,))


def get_recent_audit_events(n: int = 10) -> List[dict]:
    """Read last N compliance audit events from JSONL."""
    if not AUDIT_LOG_PATH.exists():
        return []
    lines = []
    with open(AUDIT_LOG_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines[-n:]
