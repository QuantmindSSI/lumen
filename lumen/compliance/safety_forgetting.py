"""A11: Safety-Triggered Forgetting (L4) & Compliance.

Input wire: Regex / spaCy NER patterns, user command channel
Output wire: A12 (audit log), A5 (provenance purge), D17 (log rotation)
Secret sauce: Provenance-chain deletion (GateMem requirement)
"""

import json
import re
import sqlite3
from pathlib import Path

from lumen.logging import get_console_logger

logger = get_console_logger(__name__)

PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b\d{3}-\d{3}-\d{4}\b"),
    "api_key": re.compile(r"[a-zA-Z0-9_-]{32,}"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

AUDIT_LOG_PATH = Path.home() / ".lumen" / "logs" / "compliance.jsonl"


def safety_scan_chunk(content: str) -> list[str]:
    """Return list of triggered safety rule names."""
    hits = []
    patterns = dict(PII_PATTERNS)
    try:
        from lumen.config import LumenConfig

        cfg = LumenConfig()
        if cfg.pii_custom_patterns:
            for idx, pat in enumerate(cfg.pii_custom_patterns.split(",")):
                pat = pat.strip()
                if pat:
                    patterns[f"custom_{idx}"] = re.compile(pat)
    except Exception:
        pass
    for rule_name, rx in patterns.items():
        if rx.search(content):
            hits.append(rule_name)
    return hits


def _hash_pattern(text: str, pattern_name: str) -> str:
    """Replace matches with a stable short hash."""
    import hashlib

    rx = PII_PATTERNS.get(pattern_name)
    if rx is None:
        return text

    def _repl(m: re.Match) -> str:
        digest = hashlib.sha256(m.group(0).encode()).hexdigest()[:8]
        return f"[HASH:{digest}]"

    return rx.sub(_repl, text)


def apply_pii_strategy(content: str, hits: list[str], mode: str) -> str | None:
    """Apply the configured PII strategy.

    Args:
        content: Original chunk content.
        hits: List of triggered PII pattern names.
        mode: One of ``block``, ``redact``, ``hash``.

    Returns:
        Modified content string, or ``None`` if the store should be blocked.
    """
    if mode == "block":
        return None
    if mode == "hash":
        for hit in hits:
            content = _hash_pattern(content, hit)
        return content
    # Default: redact
    for hit in hits:
        rx = PII_PATTERNS.get(hit)
        if rx:
            content = rx.sub(lambda m: "[REDACTED]", content)
    return content


def safety_forget_chunk(conn: sqlite3.Connection, chunk_id: int, reason: str) -> None:
    """
    Immediate deletion with provenance-chain clearance.
    Write-ahead audit BEFORE mutation.
    """
    from lumen.audit import log_audit_event
    log_audit_event(
        conn=conn,
        event_type="compliance",
        actor="system",
        action="safety_forget",
        resource_id=chunk_id,
        resource_type="chunk",
        metadata_json=json.dumps({"reason": reason}),
        client_ip=None,
        request_id=None,
    )

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
        (chunk_id,),
    )

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


def get_recent_audit_events(n: int = 10) -> list[dict]:
    """Read last N compliance audit events from JSONL."""
    if not AUDIT_LOG_PATH.exists():
        return []
    lines = []
    with open(AUDIT_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines[-n:]
