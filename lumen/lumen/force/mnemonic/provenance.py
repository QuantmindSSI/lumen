"""A5: Bi-Temporal Provenance Engine.

Input wire: SQLite schema (A1)
Output wire: A6 (store), C5 (context assembly), A11 (compliance purge)
Secret sauce: Engram-inspired bi-temporal model + supersession chains
"""

import sqlite3
from dataclasses import dataclass

logger = None
try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    pass


@dataclass
class ProvenanceRecord:
    provenance_id: int
    chunk_id: int
    source_type: str
    source_ref: str | None
    confidence: float
    extraction_method: str | None
    parent_provenance: int | None


def create_provenance(
    conn: sqlite3.Connection,
    chunk_id: int,
    source_type: str,
    source_ref: str | None = None,
    confidence: float = 1.0,
    extraction_method: str | None = None,
    parent_provenance: int | None = None,
) -> int:
    """Bi-temporal provenance: every memory enters with a causal chain."""
    cur = conn.execute(
        """INSERT INTO provenance
           (chunk_id, source_type, source_ref, confidence, extraction_method, parent_provenance)
           VALUES (?,?,?,?,?,?)""",
        (chunk_id, source_type, source_ref, confidence, extraction_method, parent_provenance)
    )
    prov_id = cur.lastrowid
    if logger:
        logger.info("provenance_created", chunk_id=chunk_id, prov_id=prov_id,
                    source_type=source_type, parent=parent_provenance)
    return prov_id


def get_effective_fact(
    conn: sqlite3.Connection,
    content_hash_prefix: str,
    as_of_transaction: int | None = None
) -> dict | None:
    """
    Engram merge-on-read: find the currently valid version of a fact,
    respecting supersession chains. If as_of_transaction is given, time-travel.
    """
    if as_of_transaction:
        sql = """
            SELECT c.*, p.source_type, p.confidence
            FROM chunk c
            LEFT JOIN provenance p ON c.provenance_root = p.provenance_id
            WHERE c.content_hash LIKE ? || '%'
              AND c.valid_from <= ?
              AND (c.valid_to IS NULL OR c.valid_to > ?)
            ORDER BY c.created_at DESC
            LIMIT 1
        """
        row = conn.execute(sql, (content_hash_prefix, as_of_transaction, as_of_transaction)).fetchone()
    else:
        sql = """
            SELECT c.*, p.source_type, p.confidence
            FROM chunk c
            LEFT JOIN provenance p ON c.provenance_root = p.provenance_id
            WHERE c.content_hash LIKE ? || '%'
              AND c.valid_to IS NULL
            ORDER BY c.created_at DESC
            LIMIT 1
        """
        row = conn.execute(sql, (content_hash_prefix,)).fetchone()
    return dict(row) if row else None


def supersede_chunk(conn: sqlite3.Connection, old_chunk_id: int, new_chunk_id: int) -> None:
    """Logical update: old fact is deprecated, new fact carries the chain."""
    conn.execute(
        "UPDATE chunk SET valid_to = unixepoch(), superseded_by = ? WHERE chunk_id = ?",
        (new_chunk_id, old_chunk_id)
    )
    if logger:
        logger.info("chunk_superseded", old=old_chunk_id, new=new_chunk_id)
