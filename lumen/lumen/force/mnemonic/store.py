"""A6: Store Pipeline (The Write Path).

Input wire: A1 (schema), A3 (vector backend), A5 (provenance), A9 (V(m) model), B2 (embedding model)
Output wire: A2 (FTS5 index), A7 (interference), A11 (consolidation queue)
Secret sauce: Palace-aware placement + V(m) scoring + interference check on write
"""

import hashlib
import json
import sqlite3
import uuid

import numpy as np

from lumen.force.mnemonic.provenance import create_provenance
from lumen.force.mnemonic.value_model import compute_vm
from lumen.sovereign.wear import WearAwareBatcher

logger = None
try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    pass


def _get_lexical_channel(conn: sqlite3.Connection):
    from lumen.force.mnemonic.retrieval_lexical import LexicalChannel
    return LexicalChannel(conn)


def _get_vector_channel(conn: sqlite3.Connection, config=None):
    from lumen.force.mnemonic.retrieval_dense import VectorChannel
    if config is None:
        from lumen.config import LumenConfig
        config = LumenConfig()
    return VectorChannel(config, conn)


def store_memory(
    conn: sqlite3.Connection,
    content: str,
    room_name: str,
    locus_name: str | None = None,
    source_type: str = "user_input",
    source_ref: str | None = None,
    embedding: np.ndarray | None = None,
    vm_weights: dict | None = None,
    config=None,
    batcher: WearAwareBatcher | None = None,
) -> int:
    """
    Atomic store: schema + vector + lexical + provenance in one transaction.

    ``batcher`` is accepted for API uniformity but is not used here because
    single-store writes depend on ``lastrowid`` from chunk and provenance
    inserts, which is incompatible with deferred SQL batching.  The batcher
    is wired into bulk callers (consolidation, decay, eviction) where it
    provides real SD/eMMC endurance benefits.
    """
    # Pre-store safety scan — redact PII at the write path
    from lumen.compliance.safety_forgetting import safety_scan_chunk

    scan_hits = safety_scan_chunk(content)
    if scan_hits:
        if logger:
            logger.warning("safety_scan_triggered", room=room_name, hits=scan_hits)
        # Redact the content but still store a metadata stub
        for hit in scan_hits:
            content = _redact_pattern(content, hit)

    with conn:
        # 1. Resolve room
        row = conn.execute("SELECT room_id FROM room WHERE name = ?", (room_name,)).fetchone()
        if not row:
            cur = conn.execute(
                "INSERT INTO room(name, room_type) VALUES (?, 'domain')",
                (room_name,)
            )
            room_id = cur.lastrowid
        else:
            room_id = row[0]

        # 2. Deduplication
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        existing = conn.execute(
            "SELECT chunk_id FROM chunk WHERE content_hash = ? AND valid_to IS NULL",
            (content_hash,)
        ).fetchone()
        if existing:
            if logger:
                logger.info("store_dedup", room=room_name, hash=content_hash[:16])
            return existing[0]

        # 3. Locus resolution
        locus_id = _resolve_locus(conn, room_id, locus_name, embedding)

        # 4. Compute V(m)
        vm_score, vm_factors = compute_vm(content, vm_weights, source_type)

        # 5. Insert chunk
        cur = conn.execute(
            """INSERT INTO chunk
               (locus_id, room_id, content, content_hash, vm_score, vm_factors, resolution)
               VALUES (?,?,?,?,?,?,?)""",
            (locus_id, room_id, content, content_hash, vm_score,
             json.dumps(vm_factors), "FP32")
        )
        chunk_id = cur.lastrowid

        # 6. Provenance
        prov_id = create_provenance(conn, chunk_id, source_type, source_ref)
        conn.execute("UPDATE chunk SET provenance_root = ? WHERE chunk_id = ?", (prov_id, chunk_id))

        # 7. Vector index
        if embedding is not None:
            _get_vector_channel(conn, config).add(chunk_id, embedding)

        # 8. Lexical index
        _get_lexical_channel(conn).index_chunk(chunk_id, content)

        # 9. Interference check
        _trigger_interference_check(conn, room_id, locus_id, chunk_id, embedding)

        if logger:
            logger.info("memory_stored", chunk_id=chunk_id, room=room_name,
                        locus=locus_name, vm=vm_score)
        return chunk_id


def _resolve_locus(conn, room_id, locus_name, embedding):
    if locus_name:
        row = conn.execute(
            "SELECT locus_id FROM locus WHERE room_id=? AND name=?",
            (room_id, locus_name)
        ).fetchone()
        if row:
            return row[0]
        cur = conn.execute(
            "INSERT INTO locus(room_id, name) VALUES (?,?)",
            (room_id, locus_name)
        )
        return cur.lastrowid
    # Auto-placement
    cur = conn.execute(
        "INSERT INTO locus(room_id, name) VALUES (?,?)",
        (room_id, f"auto_{uuid.uuid4().hex[:8]}")
    )
    return cur.lastrowid


def _trigger_interference_check(conn, room_id, locus_id, new_chunk_id, embedding):
    from lumen.force.mnemonic.forgetting_l2_interference import check_locus_interference
    if embedding is not None:
        check_locus_interference(conn, room_id, locus_id, new_chunk_id, embedding)


def _redact_pattern(text: str, pattern_name: str) -> str:
    """Simple string-level redaction for known PII patterns."""
    import re

    patterns = {
        "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "phone": re.compile(r"\b\d{3}-\d{3}-\d{4}\b"),
        "api_key": re.compile(r"[a-zA-Z0-9_-]{32,}"),
    }
    rx = patterns.get(pattern_name)
    if rx is None:
        return text
    return rx.sub(lambda m: "[REDACTED]", text)
