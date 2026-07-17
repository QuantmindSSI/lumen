"""A6: Store Pipeline (The Write Path).

Input wire: A1 (schema), A3 (vector backend), A5 (provenance), A9 (V(m) model), B2 (embedding model)
Output wire: A2 (FTS5 index), A7 (interference), A11 (consolidation queue)
Secret sauce: Palace-aware placement + V(m) scoring + interference check on write
"""

import hashlib
import json
import uuid
from typing import Optional

import numpy as np
import sqlite3

logger = None
try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    pass

from lumen.brand.errors import PalaceError
from lumen.force.mnemonic.provenance import create_provenance
from lumen.force.mnemonic.value_model import compute_vm


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
    locus_name: Optional[str] = None,
    source_type: str = "user_input",
    source_ref: Optional[str] = None,
    embedding: Optional[np.ndarray] = None,
    vm_weights: Optional[dict] = None,
    config=None,
) -> int:
    """
    Atomic store: schema + vector + lexical + provenance in one transaction.
    """
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
