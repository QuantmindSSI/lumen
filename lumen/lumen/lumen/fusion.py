"""C2: Fusion & Reranking Engine.

Input wire: A2 (BM25 hits), A3 (dense hits), A4 (FRQAD rerank), B1 (goal-tree), A9 (V(m))
Output wire: B1 (context assembly)
Secret sauce: Multi-channel RRF × V(m) × recency × FRQAD rerank
"""

import sqlite3
from dataclasses import dataclass

import numpy as np

from lumen.force.mnemonic.retrieval_dense import DenseHit
from lumen.force.mnemonic.retrieval_graph import GraphHit
from lumen.force.mnemonic.retrieval_lexical import LexicalHit

logger = None
try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    pass


@dataclass
class RetrievedChunk:
    chunk_id: int
    room_name: str
    locus_name: str
    content: str
    provenance_id: int | None
    rrf_score: float
    vm_score: float
    frqad_score: float
    recency_hours: float
    final_score: float


def fuse_and_rerank(
    lexical_hits: list[LexicalHit],
    dense_hits: list[DenseHit],
    goal_tree_keywords: list[str],
    conn: sqlite3.Connection,
    budget_candidates: int = 200,
    query_embedding: np.ndarray | None = None,
    graph_hits: list[GraphHit] | None = None,
) -> list[RetrievedChunk]:
    """
    Stage 3: Reciprocal Rank Fusion + V(m) + FRQAD rerank + recency boost.
    """
    k_rrf = 60

    # Build candidate pool with RRF
    rrfs: dict[int, float] = {}
    for rank, hit in enumerate(lexical_hits, 1):
        rrfs[hit.chunk_id] = rrfs.get(hit.chunk_id, 0.0) + 1.0 / (k_rrf + rank)
    for rank, hit in enumerate(dense_hits, 1):
        rrfs[hit.chunk_id] = rrfs.get(hit.chunk_id, 0.0) + 1.0 / (k_rrf + rank)

    if graph_hits:
        for rank, hit in enumerate(graph_hits, 1):
            rrfs[hit.chunk_id] = rrfs.get(hit.chunk_id, 0.0) + 1.0 / (k_rrf + rank) * 0.8

    chunk_ids = list(rrfs.keys())[:budget_candidates]
    if not chunk_ids:
        return []

    placeholders = ",".join("?" for _ in chunk_ids)
    rows = conn.execute(
        f"""SELECT chunk_id, room_id, locus_id, content, vm_score, provenance_root,
                   (strftime('%s','now') - created_at) / 3600.0 AS age_hours,
                   resolution
            FROM chunk WHERE chunk_id IN ({placeholders}) AND valid_to IS NULL""",
        chunk_ids
    ).fetchall()

    # Pre-fetch query embedding for FRQAD if provided
    query_vec = query_embedding
    results: list[RetrievedChunk] = []

    for row in rows:
        cid, rid, lid, content, vm, prov, age_hours, res = row
        # Goal bonus
        goal_bonus = 1.0 + (0.2 if any(kw.lower() in (content or "").lower() for kw in goal_tree_keywords) else 0.0)

        # FRQAD / cosine rerank
        frqad_sim = 0.5
        if query_vec is not None:
            cand_vec = _get_embedding(conn, cid)
            if cand_vec is not None:
                try:
                    from lumen.sovereign.frqad import compute_frqad
                    frqad = compute_frqad(query_vec, cand_vec, res or "FP32")
                    frqad_sim = 1.0 - (frqad / (np.pi / 2))
                except Exception as exc:
                    if logger:
                        logger.debug("frqad_fallback_to_cosine", error=str(exc))
                    from scipy.spatial.distance import cosine
                    frqad_sim = 1.0 - cosine(query_vec, cand_vec)
            else:
                frqad_sim = 0.5

        rrf = rrfs.get(cid, 0.0)
        recency_boost = float(np.exp(-(age_hours or 0) / 168.0))

        final = rrf * (vm + 0.1) * (frqad_sim + 0.1) * recency_boost * goal_bonus

        # Resolve names
        room_row = conn.execute("SELECT name FROM room WHERE room_id=?", (rid,)).fetchone()
        room_name = room_row[0] if room_row else "unknown"
        locus_name = "none"
        if lid:
            loc_row = conn.execute("SELECT name FROM locus WHERE locus_id=?", (lid,)).fetchone()
            if loc_row:
                locus_name = loc_row[0]

        results.append(RetrievedChunk(
            chunk_id=cid, room_name=room_name, locus_name=locus_name,
            content=content, provenance_id=prov,
            rrf_score=rrf, vm_score=vm, frqad_score=frqad_sim,
            recency_hours=age_hours or 0.0, final_score=final
        ))

    results.sort(key=lambda x: x.final_score, reverse=True)
    if logger:
        logger.info("fusion_rerank", candidates=len(results),
                    top_score=results[0].final_score if results else None)
    return results


def _get_embedding(conn: sqlite3.Connection, chunk_id: int) -> np.ndarray | None:
    row = conn.execute(
        "SELECT embedding FROM vec_fallback WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    if row and row[0]:
        return np.frombuffer(row[0], dtype=np.float32)
    try:
        row2 = conn.execute(
            "SELECT embedding FROM vec_chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if row2 and row2[0]:
            return np.frombuffer(row2[0], dtype=np.float32)
    except Exception as exc:
        if logger:
            logger.debug("vec_chunks_embedding_fetch_failed", chunk_id=chunk_id, error=str(exc))
    return None
