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
from lumen.logging import get_console_logger

logger = get_console_logger(__name__)


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
    enable_frqad: bool | None = None,
) -> list[RetrievedChunk]:
    """
    Stage 3: Reciprocal Rank Fusion + V(m) + FRQAD rerank + recency boost.
    """
    k_rrf = 60  # standard RRF constant from Cormack et al. (2009)

    # Build candidate pool with RRF
    rrfs: dict[int, float] = {}
    for rank, hit in enumerate(lexical_hits, 1):
        rrfs[hit.chunk_id] = rrfs.get(hit.chunk_id, 0.0) + 1.0 / (k_rrf + rank)
    for rank, dhit in enumerate(dense_hits, 1):
        rrfs[dhit.chunk_id] = rrfs.get(dhit.chunk_id, 0.0) + 1.0 / (k_rrf + rank)

    if graph_hits:
        for rank, ghit in enumerate(graph_hits, 1):
            rrfs[ghit.chunk_id] = rrfs.get(ghit.chunk_id, 0.0) + 1.0 / (k_rrf + rank) * 0.8

    chunk_ids = list(rrfs.keys())[:budget_candidates]
    if not chunk_ids:
        return []

    placeholders = "?," * (len(chunk_ids) - 1) + "?"
    rows = conn.execute(
        f"SELECT c.chunk_id, r.name AS room_name, l.name AS locus_name,"
        f" c.content, c.vm_score, c.provenance_root,"
        f" (strftime('%s','now') - c.created_at) / 3600.0 AS age_hours,"
        f" c.resolution"
        f" FROM chunk c"
        f" LEFT JOIN room r ON r.room_id = c.room_id"
        f" LEFT JOIN locus l ON l.locus_id = c.locus_id"
        f" WHERE c.chunk_id IN ({placeholders}) AND c.valid_to IS NULL",
        chunk_ids,
    ).fetchall()

    # Pre-fetch query embedding for FRQAD if provided
    query_vec = query_embedding
    results: list[RetrievedChunk] = []

    # Batch-fetch all embeddings in one query to avoid N+1
    embedding_map: dict[int, np.ndarray] = {}
    if query_vec is not None:
        emb_rows = conn.execute(
            f"SELECT chunk_id, embedding FROM vec_fallback WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        for cid, emb_blob in emb_rows:
            if emb_blob:
                embedding_map[cid] = np.frombuffer(emb_blob, dtype=np.float32)
        missing = [cid for cid in chunk_ids if cid not in embedding_map]
        if missing:
            try:
                miss_placeholders = "?," * (len(missing) - 1) + "?"
                emb_rows2 = conn.execute(
                    f"SELECT chunk_id, embedding FROM vec_chunks WHERE chunk_id IN ({miss_placeholders})",
                    missing,
                ).fetchall()
                for cid, emb_blob in emb_rows2:
                    if emb_blob:
                        embedding_map[cid] = np.frombuffer(emb_blob, dtype=np.float32)
            except Exception as exc:
                logger.debug("vec_chunks_embedding_fetch_failed", error=str(exc))

    for row in rows:
        cid, room_name, locus_name, content, vm, prov, age_hours, res = row
        # Goal bonus
        goal_bonus = 1.0 + (
            0.2 if any(kw.lower() in (content or "").lower() for kw in goal_tree_keywords) else 0.0
        )

        # FRQAD / cosine rerank
        frqad_sim = 0.5
        if query_vec is not None:
            cand_vec = embedding_map.get(cid)
            if cand_vec is not None:
                use_frqad = enable_frqad if enable_frqad is not None else True
                try:
                    if use_frqad:
                        from lumen.sovereign.frqad import compute_frqad

                        frqad = compute_frqad(query_vec, cand_vec, res or "FP32")
                        frqad_sim = 1.0 - (frqad / (np.pi / 2))
                    else:
                        from scipy.spatial.distance import cosine

                        frqad_sim = 1.0 - cosine(query_vec, cand_vec)
                except Exception as exc:
                    logger.debug("frqad_fallback_to_cosine", error=str(exc))
                    from scipy.spatial.distance import cosine

                    frqad_sim = 1.0 - cosine(query_vec, cand_vec)
            else:
                frqad_sim = 0.5

        rrf = rrfs.get(cid, 0.0)
        recency_boost = float(np.exp(-(age_hours or 0) / 168.0))

        final = rrf * (vm + 0.1) * (frqad_sim + 0.1) * recency_boost * goal_bonus

        results.append(
            RetrievedChunk(
                chunk_id=cid,
                room_name=room_name or "unknown",
                locus_name=locus_name or "none",
                content=content,
                provenance_id=prov,
                rrf_score=rrf,
                vm_score=vm,
                frqad_score=frqad_sim,
                recency_hours=age_hours or 0.0,
                final_score=final,
            )
        )

    results.sort(key=lambda x: x.final_score, reverse=True)
    logger.info(
            "fusion_rerank",
            candidates=len(results),
            top_score=results[0].final_score if results else None,
        )
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
        logger.debug("vec_chunks_embedding_fetch_failed", chunk_id=chunk_id, error=str(exc))
    return None
