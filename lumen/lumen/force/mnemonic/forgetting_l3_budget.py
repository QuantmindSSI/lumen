"""A10: Budget-Curated Forgetting (L3).

Input wire: SQLite schema, psutil (for RAM), config.memory_limit_mb
Output wire: A12 (optical degradation / release scheduler), D10 (actual vector mutation)
Secret sauce: Net-value-per-byte eviction
"""

import contextlib
import sqlite3

from lumen.sovereign.wear import WearAwareBatcher

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

logger = None
try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    pass


def budget_curated_eviction(conn: sqlite3.Connection, config, target_ram_mb: float | None = None, batcher: WearAwareBatcher | None = None):
    """
    When resident memory footprint exceeds trigger, evict lowest V(m)/byte candidates.
    Eviction = optical degradation (FP32→FP16→INT8→BINARY→RELEASED).
    """
    memory_limit_mb = getattr(config, "memory_limit_mb", 512)
    if target_ram_mb is None:
        target_ram_mb = memory_limit_mb * 0.85

    rss_mb = 0.0
    if _HAS_PSUTIL:
        try:
            proc = psutil.Process()
            rss_mb = proc.memory_info().rss / (1024 * 1024)
        except Exception:
            pass

    if rss_mb < target_ram_mb:
        return 0

    needed_eviction_mb = rss_mb - (memory_limit_mb * 0.75)
    bytes_per_chunk = 1800
    chunks_to_evict = max(1, int((needed_eviction_mb * 1024 * 1024) / bytes_per_chunk))

    rows = conn.execute(
        """SELECT chunk_id, vm_score, resolution,
                  (julianday('now') - julianday(datetime(created_at, 'unixepoch'))) AS age_days
           FROM chunk
           WHERE valid_to IS NULL AND optical_level < 2
           ORDER BY (vm_score * (1.0 / (age_days + 1.0))) ASC
           LIMIT ?""", (chunks_to_evict,)
    ).fetchall()

    evicted = 0
    if batcher:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        has_vec_fallback = "vec_fallback" in tables
        has_vec_chunks = "vec_chunks" in tables
        for chunk_id, _vm, res, _age in rows:
            new_res = _next_resolution(res)
            if new_res == "RELEASED":
                batcher.queue.append(
                    (
                        "UPDATE chunk SET optical_level = 2, valid_to = unixepoch() WHERE chunk_id = ?",
                        (chunk_id,),
                    )
                )
            else:
                batcher.queue.append(
                    (
                        "UPDATE chunk SET resolution = ?, optical_level = 1 WHERE chunk_id = ?",
                        (new_res, chunk_id),
                    )
                )
            if has_vec_fallback:
                batcher.queue.append(("DELETE FROM vec_fallback WHERE chunk_id = ?", (chunk_id,)))
            if has_vec_chunks:
                batcher.queue.append(("DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk_id,)))
            evicted += 1
    else:
        for chunk_id, _vm, res, _age in rows:
            new_res = _next_resolution(res)
            if new_res == "RELEASED":
                conn.execute(
                    "UPDATE chunk SET optical_level = 2, valid_to = unixepoch() WHERE chunk_id = ?",
                    (chunk_id,)
                )
            else:
                conn.execute(
                    "UPDATE chunk SET resolution = ?, optical_level = 1 WHERE chunk_id = ?",
                    (new_res, chunk_id)
                )
            # Remove from vector index as resolution degraded
            try:
                # Simple table-based removal via direct SQL so we don't need a global channel
                conn.execute("DELETE FROM vec_fallback WHERE chunk_id = ?", (chunk_id,))
                with contextlib.suppress(Exception):
                    conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk_id,))
            except Exception:
                pass
            evicted += 1

    if logger:
        logger.info("budget_eviction", evicted=evicted, triggered_at_mb=round(rss_mb, 1),
                    target_mb=target_ram_mb)
    return evicted


def _next_resolution(current: str) -> str:
    chain = {"FP32": "FP16", "FP16": "INT8", "INT8": "BINARY", "BINARY": "RELEASED"}
    return chain.get(current, "RELEASED")
