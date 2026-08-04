"""D13: Curiosity-Driven Exploration Scheduler.

Input wire: A1 (schema), TFC state, APScheduler
Output wire: C8 (prefetch), A6 (reconsolidation trigger)
Secret sauce: Free memory maintenance during idle time
"""

from __future__ import annotations

import sqlite3
import time

from lumen.logging import get_console_logger

logger = get_console_logger(__name__)


def curiosity_probe(conn: sqlite3.Connection, limit: int = 5) -> list[int]:
    """Return chunk_ids worth surfacing during idle time.

    Runs opportunistically (during sleep phase or when context padding > 50%).

    Signals:
      1. Oldest last_access (NULL treated as epoch 0).
      2. Highest variance in V(m) within a room (unresolved/conflicting).
      3. Lowest access_count among valid chunks (unseen/orphan).

    Args:
        conn: SQLite connection.
        limit: Maximum number of chunk_ids to return.

    Returns:
        Deduplicated list of chunk_ids, interleaved from each signal.
    """
    logger.debug("curiosity_probe_start", limit=limit)

    # Signal 1: oldest last_access
    old = conn.execute(
        """
        SELECT chunk_id FROM chunk
        WHERE valid_to IS NULL
        ORDER BY COALESCE(last_access_at, 0) ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    signal_old = [r[0] for r in old]

    # Signal 2: highest variance in V(m) — pick rooms with wide spread,
    # then chunks whose vm is furthest from the room mean.
    variance = conn.execute(
        """
        SELECT chunk_id FROM chunk
        WHERE valid_to IS NULL
          AND room_id IN (
              SELECT room_id FROM chunk
              WHERE valid_to IS NULL
              GROUP BY room_id
              HAVING MAX(vm_score) - MIN(vm_score) > 0.3
          )
        ORDER BY ABS(vm_score - (
            SELECT AVG(vm_score) FROM chunk AS c2
            WHERE c2.room_id = chunk.room_id AND c2.valid_to IS NULL
        )) DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    signal_variance = [r[0] for r in variance]

    # Signal 3: lowest access_count (unseen)
    unseen = conn.execute(
        """
        SELECT chunk_id FROM chunk
        WHERE valid_to IS NULL
        ORDER BY access_count ASC, created_at ASC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    signal_unseen = [r[0] for r in unseen]

    # Round-robin interleave with deduplication
    seen: set[int] = set()
    result: list[int] = []
    signals = [signal_old, signal_variance, signal_unseen]
    max_len = max(len(s) for s in signals) if signals else 0
    for i in range(max_len):
        for sig in signals:
            if len(result) >= limit:
                break
            if i < len(sig) and sig[i] not in seen:
                seen.add(sig[i])
                result.append(sig[i])
        if len(result) >= limit:
            break

    logger.debug(
            "curiosity_probe_done",
            returned=len(result),
            signals={"old": signal_old, "variance": signal_variance, "unseen": signal_unseen},
        )
    return result


def curiosity_score(conn: sqlite3.Connection, chunk_id: int) -> float:
    """Compute a composite curiosity score for a single chunk."""
    row = conn.execute(
        """
        SELECT vm_score, access_count, last_access_at, valid_to, created_at
        FROM chunk WHERE chunk_id = ?
        """,
        (chunk_id,),
    ).fetchone()
    if not row or row["valid_to"] is not None:
        return 0.0

    now = int(time.time())
    age = now - (row["created_at"] or now)
    age_days = age / 86400.0
    access = row["access_count"] or 0
    last_access = row["last_access_at"] or 0
    recency_days = (now - last_access) / 86400.0 if last_access else age_days

    # 40% age, 30% unseen, 30% recency
    age_norm = min(1.0, age_days / 30.0)
    unseen_norm = min(1.0, 1.0 / (1.0 + access / 10.0))
    recency_norm = min(1.0, recency_days / 30.0)

    return round(0.4 * age_norm + 0.3 * unseen_norm + 0.3 * recency_norm, 4)
