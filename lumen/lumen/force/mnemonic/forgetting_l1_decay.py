"""A8: Ebbinghaus Passive Decay (L1).

Input wire: APScheduler + SQLite (A1)
Output wire: A9 (V(m) scalar updates)
Secret sauce: User-specific decay rate (D16 user_profile.ebbinghaus_half_life_days)
"""

import math
import sqlite3
from datetime import datetime, timezone

from lumen.logging import get_console_logger
from lumen.sovereign.wear import WearAwareBatcher

logger = get_console_logger(__name__)


def ebbinghaus_decay(
    conn: sqlite3.Connection,
    user_half_life_days: float = 7.0,
    now: datetime | None = None,
    batcher: WearAwareBatcher | None = None,
) -> int:
    """
    R(t) = e^(-t / (h * ln(2))) where h = user-specific half-life in days.
    Applied as a multiplicative penalty to vm_score.
    Chunks with vm_score below 0.05 are queued for release.
    """
    now = now or datetime.now(timezone.utc)
    unix_now = int(now.timestamp())

    rows = conn.execute(
        """SELECT chunk_id, vm_score, last_access_at, created_at
           FROM chunk
           WHERE valid_to IS NULL AND optical_level < 2
        """
    ).fetchall()

    hl_sec = user_half_life_days * 86400.0
    updates = []
    released = 0
    for chunk_id, vm, last_access, created_at in rows:
        created = created_at if created_at is not None else unix_now
        age_sec = unix_now - created
        retention = math.exp(-age_sec / (hl_sec * math.log(2)))
        # Recency boost
        if last_access:
            recency_hours = (unix_now - last_access) / 3600.0
            recency_boost = math.exp(-recency_hours / 24.0)
            retention = max(retention, recency_boost)
        new_vm = vm * retention
        if new_vm < 0.05:
            new_vm = 0.0
            released += 1
        updates.append((new_vm, chunk_id))

    if batcher:
        for new_vm, chunk_id in updates:
            batcher.queue.append(
                ("UPDATE chunk SET vm_score = ? WHERE chunk_id = ?", (new_vm, chunk_id))
            )
    else:
        conn.executemany("UPDATE chunk SET vm_score = ? WHERE chunk_id = ?", updates)
    if logger:
        logger.info(
            "decay_applied",
            chunks=len(updates),
            released=released,
            half_life_days=user_half_life_days,
        )
    return released
