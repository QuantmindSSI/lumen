"""D11: Consolidation Pass Logic (Preference → Profile).

Input wire: D2 (event_buffer), A5 (provenance), A6 (store), B2 (embed),
            D9 (local_llm), D3 (feedback_log)
Output wire: C5 (context assembly), D16 (user_profile)
Secret sauce: MARS lifecycle applied during idle time (narrative generation
requires LocalLLM; dedup merging runs without it).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lumen.data.schema import get_connection
from lumen.force.contextual.embed import FallbackEmbedder
from lumen.force.mnemonic.provenance import supersede_chunk
from lumen.force.mnemonic.store import store_memory
from lumen.sovereign.local_llm import LocalLLM
from lumen.sovereign.wear import WearAwareBatcher

if TYPE_CHECKING:
    from lumen.config import LumenConfig
    from lumen.force.mnemonic.event_buffer import EventMemoryBuffer

from lumen.logging import get_console_logger

logger = get_console_logger(__name__)


@dataclass(frozen=True)
class ChunkProxy:
    chunk_id: int
    room_id: int
    content_hash: str
    vm_score: float


def _group_similar_chunks(
    conn: sqlite3.Connection, threshold: float = 0.95
) -> list[list[ChunkProxy]]:
    """Group valid chunks by exact content_hash match for deduplication.

    Args:
        conn: SQLite connection.
        threshold: Unused; kept for API compatibility with future similarity-based
            grouping.

    Returns:
        List of chunk groups where each group contains at least one chunk.
    """
    rows = conn.execute(
        """
        SELECT chunk_id, room_id, content_hash, vm_score
        FROM chunk
        WHERE valid_to IS NULL
        """
    ).fetchall()

    hash_groups: dict[str, list[ChunkProxy]] = {}
    for row in rows:
        proxy = ChunkProxy(
            chunk_id=row["chunk_id"],
            room_id=row["room_id"],
            content_hash=row["content_hash"],
            vm_score=row["vm_score"],
        )
        hash_groups.setdefault(proxy.content_hash, []).append(proxy)

    return list(hash_groups.values())


def _get_rooms_with_new_activity(
    conn: sqlite3.Connection,
) -> list[tuple[int, str]]:
    """Return rooms that have valid chunks created in the last 7 days."""
    rows = conn.execute(
        """
        SELECT DISTINCT r.room_id, r.name
        FROM room r
        JOIN chunk c ON c.room_id = r.room_id
        WHERE c.created_at >= unixepoch() - 7 * 86400
          AND c.valid_to IS NULL
        """
    ).fetchall()
    return [(row["room_id"], row["name"]) for row in rows]


def _get_recent_fact_bullets(conn: sqlite3.Connection, room_id: int, days: int = 7) -> list[str]:
    """Return recent chunk contents for a room, ordered by V(m) score.

    Args:
        conn: SQLite connection.
        room_id: Room to query.
        days: Look-back window in days.

    Returns:
        List of chunk content strings (max 20).
    """
    rows = conn.execute(
        """
        SELECT content FROM chunk
        WHERE room_id = ?
          AND created_at >= unixepoch() - ? * 86400
          AND valid_to IS NULL
        ORDER BY vm_score DESC
        LIMIT 20
        """,
        (room_id, days),
    ).fetchall()
    return [row["content"] for row in rows]


def run_consolidation_pass(
    config: LumenConfig,
    event_buffer: EventMemoryBuffer | None = None,
    embedder: object | None = None,
    batcher: WearAwareBatcher | None = None,
) -> int:
    """Run one full consolidation pass: drain events, dedup, generate narratives.

    Args:
        config: LumenConfig instance.
        event_buffer: Optional EventMemoryBuffer to drain. If None, event-tier
            draining is skipped.
        embedder: Optional embedder with ``encode_single(text)``. If None, a
            FallbackEmbedder is instantiated.
        batcher: Optional WearAwareBatcher for bulk dedup writes.

    Returns:
        Number of operations performed: dedup merges + narratives generated.
    """
    conn = batcher.conn if batcher else get_connection(config)
    should_close = batcher is None
    try:
        if embedder is None:
            embedder = FallbackEmbedder(dims=config.embedding_dims)

        operations = 0

        # 0. Drain event buffer into Preference tier
        if event_buffer is not None:
            events = event_buffer.drain_expired()
            for event in events:
                room_name = event.session_id or "events"
                emb = embedder.encode_single(event.raw_text)
                store_memory(
                    conn,
                    content=event.raw_text,
                    room_name=room_name,
                    source_type="consolidation",
                    embedding=emb,
                    config=config,
                    batcher=batcher,
                )

        # 2. Dedup-based merge: group by semantic hash
        groups = _group_similar_chunks(conn, threshold=0.95)
        if batcher:
            batch = []
            for group in groups:
                if len(group) <= 1:
                    continue
                winner = max(group, key=lambda c: c.vm_score)
                for loser in group:
                    if loser.chunk_id != winner.chunk_id:
                        batch.append(
                            (
                                "UPDATE chunk SET valid_to = unixepoch(), superseded_by = ? WHERE chunk_id = ?",
                                (winner.chunk_id, loser.chunk_id),
                            )
                        )
                        operations += 1
            if batch:
                batcher.flush_sync(batch)
        else:
            for group in groups:
                if len(group) <= 1:
                    continue
                winner = max(group, key=lambda c: c.vm_score)
                for loser in group:
                    if loser.chunk_id != winner.chunk_id:
                        supersede_chunk(conn, loser.chunk_id, winner.chunk_id)
                        operations += 1

        # 5. Resynthesis: generate Profile Memory narrative per room
        llm_available = LocalLLM.is_available()
        if llm_available:
            llm = LocalLLM(config)
            for room_id, room_name in _get_rooms_with_new_activity(conn):
                recent_facts = _get_recent_fact_bullets(conn, room_id, days=7)
                if len(recent_facts) > 5:
                    narrative = llm.summarize(
                        recent_facts,
                        instruction=(
                            f"Summarize the user's recent activity in "
                            f"'{room_name}' as 2-3 concise sentences."
                        ),
                    )
                    emb = embedder.encode_single(narrative)
                    store_memory(
                        conn,
                        content=narrative,
                        room_name=room_name,
                        source_type="consolidation",
                        embedding=emb,
                        config=config,
                        batcher=batcher,
                    )
                    operations += 1
        else:
            logger.debug("consolidation_skip_narrative", reason="llm_unavailable")

        if should_close:
            conn.commit()
        return operations
    finally:
        if should_close:
            conn.close()
