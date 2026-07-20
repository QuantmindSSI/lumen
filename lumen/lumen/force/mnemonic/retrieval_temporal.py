"""D7: Temporal Search & Bi-Temporal Query Engine.

Input wire: SQLite chunk table with bi-temporal columns (valid_from, valid_to,
superseded_by)
Output wire: Brainstorm 9.5 temporal queries ("What happened last Tuesday?")
"""

import sqlite3
from dataclasses import dataclass

logger = None
try:
    import structlog

    logger = structlog.get_logger()
except Exception:
    pass


@dataclass(frozen=True)
class TemporalHit:
    """A single memory fact anchored in bi-temporal validity."""

    chunk_id: int
    content: str
    valid_from: int | None
    valid_to: int | None
    superseded_by: int | None
    provenance_root: int | None


def temporal_point_query(
    conn: sqlite3.Connection,
    content_keywords: list[str],
    as_of_unix: int | None = None,
    include_superseded: bool = False,
) -> list[TemporalHit]:
    """Find facts that were valid at a specific point in time.

    Args:
        conn: SQLite connection initialised with the Lumen schema.
        content_keywords: List of substrings to search for in ``chunk.content``.
            If empty, an empty list is returned immediately.
        as_of_unix: Unix timestamp defining the point-in-time query horizon.
            When ``None``, only currently valid chunks (``valid_to IS NULL``)
            are returned.
        include_superseded: When ``True``, the full ``superseded_by`` chain
            for every qualifying hit is expanded and de-duplicated into the
            result list.

    Returns:
        A list of :class:`TemporalHit` ordered first by ``valid_from DESC``
        and then by the order in which they appear inside a supersession
        chain.
    """
    if not content_keywords:
        return []

    clauses: list[str] = []
    params: list[str | int] = []
    for kw in content_keywords:
        clauses.append("chunk.content LIKE ?")
        params.append(f"%{kw}%")

    if as_of_unix is not None:
        time_clause = (
            "chunk.valid_from <= ? AND (chunk.valid_to IS NULL OR chunk.valid_to > ?)"
        )
        params.extend([as_of_unix, as_of_unix])
    else:
        time_clause = "chunk.valid_to IS NULL"

    sql = f"""
        SELECT chunk_id, content, valid_from, valid_to, superseded_by, provenance_root
        FROM chunk
        WHERE ({' OR '.join(clauses)}) AND {time_clause}
        ORDER BY valid_from DESC
    """

    rows = conn.execute(sql, params).fetchall()
    hits: list[TemporalHit] = []
    seen: set[int] = set()

    for row in rows:
        hit = TemporalHit(*row)
        if include_superseded:
            chain = walk_supersession_chain(conn, hit.chunk_id)
            for c in chain:
                if c.chunk_id not in seen:
                    seen.add(c.chunk_id)
                    hits.append(c)
        else:
            if hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                hits.append(hit)

    if logger:
        logger.info(
            "temporal_point_query",
            keywords=content_keywords,
            as_of=as_of_unix,
            hits=len(hits),
            include_superseded=include_superseded,
        )

    return hits


def walk_supersession_chain(
    conn: sqlite3.Connection,
    chunk_id: int,
) -> list[TemporalHit]:
    """Follow the ``superseded_by`` chain starting at *chunk_id*.

    Traverses forward from the supplied chunk to the most recent version in
    the chain, reconstructing the evolution of that belief.

    Args:
        conn: SQLite connection.
        chunk_id: Starting chunk in the chain.

    Returns:
        Ordered list of :class:`TemporalHit` beginning with *chunk_id* and
        ending with the most recent chunk whose ``superseded_by`` is ``NULL``.
    """
    chain: list[TemporalHit] = []
    current_id: int | None = chunk_id
    visited: set[int] = set()

    while current_id is not None and current_id not in visited:
        visited.add(current_id)
        row = conn.execute(
            """
            SELECT chunk_id, content, valid_from, valid_to, superseded_by, provenance_root
            FROM chunk
            WHERE chunk_id = ?
            """,
            (current_id,),
        ).fetchone()
        if row is None:
            break
        chain.append(TemporalHit(*row))
        current_id = row[4]  # superseded_by column

    return chain
