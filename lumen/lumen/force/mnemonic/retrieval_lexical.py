"""A2: BM25 Lexical Channel (SQLite FTS5 Bridge).

Input wire: SQLite FTS5 + rank_bm25 library
Output wire: C2 (fusion engine)
Secret sauce: None — this is commodity BM25, but wired into the palace schema
"""

import sqlite3
from dataclasses import dataclass
from typing import List

logger = None
try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    pass


@dataclass(frozen=True)
class LexicalHit:
    chunk_id: int
    rank: float          # BM25 score from FTS5 (lower is better in raw FTS5 rank)
    match_info: bytes    # raw FTS5 matchinfo for phrase highlighting


class LexicalChannel:
    """Palace-native BM25 over chunk_fts. Zero ML dependency."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        try:
            self.conn.execute("PRAGMA optimize")
        except Exception:
            pass

    def search(self, query: str, k: int = 20) -> List[LexicalHit]:
        """FTS5 MATCH with bm25 ranking."""
        rows = self.conn.execute(
            """
            SELECT rowid, rank
            FROM chunk_fts
            WHERE chunk_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, k)
        ).fetchall()

        hits = [LexicalHit(cid, rank, b"") for cid, rank in rows]
        if logger:
            logger.info("lexical_retrieve", query=query, hits=len(hits),
                        top_score=hits[0].rank if hits else None)
        return hits

    def index_chunk(self, chunk_id: int, content: str) -> None:
        """Called by A6 (store pipeline) after chunk insert."""
        import hashlib
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        self.conn.execute(
            "INSERT INTO chunk_fts(rowid, content, content_hash) VALUES (?,?,?)",
            (chunk_id, content, content_hash)
        )
