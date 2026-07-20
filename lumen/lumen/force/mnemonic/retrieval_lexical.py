"""A2: BM25 Lexical Channel (SQLite FTS5 Bridge).

Input wire: SQLite FTS5 + rank_bm25 library
Output wire: C2 (fusion engine)
Secret sauce: None — this is commodity BM25, but wired into the palace schema
"""

import contextlib
import re
import sqlite3
from dataclasses import dataclass

logger = None
try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    pass

# Characters that FTS5 treats as operators/special and that commonly
# appear in user queries (e.g. question marks, quotes, asterisks).
_FTS5_SPECIAL_RE = re.compile(r'[?"*~^()+-]')


@dataclass(frozen=True)
class LexicalHit:
    chunk_id: int
    rank: float          # BM25 score from FTS5 (lower is better in raw FTS5 rank)
    match_info: bytes    # raw FTS5 matchinfo for phrase highlighting


class LexicalChannel:
    """Palace-native BM25 over chunk_fts. Zero ML dependency."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        with contextlib.suppress(Exception):
            self.conn.execute("PRAGMA optimize")

    @staticmethod
    def _sanitize(query: str) -> str:
        """Remove FTS5-special characters and convert to OR-query format."""
        cleaned = _FTS5_SPECIAL_RE.sub(" ", query)
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            return ""
        tokens = cleaned.split()
        if len(tokens) == 1:
            return tokens[0]
        return " OR ".join(tokens)

    def search(self, query: str, k: int = 20) -> list[LexicalHit]:
        """FTS5 MATCH with bm25 ranking."""
        safe_query = self._sanitize(query)
        if not safe_query:
            return []
        rows = self.conn.execute(
            """
            SELECT rowid, rank
            FROM chunk_fts
            WHERE chunk_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (safe_query, k)
        ).fetchall()

        hits = [LexicalHit(cid, rank, b"") for cid, rank in rows]
        if logger:
            logger.info("lexical_retrieve", query=safe_query, hits=len(hits),
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
