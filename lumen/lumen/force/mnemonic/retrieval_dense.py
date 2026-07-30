"""A3: Vector Channel (sqlite-vec + USearch Adapter).

Input wire: sqlite-vec (<=50k) OR USearch (50k-500k)
Output wire: C2 (fusion engine), D10 (progressive quantization)
Secret sauce: FRQAD distance metric switch-in (Task A4)
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from lumen.logging import get_console_logger

logger = get_console_logger(__name__)


@dataclass(frozen=True)
class DenseHit:
    chunk_id: int
    score: float  # FRQAD geodesic distance (lower = closer) or cosine (higher = closer)
    vector: np.ndarray


class VectorBackend(Protocol):
    def add(self, chunk_id: int, vector: np.ndarray) -> None: ...
    def search(self, query_vector: np.ndarray, k: int) -> list[DenseHit]: ...
    def remove(self, chunk_id: int) -> None: ...
    def degrade(self, chunk_id: int, new_resolution: str) -> None: ...


class SqliteVecBackend:
    """For < 50k memories. Single-file, zero process."""

    def __init__(self, conn: sqlite3.Connection, dims: int):
        self.conn = conn
        self.dims = dims
        self._has_sqlite_vec = False
        try:
            import sqlite_vec

            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self._has_sqlite_vec = True
        except Exception as exc:
            if logger:
                logger.debug("sqlite_vec_load_failed", error=str(exc))

        # Always maintain vec_fallback as a direct blob-access mirror
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vec_fallback (
                chunk_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            )
        """)

        if self._has_sqlite_vec:
            conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                    chunk_id INTEGER PRIMARY KEY,
                    embedding float[{dims}] distance_metric=cosine
                )
            """)

    def add(self, chunk_id: int, vector: np.ndarray) -> None:
        blob = vector.astype(np.float32).tobytes()
        self.conn.execute(
            "INSERT OR REPLACE INTO vec_fallback(chunk_id, embedding) VALUES (?,?)",
            (chunk_id, blob),
        )
        if self._has_sqlite_vec:
            self.conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk_id,))
            self.conn.execute(
                "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?,?)", (chunk_id, blob)
            )

    def search(self, query_vector: np.ndarray, k: int) -> list[DenseHit]:
        blob = query_vector.astype(np.float32).tobytes()
        if self._has_sqlite_vec:
            rows = self.conn.execute(
                "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (blob, k),
            ).fetchall()
            return [DenseHit(cid, 1.0 - (dist / 2.0), np.array([])) for cid, dist in rows]
        else:
            return self._brute_force_search(query_vector, k)

    def _brute_force_search(self, query_vector: np.ndarray, k: int) -> list[DenseHit]:
        rows = self.conn.execute("SELECT chunk_id, embedding FROM vec_fallback").fetchall()
        q = query_vector.astype(np.float32)
        qn = np.linalg.norm(q)
        hits = []
        for cid, emb_blob in rows:
            vec = np.frombuffer(emb_blob, dtype=np.float32)
            vn = np.linalg.norm(vec)
            sim = 0.0 if vn == 0 or qn == 0 else float(np.dot(q, vec) / (qn * vn))
            hits.append(DenseHit(cid, sim, vec))
        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:k]

    def remove(self, chunk_id: int) -> None:
        self.conn.execute("DELETE FROM vec_fallback WHERE chunk_id = ?", (chunk_id,))
        if self._has_sqlite_vec:
            self.conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk_id,))

def degrade(self, chunk_id: int, new_resolution: str) -> None:
        """Re-quantize the vector from FP32->FP16->INT8->BINARY.

        Falls back to removal if the quantizer is unavailable.
        """
        try:
            from lumen.sovereign.optical import QUANTIZERS

            if new_resolution in QUANTIZERS:
                row = self.conn.execute(
                    "SELECT embedding FROM vec_fallback WHERE chunk_id = ?",
                    (chunk_id,),
                ).fetchone()
                if row:
                    original = np.frombuffer(row[0], dtype=np.float32)
                    quantized = QUANTIZERS[new_resolution](original)
                    blob = np.asarray(quantized, dtype=np.float32).tobytes()
                    self.conn.execute(
                        "UPDATE vec_fallback SET embedding = ? WHERE chunk_id = ?",
                        (blob, chunk_id),
                    )
                    logger.info(
                        "vector_degraded",
                        chunk_id=chunk_id,
                        resolution=new_resolution,
                    )
                    return
        except Exception:
            logger.exception(
                "vector_degrade_failed",
                chunk_id=chunk_id,
                resolution=new_resolution,
            )
        self.remove(chunk_id)


class FakeSqliteVecBackend(SqliteVecBackend):
    """Explicit fallback brute-force backend with identical interface."""

    def __init__(self, conn: sqlite3.Connection, dims: int):
        self.conn = conn
        self.dims = dims
        self._has_sqlite_vec = False
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vec_fallback (
                chunk_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            )
        """)


class USearchBackend:
    """For 50k-500k memories. Memory-mapped HNSW."""

    def __init__(self, path: Path, dims: int):
        try:
            from usearch.index import Index, MetricKind, ScalarKind
        except Exception as exc:
            raise NotImplementedError("usearch is not installed") from exc
        self.path = path
        self.dims = dims
        self.index = Index(
            ndim=dims,
            metric=MetricKind.Cos,
            dtype=ScalarKind.F32,
            expansion_add=128,
            expansion_search=64,
        )
        if path.exists():
            self.index.view(str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.index.save(str(path))

    def add(self, chunk_id: int, vector: np.ndarray) -> None:
        self.index.add(chunk_id, vector.astype(np.float32))

    def search(self, query_vector: np.ndarray, k: int) -> list[DenseHit]:
        matches = self.index.search(query_vector.astype(np.float32), k)
        return [DenseHit(int(m.key), float(m.distance), np.array([])) for m in matches]

    def remove(self, chunk_id: int) -> None:
        try:
            self.index.remove(chunk_id)
        except Exception as exc:
            if logger:
                logger.debug("usearch_remove_failed", chunk_id=chunk_id, error=str(exc))

    def degrade(self, chunk_id: int, new_resolution: str) -> None:
        try:
            from lumen.sovereign.optical import QUANTIZERS

            if new_resolution in QUANTIZERS:
                logger.warning(
                    "usearch_degrade_not_supported",
                    chunk_id=chunk_id,
                    resolution=new_resolution,
                )
                # USearch cannot efficiently extract a single vector.
                # The L3 eviction caller should handle the chunk-table
                # optical_level update.
                self.remove(chunk_id)
                return
        except Exception:
            logger.exception("usearch_degrade_failed", chunk_id=chunk_id)
        self.remove(chunk_id)


class VectorChannel:
    """Runtime switchable backend; config decides, not the caller."""

    def __init__(self, config, conn: sqlite3.Connection):
        from lumen.config import LumenConfig

        cfg: LumenConfig = config
        if cfg.vector_index == "sqlite-vec":
            self.backend: VectorBackend = SqliteVecBackend(conn, cfg.embedding_dims)
        elif cfg.vector_index == "usearch":
            try:
                self.backend = USearchBackend(cfg.store_path / "vectors.usearch", cfg.embedding_dims)
            except Exception as exc:
                logger.error(
                    "usearch_unavailable_falling_back_to_brute_force",
                    error=str(exc),
                )
                self.backend = FakeSqliteVecBackend(conn, cfg.embedding_dims)
        else:
            from lumen.brand.errors import RetrievalEmptyError

            raise RetrievalEmptyError(f"Unknown vector backend: {cfg.vector_index}")

    def add(self, chunk_id: int, vector: np.ndarray) -> None:
        self.backend.add(chunk_id, vector)

    def search(self, query_vector: np.ndarray, k: int) -> list[DenseHit]:
        return self.backend.search(query_vector, k)

    def remove(self, chunk_id: int) -> None:
        self.backend.remove(chunk_id)

    def degrade(self, chunk_id: int, new_resolution: str) -> None:
        self.backend.degrade(chunk_id, new_resolution)
