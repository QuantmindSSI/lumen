# Lumen: Atomic Engineering Tasks
## Wiring Open Source Foundations into a Proprietary Product

**Status:** Engineering Specification | Version: 0.1.0-alpha  
**Companion Docs:** `agentic-memory-brainstorm.md`, `agentic-memory-brand-bible.md`, `lumen-open-source-foundation.md`

---

## 0. Principles of Atomic Wiring

This document breaks the Lumen build into **atomic tasks**: indivisible units of work with a single deliverable, a single owner, and a single pass/fail test. Each task specifies:

1. **Input wire** — exactly which OSS library/API it consumes
2. **Transform logic** — the proprietary algorithm, schema, or policy that converts the OSS output into Lumen behavior
3. **Output wire** — exactly which downstream task consumes this deliverable
4. **Secret sauce boundary** — a ✦ marking where commodity code ends and Lumen differentiation begins

**Engineering mantra:** *We do not wrap libraries. We wire them through a proprietary data model that turns them into a memory palace.*

---

## PART I: Force A — Mnemonic (The Palace)

### Task A1: SQLite Palace Schema Design
**Owner:** Data engineer  
**Estimated:** 2 days

```sql
-- File: lumen/data/schema.sql
-- Input wire: SQLite 3.45+ (WAL mode)
-- Output wire: A2 (FTS5 bridge), A3 (vector table), A5 (provenance), C5 (context assembly)

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE room (
    room_id     INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,          -- e.g. "preferences", "travel:japan"
    created_at  INTEGER DEFAULT (unixepoch()),
    last_entry_at INTEGER,
    locus_count INTEGER DEFAULT 0,
    room_type   TEXT CHECK(room_type IN ('domain','project','person','ephemeral')),
    topological_order REAL DEFAULT 0.0         -- for palace map rendering
);

CREATE TABLE locus (
    locus_id    INTEGER PRIMARY KEY,
    room_id     INTEGER NOT NULL REFERENCES room(room_id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    description TEXT,
    vector_mean BLOB,                          -- cached centroid of resident chunks (optical degrade)
    created_at  INTEGER DEFAULT (unixepoch()),
    access_count INTEGER DEFAULT 0,
    last_access_at INTEGER,
    UNIQUE(room_id, name)
);

CREATE TABLE chunk (
    chunk_id        INTEGER PRIMARY KEY,
    locus_id        INTEGER REFERENCES locus(locus_id) ON DELETE SET NULL,
    room_id         INTEGER NOT NULL REFERENCES room(room_id),
    content         TEXT NOT NULL,             -- raw text / memory payload
    content_hash    TEXT NOT NULL UNIQUE,      -- SHA-256 of content for dedup
    created_at      INTEGER DEFAULT (unixepoch()),
    valid_from      INTEGER DEFAULT (unixepoch()),  -- bi-temporal: Engram pattern
    valid_to        INTEGER,                        -- NULL = still valid
    superseded_by   INTEGER REFERENCES chunk(chunk_id),
    resolution      TEXT DEFAULT 'FP32' CHECK(resolution IN ('FP32','FP16','INT8','BINARY','RELEASED')),
    vm_score        REAL DEFAULT 0.5,          -- V(m) scalar (A9)
    vm_factors      BLOB,                      -- JSON: {goal_relevance:0.8, ...}
    access_count    INTEGER DEFAULT 0,
    last_access_at  INTEGER,
    optical_level   INTEGER DEFAULT 0,         -- 0=full, 1=degraded, 2=forgotten
    provenance_root INTEGER REFERENCES provenance(provenance_id)
);

CREATE TABLE provenance (
    provenance_id   INTEGER PRIMARY KEY,
    chunk_id        INTEGER NOT NULL REFERENCES chunk(chunk_id) ON DELETE CASCADE,
    source_type     TEXT CHECK(source_type IN ('user_input','agent_reasoning','consolidation','import','p2p_share')),
    source_ref      TEXT,                      -- e.g. session_id, turn_number
    confidence      REAL DEFAULT 1.0,
    extraction_method TEXT,
    parent_provenance INTEGER REFERENCES provenance(provenance_id)
);

CREATE INDEX idx_chunk_locus ON chunk(locus_id, vm_score DESC);
CREATE INDEX idx_chunk_room ON chunk(room_id, created_at DESC);
CREATE INDEX idx_chunk_valid ON chunk(valid_from, valid_to) WHERE valid_to IS NULL;
CREATE INDEX idx_chunk_hash ON chunk(content_hash);

-- Full-text search bridge (A2)
CREATE VIRTUAL TABLE chunk_fts USING fts5(
    content, content_hash UNINDEXED,
    content_rowid=chunk_id,
    tokenize='porter unicode61'
);
```

**Acceptance test:**
```python
# tests/test_a1_schema.py
import sqlite3
from lumen.data.schema import init_db

def test_schema_roundtrip():
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cursor.fetchall()}
    assert {'room','locus','chunk','provenance','chunk_fts'} <= tables
```

---

### Task A2: BM25 Lexical Channel (SQLite FTS5 Bridge)
**Owner:** Search engineer  
**Estimated:** 1 day

```python
# File: lumen/force/mnemonic/retrieval_lexical.py
# Input wire: SQLite FTS5 + rank_bm25 library
# Output wire: C2 (fusion engine)
# Secret sauce: None — this is commodity BM25, but wired into the palace schema

import sqlite3
from dataclasses import dataclass
from typing import List
import structlog
from lumen.data.schema import get_connection
from lumen.brand import log_format

logger = structlog.get_logger()

@dataclass(frozen=True)
class LexicalHit:
    chunk_id: int
    rank: float          # BM25 score from FTS5
    match_info: bytes    # raw FTS5 matchinfo for phrase highlighting

class LexicalChannel:
    """Palace-native BM25 over chunk_fts. Zero ML dependency."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.conn.execute("PRAGMA optimize")

    def search(self, query: str, k: int = 20) -> List[LexicalHit]:
        # FTS5 rank is bm25 by default when compiled with rank support
        rows = self.conn.execute(
            """
            SELECT chunk_id, rank, matchinfo(chunk_fts, 'pcxnal')
            FROM chunk_fts
            WHERE chunk_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, k)
        ).fetchall()

        hits = [LexicalHit(cid, rank, mi) for cid, rank, mi in rows]
        logger.info("lexical_retrieve", query=query, hits=len(hits),
                    top_score=hits[0].rank if hits else None)
        return hits

    def index_chunk(self, chunk_id: int, content: str):
        """Called by A6 (store pipeline) after chunk insert."""
        self.conn.execute(
            "INSERT INTO chunk_fts(rowid, content, content_hash) VALUES (?,?,?)",
            (chunk_id, content, "")
        )
```

**Wiring rule:** A6 (store pipeline) must call `index_chunk()` synchronously before logging the store as complete. FTS5 and the `chunk` table must stay consistent within the same SQLite transaction.

---

### Task A3: Vector Channel (sqlite-vec + USearch Adapter)
**Owner:** Search engineer  
**Estimated:** 3 days

```python
# File: lumen/force/mnemonic/retrieval_dense.py
# Input wire: sqlite-vec (<=50k) OR USearch (50k-500k)
# Output wire: C2 (fusion engine)
# ✦ Secret sauce: FRQAD distance metric switch-in (Task A4)

from enum import Enum
from pathlib import Path
from typing import List, Protocol
import numpy as np
import structlog
from lumen.config import LumenConfig
from lumen.brand import RetrievalError

logger = structlog.get_logger()

@dataclass(frozen=True)
class DenseHit:
    chunk_id: int
    score: float      # FRQAD geodesic distance (lower = closer) or cosine (higher = closer)
    vector: np.ndarray

class VectorBackend(Protocol):
    def add(self, chunk_id: int, vector: np.ndarray): ...
    def search(self, query_vector: np.ndarray, k: int) -> List[DenseHit]: ...
    def remove(self, chunk_id: int): ...
    def degrade(self, chunk_id: int, new_resolution: str): ...

class SqliteVecBackend:
    """For < 50k memories. Single-file, zero process."""

    def __init__(self, conn: sqlite3.Connection, dims: int):
        self.conn = conn
        self.dims = dims
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[{dims}] distance_metric=cosine
            )
        """)

    def add(self, chunk_id: int, vector: np.ndarray):
        blob = vector.astype(np.float32).tobytes()
        self.conn.execute(
            "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?,?)",
            (chunk_id, blob)
        )

    def search(self, query_vector: np.ndarray, k: int) -> List[DenseHit]:
        blob = query_vector.astype(np.float32).tobytes()
        rows = self.conn.execute(
            "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (blob, k)
        ).fetchall()
        # sqlite-vec returns cosine DISTANCE (0=identical, 2=opposite)
        # We invert to pseudo-similarity for downstream fusion
        return [DenseHit(cid, 1.0 - (dist/2.0), np.array([])) for cid, dist in rows]

class USearchBackend:
    """For 50k-500k memories. Memory-mapped HNSW."""

    def __init__(self, path: Path, dims: int):
        from usearch.index import Index, MetricKind, ScalarKind
        self.path = path
        self.dims = dims
        self.index = Index(
            ndim=dims,
            metric=MetricKind.Cos,      # ✦ Will be replaced by FRQAD custom metric in A4
            dtype=ScalarKind.F32,
            expansion_add=128,
            expansion_search=64,
        )
        if path.exists():
            self.index.view(str(path))
        else:
            self.index.save(str(path))

    def add(self, chunk_id: int, vector: np.ndarray):
        self.index.add(chunk_id, vector.astype(np.float32))

    def search(self, query_vector: np.ndarray, k: int) -> List[DenseHit]:
        matches = self.index.search(query_vector.astype(np.float32), k)
        return [DenseHit(int(m.key), float(m.distance), np.array([])) for m in matches]

class VectorChannel:
    """Runtime switchable backend; config decides, not the caller."""

    def __init__(self, config: LumenConfig, conn: sqlite3.Connection):
        if config.vector_index == "sqlite-vec":
            self.backend: VectorBackend = SqliteVecBackend(conn, config.embedding_dims)
        elif config.vector_index == "usearch":
            self.backend = USearchBackend(config.store_path / "vectors.usearch", config.embedding_dims)
        else:
            raise RetrievalError(f"Unknown vector backend: {config.vector_index}")

    def search(self, query_vector: np.ndarray, k: int) -> List[DenseHit]:
        return self.backend.search(query_vector, k)
```

**Key integration point:** The `VectorBackend` Protocol is the seam where FRQAD (A4) is injected without changing upstream/downstream code.

---

### Task A4: FRQAD Kernel (Fisher-Rao Quantization-Aware Distance)
**Owner:** Performance engineer (Rust or Numba)  
**Estimated:** 5 days (3 research + 2 implementation)

```python
# File: lumen/sovereign/frqad.py
# Input wire: NumPy arrays (embedding vectors)
# Output wire: A3 (VectorBackend.search), C2 (fusion reranking)
# ✦✦ Secret sauce: This is the single highest-impact search differentiator.

"""
FRQAD treats each normalized embedding as a point on the statistical manifold
of multivariate Gaussian distributions under Fisher information metric.

For unit-norm embeddings x, y in R^d:
    Treat x as center of Gaussian with covariance = identity.
    Fisher-Rao geodesic between N(x, I) and N(y, I) reduces to:
        d_FR(x, y) = arccos( x · y )
    This is exactly the great-circle distance on the sphere.

✦ Extension for quantization awareness:
    When comparing a high-fidelity FP32 vector x against a degraded INT8 vector y,
    we model y as x corrupted by uniform quantization noise with variance
    sigma_q^2 = delta^2 / 12, where delta = quantization bin width.
    The FR distance becomes:
        d_FRQ(x, y) = arccos( x·y / sqrt( (1+sigma_q^2)(1+sigma_q^2) ) )
    This penalizes matching a high-fidelity query to a degraded memory,
    automatically preferring higher-resolution candidates at equal cosine.

Reference implementation below. Production: JIT compile with Numba or
export as PyO3 extension with ARM NEON intrinsics.
"""

import numpy as np
from numba import njit, prange

@njit(fastmath=True, parallel=True, cache=True)
def _frqad_matrix(queries: np.ndarray, candidates: np.ndarray, sigma_q: float) -> np.ndarray:
    """
    queries:    (nq, d)   — typically FP32 query embeddings
    candidates: (nc, d)   — mixed resolution, but cast to float for dot product
    sigma_q:    float     — quantization noise std dev for the candidate batch
    Returns:    (nq, nc) distance matrix
    """
    nq = queries.shape[0]
    nc = candidates.shape[0]
    d = queries.shape[1]
    out = np.empty((nq, nc), dtype=np.float32)

    norm_factor = 1.0 + (sigma_q * sigma_q)

    for i in prange(nq):
        q = queries[i]
        q_norm = 0.0
        for k in range(d):
            q_norm += q[k] * q[k]
        q_norm = np.sqrt(q_norm)

        for j in range(nc):
            c = candidates[j]
            dot = 0.0
            c_norm = 0.0
            for k in range(d):
                dot += q[k] * c[k]
                c_norm += c[k] * c[k]
            c_norm = np.sqrt(c_norm)

            denom = q_norm * c_norm * norm_factor
            if denom < 1e-8:
                out[i, j] = np.pi / 2.0
            else:
                cos_theta = dot / denom
                # Clamp against floating-point drift
                if cos_theta >= 1.0:
                    out[i, j] = 0.0
                elif cos_theta <= -1.0:
                    out[i, j] = np.pi
                else:
                    out[i, j] = np.arccos(cos_theta)
    return out

def compute_frqad(query: np.ndarray, candidate: np.ndarray, resolution: str = "FP32") -> float:
    """Scalar wrapper for single comparisons in reranking (C2)."""
    sigma_map = {"FP32": 0.0, "FP16": 1e-4, "INT8": 0.02, "BINARY": 0.3}
    sigma = sigma_map.get(resolution, 0.0)
    dot = np.dot(query, candidate)
    qn = np.linalg.norm(query)
    cn = np.linalg.norm(candidate)
    denom = qn * cn * (1.0 + sigma * sigma)
    if denom < 1e-8:
        return np.pi / 2.0
    cos_t = np.clip(dot / denom, -1.0, 1.0)
    return float(np.arccos(cos_t))

# To wire into A3 USearch backend:
# USearch's Python binding currently does NOT support custom metrics in Index().
# Therefore, FRQAD with USearch uses a TWO-STAGE approach:
#   Stage 1: USearch with cosine retrieves top 10*k candidates quickly.
#   Stage 2: FRQAD reranks the 10*k subset in pure Numba.
# This is the engineering compromise for using an OSS ANN library.
```

**Rust/PyO3 fast path for ARM NEON (production target):**

```rust
// File: lumen_frqad/src/lib.rs
// Build: maturin develop --release
// Call from Python: import lumen_frqad; lumen_frqad.frqad_neon(q, cands, sigma)

use numpy::{PyArray1, PyArray2, PyReadonlyArray2, IntoPyArray};
use pyo3::prelude::*;
use std::arch::aarch64::*;  // ARM NEON intrinsics

#[pyfunction]
fn frqad_neon<'py>(
    py: Python<'py>,
    queries: PyReadonlyArray2<f32>,
    candidates: PyReadonlyArray2<f32>,
    sigma: f32,
) -> &'py PyArray2<f32> {
    // ... NEON-vectorized dot products + arccos lookup table ...
    // Target: 4–8× speedup over Numba scalar on RPi5 Cortex-A76
    unimplemented!("NEON kernel: see architecture doc for pseudocode")
}

#[pymodule]
fn lumen_frqad(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(frqad_neon, m)?)?;
    Ok(())
}
```

---

### Task A5: Bi-Temporal Provenance Engine
**Owner:** Data engineer  
**Estimated:** 2 days

```python
# File: lumen/force/mnemonic/provenance.py
# Input wire: SQLite schema (A1)
# Output wire: A6 (store), C5 (context assembly), A12 (compliance purge)
# ✦ Secret sauce: Engram-inspired bi-temporal model + supersession chains

from dataclasses import dataclass
from typing import Optional, List
import structlog
from lumen.data.schema import get_connection

logger = structlog.get_logger()

@dataclass
class ProvenanceRecord:
    provenance_id: int
    chunk_id: int
    source_type: str
    source_ref: Optional[str]
    confidence: float
    extraction_method: Optional[str]
    parent_provenance: Optional[int]

def create_provenance(
    conn: sqlite3.Connection,
    chunk_id: int,
    source_type: str,
    source_ref: Optional[str] = None,
    confidence: float = 1.0,
    extraction_method: Optional[str] = None,
    parent_provenance: Optional[int] = None,
) -> int:
    """Bi-temporal provenance: every memory enters with a causal chain."""
    cur = conn.execute(
        """INSERT INTO provenance
           (chunk_id, source_type, source_ref, confidence, extraction_method, parent_provenance)
           VALUES (?,?,?,?,?,?)""",
        (chunk_id, source_type, source_ref, confidence, extraction_method, parent_provenance)
    )
    prov_id = cur.lastrowid
    logger.info("provenance_created", chunk_id=chunk_id, prov_id=prov_id,
                source_type=source_type, parent=parent_provenance)
    return prov_id

def get_effective_fact(
    conn: sqlite3.Connection,
    content_hash_prefix: str,
    as_of_transaction: Optional[int] = None
) -> Optional[dict]:
    """
    ✦ Engram merge-on-read: find the currently valid version of a fact,
    respecting supersession chains. If as_of_transaction is given, time-travel.
    """
    sql = """
        SELECT c.*, p.source_type, p.confidence
        FROM chunk c
        LEFT JOIN provenance p ON c.provenance_root = p.provenance_id
        WHERE c.content_hash LIKE ? || '%'
          AND c.valid_to IS NULL
        ORDER BY c.created_at DESC
        LIMIT 1
    """
    if as_of_transaction:
        # Temporal query: only facts valid at that transaction time
        sql = sql.replace("c.valid_to IS NULL", "c.valid_from <= ? AND (c.valid_to IS NULL OR c.valid_to > ?)")
        row = conn.execute(sql, (content_hash_prefix, as_of_transaction, as_of_transaction)).fetchone()
    else:
        row = conn.execute(sql, (content_hash_prefix,)).fetchone()
    return dict(row) if row else None

def supersede_chunk(conn: sqlite3.Connection, old_chunk_id: int, new_chunk_id: int):
    """Logical update: old fact is deprecated, new fact carries the chain."""
    conn.execute("UPDATE chunk SET valid_to = unixepoch(), superseded_by = ? WHERE chunk_id = ?",
                 (new_chunk_id, old_chunk_id))
    logger.info("chunk_superseded", old=old_chunk_id, new=new_chunk_id)
```

---

### Task A6: Store Pipeline (The Write Path)
**Owner:** Core engineer  
**Estimated:** 3 days

```python
# File: lumen/force/mnemonic/store.py
# Input wire: A1 (schema), A3 (vector backend), A5 (provenance), A9 (V(m) model), B2 (embedding model)
# Output wire: A2 (FTS5 index), A7 (interference), A11 (consolidation queue)
# ✦ Secret sauce: Palace-aware placement + V(m) scoring + interference check on write

from dataclasses import dataclass
import json
import structlog
from lumen.data.schema import get_connection
from lumen.force.mnemonic.retrieval_lexical import LexicalChannel
from lumen.force.mnemonic.retrieval_dense import VectorChannel
from lumen.force.mnemonic.provenance import create_provenance
from lumen.force.mnemonic.value_model import compute_vm
from lumen.brand import PalaceError

logger = structlog.get_logger()

def store_memory(
    conn: sqlite3.Connection,
    content: str,
    room_name: str,
    locus_name: Optional[str] = None,
    source_type: str = "user_input",
    source_ref: Optional[str] = None,
    embedding: Optional[np.ndarray] = None,     # from B2
    vm_weights: Optional[dict] = None,          # from A9
) -> int:
    """
    Atomic store: schema + vector + lexical + provenance in one transaction.
    ✦ Palace placement policy: if locus_name omitted, select by vector similarity
    to existing loci within the room. If no match > 0.85 cosine, create new locus.
    """
    with conn:  # transaction
        # 1. Resolve room
        row = conn.execute("SELECT room_id FROM room WHERE name = ?", (room_name,)).fetchone()
        if not row:
            raise PalaceError(f"LME-1001 RoomNotFound: {room_name}")
        room_id = row[0]

        # 2. Deduplication
        import hashlib
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        existing = conn.execute("SELECT chunk_id FROM chunk WHERE content_hash = ? AND valid_to IS NULL",
                                (content_hash,)).fetchone()
        if existing:
            logger.info("store_dedup", room=room_name, hash=content_hash[:16])
            return existing[0]

        # 3. Locus resolution (palace placement policy)
        locus_id = _resolve_locus(conn, room_id, locus_name, embedding)

        # 4. Compute V(m)
        vm_score, vm_factors = compute_vm(content, vm_weights, source_type)

        # 5. Insert chunk
        cur = conn.execute(
            """INSERT INTO chunk
               (locus_id, room_id, content, content_hash, vm_score, vm_factors, resolution)
               VALUES (?,?,?,?,?,?,?)""",
            (locus_id, room_id, content, content_hash, vm_score,
             json.dumps(vm_factors), "FP32")
        )
        chunk_id = cur.lastrowid

        # 6. Provenance
        prov_id = create_provenance(conn, chunk_id, source_type, source_ref)
        conn.execute("UPDATE chunk SET provenance_root = ? WHERE chunk_id = ?", (prov_id, chunk_id))

        # 7. Vector index (if embedding provided)
        if embedding is not None:
            # VectorChannel instance is managed at runtime; here we pass through
            _get_vector_channel(conn).add(chunk_id, embedding)

        # 8. Lexical index
        _get_lexical_channel(conn).index_chunk(chunk_id, content)

        # 9. ✦ Interference check (A7)
        _trigger_interference_check(conn, room_id, locus_id, chunk_id, embedding)

        logger.info("memory_stored", chunk_id=chunk_id, room=room_name,
                    locus=locus_name, vm=vm_score)
        return chunk_id

def _resolve_locus(conn, room_id, locus_name, embedding):
    if locus_name:
        row = conn.execute("SELECT locus_id FROM locus WHERE room_id=? AND name=?",
                          (room_id, locus_name)).fetchone()
        if row:
            return row[0]
        cur = conn.execute("INSERT INTO locus(room_id, name) VALUES (?,?)",
                          (room_id, locus_name))
        return cur.lastrowid
    # ✦ Auto-placement by vector similarity to locus centroids
    # Stub: in production, query cached locus means and pick best, or create
    # new locus if max similarity < threshold (e.g. 0.82)
    cur = conn.execute("INSERT INTO locus(room_id, name) VALUES (?,?)",
                      (room_id, f"auto_{uuid.uuid4().hex[:8]}"))
    return cur.lastrowid

def _trigger_interference_check(conn, room_id, locus_id, new_chunk_id, embedding):
    """✦ L2 forgetting: same-locus occupancy creates interference weakening."""
    # See Task A7 for full implementation
    pass
```

---

### Task A7: Interference-Based Forgetting (L2)
**Owner:** Memory systems engineer  
**Estimated:** 2 days

```python
# File: lumen/force/mnemonic/forgetting_l2_interference.py
# Input wire: A1 (schema), A6 (store trigger)
# Output wire: A9 (V(m) recalculation), A11 (consolidation)
# ✦ Secret sauce: Locus occupancy causes Weakening — no other memory system does this

from lumen.data.schema import get_connection
import structlog

logger = structlog.get_logger()

INTERFERENCE_THRESHOLD = 0.85  # cosine — when two chunks in same locus are this similar,
                               # older must weaken

def check_locus_interference(conn: sqlite3.Connection, room_id: int, locus_id: int,
                             new_chunk_id: int, new_embedding: np.ndarray):
    """
    When a new memory occupies a locus, check existing residents.
    If similarity > threshold, older chunks suffer Vm penalty (interference decay).
    """
    rows = conn.execute(
        "SELECT chunk_id, vm_score FROM chunk WHERE locus_id = ? AND chunk_id != ? AND valid_to IS NULL",
        (locus_id, new_chunk_id)
    ).fetchall()

    for old_chunk_id, old_vm in rows:
        old_emb = _get_embedding(conn, old_chunk_id)
        if old_emb is None:
            continue
        sim = np.dot(new_embedding, old_emb) / (np.linalg.norm(new_embedding) * np.linalg.norm(old_emb))
        if sim > INTERFERENCE_THRESHOLD:
            penalty = 0.15 * sim  # stronger similarity = stronger interference
            new_vm = max(0.0, old_vm - penalty)
            conn.execute("UPDATE chunk SET vm_score = ? WHERE chunk_id = ?", (new_vm, old_chunk_id))
            logger.info("interference_weakened", old_chunk=old_chunk_id,
                        new_chunk=new_chunk_id, similarity=sim, new_vm=new_vm)
```

---

### Task A8: Ebbinghaus Passive Decay (L1)
**Owner:** Systems engineer  
**Estimated:** 2 days

```python
# File: lumen/force/mnemonic/forgetting_l1_decay.py
# Input wire: APScheduler + SQLite (A1)
# Output wire: A9 (V(m) scalar updates)
# Secret sauce: User-specific decay rate (A13 learns it)

import math
from datetime import datetime, timezone
import structlog
from lumen.data.schema import get_connection

logger = structlog.get_logger()

def ebbinghaus_decay(
    conn: sqlite3.Connection,
    user_half_life_days: float = 7.0,   # ✦ learned per-user (A13)
    now: Optional[datetime] = None,
):
    """
    R(t) = e^(-t / hln(2))  where h = user-specific half-life in days.
    Applied as a multiplicative penalty to vm_score every scheduler tick.
    Chunks with vm_score below 0.05 are queued for release.
    """
    now = now or datetime.now(timezone.utc)
    unix_now = int(now.timestamp())

    rows = conn.execute(
        """SELECT chunk_id, vm_score, last_access_at,
                  ( ? - created_at ) / 86400.0 AS age_days
           FROM chunk
           WHERE valid_to IS NULL AND optical_level < 2
        """, (unix_now,)
    ).fetchall()

    hl_sec = user_half_life_days * 86400.0
    updates = []
    for chunk_id, vm, last_access, age_days in rows:
        retention = math.exp(- (age_days * 86400.0) / (hl_sec * math.log(2)))
        # Boost retention if recently accessed
        if last_access:
            recency_hours = (unix_now - last_access) / 3600.0
            recency_boost = math.exp(-recency_hours / 24.0)  # 24h half-life for recency
            retention = max(retention, recency_boost)
        new_vm = vm * retention
        updates.append((new_vm, chunk_id))

    conn.executemany("UPDATE chunk SET vm_score = ? WHERE chunk_id = ?", updates)
    logger.info("decay_applied", chunks=len(updates), half_life_days=user_half_life_days)
```

---

### Task A9: 7-Factor Value Model V(m)
**Owner:** ML engineer  
**Estimated:** 4 days

```python
# File: lumen/force/mnemonic/value_model.py
# Input wire: spaCy / sklearn (B4), user interaction history (C6)
# Output wire: A6 (store pipeline), C3 (fusion reranking), C7 (TFC)
# ✦ Secret sauce: Per-user learned weights, no API calls, CPU-only

import numpy as np
from dataclasses import dataclass
from typing import Dict
from lumen.data.schema import get_connection

# Default weights for cold-start user (untrained)
DEFAULT_WEIGHTS = {
    "goal_relevance":   0.20,
    "value_alignment":  0.15,
    "self_relevance":   0.15,
    "task_utility":     0.15,
    "emotional_intensity": 0.15,
    "reliability":      0.10,
    "usage_history":    0.10,
}

FACTOR_KEYS = list(DEFAULT_WEIGHTS.keys())

@dataclass
class ValueFactors:
    goal_relevance:     float
    value_alignment:    float
    self_relevance:     float
    task_utility:       float
    emotional_intensity: float
    reliability:        float
    usage_history:      float

def extract_factors(
    content: str,
    source_type: str,
    user_goals: list[str],
    user_values: list[str],
    sentiment_pipeline,   # from spaCy / TextBlob (B4)
) -> Dict[str, float]:
    """
    Compute raw factor scores from content and user profile.
    All operations are local, no API calls.
    """
    # 1. Goal relevance: max cosine similarity to user goal embeddings
    g_rel = _max_similarity_to_phrases(content, user_goals) if user_goals else 0.5

    # 2. Value alignment: keyword overlap with user's stated values
    v_align = _jaccard_overlap(content, user_values) if user_values else 0.5

    # 3. Self/user relevance: pronoun density (I/me/my) as proxy for ego-involvement
    self_words = {"i","me","my","myself"}
    tokens = content.lower().split()
    self_rel = min(1.0, sum(1 for t in tokens if t in self_words) / max(len(tokens), 10))

    # 4. Task utility: does it contain actionable information?
    action_verbs = {"schedule","book","buy","call","email","remind","need","must","should"}
    task_u = 1.0 if any(v in tokens for v in action_verbs) else 0.3

    # 5. Emotional intensity: |sentiment polarity| from TextBlob / spaCy
    pol = abs(sentiment_pipeline(content).sentiment.polarity) if sentiment_pipeline else 0.3
    emo = max(0.3, pol)

    # 6. Reliability: provenance-based (A5 sets this); user_input=0.9, agent_reasoning=0.7, p2p=0.5
    rel_map = {"user_input":0.9, "agent_reasoning":0.7, "consolidation":0.75, "import":0.6, "p2p_share":0.5}
    rel = rel_map.get(source_type, 0.5)

    # 6. Usage history: starts at 0.5, updated by retrieval feedback (C6)
    usage = 0.5

    return {
        "goal_relevance": round(g_rel, 3),
        "value_alignment": round(v_align, 3),
        "self_relevance": round(self_rel, 3),
        "task_utility": round(task_u, 3),
        "emotional_intensity": round(emo, 3),
        "reliability": round(rel, 3),
        "usage_history": round(usage, 3),
    }

def compute_vm(content: str, user_weights: Optional[Dict[str,float]], source_type: str) -> Tuple[float, Dict]:
    """Scalar V(m) = sigmoid( dot(weights, factors) )."""
    weights = {**DEFAULT_WEIGHTS, **(user_weights or {})}
    factors = extract_factors(content, source_type, [], [], None)  # goals/values passed at runtime
    vec = np.array([factors[k] for k in FACTOR_KEYS])
    w = np.array([weights[k] for k in FACTOR_KEYS])
    z = float(np.dot(w, vec))
    vm = 1.0 / (1.0 + math.exp(-z))  # sigmoid to [0,1]
    return vm, factors

def learn_weights_from_feedback(
    conn: sqlite3.Connection,
    user_id: str = "default",
    method: str = "nelder-mead",  # or "nevergrad"
) -> Dict[str, float]:
    """
    ✦ Learn per-user weights from click/retrieval-success feedback.
    Objective: maximise V(m) of chunks that the user *actually* clicked or referenced.
    Gradient-free because n=7 is small and evaluation is a DB query.
    """
    # Extract feedback pairs: (chunk_id, was_useful: bool)
    rows = conn.execute(
        """SELECT c.vm_factors, f.positive
           FROM feedback_log f JOIN chunk c ON f.chunk_id = c.chunk_id
           WHERE f.user_id = ?""", (user_id,)
    ).fetchall()
    if len(rows) < 10:
        return DEFAULT_WEIGHTS  # not enough signal

    # Objective: AUC-like separation between positive and negative
    def loss(w_array):
        w = dict(zip(FACTOR_KEYS, w_array))
        pos_scores = []
        neg_scores = []
        for vm_factors_json, positive in rows:
            factors = json.loads(vm_factors_json)
            vec = np.array([factors[k] for k in FACTOR_KEYS])
            score = 1.0 / (1.0 + math.exp(-np.dot(w_array, vec)))
            if positive:
                pos_scores.append(score)
            else:
                neg_scores.append(score)
        # Simple separability loss
        mean_pos = np.mean(pos_scores) if pos_scores else 0.5
        mean_neg = np.mean(neg_scores) if neg_scores else 0.5
        return -(mean_pos - mean_neg)  # maximise gap

    from scipy.optimize import minimize
    x0 = np.array([DEFAULT_WEIGHTS[k] for k in FACTOR_KEYS])
    result = minimize(loss, x0, method="Nelder-Mead",
                      bounds=[(0.01, 0.99)] * len(FACTOR_KEYS))
    learned = dict(zip(FACTOR_KEYS, result.x.tolist()))
    # Renormalise to sum 1.0
    s = sum(learned.values())
    return {k: round(v/s, 4) for k, v in learned.items()}
```

---

### Task A10: Budget-Curated Forgetting (L3)
**Owner:** Systems engineer  
**Estimated:** 2 days

```python
# File: lumen/force/mnemonic/forgetting_l3_budget.py
# Input wire: SQLite schema, psutil (for RAM), config.memory_limit_mb
# Output wire: A12 (optical degradation / release scheduler)
# ✦ Secret sauce: Net-value-per-byte eviction (Wu et al. inspiration)

import sqlite3
import psutil
import structlog
from lumen.data.schema import get_connection
from lumen.config import LumenConfig

logger = structlog.get_logger()

def budget_curated_eviction(
    conn: sqlite3.Connection,
    config: LumenConfig,
    target_ram_mb: Optional[float] = None,
):
    """
    When resident memory footprint exceeds trigger, evict lowest V(m)/byte candidates.
    Eviction = optical degradation (FP32→FP16→INT8→BINARY→RELEASED), not deletion.
    """
    if target_ram_mb is None:
        target_ram_mb = config.memory_limit_mb * 0.85  # 85% trigger

    proc = psutil.Process()
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    if rss_mb < target_ram_mb:
        return 0

    needed_eviction_mb = rss_mb - (config.memory_limit_mb * 0.75)
    # Approximate bytes per chunk: text_avg 200 B + vector 1536 B (FP32 384-dim)
    bytes_per_chunk = 1800
    chunks_to_evict = int((needed_eviction_mb * 1024 * 1024) / bytes_per_chunk)

    # Select lowest V(m)/byte — here simplified to lowest V(m) × age
    rows = conn.execute(
        """SELECT chunk_id, vm_score, resolution,
                  (julianday('now') - julianday(datetime(created_at, 'unixepoch'))) AS age_days
           FROM chunk
           WHERE valid_to IS NULL AND optical_level < 2
           ORDER BY (vm_score * (1.0 / (age_days + 1.0))) ASC
           LIMIT ?""", (chunks_to_evict,)
    ).fetchall()

    evicted = 0
    for chunk_id, vm, res, age in rows:
        new_res = _next_resolution(res)
        if new_res == "RELEASED":
            conn.execute("UPDATE chunk SET optical_level = 2, valid_to = unixepoch() WHERE chunk_id = ?",
                        (chunk_id,))
        else:
            conn.execute("UPDATE chunk SET resolution = ?, optical_level = optical_level + 1 WHERE chunk_id = ?",
                        (new_res, chunk_id))
        evicted += 1

    logger.info("budget_eviction", evicted=evicted, triggered_at_mb=round(rss_mb,1),
                target_mb=target_ram_mb)
    return evicted

def _next_resolution(current: str) -> str:
    chain = {"FP32":"FP16", "FP16":"INT8", "INT8":"BINARY", "BINARY":"RELEASED"}
    return chain.get(current, "RELEASED")
```

---

### Task A11: Safety-Triggered Forgetting (L4) & Compliance
**Owner:** Security / compliance engineer  
**Estimated:** 2 days

```python
# File: lumen/compliance/safety_forgetting.py
# Input wire: Regex / spaCy NER patterns, user command channel
# Output wire: A12 (audit log), A5 (provenance purge)
# Secret sauce: Provenance-chain deletion (GateMem requirement)

import re
import sqlite3
import structlog
from datetime import datetime, timezone
from lumen.data.schema import get_connection
from lumen.brand import ComplianceError

logger = structlog.get_logger()

# Standard PII patterns + user-defined safety rules
PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b\d{3}-\d{3}-\d{4}\b"),
    "api_key": re.compile(r"[a-zA-Z0-9_-]{32,}"),
}

def safety_scan_chunk(content: str) -> List[str]:
    """Return list of triggered safety rule names."""
    hits = []
    for rule_name, pattern in PII_PATTERNS.items():
        if pattern.search(content):
            hits.append(rule_name)
    return hits

def safety_forget_chunk(conn: sqlite3.Connection, chunk_id: int, reason: str):
    """
    Immediate deletion with provenance-chain clearance.
    Write-ahead audit BEFORE mutation.
    """
    audit = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "safety_triggered_forget",
        "chunk_id": chunk_id,
        "reason": reason,
        "provenance_cleared": True,
    }
    # Write audit FIRST (append-only JSONL)
    with open("~/.lumen/logs/compliance.jsonl", "a") as f:
        f.write(json.dumps(audit) + "\n")

    # Clear provenance chain recursively
    _clear_provenance_tree(conn, chunk_id)

    # Remove from vector index (A3 backend)
    # Remove from FTS5
    conn.execute("DELETE FROM chunk_fts WHERE rowid = ?", (chunk_id,))
    # Logical delete in chunk table
    conn.execute("UPDATE chunk SET valid_to = unixepoch(), content = '[REDACTED]', optical_level = 2 WHERE chunk_id = ?",
                (chunk_id,))

    logger.warning("safety_forget_executed", chunk_id=chunk_id, reason=reason)

def _clear_provenance_tree(conn: sqlite3.Connection, chunk_id: int):
    """Walk provenance parent/child links and anonymise."""
    conn.execute("DELETE FROM provenance WHERE chunk_id = ?", (chunk_id,))
    # In production, also walk parent_provenance to redact upstream
```

---

### Task A12: Wear-Aware Write Batcher
**Owner:** Systems engineer  
**Estimated:** 2 days

```python
# File: lumen/sovereign/wear.py
# Input wire: async queue (Python asyncio), SQLite WAL
# Output wire: Disk I/O
# Secret sauce: SD/eMMC endurance optimisation

import asyncio
import sqlite3
from collections import deque
import structlog

logger = structlog.get_logger()

class WearAwareBatcher:
    """
    Collects write operations (chunk inserts, VM updates, provenance)
    and flushes as large sequential SQLite transactions.
    Target: write amplification < 1.1x vs naive per-query writes.
    """

    def __init__(self, conn: sqlite3.Connection, max_batch_size: int = 100,
                 max_latency_ms: float = 500):
        self.conn = conn
        self.queue = deque()
        self.max_batch = max_batch_size
        self.max_latency = max_latency_ms / 1000.0
        self._lock = asyncio.Lock()
        self._flush_event = asyncio.Event()

    async def enqueue(self, sql: str, params: tuple):
        async with self._lock:
            self.queue.append((sql, params))
            if len(self.queue) >= self.max_batch:
                self._flush_event.set()

    async def run(self):
        while True:
            try:
                await asyncio.wait_for(self._flush_event.wait(), timeout=self.max_latency)
            except asyncio.TimeoutError:
                pass
            await self._flush()

    async def _flush(self):
        async with self._lock:
            if not self.queue:
                return
            batch = list(self.queue)
            self.queue.clear()
            self._flush_event.clear()

        # Synchronous SQLite write in executor thread
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_flush, batch)

    def _sync_flush(self, batch: List[Tuple[str, tuple]]):
        with self.conn:
            for sql, params in batch:
                self.conn.execute(sql, params)
            # WAL checkpoint every batch to amortise fsync cost
            self.conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        logger.debug("wear_flush", count=len(batch))
```

---

## PART II: Force B — Contextual (The Window)

### Task B1: Context Assembly Jinja Schema
**Owner:** Frontend / context engineer  
**Estimated:** 2 days

```python
# File: lumen/force/contextual/assembly.py
# Input wire: Jinja2, retrieval results (C2), goal-tree (C6), TFC state (C7)
# Output wire: LLM prompt string
# ✦ Secret sauce: Palace minimap injection, provenance tagging, budget enforcement

from jinja2 import Environment, PackageLoader
from typing import List
from lumen.config import LumenConfig
from lumen.brand import ContextError

env = Environment(loader=PackageLoader("lumen", "templates"))
ASSEMBLY_TEMPLATE = env.get_template("context_assembly.j2")

def assemble_context(
    query: str,
    retrieved_chunks: List[RetrievedChunk],   # from C2 fusion
    active_goals: GoalTreeNode,               # from C6
    tfc_state: TFCState,                      # from C7
    config: LumenConfig,
) -> str:
    """
    Build the working window. Enforce token budget via character proxy
    (1 token ≈ 4 chars for English text on BPE tokenisers).
    """
    budget_chars = config.context_budget * 4
    system_prompt = _build_system_prompt(tfc_state)
    minimap = _build_minimap(retrieved_chunks)
    goal_block = _render_goal_tree(active_goals)

    # Pack chunks by V(m) descending until budget exhausted
    packed = []
    used = len(system_prompt) + len(minimap) + len(goal_block) + len(query)
    for rc in sorted(retrieved_chunks, key=lambda x: x.final_score, reverse=True):
        chunk_text = f"[Room:{rc.room_name} Locus:{rc.locus_name} Prov:{rc.provenance_id}]\n{rc.content}\n\n"
        if used + len(chunk_text) > budget_chars:
            break
        packed.append(chunk_text)
        used += len(chunk_text)

    if not packed and retrieved_chunks:
        raise ContextError("LCX-2001 BudgetExceeded: even top chunk exceeds context budget")

    return ASSEMBLY_TEMPLATE.render(
        system=system_prompt,
        minimap=minimap,
        goals=goal_block,
        memories="".join(packed),
        query=query,
        tfc=tfc_state,
    )

def _build_minimap(chunks: List[RetrievedChunk]) -> str:
    """High-level palace topology visible in context window."""
    rooms = sorted({c.room_name for c in chunks})
    return "Palace minimap: " + " → ".join(rooms) + "\n"

def _build_system_prompt(tfc: TFCState) -> str:
    personality = "builder" if tfc.e > 0.6 else "explorer"
    return (
        f"You are a sovereign agent with a memory palace. "
        f"Your mnemonic bias is {tfc.e:.2f} ({personality} mode). "
        f"Attend to retrieved memories with care."
    )

def _render_goal_tree(node: GoalTreeNode) -> str:
    # anytree rendering
    lines = []
    for pre, fill, n in RenderTree(node):
        status = "[ACTIVE]" if n.is_active else "[pending]"
        lines.append(f"{pre}{n.name} {status}")
    return "Goals:\n" + "\n".join(lines) + "\n"
```

**Template:** `lumen/templates/context_assembly.j2`

```jinja2
{{ system }}

{{ minimap }}

{{ goals }}

--- Retrieved Memories ---
{{ memories }}
--- End Memories ---

User: {{ query }}
Agent:
```

---

### Task B2: Local Embedding Pipeline (ONNX Runtime)
**Owner:** ML engineer  
**Estimated:** 2 days

```python
# File: lumen/force/contextual/embed.py
# Input wire: ONNX Runtime + Optimum-exported BGE-small (33 MB INT8)
# Output wire: A3 (vector add), A6 (store pipeline), C2 (dense query)
# Secret sauce: None — pure commodity inference, but optimised for edge

import numpy as np
from pathlib import Path
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from lumen.config import LumenConfig

class LocalEmbedder:
    def __init__(self, model_path: Path, dims: int = 384):
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = ORTModelForFeatureExtraction.from_pretrained(str(model_path))
        self.dims = dims
        self._cache = {}  # LRU for repeated queries

    def encode(self, texts: List[str]) -> np.ndarray:
        # ONNX Runtime inference on CPU; batching handled by Optimum
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                                max_length=512, return_tensors="pt")
        outputs = self.model(**inputs)
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        # L2 normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-8)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
```

---

## PART III: Lumen — Unification (Controller, Search, Life Cycle)

### Task C1: Intent Classifier (Stage 1 Router)
**Owner:** ML engineer  
**Estimated:** 1 day

```python
# File: lumen/lumen/intent.py
# Input wire: fastText (5 MB) or sklearn SGD on embeddings
# Output wire: C2 (parallel retrieval channel selection)
# Secret sauce: Deterministic fallback to TFC state if model uncertain

import fasttext
import numpy as np
from lumen.config import LumenConfig
from lumen.lumen.controller import TFCState

class IntentRouter:
    def __init__(self, model_path: Path):
        self.model = fasttext.load_model(str(model_path))

    def classify(self, query: str, tfc: TFCState) -> str:
        label, prob = self.model.predict(query.replace('\n', ' '))
        intent = label[0].replace("__label__", "")
        if prob[0] < 0.7:
            # Fallback: TFC decides when model is uncertain
            if tfc.a > 0.6:
                return "exploratory"
            return "factual"
        return intent

# Training data generation (build phase):
# 10 lines per label, fastText format:
# __label__factual What is my API key?
# __label__relational What's connected to Project X?
# __label__temporal What did I do last Tuesday?
# __label__exploratory Tell me something interesting.
# Compiled into .ftz with: fasttext supervised -input intent.txt -output intent_model
```

---

### Task C2: Fusion & Reranking Engine
**Owner:** Search engineer  
**Estimated:** 3 days

```python
# File: lumen/lumen/fusion.py
# Input wire: A2 (BM25 hits), A3 (dense hits), A4 (FRQAD rerank), B1 (goal-tree), A9 (V(m))
# Output wire: B1 (context assembly)
# ✦ Secret sauce: Multi-channel RRF × V(m) × recency × FRQAD rerank

from dataclasses import dataclass
from typing import List
import numpy as np
from lumen.force.mnemonic.value_model import compute_vm
from lumen.sovereign.frqad import compute_frqad

@dataclass
class RetrievedChunk:
    chunk_id: int
    room_name: str
    locus_name: str
    content: str
    provenance_id: Optional[int]
    rrf_score: float
    vm_score: float
    frqad_score: float
    recency_hours: float
    final_score: float

def fuse_and_rerank(
    lexical_hits: List[LexicalHit],
    dense_hits: List[DenseHit],
    goal_tree_keywords: List[str],
    conn: sqlite3.Connection,
    budget_candidates: int = 200,
) -> List[RetrievedChunk]:
    """
    Stage 3: Reciprocal Rank Fusion + V(m) + FRQAD rerank + recency boost.
    """
    k_rrf = 60

    # Build candidate pool with RRF
    rrfs = {}
    for rank, hit in enumerate(lexical_hits, 1):
        rrfs[hit.chunk_id] = rrfs.get(hit.chunk_id, 0) + 1.0 / (k_rrf + rank)
    for rank, hit in enumerate(dense_hits, 1):
        rrfs[hit.chunk_id] = rrfs.get(hit.chunk_id, 0) + 1.0 / (k_rrf + rank)

    chunk_ids = list(rrfs.keys())[:budget_candidates]
    if not chunk_ids:
        return []

    # Fetch full metadata
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = conn.execute(
        f"""SELECT chunk_id, room_id, locus_id, content, vm_score, provenance_root,
                   (strftime('%s','now') - created_at) / 3600.0 AS age_hours,
                   resolution
            FROM chunk WHERE chunk_id IN ({placeholders}) AND valid_to IS NULL""",
        chunk_ids
    ).fetchall()

    # Pre-fetch embeddings for FRQAD rerank
    query_vec = _get_query_embedding()  # from B2

    results = []
    for row in rows:
        cid, rid, lid, content, vm, prov, age_hours, res = row
        # Goal bonus: if content overlaps active goal keywords, boost
        goal_bonus = 1.0 + (0.2 if any(kw in content for kw in goal_tree_keywords) else 0.0)

        # FRQAD rerank on top candidates
        cand_vec = _get_embedding(conn, cid)
        frqad = compute_frqad(query_vec, cand_vec, res) if cand_vec is not None else np.pi/2
        frqad_sim = 1.0 - (frqad / (np.pi/2))  # convert distance to similarity [0,1]

        rrf = rrfs.get(cid, 0)
        recency_boost = np.exp(-age_hours / 168.0)  # 1 week half-life

        final = rrf * (vm + 0.1) * (frqad_sim + 0.1) * recency_boost * goal_bonus

        # Resolve names
        room = conn.execute("SELECT name FROM room WHERE room_id=?", (rid,)).fetchone()[0]
        locus = conn.execute("SELECT name FROM locus WHERE locus_id=?", (lid,)).fetchone()[0] if lid else "none"

        results.append(RetrievedChunk(
            chunk_id=cid, room_name=room, locus_name=locus,
            content=content, provenance_id=prov,
            rrf_score=rrf, vm_score=vm, frqad_score=frqad_sim,
            recency_hours=age_hours, final_score=final
        ))

    results.sort(key=lambda x: x.final_score, reverse=True)
    return results
```

---

### Task C3: Epistemic State Tracker
**Owner:** Core engineer  
**Estimated:** 2 days

```python
# File: lumen/lumen/epistemic.py
# Input wire: LMDB, context assembly output
# Output wire: C2 (search bias), C7 (TFC)
# Secret sauce: Known / Gap / Truth triage influencing retrieval

import lmdb
from dataclasses import dataclass, field
from typing import Set
import json

@dataclass
class EpistemicState:
    known_facts: Set[str] = field(default_factory=set)       # retrieved this session
    assumed_gaps: Set[str] = field(default_factory=set)     # plan steps missing info
    established_truths: Set[str] = field(default_factory=set)  # confirmed reliability > 0.9

class EpistemicTracker:
    """Session-resident; optionally flushed to LMDB for cross-session continuity."""

    def __init__(self, env_path: Path):
        self.env = lmdb.open(str(env_path), map_size=10*1024*1024)  # 10 MB
        self.state = EpistemicState()

    def mark_known(self, chunk_ids: List[int]):
        self.state.known_facts.update(str(c) for c in chunk_ids)

    def mark_gap(self, description: str):
        self.state.assumed_gaps.add(description)

    def mark_truth(self, content_hash: str):
        self.state.established_truths.add(content_hash)

    def search_bias(self, query_type: str) -> str:
        """
        Return retrieval strategy hint:
        - 'skip' if query is subset of known
        - 'uncertain' if query intersects gaps
        - 'contradiction' if query opposes established truth
        """
        # Simplified: actual implementation does fuzzy matching over embeddings
        return "default"
```

---

### Task C4: Goal-Tree Tracker
**Owner:** Core engineer  
**Estimated:** 2 days

```python
# File: lumen/lumen/goals.py
# Input wire: anytree, agent reasoning output
# Output wire: C2 (goal-guided retrieval), B1 (context assembly)

from anytree import Node, RenderTree
from typing import Optional

class GoalNode(Node):
    def __init__(self, name: str, parent=None, is_active: bool = False):
        super().__init__(name, parent)
        self.is_active = is_active
        self.completion = 0.0

class GoalTree:
    def __init__(self):
        self.root = GoalNode("root")
        self._active = None

    def add_goal(self, name: str, parent_name: Optional[str] = None) -> GoalNode:
        parent = next(
            (n for n in self.root.descendants if n.name == parent_name), self.root
        )
        return GoalNode(name, parent=parent)

    def set_active(self, name: str):
        for n in self.root.descendants:
            n.is_active = (n.name == name)
        self._active = name

    def active_path_keywords(self) -> List[str]:
        """Return names along the active branch for retrieval boosting."""
        if not self._active:
            return []
        node = next(n for n in self.root.descendants if n.name == self._active)
        return [a.name for a in node.ancestors if a.name != "root"] + [node.name]
```

---

### Task C5: Twin-Force Controller (TFC)
**Owner:** Core architect  
**Estimated:** 4 days

```python
# File: lumen/lumen/controller.py
# Input wire: transitions (FSM), pydantic, psutil
# Output wire: A8 (decay rate), A10 (budget trigger), B1 (context assembly personality), C1 (intent fallback)
# ✦✦ Secret sauce: The dynamic equilibrium governing the entire agent

from pydantic import BaseModel, Field
from transitions import Machine
from lumen.config import LumenConfig
import structlog

logger = structlog.get_logger()

class TFCState(BaseModel):
    """
    e: mnemonic conservation bias     [0 = pure flow, 1 = pure structure]
    a: attentional temperature         [0 = narrow focus, 1 = broad scan]
    tau: temporal horizon (days)       [how far back to retrieve]
    r: resolution level                [current quantisation tier for context window]
    """
    e: float = Field(0.5, ge=0.0, le=1.0)
    a: float = Field(0.5, ge=0.0, le=1.0)
    tau: int = Field(7, ge=1, le=365)
    r: int = Field(3, ge=0, le=4)

class TwinForceController:
    states = ["balanced", "encoding", "consolidating", "exploring", "compressing"]

    def __init__(self, config: LumenConfig):
        self.config = config
        self.state = TFCState()
        self.machine = Machine(model=self, states=TwinForceController.states, initial="balanced")
        self.machine.add_transition("detect_novelty", "balanced", "encoding")
        self.machine.add_transition("detect_repetition", "balanced", "consolidating")
        self.machine.add_transition("context_pressure", "balanced", "compressing")
        self.machine.add_transition("goal_shift", "*", "balanced")
        self.machine.add_transition("satisfaction_drop", "*", "exploring")
        self._history = []

    def update(self, interaction_signal: dict):
        """
        Evaluate after every agent turn.
        interaction_signal keys:
            - novelty_score: float (0..1)
            - repetition_flag: bool
            - context_tokens_used: int
            - context_budget: int
            - user_satisfaction_delta: float (-1..+1)
            - goal_changed: bool
        """
        sig = interaction_signal
        novelty = sig.get("novelty_score", 0.0)
        rep = sig.get("repetition_flag", False)
        pressure = sig.get("context_tokens_used", 0) / max(sig.get("context_budget", 1), 1)
        sat = sig.get("user_satisfaction_delta", 0.0)
        goal_changed = sig.get("goal_changed", False)

        if goal_changed:
            self.goal_shift()
            self.state.tau = 7
            self.state.r = 3

        elif novelty > 0.7:
            self.detect_novelty()
            self.state.e = max(0.0, self.state.e - 0.1)  # favour encoding new rooms
            self.state.a = min(1.0, self.state.a + 0.1)  # wider attention

        elif rep:
            self.detect_repetition()
            self.state.e = min(1.0, self.state.e + 0.1)  # consolidate
            self.state.a = max(0.0, self.state.a - 0.1)  # narrow focus

        elif pressure > 0.9:
            self.context_pressure()
            self.state.r = max(0, self.state.r - 1)  # degrade resolution

        elif sat < -0.3:
            self.satisfaction_drop()
            self.state.a = min(1.0, self.state.a + 0.2)  # broader net next time

        logger.info("tfc_update", state=self.state.state, e=self.state.e,
                    a=self.state.a, tau=self.state.tau, r=self.state.r)

    def to_env(self) -> dict:
        return {
            "LUMEN_TFC_E": str(self.state.e),
            "LUMEN_TFC_A": str(self.state.a),
            "LUMEN_TFC_TAU": f"{self.state.tau}d",
            "LUMEN_TFC_R": str(self.state.r),
        }
```

---

### Task C6: Sleep-Phase Consolidation Scheduler
**Owner:** Systems engineer  
**Estimated:** 2 days

```python
# File: lumen/lumen/sleep.py
# Input wire: APScheduler, psutil, llama-cpp-python
# Output wire: A1 (schema updates), A3 (vector index rebuilds)
# ✦ Secret sauce: Idle-time palace maintenance; zero latency impact

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import psutil
import structlog
from lumen.config import LumenConfig
from lumen.force.mnemonic.consolidation import run_consolidation_pass

logger = structlog.get_logger()

class SleepScheduler:
    def __init__(self, config: LumenConfig):
        self.config = config
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(
            self._maybe_consolidate,
            trigger=CronTrigger(hour=3, minute=0),  # 3 AM default
            id="sleep_consolidation",
            replace_existing=True,
        )
        # Also opportunistic: when CPU/disk idle for > 5 minutes
        self.scheduler.add_job(
            self._opportunistic_check,
            trigger="interval", minutes=5,
            id="opportunistic",
        )

    def start(self):
        self.scheduler.start()

    def _maybe_consolidate(self):
        if not self._should_run():
            return
        logger.info("sleep_consolidation_start")
        run_consolidation_pass(self.config)
        logger.info("sleep_consolidation_end")

    def _opportunistic_check(self):
        # Idle detection: < 10% CPU avg over last minute and AC power
        cpu = psutil.cpu_percent(interval=1)
        battery = psutil.sensors_battery()
        on_ac = battery is None or battery.power_plugged
        if cpu < 10.0 and on_ac:
            self._maybe_consolidate()

    def _should_run(self) -> bool:
        # Gate: don't consolidate if low battery (< 50%)
        battery = psutil.sensors_battery()
        if battery and not battery.power_plugged and battery.percent < 50:
            return False
        return True
```

---

### Task C7: Palace Construction Pipeline (Onboarding → Blueprint)
**Owner:** NLP engineer  
**Estimated:** 4 days

```python
# File: lumen/lumen/illuminate.py
# Input wire: spaCy, Rich CLI wizard, SQLite schema
# Output wire: A1 (room/locus creation)
# ✦ Secret sauce: Cognitive mapping from user research to palace topology

import spacy
from rich.console import Console
from rich.prompt import Prompt, Confirm
from lumen.data.schema import get_connection
from lumen.brand import log_format

console = Console()
nlp = spacy.load("en_core_web_sm")

def run_onboarding_wizard(conn: sqlite3.Connection):
    """
    5-minute structured interaction. Extracts domain taxonomy.
    No LLM calls. Pure rule-based NLP + pairwise comparison.
    """
    console.print("[bold #3D5A80]Welcome to Lumen. Let's build your memory palace.[/]")

    # Step 1: Free-text self-description
    desc = Prompt.ask("Describe what you do in 2-3 sentences")
    doc = nlp(desc)

    # Step 2: Noun-phrase extraction for room candidates
    noun_chunks = [nc.text.lower() for nc in doc.noun_chunks if len(nc.text) > 3]
    # Frequency rank
    from collections import Counter
    top_domains = [item for item, _ in Counter(noun_chunks).most_common(7)]

    console.print(f"\n[#E8A838]I detected these domains in your work:[/] {', '.join(top_domains)}")
    confirmed = []
    for domain in top_domains:
        if Confirm.ask(f"Is '{domain}' a major area of your work?", default=True):
            confirmed.append(domain)

    # Step 3: Pairwise priority comparison (5-10 comparisons)
    # Generates topological_order for rooms via Elo-like rating
    rankings = _pairwise_rank(confirmed)

    # Step 4: Room creation with locus seeding
    for rank, domain in enumerate(rankings, 1):
        cur = conn.execute("INSERT INTO room(name, room_type, topological_order) VALUES (?,?,?)",
                          (domain, "domain", float(rank)))
        room_id = cur.lastrowid
        # Seed locus: extract sub-entities from description that co-occur with domain
        seed_loci = _extract_sub_entities(doc, domain)
        for locus in seed_loci[:5]:
            conn.execute("INSERT INTO locus(room_id, name) VALUES (?,?)", (room_id, locus))

    conn.commit()
    console.print("\n[bold #2D8A5E]Palace blueprint created.[/] Rooms: {}".format(len(rankings)))

def _pairwise_rank(items: List[str]) -> List[str]:
    """Simple bubble sort by user preference — 5 min max for ≤7 items."""
    if len(items) <= 1:
        return items
    arr = items[:]
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            pref = Prompt.ask(f"Which is more important? 1) {arr[i]}  2) {arr[j]}",
                              choices=["1","2"], default="1")
            if pref == "2":
                arr[i], arr[j] = arr[j], arr[i]
    return arr

def _extract_sub_entities(doc, domain: str) -> List[str]:
    """Heuristic: tokens that appear near the domain noun in the description."""
    loci = set()
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if domain in sent_text:
            for ent in sent.ents:
                if ent.label_ in {"PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART"}:
                    loci.add(ent.text)
            for token in sent:
                if token.pos_ == "NOUN" and token.text != domain:
                    loci.add(token.lemma_)
    return list(loci)[:5]
```

---

### Task C8: Full Composed Search Pipeline (Stages 1–5 Runtime)
**Owner:** Core architect  
**Estimated:** 3 days

```python
# File: lumen/lumen/search.py
# Input wire: C1 (intent), A2 (BM25), A3 (dense), A4 (FRQAD), A5 (provenance), C2 (fusion), C4 (goals), C5 (TFC)
# Output wire: B1 (context assembly)
# This is the orchestration layer — no new algorithms, just precise wiring.

import time
import asyncio
from typing import List, Optional
import structlog
from lumen.lumen.intent import IntentRouter
from lumen.lumen.fusion import fuse_and_rerank
from lumen.lumen.controller import TwinForceController
from lumen.lumen.goals import GoalTree
from lumen.force.mnemonic.retrieval_lexical import LexicalChannel
from lumen.force.mnemonic.retrieval_dense import VectorChannel
from lumen.force.contextual.embed import LocalEmbedder
from lumen.lumen.epistemic import EpistemicTracker
from lumen.config import LumenConfig

logger = structlog.get_logger()

class SearchPipeline:
    def __init__(self, config: LumenConfig, conn, embedder: LocalEmbedder,
                 tfc: TwinForceController, goals: GoalTree, epistemic: EpistemicTracker):
        self.config = config
        self.conn = conn
        self.embedder = embedder
        self.tfc = tfc
        self.goals = goals
        self.epistemic = epistemic
        self.intent = IntentRouter(config.model_path / "intent_classifier.ftz")
        self.lexical = LexicalChannel(conn)
        self.dense = VectorChannel(config, conn)

    async def execute(self, query: str) -> List[RetrievedChunk]:
        t0 = time.perf_counter()

        # Stage 1: Intent Classification
        intent = self.intent.classify(query, self.tfc.state)
        logger.info("search_intent", intent=intent, query=query[:50])

        # Stage 2: Parallel Retrieval (asyncio thread pool)
        qvec = self.embedder.encode_single(query)

        loop = asyncio.get_event_loop()
        lex_future = loop.run_in_executor(None, self.lexical.search, query, 50)
        dense_future = loop.run_in_executor(None, self.dense.search, qvec, 50)

        lex_hits = await lex_future
        dense_hits = await dense_future

        # Stage 3: Fusion & Rerank
        goal_keywords = self.goals.active_path_keywords()
        fused = fuse_and_rerank(lex_hits, dense_hits, goal_keywords, self.conn)

        # Stage 4: Context Budget Enforcement
        budget = self.config.context_budget
        # Estimate tokens: 4 chars per token heuristic
        used = 0
        final = []
        for rc in fused:
            cost = len(rc.content) // 4 + 10  # +10 for metadata overhead
            if used + cost > budget:
                logger.info("search_budget_cut", included=len(final), excluded=len(fused)-len(final))
                break
            final.append(rc)
            used += cost

        # Stage 5: Feedback & TFC Update
        latency_ms = (time.perf_counter() - t0) * 1000
        self.tfc.update({
            "novelty_score": self._estimate_novelty(qvec),
            "context_tokens_used": used,
            "context_budget": budget,
            "user_satisfaction_delta": 0.0,  # filled by caller after agent turn
        })

        logger.info("search_complete", latency_ms=round(latency_ms,1),
                    candidates=len(fused), included=len(final), intent=intent)
        return final

    def _estimate_novelty(self, qvec: np.ndarray) -> float:
        """Novelty ≈ distance to nearest known fact centroid."""
        # Placeholder: in production, keep session centroid and compare
        return 0.5
```

---

### Task C9: CLI Implementation (Typer + Rich)
**Owner:** UX engineer  
**Estimated:** 3 days

```python
# File: lumen/cli/main.py  (excerpt — full tree in brand bible)
# Input wire: Typer, Rich
# Output wire: All subsystems via API calls

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from lumen.config import load_config
from lumen.lumen.controller import TwinForceController
from lumen.data.schema import get_connection

app = typer.Typer(name="lumen", help="Twin-force memory and context framework")
console = Console()

@app.command()
def status():
    config = load_config()
    conn = get_connection(config)
    tfc = TwinForceController(config)

    # Palace overview
    rooms = conn.execute("SELECT name, locus_count FROM room").fetchall()
    total_chunks = conn.execute("SELECT COUNT(*) FROM chunk WHERE valid_to IS NULL").fetchone()[0]

    table = Table(title="Palace Status", border_style="#1B2A4A")
    table.add_column("Room", style="#3D5A80")
    table.add_column("Loci", justify="right")
    for name, lcount in rooms:
        table.add_row(name, str(lcount))

    console.print(Panel(
        f"[bold #E8A838]⚡ Twin-Force Controller: ACTIVE[/]\n"
        f"Force A (Mnemonic) → e={tfc.state.e:.2f}\n"
        f"Force B (Contextual) → a={tfc.state.a:.2f}\n"
        f"τ={tfc.state.tau}d  r={tfc.state.r}\n\n"
        f"Memory Palace: {len(rooms)} rooms, {total_chunks} chunks lit\n"
        f"Context Window: warm and focused",
        title="Lumen",
        border_style="#7C5CBF"
    ))
    console.print(table)

@app.command()
def init(device: str = typer.Option("generic", "--device", "-d")):
    config = load_config(device=device)
    conn = get_connection(config)
    from lumen.data.schema import init_db
    init_db(conn)
    console.print(f"[bold #2D8A5E]Lumen initialised for device: {device}[/]")

# ... additional commands for palace, context, memory, tfc, compliance, p2p
```

---

### Task C10: P2P Memory Sharing (Beam Protocol)
**Owner:** Distributed systems engineer  
**Estimated:** 3 days

```python
# File: lumen/p2p/beam.py
# Input wire: zeroconf, asyncio, msgspec, pynacl
# Output wire: A6 (store pipeline with source_type='p2p_share')
# Secret sauce: Household-only, encrypted, ephemeral share with permission decay

import asyncio
import zeroconf
from pynacl.public import PrivateKey, Box
import msgspec
import structlog
from lumen.config import LumenConfig
from lumen.data.schema import get_connection

logger = structlog.get_logger()

class BeamNode:
    """
    SHARE protocol from Forget to Improve, adapted for Lumen.
    - Discovers peers via mDNS (_lumen-beam._tcp.local.)
    - Shares room-scoped memories with TTL
    - Receives memories into local palace with lowered reliability (0.5)
    """

    def __init__(self, config: LumenConfig):
        self.config = config
        self.zc = zeroconf.Zeroconf()
        self.service_type = "_lumen-beam._tcp.local."
        self.private_key = PrivateKey.generate()
        self.peers = {}  # hostname -> public_key

    async def start(self):
        # Advertise self
        info = zeroconf.ServiceInfo(
            self.service_type,
            f"{self.config.device_name}.{self.service_type}",
            addresses=[self.config.local_ip],
            port=8847,
        )
        self.zc.register_service(info)
        # Browse for peers
        browser = zeroconf.ServiceBrowser(self.zc, self.service_type, self)
        # Start server
        server = await asyncio.start_server(self._handle_peer, "0.0.0.0", 8847)
        async with server:
            await server.serve_forever()

    async def _handle_peer(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        # NaCl encrypted stream
        # Protocol: [4-byte length][msgspec-encoded MemoryPacket]
        pass

    async def share_room(self, room_name: str, ttl_hours: int = 24):
        conn = get_connection(self.config)
        rows = conn.execute(
            "SELECT chunk_id, content, vm_score, content_hash FROM chunk "
            "JOIN room USING(room_id) WHERE room.name = ? AND valid_to IS NULL",
            (room_name,)
        ).fetchall()
        packet = {
            "room": room_name,
            "ttl": ttl_hours,
            "chunks": [{"content": r[1], "vm": r[2], "hash": r[3]} for r in rows],
        }
        for peer_addr in self.peers.values():
            await self._send(peer_addr, packet)

    async def _send(self, addr, packet):
        # ... encrypted send
        pass
```

---

## PART IV: Integration Matrix & Dependency Order

| Phase | Task | Depends On | Blocked By | Owner | Est. Days |
|---|---|---|---|---|---|
| Foundation | A1 | None | — | Data | 2 |
| Foundation | A2 | A1 | — | Search | 1 |
| Foundation | A3 | A1 | — | Search | 3 |
| Foundation | B2 | None | — | ML | 2 |
| Foundation | A5 | A1 | — | Data | 2 |
| Foundation | A6 | A1,A2,A3,A5,B2 | — | Core | 3 |
| Foundation | A12 | A1 | — | Systems | 2 |
| Foundation | A7 | A6 | — | Memory | 2 |
| Foundation | A8 | A1 | — | Systems | 2 |
| Foundation | A9 | A1 | — | ML | 4 |
| Foundation | A10 | A1,A9 | — | Systems | 2 |
| Foundation | A11 | A1,A5 | — | Security | 2 |
| Foundation | A4 | A3 | — | Perf | 5 |
| Foundation | C1 | None | B2 (embeddings)| ML | 1 |
| Foundation | C2 | A2,A3,A4,A9 | — | Search | 3 |
| Foundation | C3 | None | — | Core | 2 |
| Foundation | C4 | None | — | Core | 2 |
| Foundation | B1 | C2,C3,C4 | — | Context | 2 |
| Foundation | C5 | None | — | Architect | 4 |
| Foundation | C6 | A1,B1 | B2 (local LLM) | Systems | 2 |
| Foundation | C7 | A1 | B4 (spaCy) | NLP | 4 |
| Foundation | C8 | C1,C2,B1,C4,C5 | — | Architect | 3 |
| Foundation | C9 | All above | — | UX | 3 |
| Foundation | C10 | A6 | — | DistSys | 3 |

---

## PART V: Acceptance Criteria by Milestone

### Milestone 1: Store & Retrieve (~Week 3)
- [ ] `lumen init --device rpi5` creates `.lumen/` with SQLite schema
- [ ] `lumen memory store "..." --room prefs` inserts row, FTS5 row, vector index entry
- [ ] `lumen memory retrieve "theme"` returns BM25 + dense fusion results in < 100 ms on RPi5
- [ ] FRQAD reranks top-20 subset with Numba JIT (< 5 ms additional)
- [ ] Wear batcher collects 50 writes and flushes in a single WAL transaction

### Milestone 2: Palace & Forget (~Week 6)
- [ ] Onboarding wizard creates ≥3 rooms with ≥2 loci each from 2-sentence description
- [ ] 7-factor V(m) assigned on store; weights learnable from 10 feedback signals
- [ ] Memory degrades FP32→FP16→INT8 after configurable optical schedule
- [ ] L2 interference penalty measurable: two similar chunks in same locus → older Vm drops
- [ ] L4 safety scan redacts PII on detection with JSONL audit trail

### Milestone 3: Context & Twin Force (~Week 9)
- [ ] Context assembly produces prompt with minimap, goal tree, provenance tags
- [ ] TFC auto-adjusts e/a/τ/r after 10 interactions and can be queried via `lumen tfc show`
- [ ] Sleep consolidation triggers at 03:00 or when CPU < 10% for 5 minutes
- [ ] Budget eviction reduces RAM footprint when psutil RSS > threshold
- [ ] Retrieval latency p95 < 75 ms on RPi5 for 10k-memory palace

### Milestone 4: Network & Polish (~Week 12)
- [ ] `lumen p2p share --room travel` sends encrypted memory packet to discovered peer
- [ ] Two Lumen agents on same LAN share a room and retrieve each other's facts
- [ ] Full CLI brand identity: vesica colors in `lumen status`, structured JSONL logs
- [ ] 100% CPU-only, zero network calls when `LUMEN_SOVEREIGN=true`

---

*End of Atomic Engineering Tasks v0.1.0*
