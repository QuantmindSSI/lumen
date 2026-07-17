# Lumen: Overlooked Atomic Tasks — Addendum v0.1.1
## Critical gaps discovered in review of `lumen-atomic-engineering-tasks.md`

**Reviewer:** Architecture audit  
**Date:** 2026-07-17  
**Action required:** Insert these tasks into the engineering backlog before Milestone 1 begins. Several are blockers for tasks already defined.

---

## 0. Config System — The Missing Foundation

**Impact:** BLOCKER. Referenced by A3, A6, A10, B2, C5, C6, C8, C9, C10 but never atomized.

### Task D1: Configuration Schema & Runtime Loader
**Owner:** Core engineer  
**Estimated:** 1 day  
**Priority:** P0 (must exist before any other task compiles)

```python
# File: lumen/config.py
# Input wire: pydantic-settings, tomli, env vars
# Output wire: EVERY other task
# Secret sauce: Device-specific defaults (RPi5 vs Jetson vs generic)

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Literal

class LumenConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LUMEN_",
        toml_file=[".lumen/config.toml", "~/.lumen/config.toml"],
    )

    device: Literal["rpi5","jetson-orin","orange-pi","generic"] = "generic"
    context_budget: int = 2048
    memory_limit_mb: int = 300
    embedding_model: str = "bge-small-en-v1.5"
    embedding_dims: int = 384
    vector_index: Literal["sqlite-vec","usearch"] = "sqlite-vec"
    enable_kuzu: bool = False
    enable_frqad: bool = False
    enable_local_llm: bool = False
    local_llm_model: str = "Qwen3-1.7B-Q4_K_M.gguf"
    consolidation_cpu_percent: float = 5.0
    scheduler_granularity: int = 300
    store_path: Path = Path.home() / ".lumen" / "store"
    model_path: Path = Path.home() / ".lumen" / "models"
    cache_path: Path = Path.home() / ".lumen" / "cache"
    sovereign: bool = True
    log_level: str = "info"

    @property
    def db_uri(self) -> str:
        return f"{self.store_path}/lumen.db"
```

**Acceptance test:**
```python
def test_config_env_override():
    import os
    os.environ["LUMEN_DEVICE"] = "rpi5"
    os.environ["LUMEN_CONTEXT_BUDGET"] = "4096"
    cfg = LumenConfig()
    assert cfg.device == "rpi5"
    assert cfg.context_budget == 4096
```

---

## 1. Data Model Gaps

### Task D2: Event Memory Circular Buffer (RAM Tier)
**Owner:** Systems engineer  
**Estimated:** 1 day  
**Priority:** P1

The brainstorm establishes a 3-tier hierarchy (Event → Preference → Profile). A1 schema only implements Preference (SQLite) and implicitly Profile (consolidation summaries). The raw event tier — a lossless, short-lived circular buffer in RAM — is missing entirely.

```python
# File: lumen/force/mnemonic/event_buffer.py
# Input wire: Python collections.deque
# Output wire: A6 (store pipeline), A11 (consolidation)
# Secret sauce: Configurable duration (default 24h), RAM-only, feeds downstream tiers

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import threading

@dataclass
class Event:
    event_id: str          # UUID
    timestamp: float       # unixepoch
    raw_text: str          # verbatim user input or agent output
    source: str            # "user" | "agent"
    session_id: str
    metadata: dict

class EventMemoryBuffer:
    """
    Lossless circular buffer of raw interactions. NOT persisted to SQLite.
    Consolidation (Task D8) drains old events into the Preference tier.
    """
    def __init__(self, max_events: int = 10_000, max_age_hours: float = 24.0):
        self._buffer = deque(maxlen=max_events)
        self.max_age_hours = max_age_hours
        self._lock = threading.RLock()

    def append(self, event: Event):
        with self._lock:
            self._buffer.append(event)

    def query_since(self, since: float) -> list[Event]:
        with self._lock:
            return [e for e in self._buffer if e.timestamp >= since]

    def drain_expired(self) -> list[Event]:
        """Return events older than max_age_hours for consolidation."""
        cutoff = (datetime.now(timezone.utc).timestamp()) - (self.max_age_hours * 3600)
        with self._lock:
            expired = [e for e in self._buffer if e.timestamp < cutoff]
            self._buffer = deque([e for e in self._buffer if e.timestamp >= cutoff],
                                 maxlen=self._buffer.maxlen)
        return expired
```

---

### Task D3: Feedback Log Schema & Capture API
**Owner:** Data engineer  
**Estimated:** 0.5 day  
**Priority:** P1

Task A9 references `feedback_log` table which does not exist in A1 schema. V(m) learning is impossible without it.

```sql
-- Add to A1 schema
CREATE TABLE feedback_log (
    feedback_id INTEGER PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunk(chunk_id),
    user_id TEXT DEFAULT 'default',
    positive INTEGER NOT NULL CHECK(positive IN (0,1)),
    feedback_type TEXT DEFAULT 'implicit' CHECK(feedback_type IN ('implicit','explicit','repair')),
    session_id TEXT,
    turn_id INTEGER,
    created_at INTEGER DEFAULT (unixepoch())
);
CREATE INDEX idx_feedback_chunk ON feedback_log(chunk_id, positive);
```

**Capture API:**
```python
# File: lumen/force/contextual/feedback.py
def log_implicit_feedback(conn: sqlite3.Connection, chunk_id: int, was_used: bool):
    """Called by C8 Stage 5 when agent cites a retrieved chunk in its response."""
    conn.execute(
        "INSERT INTO feedback_log(chunk_id, positive, feedback_type) VALUES (?,?,?)",
        (chunk_id, 1 if was_used else 0, "implicit")
    )
```

---

### Task D4: Schema Migration Tool
**Owner:** Data engineer  
**Estimated:** 1 day  
**Priority:** P1

Sovereign AI = user data must survive schema upgrades. No Alembic (too heavy). Custom lightweight migrator.

```python
# File: lumen/data/migrate.py
# Input wire: SQLite, versioned SQL scripts in lumen/data/migrations/
# Output wire: A1 (schema at target version)
# Secret sauce: Single-file manifest tracking; zero external dependencies

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

def migrate(conn: sqlite3.Connection, target_version: int = CURRENT_SCHEMA_VERSION):
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for v in range(current + 1, target_version + 1):
        path = MIGRATIONS_DIR / f"{v:04d}.sql"
        if path.exists():
            conn.executescript(path.read_text())
            conn.execute(f"PRAGMA user_version = {v}")
            logger.info("migration_applied", from_version=current, to_version=v)
```

---

## 2. Retrieval Gaps

### Task D5: Graph Retrieval Channel (Kùzu Entity KG)
**Owner:** Search engineer  
**Estimated:** 2 days  
**Priority:** P1

The foundation doc lists Kùzu. The brainstorm describes graph traversal (BFS/DFS), temporal KG, spreading activation. No atomic task implements any graph-based retrieval. A3 only wires vector indexes.

```python
# File: lumen/force/mnemonic/retrieval_graph.py
# Input wire: Kùzu embedded graph DB (or NetworkX fallback)
# Output wire: C2 (fusion engine)
# ✦ Secret sauce: Seeded graph traversal; never run unseeded on SBC

class GraphChannel:
    def __init__(self, config: LumenConfig):
        if config.enable_kuzu:
            import kuzu
            self.db = kuzu.Database(str(config.store_path / "graph.kuzu"))
            self.conn = kuzu.Connection(self.db)
            self._init_schema()
        else:
            import networkx as nx
            self.nx = nx.DiGraph()  # in-memory, rebuilt from SQLite on startup
            self._rebuild_from_sqlite()

    def traverse_from_seed(self, seed_chunk_id: int, hops: int = 2) -> List[GraphHit]:
        """
        Only run AFTER dense/lexical retrieval has established a seed set.
        Returns related chunks via entity relationships.
        """
        if hasattr(self, 'conn'):
            # Kùzu Cypher
            result = self.conn.execute(
                f"MATCH (c:Chunk)-[*1..{hops}]-(n:Chunk) WHERE c.id = {seed_chunk_id} RETURN n.id"
            )
            return [GraphHit(r[0], 0.5) for r in result]  # placeholder scoring
        else:
            # NetworkX BFS
            if seed_chunk_id not in self.nx:
                return []
            nodes = set()
            for _, successors in nx.bfs_successors(self.nx, seed_chunk_id, depth_limit=hops):
                nodes.update(successors)
            return [GraphHit(int(n), 0.5) for n in nodes]
```

---

### Task D6: Spreading Activation Module
**Owner:** Search engineer  
**Estimated:** 2 days  
**Priority:** P2

Brainstorm 9.2.6 describes spreading activation from SLM V3.3. Not atomized anywhere. This warms up related loci before the agent queries.

```python
# File: lumen/force/mnemonic/spreading.py
# Input wire: Kùzu / NetworkX entity graph, TFC attentional temperature `a`
# Output wire: C8 (prefetch buffer)
# ✦ Secret sauce: Controlled echo-location; γ decay prevents explosion

import numpy as np

SPREAD_GAMMA = 0.4

def spread_activation(graph, seed_ids: list[int], a: float) -> dict[int, float]:
    """
    a ∈ [0,1] maps directly to TFC attentional temperature.
    Higher a → wider spread. Formula from brainstorm 9.2.6.
    """
    activations = {sid: 1.0 for sid in seed_ids}
    threshold = max(0.1, 1.0 - a)  # a=1 → threshold=0 (max spread)
    frontier = set(seed_ids)

    for hop in range(1, 3):  # max 2 hops on SBC
        next_frontier = set()
        for node in frontier:
            for neighbor in graph.neighbors(node):
                weight = graph[node][neighbor].get("weight", 1.0)
                activation = activations.get(node, 0) * (SPREAD_GAMMA ** hop) * weight
                if activation > threshold:
                    activations[neighbor] = max(activations.get(neighbor, 0), activation)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return activations
```

---

### Task D7: Temporal Search & Bi-Temporal Query Engine
**Owner:** Data engineer  
**Estimated:** 2 days  
**Priority:** P2

Brainstorm 9.5 includes temporal queries ("What happened last Tuesday?" / "Why did we decide Z?"). A5 defines bi-temporal fields but no query engine uses them.

```python
# File: lumen/force/mnemonic/retrieval_temporal.py
# Input wire: SQLite schema (valid_from, valid_to, superseded_by)
# Output wire: C2 (fusion engine)
# Secret sauce: Engram merge-on-read; point-in-time + supersession chain walking

def temporal_point_query(
    conn: sqlite3.Connection,
    content_keywords: list[str],
    as_of_unix: Optional[int] = None,
    include_superseded: bool = False,
) -> list[TemporalHit]:
    """Find facts that were valid at a specific point in time."""
    clauses = []
    params = []
    for kw in content_keywords:
        clauses.append("chunk.content LIKE ?")
        params.append(f"%{kw}%")

    time_clause = "chunk.valid_to IS NULL"
    if as_of_unix:
        time_clause = "chunk.valid_from <= ? AND (chunk.valid_to IS NULL OR chunk.valid_to > ?)"
        params.extend([as_of_unix, as_of_unix])

    sql = f"""
        SELECT chunk_id, content, valid_from, valid_to, superseded_by, provenance_root
        FROM chunk
        WHERE ({' OR '.join(clauses)}) AND {time_clause}
        ORDER BY valid_from DESC
    """
    rows = conn.execute(sql, params).fetchall()
    # If include_superseded, walk superseded_by chain to reconstruct belief evolution
    hits = []
    for r in rows:
        hits.append(TemporalHit(r[0], r[1], r[2], r[3], r[4]))
    return hits
```

---

## 3. Model & Asset Provisioning

### Task D8: Model Download & ONNX Export Pipeline
**Owner:** ML engineer  
**Estimated:** 1 day  
**Priority:** P0 (BLOCKER for B2)

B2 assumes model exists at `model_path` but there's no task for acquiring it. SBCs cannot run `optimum-cli` at install time (too slow / may need build tools).

```python
# File: lumen/cli/models.py  (invoked by `lumen init`)
# Input wire: Hugging Face Hub (python), Optimum, ONNX Runtime
# Output wire: ~/.lumen/models/
# Secret sauce: Pre-quantized INT8 artifacts bundled or downloaded; zero compile on Pi

from huggingface_hub import hf_hub_download
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
import shutil

KNOWN_MODELS = {
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "all-MiniLM-L6-v2":  "sentence-transformers/all-MiniLM-L6-v2",
}

def provision_embedding_model(model_name: str, dest: Path) -> Path:
    repo_id = KNOWN_MODELS.get(model_name, model_name)
    cache = dest / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    # 1. Download ONNX export if available on HF Hub (Lumen community space)
    try:
        onnx_path = hf_hub_download(repo_id=f"lumen-ai/{model_name}-onnx-int8",
                                    filename="model.onnx",
                                    local_dir=dest,
                                    cache_dir=cache)
    except Exception:
        # 2. Fallback: export locally on developer machine, NEVER on SBC
        raise RuntimeError(
            "Model not pre-exported. Run `lumen model export` on a workstation, "
            "then copy ~/.lumen/models/ to the SBC."
        )
    return Path(onnx_path)
```

---

### Task D9: Local LLM Runtime Wrapper (llama-cpp-python)
**Owner:** Systems engineer  
**Estimated:** 2 days  
**Priority:** P1 (BLOCKER for C6 sleep consolidation, C7 palace rebuild)

C6 references `llama-cpp-python` but there's no atomic task for model loading, prompt templates, memory management (load/unload to free RAM), or GGUF discovery.

```python
# File: lumen/sovereign/local_llm.py
# Input wire: llama-cpp-python
# Output wire: C6 (consolidation summaries), C7 (palace rebuild NLP)
# Secret sauce: Demand-loaded; unloads when not in use to save SBC RAM

from llama_cpp import Llama
from lumen.config import LumenConfig
import structlog

logger = structlog.get_logger()

class LocalLLM:
    def __init__(self, config: LumenConfig):
        self.config = config
        self._model: Optional[Llama] = None
        self._path = config.model_path / config.local_llm_model

    def _load(self):
        if self._model is None:
            logger.info("local_llm_loading", path=str(self._path))
            self._model = Llama(
                model_path=str(self._path),
                n_ctx=2048,
                verbose=False,
                # n_threads=4 for RPi5 / Jetson
            )

    def _unload(self):
        self._model = None
        import gc
        gc.collect()

    def summarize(self, texts: list[str], instruction: str) -> str:
        """Used by sleep consolidation to generate Profile Memory narrative."""
        self._load()
        prompt = f"<|im_start|>user\n{instruction}\n" + "\n".join(texts) + "<|im_end|>\n<|im_start|>assistant\n"
        output = self._model(prompt, max_tokens=256, stop=["<|im_end|>"])
        result = output["choices"][0]["text"].strip()
        self._unload()  # ✦ Free RAM immediately after inference
        return result
```

---

## 4. Optical Forgetting Mechanics

### Task D10: Progressive Quantization Pipeline (Actual Vector Mutation)
**Owner:** Performance engineer  
**Estimated:** 2 days  
**Priority:** P1

A10 triggers optical degradation but only updates `chunk.resolution` metadata. The actual vectors in sqlite-vec / USearch remain FP32. This task defines the Numba quantization pipeline and re-insertion.

```python
# File: lumen/sovereign/optical.py
# Input wire: NumPy, Numba, VectorBackend (A3)
# Output wire: A3 (degraded re-insert), A10 (triggered by budget)
# ✦ Secret sauce: Quantization-aware storage schedule

import numpy as np
from numba import njit

@njit(cache=True)
def quantize_fp16(vec: np.ndarray) -> np.ndarray:
    return vec.astype(np.float16).astype(np.float32)  # round-trip for storage uniformity

@njit(cache=True)
def quantize_int8(vec: np.ndarray) -> np.ndarray:
    mn, mx = vec.min(), vec.max()
    if mx - mn < 1e-8:
        return np.zeros_like(vec)
    scaled = (vec - mn) / (mx - mn) * 255.0 - 128.0
    return scaled.astype(np.int8).astype(np.float32)  # storage as float32, but info lost

@njit(cache=True)
def quantize_binary(vec: np.ndarray) -> np.ndarray:
    return np.where(vec >= 0, 1.0, -1.0).astype(np.float32)

QUANTIZERS = {
    "FP32": lambda v: v,
    "FP16": quantize_fp16,
    "INT8": quantize_int8,
    "BINARY": quantize_binary,
}

def degrade_chunk_vector(
    backend: VectorBackend,
    chunk_id: int,
    current_res: str,
    target_res: str,
    vector: np.ndarray,
):
    """
    Called by A10 budget eviction and by A8 decay scheduler.
    Re-quantizes and re-inserts into vector index.
    """
    quantized = QUANTIZERS[target_res](vector)
    backend.remove(chunk_id)   # A3 protocol method
    backend.add(chunk_id, quantized)
    logger.info("optical_degrade", chunk_id=chunk_id, from_res=current_res, to_res=target_res)
```

---

## 5. Consolidation & Hierarchy

### Task D11: Consolidation Pass Logic (Preference → Profile)
**Owner:** Core architect  
**Estimated:** 3 days  
**Priority:** P1 (BLOCKER for C6)

C6 schedules consolidation but `run_consolidation_pass` is undefined. The brainstorm describes dedup-based consolidation (58% store reduction) and NL narrative generation.

```python
# File: lumen/force/mnemonic/consolidation.py
# Input wire: A1 (schema), D2 (event buffer), D9 (local LLM), A5 (provenance)
# Output wire: A1 (chunk updates, profile narrative insert)
# ✦ Secret sauce: 6-operation MARS lifecycle applied during idle time

def run_consolidation_pass(config: LumenConfig):
    conn = get_connection(config)
    events = event_buffer.drain_expired()  # D2
    if not events:
        return

    # 1. Extraction: raw events → facts (already done at store time in A6)
    # 2. Dedup-based merge: group by semantic hash
    groups = _group_similar_chunks(conn, threshold=0.95)
    for group in groups:
        if len(group) > 1:
            winner = max(group, key=lambda c: c.vm_score)
            for loser in group:
                if loser.chunk_id != winner.chunk_id:
                    supersede_chunk(conn, loser.chunk_id, winner.chunk_id)  # A5

    # 3. Weakening: old facts decay (handled by A8)
    # 4. Forgetting: optical degrade old chunks (handled by A10)
    # 5. Resynthesis: generate Profile Memory narrative per room
    llm = LocalLLM(config)  # D9
    for room_id, room_name in _get_rooms_with_new_activity(conn):
        recent_facts = _get_recent_fact_bullets(conn, room_id, days=7)
        if len(recent_facts) > 5:
            narrative = llm.summarize(
                recent_facts,
                instruction=f"Summarize the user's recent activity in '{room_name}' as 2-3 concise sentences."
            )
            # Insert narrative as a high-level Profile Memory chunk
            store_memory(conn, content=narrative, room_name=room_name,
                         source_type="consolidation", embedding=embedder.encode_single(narrative))

    conn.commit()
```

---

## 6. Search Repair & Curiosity

### Task D12: Search Repair Loop (Stage 5 Feedback)
**Owner:** Search engineer  
**Estimated:** 1 day  
**Priority:** P2

Brainstorm 9.3 Stage 5 describes feedback & repair when retrieval fails. C8 implements Stages 1–4 but Stage 5 is stubbed.

```python
# File: lumen/lumen/repair.py
# Input wire: C8 (failed search signal), TFC state (C5)
# Output wire: C8 (re-execute with wider net), A9 (feedback_log)
# Secret sauce: Self-healing retrieval; teaches TFC to widen `a`

class SearchRepair:
    def __init__(self, tfc: TwinForceController, pipeline: SearchPipeline):
        self.tfc = tfc
        self.pipeline = pipeline

    async def attempt_repair(self, query: str, reason: str) -> list[RetrievedChunk]:
        if reason == "empty_results":
            # Widen net: drop threshold, increase k, maybe switch to sparse-only
            self.tfc.state.a = min(1.0, self.tfc.state.a + 0.3)
            return await self.pipeline.execute(query)
        elif reason == "budget_exceeded":
            self.tfc.state.r -= 1  # degrade resolution to fit more
            return await self.pipeline.execute(query)
        return []
```

---

### Task D13: Curiosity-Driven Exploration Scheduler
**Owner:** Core engineer  
**Estimated:** 1 day  
**Priority:** P2

Brainstorm 9.4.3 describes curiosity signals (oldest access, high V(m) variance, fewest connections). Not atomized.

```python
# File: lumen/lumen/curiosity.py
# Input wire: A1 (schema), Kùzu/NetworkX graph, TFC state, APScheduler
# Output wire: C8 (prefetch), A6 (reconsolidation trigger)
# Secret sauce: Free memory maintenance during idle time

def curiosity_probe(conn: sqlite3.Connection) -> list[int]:
    """
    Runs opportunistically (during sleep phase or when context padding > 50%).
    Returns chunk_ids worth surfacing.
    """
    # Signal 1: oldest last_access
    old = conn.execute(
        "SELECT chunk_id FROM chunk WHERE valid_to IS NULL ORDER BY last_access_at ASC LIMIT 3"
    ).fetchall()
    # Signal 2: highest variance in V(m) (unresolved/conflicting memories)
    # Signal 3: graph nodes with fewest connections (orphan facts)
    return [r[0] for r in old]
```

---

## 7. Testing & Infrastructure

### Task D14: In-Memory Test Fixtures & Harness
**Owner:** QA engineer  
**Estimated:** 2 days  
**Priority:** P1

No testing infrastructure is atomized. Every task needs a fast way to test against in-memory palace + mock embedder.

```python
# File: tests/conftest.py
# Input wire: pytest, SQLite :memory:, mock embedding model
# Output wire: All test files

import pytest
import sqlite3
import numpy as np
from lumen.data.schema import init_db
from lumen.config import LumenConfig

@pytest.fixture
def memory_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()

@pytest.fixture
def mock_embedder():
    class MockEmbedder:
        dims = 384
        def encode_single(self, text: str) -> np.ndarray:
            # Deterministic pseudo-embedding from text hash
            h = hash(text) % (2**31)
            np.random.seed(h)
            vec = np.random.randn(self.dims).astype(np.float32)
            return vec / np.linalg.norm(vec)
    return MockEmbedder()

@pytest.fixture
def test_config():
    return LumenConfig(vector_index="sqlite-vec", device="generic", store_path="/tmp/.lumen-test")
```

---

## 8. Error Registry & Brand Compliance

### Task D15: Centralized Error/Exception Hierarchy
**Owner:** Core engineer  
**Estimated:** 0.5 day  
**Priority:** P1

Brand bible specifies LME-/LCX-/LLM- error codes. Scattered across snippets but no centralized registry.

```python
# File: lumen/brand/errors.py

class LumenError(Exception):
    code = "LME-0000"

class PalaceError(LumenError):
    pass

class RoomNotFoundError(PalaceError):
    code = "LME-1001"
    def __init__(self, room: str):
        super().__init__(f"The room '{room}' has faded. Let me search nearby loci.")

class LocusConflictError(PalaceError):
    code = "LME-1002"

class ContextError(LumenError):
    pass

class BudgetExceededError(ContextError):
    code = "LCX-2001"

class RetrievalEmptyError(ContextError):
    code = "LCX-2002"

class TFCError(LumenError):
    code = "LLM-3001"

class SovereignViolationError(LumenError):
    code = "LLM-3003"
    def __init__(self, attempted_call: str):
        super().__init__(f"Sovereign boundary: external API call blocked ({attempted_call}).")
```

---

## 9. User Profile Storage (Goals & Values Embeddings)
**Owner:** Data engineer  
**Estimated:** 1 day  
**Priority:** P2

A9 references `user_goals` and `user_values` but there's no schema to store them, nor embeddings for them.

```sql
-- Add to A1 schema
CREATE TABLE user_profile (
    user_id TEXT PRIMARY KEY DEFAULT 'default',
    goals_json TEXT DEFAULT '[]',          -- list of goal strings
    values_json TEXT DEFAULT '[]',         -- list of value strings
    goal_embeddings BLOB,                  -- np.ndarray of shape (n_goals, 384)
    value_embeddings BLOB,
    vm_weights_json TEXT,                  -- serialized 7-factor weights
    ebbinghaus_half_life_days REAL DEFAULT 7.0
);
```

**API:**
```python
# File: lumen/force/mnemonic/user_profile.py
def get_goal_embeddings(conn: sqlite3.Connection, user_id: str = "default") -> np.ndarray:
    row = conn.execute("SELECT goal_embeddings FROM user_profile WHERE user_id=?", (user_id,)).fetchone()
    if row and row[0]:
        return np.frombuffer(row[0], dtype=np.float32).reshape(-1, 384)
    return np.zeros((0, 384), dtype=np.float32)
```

---

## 10. Audit Log Rotation & Disk Guard
**Owner:** Systems engineer  
**Estimated:** 1 day  
**Priority:** P1

A11 writes unbounded JSONL to `~/.lumen/logs/compliance.jsonl`. On an SD card this will eventually exhaust storage.

```python
# File: lumen/sovereign/log_rotation.py
# Input wire: Python stdlib, APScheduler
# Output wire: A11, all structlog sinks

from pathlib import Path
import gzip
import shutil
from datetime import datetime, timedelta

def rotate_jsonl_logs(max_uncompressed_mb: float = 50.0):
    log_dir = Path.home() / ".lumen" / "logs"
    for log_file in log_dir.glob("*.jsonl"):
        size_mb = log_file.stat().st_size / (1024 * 1024)
        if size_mb > max_uncompressed_mb:
            archive = log_file.with_suffix(f".jsonl.{datetime.now():%Y%m%d}.gz")
            with open(log_file, "rb") as f_in, gzip.open(archive, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            log_file.write_text("")  # truncate
```

---

## Summary Table: New Atomic Tasks

| ID | Name | Owner | Est. | Priority | Blocker For |
|---|---|---|---|---|---|
| D1 | Config System | Core | 1d | P0 | All tasks |
| D2 | Event Memory Buffer | Systems | 1d | P1 | D11 (consolidation) |
| D3 | Feedback Log Schema | Data | 0.5d | P1 | A9 (V(m) learning) |
| D4 | Schema Migration Tool | Data | 1d | P1 | Post-M1 upgrades |
| D5 | Graph Retrieval Channel | Search | 2d | P1 | C2 fusion (graph path) |
| D6 | Spreading Activation | Search | 2d | P2 | Prefetch & curiosity |
| D7 | Temporal Search Engine | Data | 2d | P2 | "Last week" queries |
| D8 | Model Provisioning Pipeline | ML | 1d | P0 | B2 (embedder init) |
| D9 | Local LLM Runtime Wrapper | Systems | 2d | P1 | C6, D11, C7 |
| D10 | Progressive Quantization Pipeline | Perf | 2d | P1 | A10 (actual degrade) |
| D11 | Consolidation Pass Logic | Architect | 3d | P1 | C6 (sleep scheduler) |
| D12 | Search Repair Loop | Search | 1d | P2 | C8 Stage 5 |
| D13 | Curiosity Scheduler | Core | 1d | P2 | Idle exploration |
| D14 | Test Fixtures & Harness | QA | 2d | P1 | All acceptance tests |
| D15 | Error Registry | Core | 0.5d | P1 | User-facing error identity |
| D16 | User Profile Storage | Data | 1d | P2 | A9 (goal/value similarity) |
| D17 | Audit Log Rotation | Systems | 1d | P1 | SD card survival |

**Note:** Task references in the original doc to "A13" (user-specific Ebbinghaus decay estimation) should be mapped to **D16** (User Profile Storage) which includes the `ebbinghaus_half_life_days` field.

---

*End of Addendum v0.1.1*
