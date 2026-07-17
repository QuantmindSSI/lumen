# Lumen Open Source Foundation
## A Sovereign Build Blueprint: Memory, Context & Forging on the Edge

**Status:** Build Reference | Classification: Public Architecture  
**Companion Docs:** `agentic-memory-brainstorm.md`, `agentic-memory-brand-bible.md`

---

## Executive Summary

Lumen is designed as a sovereign-first, edge-native agentic memory framework. The architecture in our brainstorm document is ambitious — palaces, twin forces, biologically-grounded forgetting, and Fisher-Rao distances. This document maps that ambition to a **concrete, installable foundation of battle-tested open-source projects**.

**The rule:** prefer embedded, zero-server, CPU-only libraries with permissive licenses. No clouds. No daemons. No Docker on the Pi.

---

## 1. Foundation Philosophy

| Criterion | Open Source Selection Rule |
|---|---|
| **Sovereign** | No network calls required after install. No SaaS wrappers. |
| **Edge-native** | Must run on ≤ 8 GB RAM, ≤ 4 ARM cores, no GPU. |
| **Embedded storage** | File-based or in-process. No PostgreSQL, no Redis server, no Chroma server. |
| **Quantisation-friendly** | Must support INT8 / binary / ONNX runtimes out of the box. |
| **Permissive license** | Apache-2.0, MIT, or BSD preferred. No GPL-3 in the core runtime. |

---

## 2. The Open Source Stack Map

### 2.1 Force A: Mnemonic (The Palace)

| Lumen Component | Open Source Project | Role | Why It Fits |
|---|---|---|---|
| **Relational schema** (rooms, loci, chunks, provenance) | **SQLite** (built-in) + **SQLModel** | Palace topology, bi-temporal provenance chains, adjacency lists for graphs | Zero process, zero port, WAL mode for concurrent reads, battle-hardened on SBCs |
| **Vector index** (< 50k vectors) | **sqlite-vec** | In-SQLite vector search with cosine/dot-product distance | Single file, zero dependencies, crash-safe, no server |
| **Vector index** (50k–500k vectors) | **USearch** (via `usearch` Python/Rust) | HNSW ANN with memory-mapped indices | Pure C++, SIMD on ARM NEON, smaller RAM than FAISS, no server process |
| **Vector index** (fallback) | **hnswlib** | In-memory HNSW graph | Simpler API than USearch, widely used, pickle-serialisable |
| **Embedding inference** | **ONNX Runtime** (`onnxruntime`) + **Optimum** (`optimum[onnxruntime]`) | Run quantised embedding models locally | 5–20× faster than PyTorch on CPU, INT8 quantisation in one line, no CUDA required |
| **Embedding models** | **BGE-small-en-v1.5** (ONNX-exported) or **all-MiniLM-L6-v2** | 384-dim sentence embeddings | BGE: best quality/size trade-off (33 MB INT8). MiniLM: universal fallback (22 MB) |
| **Embedding export** | **Hugging Face `optimum` CLI** | Export `sentence-transformers` models → ONNX → quantise | `optimum-cli export onnx --model BAAI/bge-small-en-v1.5 ./bge_onnx` |
| **Sparse lexical search** | **SQLite FTS5** (built-in) + **rank_bm25** (Python) | BM25 keyword retrieval inside SQLite | Zero dependency, handles names/code/identifiers that dense vectors smear |
| **Graph traversal** (entity KG, palace topology) | **Kùzu** (`kuzu`) | Embedded graph database (Cypher query language) | No server, columnar storage, fits in-process, outperforms NetworkX at scale |
| **Graph analytics** (small scale, in-memory) | **NetworkX** | Spreading activation, BFS/DFS on palace rooms | Pure Python, easy adjacency logic, good for < 10k node subgraphs |
| **Progressive quantisation** (optical forgetting) | **NumPy** + **Numba** (`numba`) | Embedding resolution reduction FP32→FP16→INT8→binary | Numba JIT compiles to native ARM, handles quantisation loops at near-C speed |
| **Forgetting scheduler** | **APScheduler** (`APScheduler`) | Background decay, consolidation cron, idle detection | Mature, thread-safe, supports cron-like schedules and jitter |
| **Sleep-phase consolidation** | **APScheduler** + **psutil** | Trigger when disk idle / CPU low / AC power | psutil reads `iostat`-like metrics cross-platform |
| **Wear-aware I/O batching** | **Python `sqlite3`** (WAL mode) + **aiofiles** | Sequential write batching, async append-only audit logs | WAL = write-ahead logging minimises random I/O on SD/eMMC |
| **KV cache / fast lookup** | **LMDB** (`lmdb`) | Embedding cache, V(m) weights store, TFC state | Memory-mapped, transaction-safe, faster than SQLite for simple key-value |

### 2.2 Force B: Contextual (The Window)

| Lumen Component | Open Source Project | Role | Why It Fits |
|---|---|---|---|
| **Context assembly** | **Jinja2** (built-in) | Template context blocks with palace room labels | Lightweight, proven, allows structured "minimap" injection into prompts |
| **Intent routing** (Stage 1) | **scikit-learn** (`sklearn`) — SGDClassifier on embeddings, or **fastText** (`fasttext`) | Classify query intent: factual / exploratory / relational / temporal | fastText model < 5 MB, trains in seconds, no GPU. sklearn SGD: pure Python ecosystem |
| **Dense retrieval distance** | **Custom FRQAD kernel** + **Numba** | Fisher-Rao Quantization-Aware Distance | No existing open-source FRQAD implementation. Build as Numba/SIMD kernel. |
| **Hybrid fusion** (RRF, LTR) | **NumPy** / **Pandas** | Reciprocal Rank Fusion, linear interpolation, feature sorting | Lightweight data-frame operations on candidate lists (< 200 items) |
| **Epistemic state tracker** | **Python `dataclasses`** + **LMDB** | Known facts, assumed gaps, established truths | Simple structured storage, no schema migration overhead |
| **Goal-tree tracking** | **anytree** (`anytree`) | Hierarchical goal trees with active/pending states | Pure Python, small, renders to dicts for prompt injection |
| **Structured logging** | **structlog** | JSON audit logs, LME-/LCX-/LLM- error codes | Builds the brand's logging identity directly into the pipeline |
| **CLI & terminal UI** | **Typer** + **Rich** | `lumen` command tree, vesica-inspired status panels, progress bars | Rich = the standard for beautiful Python CLI; enables the brand's visual style |

### 2.3 Lumen: The Unification (Controller & Edge)

| Lumen Component | Open Source Project | Role | Why It Fits |
|---|---|---|---|
| **Twin-Force Controller (TFC)** | **Python `pydantic`** + **transitions** (`transitions`) | State machine for (e, a, τ, r) with rules | transitions library gives clean FSM syntax; pydantic validates state |
| **7-factor value model training** | **scipy.optimize** (Nelder-Mead) or **Nevergrad** (`nevergrad`) | Gradient-free per-user weight optimisation | Paper-proven on CPU; Nevergrad has excellent noise-handling for interaction data |
| **Local LLM inference** (sleep consolidation, palace rebuild, NL summaries) | **llama.cpp** (`llama-cpp-python`) | Run Qwen3-1.7B / Gemma-2B / Phi-3-mini in 4-bit on SBC | 15W power envelope, GGUF format standard, Python bindings mature |
| **Alternative local LLM runtime** | **mlc-llm** | Optimised ARM deployment with native quantisation | Slightly harder setup but 20–30% faster than llama.cpp on some ARM boards |
| **Small local classifier / NER** | **spaCy** (`spacy`) — `en_core_web_sm` model (12 MB) | Noun-phrase extraction, entity recognition for palace mining | Industry standard, CPU-fast, models are small when distilled |
| **NLP feature extraction** (onboarding mining) | **spaCy** + **TextBlob** | Sentiment, noun phrases, pairwise comparison parsing | TextBlob is naive but tiny; spaCy does the heavy lifting |
| **Configuration & schemas** | **Pydantic Settings** (`pydantic-settings`) + **tomli** | `.lumen/config.toml`, `.lumen/palace.toml`, env var parsing | Type-safe config, auto-generated CLI help, matches brand conventions |
| **Serialization (fast, low mem)** | **msgspec** (`msgspec`) | Memory chunk serialisation, cache entries | 5–10× faster than pydantic/json for structured data, lower memory |
| **P2P discovery** | **zeroconf** (`zeroconf`) | mDNS device discovery on local network | No central broker, pure Python, works on all tested SBCs |
| **P2P memory sharing (transport)** | **Python `asyncio`** streams + **msgspec** | Forget-to-Improve SHARE protocol between household devices | Low-level control, no extra dependencies, sovereign by design |
| **Battery / power sensing** | **psutil** + **UPower D-Bus** (Linux) | Gate consolidation/curiosity by battery level | psutil covers most cases; UPower for detailed battery semantics on Linux |
| **Cross-compilation / Rust core** | **PyO3** + **Maturin** (`maturin`) | Build Rust performance modules (FRQAD, USearch wrappers) with Python bindings | Officially supported by PyO3 team, builds wheels for ARM64 |

---

## 3. Core Engine Wiring: How the Pieces Fit

### 3.1 Runtime Data Flow

```
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  INTENT CLASSIFIER (fastText / sklearn SGD)                │
│  → factual? exploratory? relational? temporal?             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  PARALLEL RETRIEVAL (Python asyncio + thread pool)         │
│  ├── Channel A: BM25  (SQLite FTS5)                        │
│  ├── Channel B: FRQAD (USearch / sqlite-vec)               │
│  ├── Channel C: KG BFS (Kùzu / NetworkX)                   │
│  └── Channel D: Prefetch LRU (Python dict, 256 KB)         │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  FUSION & RERANK (NumPy + RRF)                             │
│  → score = RRF_score × V(m) × recency_boost                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  CONTEXT ASSEMBLY (Jinja2 template)                        │
│  ├── System prompt                                          │
│  ├── Palace minimap (room labels)                          │
│  ├── Retrieved memories (provenance-tagged)                │
│  └── Goal-tree active branch (anytree)                     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Agent Reasoning (local LLM via llama.cpp, or external if not sovereign)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  CONSOLIDATION (async, low priority)                       │
│  ├── Encode to palace (USearch insert + SQLite row)        │
│  ├── Compute V(m) (7-factor, NumPy dot product)            │
│  ├── Interference check (KG adjacency via Kùzu)            │
│  └── Schedule optical quantisation (APScheduler)           │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Storage Layout on Disk

```
~/.lumen/
├── config.toml                 # pydantic-settings reads this
├── palace.toml                 # topology: rooms, loci, corridors
├── user.toml                   # V(m) weights, TFC calibration
├── store/
│   ├── lumen.db                # SQLite: rooms, loci, chunks, provenance, FTS5
│   ├── lumen.db-wal            # WAL mode = wear-friendly sequential writes
│   ├── vectors.usearch         # USearch index (memory-mapped)
│   ├── graph.kuzu              # Kùzu graph DB (embedded, single directory)
│   └── audit.jsonl             # structlog structured audit trail
├── cache/
│   ├── prefetch.lmdb           # LMDB: embedding query cache
│   └── semantic.lmdb           # LMDB: V(m) scalar cache
└── models/
    ├── bge-small-en-v1.5.onnx  # Optimum-exported, INT8 quantised
    ├── intent_classifier.ftz   # fastText compressed model (< 5 MB)
    └── spacy_sm/               # en_core_web_sm extracted
```

### 3.3 Process Architecture on an SBC

There is **one process**: the Python runtime running `lumen`.

- No database server (SQLite/USearch/Kùzu are in-process).
- No vector service (embedded).
- No graph service (embedded).
- No separate scheduler (APScheduler runs inside the main thread pool).
- Optional: a second process only if running `llama.cpp` server mode for local LLM consolidation, but prefer the `llama-cpp-python` in-process bindings to avoid IPC overhead.

---

## 4. Edge Platform Stacks

### 4.1 Raspberry Pi 5 (4 GB) — Entry Stack

```toml
# ~/.lumen/config.toml  (suggested defaults)
device = "rpi5"
context_budget = 2048          # tokens
memory_limit_mb = 300
embedding_model = "bge-small-en-v1.5-int8.onnx"
embedding_dims = 384
vector_index = "sqlite-vec"    # < 50k memories, zero RAM overhead
enable_hnsw = false
enable_kuzu = false            # use NetworkX for graph < 10k nodes
consolidation_cpu_percent = 5
scheduler_granularity = 300    # 5 minutes
```

```bash
# Installation (one-time, ~3 minutes on RPi5)
pip install lumen \
  sqlite-vec usearch numpy numba \
  onnxruntime optimum[onnxruntime] \
  spacy textblob sklearn apscheduler \
  typer rich structlog pydantic-settings \
  msgspec anytree psutil zeroconf \
  llama-cpp-python --no-binary :all:

# Download spacy model
python -m spacy download en_core_web_sm
```

**Expected footprint:**  
- Python packages: ~180 MB (site-packages)  
- Runtime RAM: ~90 MB (embedding model 22 MB + SQLite + index)  
- Storage per 10k memories: ~35 MB (vectors + text + metadata)

### 4.2 Jetson Orin Nano (8 GB) — Performance Stack

```toml
device = "jetson-orin"
context_budget = 4096
memory_limit_mb = 800
vector_index = "usearch"       # HNSW for up to 500k vectors
enable_kuzu = true             # full graph DB
enable_frqad = true            # custom Numba kernel compiled for ARM NEON
enable_local_llm = true        # Qwen3-1.7B GGUF for sleep-phase consolidation
```

```bash
pip install lumen \
  usearch sqlite-vec kuzu numpy numba \
  onnxruntime optimum[onnxruntime] \
  spacy sklearn apscheduler \
  typer rich structlog pydantic-settings \
  msgspec anytree psutil zeroconf \
  llama-cpp-python

# Compile FRQAD Numba kernel ahead-of-time for ARM
python -c "import lumen.sovereign.frqad; lumen.sovereign.frqad.compile_neon()"
```

**Expected footprint:**  
- Runtime RAM: ~180 MB  
- With Qwen3-1.7B GGUF (Q4_K_M): +1.1 GB when loaded  
- Storage per 100k memories: ~280 MB

### 4.3 Generic x86 / Home Server — Developer Stack

Same as Jetson stack, but with:
- `vector_index = "usearch"` (default)
- `enable_local_llm = true` with Gemma-4B for faster consolidation
- `pytest`, `mypy`, `ruff`, `pre-commit` in dev dependencies

---

## 5. Custom Components: What We Must Build

No open-source project implements these exactly. They are the Lumen differentiation.

| Component | Why Custom | Implementation Notes |
|---|---|---|
| **FRQAD kernel** | Zero existing implementations of Fisher-Rao Quantization-Aware Distance for embeddings | Numba JIT or Rust + PyO3. Derive from Gaussian statistical manifold geodesic. Target ARM NEON SIMD. |
| **Twin-Force Controller (TFC)** | Domain-specific state machine coupling memory pressure, context budget, user satisfaction | `transitions` FSM + pydantic models. Rules are heuristics from the brainstorm, not a generic library concern. |
| **7-factor V(m) value model** | Per-user, gradient-free, interaction-history-based | `scipy.optimize` or `nevergrad`. Feature engineering is bespoke. |
| **Optical forgetting scheduler** | Progressive quantisation tied to palace topology, not generic TTL | Custom logic: every chunk has a `(resolution, last_access, room_coldness)` tuple. APScheduler triggers degrade events. |
| **Palace construction pipeline** | Onboarding → domain taxonomy → room/locus blueprint | spaCy noun-phrase extraction + pairwise comparison UI (built as CLI wizard with Rich). No library automates this cognitive mapping. |
| **Context assembly Jinja schema** | Specific provenance tagging, minimap injection, goal-tree anchoring | Jinja2 templates shipped in `lumen-core`. The schema is the product. |
| **Engram bi-temporal supersession chain** | Logical validity time + transaction time with causal chain pruning | Schema layer on top of SQLite + Kùzu. Custom merge-on-read logic. |
| **Wear-aware write batcher** | SD/eMMC-specific I/O coalescing | Custom async queue + `sqlite3` WAL checkpoint timing. |
| **P2P SHARE protocol** | Security model: household-only, encrypted, ephemeral | zeroconf for discovery + asyncio streams + NaCL (`pynacl`) for encryption. |

---

## 6. Reference Manifests

### 6.1 Python `pyproject.toml` (Core Runtime)

```toml
[project]
name = "lumen"
version = "0.1.0"
description = "Twin-force memory and context framework for sovereign AI agents"
requires-python = ">=3.10"
dependencies = [
    # Force A: Mnemonic
    "sqlite-vec>=0.1.0",
    "usearch>=2.0",
    "kuzu>=0.4",
    "networkx>=3.0",
    "numpy>=1.26",
    "numba>=0.59",
    "lmdb>=1.4",
    "apscheduler>=3.10",
    "rank-bm25>=0.2.2",
    # Force B: Contextual
    "scikit-learn>=1.4",
    "fasttext-wheel>=0.9.2",  # pre-built wheels, avoids GCC hell
    "anytree>=2.12",
    "jinja2>=3.1",
    "structlog>=24.1",
    # Lumen: Unification
    "pydantic>=2.0",
    "pydantic-settings>=2.2",
    "typer>=0.12",
    "rich>=13.0",
    "msgspec>=0.18",
    "transitions>=0.9",
    "nevergrad>=1.0",
    "psutil>=5.9",
    "zeroconf>=0.132",
    "aiofiles>=23.0",
    # NLP / embeddings
    "spacy>=3.7",
    "onnxruntime>=1.17",
    "optimum[onnxruntime]>=1.17",
    # Local LLM (optional, large)
    "llama-cpp-python>=0.2.0 ; platform_machine == 'aarch64' or platform_machine == 'x86_64'",
]

[project.optional-dependencies]
dev = ["pytest", "mypy", "ruff", "pre-commit"]
```

### 6.2 Rust `Cargo.toml` (Performance Kernel)

For FRQAD, USearch bindings, and optional cross-language speedups:

```toml
[package]
name = "lumen-frqad"
version = "0.1.0"
edition = "2021"

[dependencies]
numpy = "0.21"
ndarray = "0.15"
pyo3 = { version = "0.21", features = ["extension-module"] }
num-complex = "0.4"
# Optional: explicit SIMD on aarch64
[target.'cfg(target_arch = "aarch64")'.dependencies]
stdsimd = { git = "https://github.com/rust-lang/stdsimd" }  # or core::arch::aarch64

[lib]
crate-type = ["cdylib"]
```

### 6.3 Environment & Boot Checklist

```bash
# 1. SQLite tuning for SD/eMMC
sqlite3 ~/.lumen/store/lumen.db "PRAGMA journal_mode=WAL;"
sqlite3 ~/.lumen/store/lumen.db "PRAGMA synchronous=NORMAL;"
sqlite3 ~/.lumen/store/lumen.db "PRAGMA mmap_size=268435456;"  # 256 MB

# 2. Disable swap or minimise (prevents flash wear on Pi)
sudo dphys-swapfile swapoff
sudo systemctl disable dphys-swapfile

# 3. CPU governor for latency vs power
# Performance when plugged in
sudo cpupower frequency-set -g performance
# Powersave when on battery (handled by TFC via psutil)

# 4. Limit open files / systemd unit (if running as service)
# /etc/systemd/system/lumen.service
[Service]
ExecStart=/home/pi/.local/bin/lumen daemon
Restart=always
MemoryMax=900M
CPUQuota=80%
```

---

## 7. Development Phases with OSS Milestones

### Phase 1: Foundation (0–3 months)

**Goal:** A working `pip install lumen` on RPi5 that can store, retrieve, and forget.

| Milestone | OSS Projects Involved | Deliverable |
|---|---|---|
| Embedding pipeline | ONNX Runtime + Optimum + BGE-small | `lumen.memory.store("fact", room="prefs")` works end-to-end |
| Vector + lexical search | sqlite-vec + SQLite FTS5 + rank_bm25 | Hybrid retrieval with RRF fusion |
| Basic forgetting | APScheduler + NumPy | TTL decay + manual `forget()` |
| CLI identity | Typer + Rich | `lumen status` shows vesica-themed dashboard |
| Schema & config | Pydantic Settings + SQLModel | `.lumen/` directory structure, config validated |

### Phase 2: Palace & Personalisation (3–6 months)

| Milestone | OSS Projects Involved | Deliverable |
|---|---|---|
| Palace topology | SQLModel + Kùzu / NetworkX | `lumen palace map` renders rooms/loci |
| User research mining | spaCy + TextBlob + Rich wizard | 5-minute onboarding → palace blueprint |
| 7-factor V(m) model | scikit-learn / Nevergrad | Per-user importance scoring learned from feedback |
| Optical forgetting | NumPy + APScheduler | FP32→INT8→binary auto-degrade schedule |
| Context assembly | Jinja2 + anytree | Working context window with minimap & goal-tree |

### Phase 3: Edge Hardening (6–9 months)

| Milestone | OSS Projects Involved | Deliverable |
|---|---|---|
| FRQAD distance | Custom Numba / Rust + PyO3 | Replace cosine with FRQAD in retrieval path |
| USearch at scale | USearch + LMDB | 100k+ memory support on Jetson |
| Wear-aware writes | Custom WAL batcher | Write amplification < 1.1x measured on SD card |
| Sleep consolidation | llama-cpp-python + APScheduler + psutil | Idle-triggered consolidation with local Qwen3-1.7B |
| TFC autonomy | transitions + pydantic | Self-adjusting (e, a, τ, r) from interaction signals |

### Phase 4: Networked & Autonomous (9–12 months)

| Milestone | OSS Projects Involved | Deliverable |
|---|---|---|
| P2P memory sharing | zeroconf + asyncio + msgsync | `lumen p2p share --room travel` between devices |
| Multi-user / GateMem | Kùzu ACL + TFC policies | Shared household agent with per-user palaces |
| Self-evolving palace | llama-cpp-python + Kùzu | Palace reorganisation triggered by priority drift |
| Full benchmark suite | pytest + custom harness | Published RPi5 / Jetson latency & wear numbers |

---

## 8. Governance & Licensing

| Layer | License | Rationale |
|---|---|---|
| `lumen-core` (Python) | Apache-2.0 | Permissive, patent protection, enterprise-friendly |
| `lumen-frqad` (Rust) | Apache-2.0 | Matches core |
| `lumen-docs` | CC-BY-4.0 | Free to share, attribute Lumen |
| Brand assets (vesica logo) | Proprietary | Trademark protected; free for community use under brand guidelines |

**Dependency license audit:**

Run this before every release to ensure no GPL-3 leaks into the default install:

```bash
pip install pip-licenses
pip-licenses --format=markdown --summary --from=mixed | grep -E "GPL|CC-BY-SA"
```

Current known permissive status of key deps:
- SQLite / sqlite-vec: Public Domain / MIT
- USearch: Apache-2.0
- Kùzu: MIT
- ONNX Runtime: MIT
- spaCy: MIT
- llama.cpp / llama-cpp-python: MIT
- NumPy / scikit-learn: BSD-3
- Typer / Rich / Pydantic: MIT

---

## 9. Anti-Stack: What We Deliberately Exclude

| Category | Rejected Project | Why Not Lumen |
|---|---|---|
| Server vector DB | Chroma, Weaviate, Qdrant, Milvus | Requires a daemon process; violates sovereign zero-server rule |
| Server graph DB | Neo4j, ArangoDB | JVM or C++ daemon, RAM hungry, network port required |
| Cloud embedding API | OpenAI, Cohere, Voyage | Sovereign rule: zero API calls |
| Heavy DL framework | PyTorch (default), TensorFlow | Too large for SBCs (PyTorch CPU ~200 MB base); ONNX Runtime is the replacement |
| Container runtime | Docker, Podman | On a 4 GB Pi, Docker overhead is unacceptable; install native venv |
| Message broker | RabbitMQ, NATS, MQTT (broker) | P2P uses direct asyncio streams, no broker |
| Big-data query engine | DuckDB | Excellent project, but overlaps with SQLite + Kùzu; adds 20 MB and another SQL dialect |

---

## 10. Summary

Lumen is not built from scratch. It is **architected from scratch, composed from the best open-source foundations**, and differentiated by the custom layers only we can build: the palace topology, the twin-force controller, the biologically-grounded forgetting physics, and the FRQAD retrieval kernel.

The stack above is **installable today** on a Raspberry Pi 5. The Phase 1 milestone is achievable in weeks, not years, because we stand on the shoulders of sqlite-vec, USearch, ONNX Runtime, spaCy, Typer, Rich, and the broader Python scientific stack.

**The palace is ours to design. The light is ours to focus. The foundation is already open.**

---

*Document version: 0.1.0*  
*Last updated: 2026-07-17*  
*Maintainer: Lumen Architecture Group*
