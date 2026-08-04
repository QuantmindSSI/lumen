# Lumen — Technical Documentation

**Version:** v0.1.0-alpha
**License:** Apache 2.0
**Python:** 3.10+

---

## 1. Overview

Lumen is a local-first, self-contained memory and context framework for AI agents. It provides persistent storage, hybrid retrieval, and managed memory lifecycle for LLM-powered applications. All operations run entirely on-device — no cloud dependencies, no external APIs, no telemetry.

### Core Capabilities

| Capability | Status | Details |
|---|---|---|
| Memory storage | Working | SQLite with WAL, FTS5, bi-temporal schema |
| Hybrid retrieval | Working | BM25 (FTS5) + cosine-similarity vector + optional graph traversal with reciprocal rank fusion |
| Context assembly | Working | Jinja2 templating with token budget enforcement |
| Managed forgetting | Working | Three-layer: Ebbinghaus decay (L1), similarity interference (L2), budget eviction (L3) |
| PII detection | Working | Regex scanning (email, SSN, phone, credit card, API key, IP) with block/redact/hash strategies |
| Audit logging | Working | SQLite audit_log table with structured entries |
| Conversation tracking | Working | Turn storage, implicit feedback, context assembly |
| REST API | Working | FastAPI with auth (header key comparison), rate limiting, CORS |
| MCP server | Working | 7 tools: search, store, assemble, turn, feedback, status, dashboard |
| LangChain adapter | Working | LumenChatMemory (requires `langchain` package) |
| LangGraph adapter | Working | LumenCheckpointSaver (requires `langgraph` package) |
| Dashboard | Working | Real-time palace topology, TFC state, memory health (http://localhost:8848/dashboard) |
| P2P sharing | Working (trusted LAN) | Beam protocol over plaintext TCP; blocked by default (sovereign mode) |
| Encryption-at-rest | Not implemented | Use OS-level encryption (LUKS, FileVault, BitLocker) |

### Not Yet Implemented

- Transport encryption for P2P beam
- BEIR benchmark validation (harness exists, results pending)
- `lumen.api.backup` CLI command
- `lumen.spreading.activation` runtime integration
- Per-user adaptive decay (schema column exists, not wired)
- API versioning (`/v1/` prefix)

---

## 2. Architecture

```
User Input → IntentRouter → Parallel Retrieval (BM25 + Dense + Graph)
                                  │
                                  ▼
                          RRF Fusion × V(m) × FRQAD/cosine × Recency
                                  │
                                  ▼
                          Context Assembly (Jinja2)
                                  │
                                  ▼
                    Sleep Consolidation → Decay → Interference → Eviction
```

### Components

| Layer | Module | Purpose |
|---|---|---|
| Storage | `lumen/data/schema.py` | SQLite schema, connection management, permissions |
| Retrieval | `lumen/search.py` | Orchestrates BM25, dense, graph channels |
| Retrieval | `lumen/force/mnemonic/retrieval_lexical.py` | FTS5 full-text search |
| Retrieval | `lumen/force/mnemonic/retrieval_dense.py` | Vector similarity with sqlite-vec or usearch |
| Retrieval | `lumen/force/mnemonic/retrieval_graph.py` | NetworkX-based graph traversal |
| Fusion | `lumen/fusion.py` | RRF fusion with V(m), FRQAD/cosine, recency boost |
| Memory | `lumen/force/mnemonic/store.py` | Chunk insertion with PII scanning, dedup, provenance |
| Memory | `lumen/force/mnemonic/consolidation.py` | Background consolidation with dedup merging |
| Forgetting | `lumen/force/mnemonic/forgetting_l1_decay.py` | Ebbinghaus exponential decay |
| Forgetting | `lumen/force/mnemonic/forgetting_l2_interference.py` | Similarity-based interference weakening |
| Forgetting | `lumen/force/mnemonic/forgetting_l3_budget.py` | Budget-curated eviction |
| Context | `lumen/force/contextual/assembly.py` | Context assembly with token budget |
| Context | `lumen/force/contextual/embed.py` | ONNX embedding with real/mock fallback |
| Context | `lumen/force/contextual/token_budget.py` | Token counting |
| Controller | `lumen/controller.py` | Twin-Force Controller (mnemonic bias α / attention temperature τ) |
| Repair | `lumen/repair.py` | Self-healing retrieval with TFC-driven re-query |
| Compliance | `lumen/compliance/safety_forgetting.py` | PII detection, safety-triggered deletion |
| Audit | `lumen/audit.py` | SQLite audit event logging |
| Scheduler | `lumen/sleep.py` | Background consolidation, decay, eviction, curiosity |
| P2P | `lumen/p2p/beam.py` | LAN-scoped plaintext memory sharing |
| API | `lumen/api/server.py` | FastAPI REST API with auth, rate limiting |
| CLI | `lumen/cli/main.py` | Typer CLI for init, store, retrieve, status, daemon |
| Integrations | `lumen/integrations/` | MCP server, LangChain, LangGraph adapters |

---

## 3. Memory Palace Data Model

The memory palace organizes data in a structured hierarchy:

```
Room (domain/project/person/ephemeral)
 └── Locus (sub-location within a room)
      └── Chunk (individual memory unit)
           ├── Content (raw text)
           ├── V(m) score (dynamic value model scalar)
           ├── Resolution (FP32 → FP16 → INT8 → BINARY → RELEASED)
           ├── Provenance chain (source tracking)
           └── Embedding vector
```

### Key Tables

| Table | Purpose |
|---|---|
| `room` | Top-level organizational unit |
| `locus` | Sub-location within a room |
| `chunk` | Individual memory with bi-temporal tracking, V(m), resolution |
| `provenance` | Source chain (user_input, agent_reasoning, consolidation, import, p2p_share) |
| `feedback_log` | Explicit/implicit/repair feedback for value model training |
| `audit_log` | Structured security/compliance event log |
| `user_profile` | Per-user preferences, goals, decay parameters |
| `goals` | Active goal tree for context assembly weighting |
| `epistemic_state` | What the system knows/assumes/establishes |

---

## 4. Retrieval Pipeline

### Channels

1. **Lexical (BM25):** SQLite FTS5 with porter stemming and unicode tokenization
2. **Dense (Vector):** Cosine similarity via sqlite-vec or usearch, with ONNX embeddings
3. **Graph:** NetworkX-based traversal from seed chunks via co-occurrence edges

### Fusion

Reciprocal Rank Fusion (k=60) weighted by:
- V(m) scalar — learned memory value
- FRQAD or cosine similarity — quantization-aware or raw vector distance
- Recency boost — exponential decay over 168-hour half-life
- Goal relevance — keyword match bonus

### Repair

When retrieval yields no results, the repair module:
1. Checks if TFC state suggests exploration mode (high α)
2. Re-queries with relaxed parameters
3. Falls back to broader lexical search

---

## 5. Memory Lifecycle

### L1: Ebbinghaus Decay
- Passive exponential decay: `R(t) = e^(-t / (h × ln(2)))`
- Default half-life `h = 7` days
- Configurable via `user_profile.ebbinghaus_half_life_days`
- Chunks below `vm_score < 0.05` are queued for release (logical deletion)

### L2: Similarity Interference
- When new chunks are stored, checks for similarity-conflict within the same locus
- High-similarity chunks with lower V(m) scores are weakened

### L3: Budget Eviction
- Triggers when active chunk count approaches `memory_limit_mb` budget
- Evicts lowest-V(m) chunks, prioritizing those with low access counts
- Batched writes via `WearAwareBatcher` for SD/eMMC endurance

### L4: Safety Forgetting (Manual)
- Triggered via API endpoint or `lumen compliance audit`
- Permanently deletes chunks with provenance-chain clearance
- Writes audit events before mutation

---

## 6. Configuration

All configuration via environment variables (`LUMEN_` prefix), TOML file (`~/.lumen/config.toml`), or programmatic `LumenConfig()`.

### Key Settings

| Setting | Default | Description |
|---|---|---|
| `device` | `generic` | Device profile: rpi5, jetson-orin, orange-pi, generic |
| `context_budget` | 2048 | Token budget for context assembly |
| `memory_limit_mb` | 300 | Soft memory limit for budget eviction |
| `embedding_model` | `bge-small-en-v1.5` | ONNX embedding model |
| `embedding_dims` | 384 | Embedding vector dimensions |
| `vector_index` | `sqlite-vec` | Vector backend: sqlite-vec or usearch |
| `enable_frqad` | `true` | Use FRQAD for vector distance (fallback: cosine) |
| `enable_local_llm` | `false` | Use local LLM for narrative generation |
| `sovereign` | `true` | Block external network calls, P2P, model downloads |
| `pii_detection_enabled` | `true` | Scan stored content for PII |
| `pii_redaction_mode` | `redact` | PII handling: block, redact, or hash |
| `store_path` | `~/.lumen/store` | Database storage path |
| `model_path` | `~/.lumen/models` | Embedding model cache |
| `api_host` / `api_port` | `0.0.0.0` / `8848` | API server binding |
| `api_key` | None (disabled) | Optional API authentication |
| `api_rate_limit` | `60/minute` | Rate limiting |

---

## 7. API Reference

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| GET | `/status` | Palace overview (rooms, loci, chunks, TFC state) |
| POST | `/search` | Hybrid search (`query`, `top_k`) |
| POST | `/store` | Store memory chunk (`content`, `room`, `locus`) |
| POST | `/assemble` | Retrieve + assemble context in one call |
| POST | `/turn` | Store conversation turn with implicit feedback |
| POST | `/feedback` | Log explicit feedback on a retrieved chunk |
| GET | `/dashboard` | Real-time memory palace dashboard (HTML) |
| GET | `/dashboard-data` | Dashboard data as JSON for frontend |
| GET | `/metrics` | Machine-readable metrics for monitoring |

### Authentication

When `LUMEN_API_KEY` is set, all endpoints require `X-API-Key` header. Comparison uses constant-time string comparison.

---

## 8. CLI Reference

```bash
lumen init [--device generic|rpi5|jetson-orin|orange-pi] [--download-model]
lumen serve [--host 0.0.0.0] [--port 8848]
lumen status

# Memory operations
lumen memory store "content text" --room decisions [--locus topic]
lumen memory retrieve "search query" [--top-k 5]

# Palace operations
lumen palace rooms
lumen palace stats [--room name]

# Compliance
lumen compliance audit [--n 10]

# Daemon (background scheduler)
lumen daemon start
lumen daemon status
lumen daemon run-once

# P2P sharing (requires LUMEN_SOVEREIGN=false)
lumen p2p share --room name [--ttl 24]

# Model management
lumen model list
lumen model download bge-small-en-v1.5
lumen model export bge-small-en-v1.5
```

---

## 9. Integrations

### MCP Server

```bash
python -m lumen.integrations.mcp_server
```

Exposes 7 tools via stdio: `lumen_search`, `lumen_store`, `lumen_assemble`, `lumen_turn`, `lumen_feedback`, `lumen_status`, `lumen_dashboard`.

### LangChain

```python
from lumen.integrations.langchain import LumenChatMemory
# Requires: pip install "lumen[langchain]"
```

### LangGraph

```python
from lumen.integrations.langgraph import LumenCheckpointSaver
# Requires: pip install "lumen[langgraph]"
```

---

## 10. Benchmarks

Eight benchmark suites in `benchmarks/`. All run from the repo root.

| Suite | Status | Command |
|---|---|---|
| Retrieval | Implemented | `python -m benchmarks.retrieval.run` |
| E2E quality | Implemented | `python -m benchmarks.e2e.run` |
| Navigation | Implemented | `python -m benchmarks.navigation.run` |
| Ablation | Implemented | `python -m benchmarks.ablation.run` |
| Forgetting | Harness ready | `python -m benchmarks.forgetting.run` |
| Performance | Harness ready | `python -m benchmarks.perf.run` |
| BEIR subset | Harness ready | `python -m benchmarks.beir.run` |
| All suites | Wrapper | `python -m benchmarks.run_all` |

Benchmarks use synthetic corpora unless real datasets (MS MARCO, BEIR) are explicitly loaded. Install benchmark dependencies with `pip install "lumen[benchmarks]"`.

---

## 11. Security Model

- **Sovereign by default:** All external network calls blocked unless `LUMEN_SOVEREIGN=false`
- **Local embeddings:** ONNX Runtime, no external embedding API calls
- **PII scanning:** Regex-based at storage time with block/redact/hash strategies
- **Audit trail:** All operations logged to SQLite `audit_log` table
- **File permissions:** `~/.lumen` enforced to 700 (dirs) / 600 (files) on startup
- **API auth:** Optional `X-API-Key` header with constant-time comparison
- **No encryption-at-rest:** Use OS-level disk encryption (LUKS, FileVault, BitLocker)

Report vulnerabilities via [GitHub Security Advisories](https://github.com/QuantumindSSI/lumen/security/advisories/new).

---

## 12. Known Limitations

1. **No encryption-at-rest** — storage is plaintext SQLite
2. **P2P is plaintext** — Beam transmits over unencrypted TCP (trusted LAN only)
3. **API unversioned** — no `/v1/` prefix
4. **Benchmark results unvalidated** — numbers are from synthetic corpora; real-world BEIR results pending
5. **No CDN SRI** — dashboard loads Chart.js from CDN without integrity hash
6. **Some modules unwired** — backup, spreading, user_profile, log_rotation exist but await integration
7. **Dual scheduler risk** — running both `lumen serve` and `lumen daemon start` creates duplicate schedulers

---

## 13. Project Structure

```
lumen/
├── config.py              # Pydantic-settings configuration
├── controller.py          # Twin-Force state controller
├── search.py              # Search pipeline orchestration
├── fusion.py              # RRF fusion + FRQAD/cosine reranking
├── conversation.py        # Context assembly + turn tracking
├── repair.py              # Self-healing retrieval
├── intent.py              # Intent classifier (keyword + logistic regression)
├── sleep.py               # Background consolidation scheduler
├── audit.py               # SQLite audit event logging
├── api/                   # FastAPI server + dashboard
├── cli/                   # Typer CLI
├── data/                  # Schema, migrations, backup
├── brand/                 # Error hierarchy
├── compliance/            # Safety forgetting, PII audit
├── force/
│   ├── mnemonic/          # Store, retrieval, decay, interference, eviction, provenance
│   └── contextual/        # Embedding, token budget, assembly
├── integrations/          # LangChain, LangGraph, MCP server
├── p2p/                   # Beam P2P sharing protocol
├── sovereign/             # FRQAD, optical quantization, local LLM
tests/                     # 39 test files, 243 tests
benchmarks/                # 8 benchmark suites
```