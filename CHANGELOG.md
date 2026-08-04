# Changelog

All notable changes to Lumen will be documented in this file.

## [Unreleased]

### Added
- Domain corpus dataset (`datasets/domain_corpus.json`) — a hand-crafted knowledge corpus with 111 fact chunks across 8 rooms (machine_learning, nlp, cybersecurity, distributed_systems, healthcare_ai, quantum_computing, open_source, climate_tech), each with 4 structured loci. Designed for both palace seeding (`python -m datasets.seed`) and benchmarking.
- Palace seed script (`datasets/seed.py`) — populates a Lumen instance from the domain corpus JSON with real embeddings, creating a production-ready knowledge base in ~2 seconds.
- Palace Navigation Efficiency (PNE) benchmark (`benchmarks/navigation/`) — proprietary evaluation suite measuring latency speedup, recall retention, intent-routing accuracy, and pruning efficiency of Lumen's room/locus topology vs. flat global search. Outputs JSON + Markdown.
- End-to-End Memory Quality benchmark (`benchmarks/e2e/`) — multi-turn conversational memory persistence benchmark with 7 agent personas, 28 queries across 14 sessions. Measures R@k semantic recall, similarity scores, and per-query latency. Bridges Lumen to the SOTA evaluation standards used by MemGPT/Mem0/Zep.
- LangGraph integration (`lumen.integrations.langgraph`) — `LumenCheckpointSaver` for graph state persistence and `LumenGraphStore` for cross-thread long-term memory. Installable via `pip install lumen[langgraph]`. Includes 9 test cases.
- Centralized logging helper (`lumen.logging`) — `get_console_logger()` with structlog preference and stdlib `logging` fallback, eliminating the project-wide `except Exception: pass` anti-pattern from 32 modules.

### Changed
- Roadmap in README updated to reflect M1–M3 completion and M4 in-progress status.
- Full codebase reformatted with `ruff format` for consistent style.
- `benchmarks/run_all.py` now includes the `navigation` and `e2e` suites in the unified orchestrator.
- Dashboard retrieval metrics (`mcp_server.py`) now load benchmark values dynamically from JSON result files instead of hardcoded constants.

### Fixed
- **CRITICAL**: USearch vector backend now gracefully falls back to brute-force (FakeSqliteVecBackend) when the `usearch` package is unavailable, instead of crashing with `NotImplementedError`.
- **CRITICAL**: `SqliteVecBackend.degrade()` no longer silently swallows quantization failures — errors are logged before fallback removal.
- **CRITICAL**: `USearchBackend.degrade()` now properly logs warnings and removes vectors rather than silently doing nothing before removal.
- L2 interference checking (`forgetting_l2_interference.py`) now reads embeddings from `vec_chunks` when `vec_fallback` is empty, fixing a blind spot for sqlite-vec-only deployments.
- API `/turn` endpoint (`server.py`) now resolves real `room_name`, `locus_name`, `vm_score`, and `age_hours` from the database instead of using hardcoded placeholder values.
- LangGraph `alist()` async method now yields items as a proper async generator instead of returning a sync Iterator.
- Removed trailing whitespace in `lumen/api/server.py` docstring.
- Removed unused `os` import in `lumen/api/server.py`.
- Replaced unused local variable assignments with `_` in `tests/test_beam.py` and `tests/test_core.py`.
- Sorted and cleaned import blocks across multiple test files (auto-fixed by ruff).
- Removed dead `pass` in `benchmarks/beir/run.py` except block.
- Removed empty `if TYPE_CHECKING: pass` block in `lumen/curiosity.py`.
- Fixed FTS5 lexical search failing on apostrophes, periods, and other punctuation characters in query strings (`retrieval_lexical.py`).

## [0.1.0] — 2026-07-20

### Added — Milestone 1 (Store & Retrieve)
- A1: SQLite Palace Schema with WAL mode
- A2: BM25 lexical channel (FTS5 bridge)  
- A3: Vector channel (sqlite-vec + USearch adapter)
- A4: FRQAD distance metric
- A5: Provenance tracking
- A6: Write path (store_memory) with dedup + interference
- A7: Interference-based forgetting (L2)
- A8: Ebbinghaus passive decay (L1)
- A9: V(m) value model with learnable weights
- A10: Budget-curated eviction (L3) with optical degradation chain
- A11: Consolidation pass runner
- B1: Intent router
- B2: Mock/ONNX embedder with local model support
- B3: Context assembly with Jinja2 templates + token budget
- C1: Palace onboarding wizard (Illuminate)
- C2: Multi-channel RRF fusion engine with FRQAD rerank
- C3: Epistemic state tracker
- C4: Goal tree with SQLite persistence
- C5: Twin-Force Controller (TFC)
- C6: Sleep-phase consolidation scheduler
- C7: Conversation memory with search→store→feedback loop
- C8: Search repair loop (self-healing)
- C9: Full Typer CLI
- C10: P2P memory sharing (Beam protocol)
- D1: Pydantic-settings based config with device profiles
- D2: FastAPI server with search/store/feedback/assemble/turn endpoints
- D3: Feedback log schema
- D8: Model download & ONNX export pipeline
- D12: Search repair with feedback-driven widening
- D13: Wear-aware batch writer for SD/eMMC endurance
- LangChain integration (`LumenChatMemory` + `LumenStore`)
- 24 test suites with 177 tests
- Retrieval, forgetting, and performance benchmarks
- GitHub Actions CI workflow (lint, typecheck, test, coverage)
- Dockerfile + docker-compose for production deployment

### Fixed — Production Blocker Resolution
- Lazy imports for apscheduler/zeroconf so CLI loads without optional deps
- BM25 retrieval benchmark pid/chunk_id mismatch (was returning 0 recall)
- Forgetting eviction optical_level overflow that permanently hid chunks from the release cycle
- 9 silent `except Exception: pass` blocks replaced with structured logging
- MockEmbedder silent fallback replaced with logged ModelNotAvailableError in production paths
- 59 uncommitted files committed to create a clean reproducible state

### Added — Enterprise Readiness
- Apache-2.0 LICENSE file
- pyproject.toml metadata (license, authors, classifiers, URLs, readme)
- API key authentication middleware with configurable `LUMEN_API_KEY`
- Rate limiting via slowapi with configurable `LUMEN_API_RATE_LIMIT`
- CORS middleware with configurable `LUMEN_ALLOWED_ORIGINS`
- Request ID header propagation (`X-Request-ID`)
- Request body size limit (configurable, default 1 MB)
- Proper pydantic-settings TOML config file support
- TurnRequest Pydantic model replacing bare `dict` endpoint
- Global unhandled exception handler (no stack traces in responses)
- Structured logging for all API operations
- Health check endpoint improved with proper 503 handling
- Coverage measurement (pytest-cov, 75% baseline, 50% CI threshold)
- pre-commit config (ruff + ruff-format + general hooks)
- `.dockerignore` for efficient Docker builds
- Healthcheck instructions in Dockerfile and docker-compose

### Changed
- `LumenConfig.model_post_init` replaced with explicit `resolve_device_defaults()`
- `FallbackEmbedder` is now `MockEmbedder` with a compatibility alias
- `budget_curated_eviction` no longer increments `optical_level` per degradation step
- API server `_state` dict access guarded with null checks
- `/turn` endpoint upgraded from bare `req: dict` to validated `TurnRequest` model