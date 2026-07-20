# Changelog

All notable changes to Lumen will be documented in this file.

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