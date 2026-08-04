# Lumen: Technical Summary

**Version:** v0.1.0-alpha | **Type:** Local-first agentic memory framework | **License:** Apache 2.0

---

## What Lumen Does

Lumen provides a persistent, searchable memory store for LLM agents. It organizes information in a spatial metaphor (rooms, loci, chunks), retrieves via multi-channel hybrid search (BM25 + vector + graph), and manages memory lifecycle through controlled forgetting (decay, interference, eviction).

### Proven

- **243 tests (73% coverage)** — all passing, 2 skipped (optional integration deps)
- **End-to-end flow:** store → search → assemble → feedback → forget
- **API server:** FastAPI with auth, rate limiting, CORS
- **MCP server:** 7 tools exposed via stdio
- **LangChain + LangGraph** adapters work with respective packages installed
- **Dashboard:** live memory palace topology, TFC state, feedback stats

### Honest Limitations

- **No encryption-at-rest** — use OS-level disk encryption
- **P2P is plaintext TCP** — trusted LAN only
- **No BEIR results** — benchmarks use synthetic corpora; harness for real datasets exists
- **API unversioned** — no `/v1/` prefix yet
- **Several modules await wiring** (backup, spreading activation, user profile, log rotation)
- **Benchmark numbers are from synthetic data** — not validated on standard leaderboards

---

## Architecture

```
Store → PII Scan → Vector Embed → Dedup → Write
Search → Intent Classify → BM25+Dense+Graph → RRF Fusion → Rerank → Assemble
Background → Consolidation → L1 Decay → L2 Interference → L3 Eviction → Curiosity
```

### Technical Stack

| Component | Technology |
|---|---|
| Storage | SQLite 3.45+, WAL mode, FTS5 |
| Vector | sqlite-vec or usearch |
| Embeddings | ONNX Runtime, bge-small-en-v1.5 (384-d) |
| Lexical | SQLite FTS5, porter stemming |
| Graph | NetworkX |
| API | FastAPI, uvicorn, slowapi |
| CLI | Typer, Rich |
| Config | pydantic-settings |
| Logging | structlog |

### Key Design Decisions

- **Local-first:** No cloud dependencies. Embeddings run on-device via ONNX.
- **Single-file DB:** One SQLite file contains all data — easy backup, portable.
- **In-process:** No external daemon required. Runs in-process with your agent.
- **Bi-temporal schema:** Every chunk tracks when it was known (valid_from) and when it was superseded (valid_to).
- **Wear-aware batching:** SD card / eMMC endurance via batched writes in consolidation pass.

---

## State of Development

Lumen is **functional alpha software**. It works end-to-end for storing, searching, and assembling agent memories. It has been audited through six exhaustive rounds of truth-in-engineering review and is honest about what works and what doesn't.

### Works End-to-End

- Store memories with PII detection
- Search via BM25 + vector + graph hybrid
- Assemble context for agent consumption
- Log feedback for value model learning
- Managed forgetting (L1 decay, L2 interference, L3 eviction)
- REST API with auth and rate limiting
- MCP server for agent tool integration
- Real-time dashboard

### In Progress / Planned

- Encryption-at-rest (SQLCipher integration)
- BEIR benchmark validation
- API versioning
- Transport encryption for P2P
- Full MARS consolidation lifecycle (currently: dedup only without LocalLLM)
- Module wiring (backup, spreading, user_profile)

---

## Getting Started

```bash
pip install -e ".[dev]"
lumen init --device generic
lumen serve
# Dashboard at http://localhost:8848/dashboard
```

```python
from lumen.config import LumenConfig
from lumen.data.schema import get_connection
from lumen.force.mnemonic.store import store_memory

config = LumenConfig()
conn = get_connection(config)
store_memory(conn, content="The user prefers dark mode", room_name="preferences", config=config)
```

---

## Contributing

See CONTRIBUTING.md for setup and guidelines. High-impact areas:
- Run BEIR harness for real-world retrieval benchmarks
- Add encryption-at-rest support
- Wire unwired modules into production paths
- Add API versioning

---

## Security

Report vulnerabilities via [GitHub Security Advisories](https://github.com/QuantumindSSI/lumen/security/advisories/new).