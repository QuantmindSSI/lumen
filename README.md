<div align="center">

<pre>
        ┌───┐
     ┌──┤   ├──┐
     │  │ ✦ │  │
     └──┤   ├──┘
        └───┘
</pre>

<h1>Lumen</h1>

<p><strong>Local-first memory and context framework for sovereign AI agents.</strong></p>

<p>
  <a href="https://github.com/QuantumindSSI/lumen/actions"><img src="https://img.shields.io/github/actions/workflow/status/QuantumindSSI/lumen/ci.yml?branch=main&style=flat-square" alt="CI"></a>
  <a href="https://github.com/QuantumindSSI/lumen/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/coverage-73%25-brightgreen?style=flat-square" alt="Coverage">
  <img src="https://img.shields.io/badge/tests-243%20passed-brightgreen?style=flat-square" alt="Tests">
</p>

<p><em>Beta software. API-stable; actively developed. Contributions welcome.</em></p>

</div>

---

## What is Lumen?

Lumen is a **local-first memory store for LLM agents** — it organizes agent memories in a structured memory palace (rooms, loci, chunks) with hybrid retrieval, managed decay, and native integrations. It runs entirely on your hardware with no cloud dependencies.

- **No cloud.** Embeddings run locally via ONNX Runtime. Storage is single-file SQLite.
- **Optional daemon.** Background scheduler auto-starts with `lumen serve`; can also run standalone with `lumen daemon start`.
- **Hybrid retrieval.** BM25 (SQLite FTS5), cosine-similarity vector search, and optional graph traversal with reciprocal rank fusion.
- **Managed memory lifecycle.** Three-layer forgetting: time-based decay, similarity interference, and budget eviction.
- **Integrations.** LangGraph checkpoint saver, LangChain memory adapter, MCP server, FastAPI REST API.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/QuantumindSSI/lumen.git
cd lumen
pip install -e ".[dev]"

# Initialize
lumen init --device generic

# Start the server
lumen serve
# Dashboard at http://localhost:8848/dashboard
# API docs at http://localhost:8848/docs
```

### Store and retrieve

```python
from lumen.config import LumenConfig
from lumen.data.schema import get_connection
from lumen.force.mnemonic.store import store_memory

config = LumenConfig()
conn = get_connection(config)

chunk_id = store_memory(
    conn,
    content="User prefers dark mode and large fonts",
    room_name="preferences",
    config=config,
)
conn.close()
```

```python
from lumen.config import LumenConfig
from lumen.data.schema import get_connection
from lumen.conversation import ConversationMemory

config = LumenConfig()
conn = get_connection(config)
memory = ConversationMemory(config=config, conn=conn)

turn = memory.retrieve_and_assemble("What UI settings does the user like?")
print(turn.assembled_context)
```

```bash
$ lumen status
Lumen Status
Device: generic
Rooms: 3
Active chunks: 56
Context budget: 2048 tokens
TFC → e=0.50 a=0.50 tau=7.0 r=3
```

### API endpoints

```
GET  /health            Liveness probe (unversioned)
GET  /dashboard         Effectiveness dashboard (HTML)
GET  /metrics           Machine-readable metrics
GET  /v1/status        Palace overview
POST /v1/search       Semantic + lexical hybrid search
POST /v1/store        Store a memory chunk
POST /v1/feedback     Log explicit or implicit feedback
POST /v1/assemble     Retrieve + assemble context in one call
POST /v1/turn         Store full conversation turn
GET  /v1/dashboard-data Dashboard data as JSON
```

---

## Architecture

```
User Input → Intent Router → Parallel Retrieval (BM25 + Dense + Graph)
                                  │
                                  ▼
                          RRF Fusion × V(m) × Recency
                                  │
                                  ▼
                          Context Assembly (Jinja2)
                                  │
                                  ▼
                    Consolidation → Decay / Interference / Eviction
```

---

## State of the Project

Lumen is **beta software**. It works end-to-end with API versioning, comprehensive tests, and documented security limitations. It is suitable for evaluation, development, and trusted-LAN deployments.

| Dimension | Status | Detail |
|---|---|---|
| **Tests** | 243 passing, 2 skipped | 73% coverage. 36 test files. |
| **Storage** | Working | SQLite with WAL, FTS5, bi-temporal tracking, provenance chains. |
| **Retrieval** | Working | BM25 + dense + graph with RRF fusion. |
| **Forgetting** | Working | L1 decay (Ebbinghaus), L2 interference, L3 budget eviction. |
| **PII detection** | Working | Regex-based scanning at storage time. Configurable block/redact/hash. |
| **Audit logging** | Working | SQLite audit_log table with request tracing. |
| **API server** | Working | FastAPI with `/v1/` versioning, auth, rate limiting, CORS, security headers. |
| **MCP server** | Working | 7 tools (search, store, assemble, turn, feedback, status, dashboard). |
| **LangChain** | Working | LumenChatMemory adapter (requires `langchain` package). |
| **LangGraph** | Working | LumenCheckpointSaver (requires `langgraph` package). |
| **Encryption-at-rest** | Not implemented | Use OS-level disk encryption (LUKS, FileVault, BitLocker). Planned for v0.3.0. |
| **BEIR benchmarks** | Harness ready | Runner at `benchmarks/beir/run.py`. Full evaluation deferred to HPC. |
| **P2P sharing** | Plaintext only | Beam protocol works on trusted LANs. No transport encryption. |

---

## Benchmark Suites

All run with a single command from the repo root:

| Suite | Command | Status |
|---|---|---|
| Retrieval (R@k, nDCG, MRR) | `python -m benchmarks.retrieval.run` | Run (synthetic corpus) |
| E2E memory quality | `python -m benchmarks.e2e.run` | Run (28 queries) |
| Navigation efficiency | `python -m benchmarks.navigation.run` | Run |
| Ablation (component isolation) | `python -m benchmarks.ablation.run` | Run |
| Forgetting (90-day survival) | `python -m benchmarks.forgetting.run` | Harness ready; no results yet |
| Performance (latency/footprint) | `python -m benchmarks.perf.run` | Harness ready; no results yet |
| BEIR subset evaluation | `python -m benchmarks.beir.run` | Harness ready; no results yet |
| All suites | `python -m benchmarks.run_all` | Wraps the above |

> **Note on results:** Retrieval benchmarks currently use a synthetic keyword-overlap corpus (1,000 passages, 50 queries). The nDCG formula was fixed in a recent commit; re-run `python -m benchmarks.retrieval.run` to generate updated scores.

---

## Integrations

| Integration | What it does | How to use |
|---|---|---|
| **MCP Server** | Exposes Lumen tools to OpenCode, Claude Desktop | `python -m lumen.integrations.mcp_server` |
| **LangChain** | `LumenChatMemory` adapter | `pip install langchain` |
| **LangGraph** | `LumenCheckpointSaver` for graph state | `pip install langgraph` |
| **FastAPI** | REST API with auth/rate-limiting | `lumen serve` |
| **OpenCode** | Native skill for memory workflows | See `docs/INTEGRATIONS.md` |

---

## Project Structure

```
lumen/
├── config.py          Configuration (pydantic-settings)
├── search.py          Search pipeline orchestration
├── fusion.py          RRF fusion + reranking
├── controller.py      Twin-Force state controller
├── conversation.py    Context assembly + turn tracking
├── repair.py          Self-healing retrieval
├── intent.py          Intent router (keyword + optional LR)
├── api/               FastAPI server + dashboard
├── cli/               Typer CLI
├── data/              Schema, migrations, backup
├── force/
│   ├── mnemonic/      Store, retrieval, decay, interference, eviction, provenance
│   └── contextual/    Embedding, token budget, assembly
├── integrations/      LangChain, LangGraph, MCP server
├── p2p/               Beam P2P sharing protocol
├── sovereign/         FRQAD, optical quantization, local LLM
├── brand/             Error hierarchy
└── compliance/        Safety forgetting, PII audit
tests/                36 test files, 243 tests
benchmarks/           8 benchmark suites
```

---

## Contributing

We welcome contributions. The best way to start:

1. **Read [`CONTRIBUTING.md`](CONTRIBUTING.md)** — setup, branch naming, code standards.
2. **Pick a `good first issue`** from the [issues tracker](https://github.com/QuantumindSSI/lumen/issues).
3. **Run the tests:** `pytest tests/` (must pass with ≥50% coverage).
4. **Submit a PR** against `main`.

### High-impact areas to contribute

- **Run the BEIR harness** — generate the first real-world retrieval benchmark results.
- **Add encryption-at-rest** — SQLCipher integration is the highest-priority security feature.
- **Write tests** — 28 modules lack dedicated test files. Pick one and add coverage.

### Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/                        # Run full suite
pytest tests/ --cov=lumen            # With coverage
ruff check lumen/ tests/             # Lint
```

---

## Documentation

| Document | Purpose |
|---|---|
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | Production deployment guide |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to contribute |
| [`SECURITY.md`](SECURITY.md) | Security policy and known limitations |
| [`INTEGRATIONS.md`](INTEGRATIONS.md) | Integration guides for each platform |
| [`ROADMAP.md`](ROADMAP.md) | Development milestones and open work |
| [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) | Detailed readiness audit |
| [`docs/Lumen_Agentic_Memory_Whitepaper_v2.md`](docs/Lumen_Agentic_Memory_Whitepaper_v2.md) | Technical white paper |

---

## Community

- [GitHub Discussions](https://github.com/QuantumindSSI/lumen/discussions)
- [Issue Tracker](https://github.com/QuantumindSSI/lumen/issues)
- [Security Advisories](https://github.com/QuantumindSSI/lumen/security/advisories)
- Matrix: `#lumen:matrix.org`

---

## License

Apache 2.0. See [`LICENSE`](LICENSE).

---

<div align="center">

<p><em>Local-first memory for your agents. On your hardware. Your way.</em></p>

</div>