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
  <a href="https://pypi.org/project/lumen/"><img src="https://img.shields.io/pypi/v/lumen?style=flat-square" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python">
</p>

<p><em>Local memory for local agents. Beta — under active development.</em></p>

</div>

---

## What is Lumen?

Lumen is a **local-first memory store for LLM agents** that runs entirely on your hardware. It provides structured storage, hybrid retrieval (lexical + vector + graph), managed decay, and native integrations for LangChain, LangGraph, MCP, and FastAPI.

- **No cloud.** All embeddings, storage, and retrieval run on-device via ONNX Runtime and SQLite.
- **No daemons.** Single in-process runtime. Fits on a Raspberry Pi (~90 MB RAM).
- **Hybrid retrieval.** BM25 (SQLite FTS5), cosine-similarity vector search, and optional graph traversal with reciprocal rank fusion.
- **Managed memory lifecycle.** Time-based decay, similarity-based interference weakening, and budget-based eviction.
- **Integrations.** Native LangGraph checkpoint saver, LangChain memory adapter, MCP server, FastAPI REST API.

### Core Design

| Component | Role |
|---|---|
| **Structured storage** | Rooms, loci, and chunks with bi-temporal tracking |
| **Hybrid retrieval** | Lexical (BM25), dense (cosine), and graph channels with RRF fusion |
| **Memory lifecycle** | Heuristic decay, interference weakening, and budget eviction |
| **Context assembly** | Token-budgeted context windows with Jinja2 templates |

---

## Quick Start

### Install

```bash
# From PyPI (when published)
pip install lumen

# From source
git clone https://github.com/QuantumindSSI/lumen.git
cd lumen
pip install -e ".[dev]"

# Optional: local LLM support for sleep-phase consolidation
pip install llama-cpp-python

# Optional: LangChain integration
pip install langchain

# Optional: MCP server for OpenCode / Claude Desktop
pip install mcp
```

### Initialize

```bash
# On a Raspberry Pi 5
lumen init --device rpi5

# On a generic x86 server
lumen init --device generic

# Or use the interactive wizard (requires spacy: python -m spacy download en_core_web_sm)
lumen illuminate
```

### Store a Memory

```python
from lumen.force.mnemonic.store import store_memory
from lumen.data.schema import get_connection
from lumen.config import LumenConfig

config = LumenConfig()
conn = get_connection(config)
chunk_id = store_memory(
    conn,
    content="User prefers dark mode and large fonts",
    room_name="preferences",
    source_type="user_input",
    config=config,
)
conn.close()
```

### Retrieve Context

```python
from lumen.conversation import ConversationMemory
from lumen.data.schema import get_connection
from lumen.config import LumenConfig

config = LumenConfig()
conn = get_connection(config)
memory = ConversationMemory(config=config, conn=conn)
turn = memory.retrieve_and_assemble(
    query="What UI settings does the user like?"
)
print(turn.assembled_context)
```

### Check Status

```bash
$ lumen status

Lumen Status
Device: generic
Rooms: 3
Active chunks: 56
Context budget: 2048 tokens
TFC → e=0.50 a=0.50 tau=7.0 r=3
```

---

## Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────────┐
│  Intent Router (keyword-based)              │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Parallel Retrieval                         │
│  ├── BM25  (SQLite FTS5)                    │
│  ├── Dense (sqlite-vec / brute-force)       │
│  └── Graph (NetworkX BFS)                   │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Fusion & Rerank (RRF × V(m) × recency)    │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Context Assembly (Jinja2 + goal tree)      │
└─────────────────────────────────────────────┘
    │
    ▼
Agent Reasoning (local LLM or external)
    │
    ▼
┌─────────────────────────────────────────────┐
│  Consolidation (async, low priority)        │
│  ├── Encode to structured store             │
│  ├── Compute V(m) heuristic score           │
│  ├── Interference check                     │
│  └── Schedule decay                          │
└─────────────────────────────────────────────┘
```

---

## Platform Support

| Platform | RAM | Vector Index | Embedder | Status |
|---|---|---|---|---|
| Raspberry Pi 5 | 4 GB | sqlite-vec | BGE-small | ✅ Tested |
| Jetson Orin Nano | 8 GB | USearch | BGE-small | ✅ Tested |
| Generic x86_64 | 8 GB+ | USearch | BGE-small | ✅ Tested |
| Orange Pi 5 | 8 GB | USearch | BGE-small | 🧪 Community |
| Apple Silicon | 8 GB+ | USearch | BGE-small | 🧪 Community |

---

## Key Features

### Structured Store
- **Rooms** — top-level categories (preferences, projects, people)
- **Loci** — specific locations within rooms
- **Chunks** — atomic memory units with content, embeddings, and metadata
- **Provenance chains** — source tracking for every fact

### Memory Lifecycle
- **L1 Decay** — time-based exponential decay of memory scores
- **L2 Interference** — similar memories within a locus penalize each other
- **L3 Budget** — eviction when process memory exceeds configurable limit

### TFC State Controller
Heuristic state machine that adjusts retrieval behavior based on interaction signals:
- `e` = conservation bias (remember vs. attend)
- `a` = attentional temperature (focused vs. exploratory)
- `τ` = temporal horizon in days
- `r` = resolution level (FP32 → INT8 → binary degradation)

### Sovereign Mode
```bash
export LUMEN_SOVEREIGN=true
```
Blocks all external API calls. Embeddings run locally via ONNX Runtime. Your data never leaves your device.

### Integrations
- **MCP Server** — OpenCode, Claude Desktop, GitHub Copilot
- **LangChain** — `LumenChatMemory` adapter
- **LangGraph** — `LumenCheckpointSaver` for graph state persistence (`pip install lumen[langgraph]`)
- **FastAPI** — REST API with health checks, rate limiting, and auth
- **OpenCode** — Native skill in `.opencode/skills/lumen-memory/`

### Effectiveness Dashboard
Lumen includes a self-hosted web dashboard at `/dashboard`:

```bash
lumen serve
# Open http://localhost:8848/dashboard
```

The dashboard shows live palace topology, TFC state, and memory health metrics.

---

## Current Limitations

Lumen is **beta software** under active development. Key known limitations:

| Area | Status | Detail |
|---|---|---|
| Forgetting | **Partially addressed** | Re-access reinforcement is implemented (boosts V(m) on retrieval). Selective weight learning via feedback still heuristic |
| V(m) scoring | **Heuristic only** | 7-factor lexical scoring (word overlap, pronoun density, sentiment word lists). Weight learning requires ≥10 feedback samples |
| TFC controller | **Heuristic only** | Hand-tuned thresholds; no sensitivity analysis, RL, or Bayesian optimization |
| Retrieval benchmarks | **Internal only** | No full BEIR/MTEB leaderboard submission yet. Current evaluation is on synthetic corpora and BEIR subsets |
| Intent routing | **Partially addressed** | Keyword rules with optional trained logistic regression classifier (`intent.py`). LR requires manual training |
| Multi-user / P2P | **Experimental** | Beam P2P is household-local only; not hardened for adversarial networks |
| Encryption-at-rest | **Not implemented** | No built-in encryption. Use OS-level disk encryption (FileVault, BitLocker, LUKS) for sensitive deployments. SQLCipher support is planned for v0.2.0 |

---

## Benchmarks

Lumen ships with first-party benchmark suites:

| Benchmark | What It Measures | Run |
|---|---|---|
| **Retrieval** | R@k, nDCG@k, MAP, MRR across BM25/dense/hybrid | `python -m lumen.benchmarks.retrieval.run` |
| **Forgetting** | 90-day survival curves, interference precision | `python -m lumen.benchmarks.forgetting.run` |
| **Performance** | Query latency, ingestion throughput, footprint | `python -m lumen.benchmarks.perf.run` |
| **Navigation** | Room-constrained vs. global search efficiency | `python -m lumen.benchmarks.navigation.run` |
| **E2E Memory Quality** | Multi-turn conversational recall | `python -m lumen.benchmarks.e2e.run` |

Run all at once: `python -m lumen.benchmarks.run_all`

### Approximate Resource Footprint

Measured on single-core CPU with BGE-small embedder. Competitor numbers are from published documentation (heterogeneous sources — not controlled comparison).

| Metric | Lumen | Chroma | FAISS | Qdrant |
|---|---|---|---|---|
| Install size | ~180 MB | ~250 MB | ~300 MB | ~400 MB |
| Runtime RAM | ~90 MB | ~150 MB | ~200 MB | ~250 MB |
| 10k chunk storage | ~35 MB | ~50 MB | ~60 MB | ~80 MB |
| Cold start | 0.8 s | 2.1 s | 1.5 s | 3.2 s |

---

## Documentation

| Document | Purpose |
|---|---|
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | Production deployment guide |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to contribute code and docs |
| [`SECURITY.md`](SECURITY.md) | Security policy and vulnerability reporting |
| [`docs/Lumen_Agentic_Memory_Whitepaper_v2.md`](docs/Lumen_Agentic_Memory_Whitepaper_v2.md) | Technical white paper with full methodology and known limitations |

---

## Roadmap

| Milestone | Status | Key Deliverables |
|---|---|---|
| **M1** | ✅ Done | Store & retrieve — schema, BM25+dense fusion, wear batcher |
| **M2** | ✅ Done | Palace & lifecycle — onboarding wizard, V(m) calibration, decay pipeline |
| **M3** | ✅ Core Done | Context & TFC — assembly, sleep consolidation, TFC state machine |
| **M4** | In Progress | Network & hardening — P2P sharing (Beam) 🔄, multi-user 🔄, encryption at rest 🔄, API auth ✅, LangGraph adapter ✅ |

---

## Community

- **Discussions:** https://github.com/QuantumindSSI/lumen/discussions
- **Issues:** https://github.com/QuantumindSSI/lumen/issues
- **Matrix:** #lumen:matrix.org
- **Security:** security@lumen.ai

Contributors are recognized as **Luminary** members. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

---

## License

Apache-2.0. See [`LICENSE`](LICENSE).

---

<div align="center">

<p><em>Local-first memory for your agents. On your hardware. Your way.</em></p>

</div>