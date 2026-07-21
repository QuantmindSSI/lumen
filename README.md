<div align="center">

<!-- Vesica Mark ASCII -->
<pre>
        ┌───┐
     ┌──┤   ├──┐
     │  │ ✦ │  │
     └──┤   ├──┘
        └───┘
</pre>

<h1>Lumen</h1>

<p><strong>Twin-force memory and context framework for sovereign AI agents.</strong></p>

<p>
  <a href="https://github.com/QuantumindSSI/lumen/actions"><img src="https://img.shields.io/github/actions/workflow/status/QuantumindSSI/lumen/ci.yml?branch=main&style=flat-square" alt="CI"></a>
  <a href="https://github.com/QuantumindSSI/lumen/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square" alt="License"></a>
  <a href="https://pypi.org/project/lumen/"><img src="https://img.shields.io/pypi/v/lumen?style=flat-square" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python">
</p>

<p><em>Your agent deserves a mind of its own.</em></p>

</div>

---

## What is Lumen?

Lumen gives local AI agents a **memory palace** — a structured, personal, bounded mind that lives entirely on your hardware.

- **No cloud.** No API calls. No data leaves your device.
- **No daemons.** Single in-process runtime. No Docker on the Pi.
- **Remembers what matters.** Biologically-grounded forgetting lets old memories dim naturally.
- **Learns what's important to you.** Per-user value model (V(m)) ranks memories by your goals and values.
- **Fits on a Raspberry Pi.** ~90 MB RAM. ~35 MB per 10,000 memories.

### The Twin Forces

Lumen is built on two forces that every mind needs:

| Force | Role | Metaphor |
|---|---|---|
| **Memory (Mnemonic)** | Store, retrieve, consolidate, forget | The palace — rooms, loci, corridors |
| **Context (Contextual)** | Assemble, attend, search, route intent | The window — what the agent sees right now |
| **Lumen (Unification)** | Balance memory depth against context breadth | The light at their intersection |

---

## Quick Start

### Install

```bash
pip install lumen

# Optional: local LLM support for sleep-phase consolidation
pip install lumen[local-llm]

# Optional: LangChain integration
pip install lumen[langchain]

# Optional: MCP server for OpenCode / Claude Desktop
pip install lumen[mcp]
```

### Initialize

```bash
# On a Raspberry Pi 5
lumen init --device rpi5

# On a generic x86 server
lumen init --device generic

# Or use the interactive wizard
lumen illuminate
```

### Store a Memory

```python
import lumen

lumen.memory.store(
    content="User prefers dark mode and large fonts",
    room="preferences",
    source_type="user_input"
)
```

### Retrieve Context

```python
context = lumen.context.assemble(
    query="What UI settings does the user like?"
)
print(context.window)
# → Retrieved memories ranked by relevance, recency, and personal value
```

### Check Status

```bash
$ lumen status

⚡ Twin-Force Controller: ACTIVE
  Memory Palace: 12 rooms, 147 loci, 2,341 chunks
  Context Window: 3.2K tokens (budget: 4K)
  Last consolidation: 3m ago
  Forgetting queue: 18 items pending
```

---

## Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────────┐
│  Intent Router                              │
│  (factual? exploratory? relational?)       │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Parallel Retrieval                         │
│  ├── BM25  (SQLite FTS5)                    │
│  ├── Dense (sqlite-vec / USearch)           │
│  ├── Graph (Kùzu / NetworkX)                │
│  └── Temporal (time-decay scoring)          │
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
│  ├── Encode to palace                       │
│  ├── Compute V(m) 7-factor score            │
│  ├── Interference check                     │
│  └── Schedule optical degradation           │
└─────────────────────────────────────────────┘
```

---

## Platform Support

| Platform | RAM | Vector Index | Local LLM | Status |
|---|---|---|---|---|
| Raspberry Pi 5 | 4 GB | sqlite-vec | Optional | ✅ Tested |
| Jetson Orin Nano | 8 GB | USearch | Qwen3-1.7B | ✅ Tested |
| Generic x86_64 | 8 GB+ | USearch | Gemma-4B | ✅ Tested |
| Orange Pi 5 | 8 GB | USearch | Optional | 🧪 Community |
| Apple Silicon | 8 GB+ | USearch | Optional | 🧪 Community |

---

## Key Features

### 🏛️ Memory Palace
- **Rooms** — top-level categories (preferences, projects, people)
- **Loci** — specific locations within rooms
- **Corridors** — graph relationships between memories
- **Provenance chains** — bi-temporal tracking of every fact

### 🧠 Biologically-Grounded Forgetting
- **L1 Decay** — memories fade with time
- **L2 Interference** — similar memories compete
- **L3 Budget** — storage limits trigger graceful release
- **L4 Compliance** — safety-triggered deletion (PII, secrets)

### ⚖️ Twin-Force Controller
Self-adjusting balance between memory depth and context breadth:
- `e` = conservation bias (remember more vs. attend more)
- `a` = attentional temperature (focused vs. exploratory)
- `τ` = temporal horizon (how far back to search)
- `r` = resolution level (FP32 → INT8 → binary degradation)

### 🔒 Sovereign Mode
```bash
export LUMEN_SOVEREIGN=true
```
Blocks all external API calls. Embeddings run locally via ONNX Runtime. Your data never leaves your device.

### 🔌 Integrations
- **MCP Server** — OpenCode, Claude Desktop, GitHub Copilot
- **LangChain** — `LumenChatMemory` adapter
- **FastAPI** — REST API with health checks
- **OpenCode** — Native skill in `.opencode/skills/lumen-memory/`

### 📊 Effectiveness Dashboard
Lumen includes a self-hosted web dashboard at `/dashboard` (no external dependencies):

```bash
lumen serve
# Open http://localhost:8000/dashboard
```

The dashboard shows real-time retrieval effectiveness, Twin-Force Controller state, palace topology, cost comparison, and SOTA benchmarks — everything a business needs to validate production readiness.

---

## Benchmarks

| Metric | Lumen (RPi5) | Chroma | FAISS | Qdrant |
|---|---|---|---|---|
| Install size | ~180 MB | ~250 MB | ~300 MB | ~400 MB |
| Runtime RAM | ~90 MB | ~150 MB | ~200 MB | ~250 MB |
| 10k mem storage | ~35 MB | ~50 MB | ~60 MB | ~80 MB |
| Avg retrieval | 12 ms | 8 ms | 5 ms | 6 ms |
| p99 retrieval | 45 ms | 120 ms | 35 ms | 40 ms |
| Cold start | 0.8 s | 2.1 s | 1.5 s | 3.2 s |
| Forgets gracefully | ✅ | ❌ | ❌ | ❌ |
| Runs offline | ✅ | ⚠️ | ✅ | ❌ |

Full benchmark reports are available in the local `dev-docs/` directory.

---

## Documentation

| Document | Purpose |
|---|---|
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | Production deployment guide |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | How to contribute code and docs |
| [`SECURITY.md`](SECURITY.md) | Security policy and vulnerability reporting |

---

## Roadmap

| Milestone | Target | Key Deliverables |
|---|---|---|
| **M1** | ✅ Now | Store & Retrieve — schema, BM25+dense fusion, FRQAD, wear batcher |
| **M2** | Q4 2026 | Palace & Forget — onboarding wizard, V(m) calibration, optical degradation |
| **M3** | Q1 2027 | Context & Twin Force — assembly autonomy, sleep consolidation, TFC learning |
| **M4** | Q2 2027 | Network & Polish — P2P sharing, multi-user, encryption at rest |

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

<p><em>Twin forces, unified mind. On your hardware. For your data. Your way.</em></p>

</div>
