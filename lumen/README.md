# Lumen

**Twin-force memory and context framework for sovereign AI agents.**

Lumen turns commodity open-source libraries into a biologically-grounded memory palace that lives entirely on the edge—no cloud, no daemons, no Docker on the Pi.

---

## Quick Start

### 1. Install

```bash
cd lumen
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Verify

```bash
# Code quality
ruff check lumen/

# Test harness (empty / trivial tests should pass)
pytest tests/

# Import check
python -c "import lumen; print(lumen.__version__)"
```

### 3. Run your first atomic task

Pick any open task from `lumen-master-engineering-spec.md` (Section 5.2). Each task is self-contained with:
- A single file to create or modify
- Input/output wires documented
- An acceptance test

**Good first tasks for parallel development:**

| Task | File | Skill | Est. |
|---|---|---|---|
| A2 | `lumen/force/mnemonic/retrieval_lexical.py` | Search / SQL | 1d |
| D2 | `lumen/force/mnemonic/event_buffer.py` | Systems | 1d |
| D15 | `lumen/brand/errors.py` | Core | 0.5d |
| D14 | `tests/conftest.py` | QA | 2d |
| A4 | `lumen/sovereign/frqad.py` | Performance / Numba | 5d |

---

## Project Layout

```
lumen/
├── lumen/
│   ├── config.py                 # D1: Pydantic settings (device profiles)
│   ├── data/
│   │   ├── schema.sql            # A1 + D3 + D16: canonical SQLite schema
│   │   ├── schema.py             # init_db / get_connection helpers
│   │   └── migrate.py            # D4: lightweight migration runner
│   ├── brand/
│   │   └── errors.py             # D15: LME-/LCX-/LLM- exception hierarchy
│   ├── force/
│   │   ├── mnemonic/             # Force A: palace, store, retrieve, forget
│   │   │   └── event_buffer.py   # D2: RAM-tier circular buffer stub
│   │   └── contextual/           # Force B: context assembly, embeddings
│   ├── lumen/                    # Unification: controller, search, CLI
│   ├── sovereign/                # Custom kernels (FRQAD, optical, wear)
│   ├── p2p/                      # Beam protocol
│   └── cli/                      # Typer + Rich commands
├── tests/
│   └── conftest.py               # D14: in-memory DB, mock embedder, test config
├── pyproject.toml                # Dependencies, ruff, mypy, pytest config
└── lumen-master-engineering-spec.md   # Canonical 39-task build plan
```

---

## Dependencies

Runtime and dev dependencies are declared in `pyproject.toml`. Key stacks:

- **Storage:** SQLite (WAL), sqlite-vec, USearch, Kùzu, LMDB
- **ML / NLP:** ONNX Runtime, spaCy, scikit-learn, fastText, Numba
- **Context:** Jinja2, anytree, structlog, pydantic-settings
- **CLI / UX:** Typer, Rich, APScheduler, psutil
- **Dev:** pytest, mypy, ruff, pre-commit

---

## How to Pick Up an Atomic Task

1. **Read** the task in `lumen-master-engineering-spec.md`.
2. **Check** the dependency graph (Section 5.2) to ensure its input wires are satisfied.
3. **Create** the specified file with the stub/interface signatures already provided.
4. **Write** the acceptance test in `tests/test_<task>.py`.
5. **Run** `pytest tests/test_<task>.py` until green.
6. **Run** `ruff check lumen/` before committing.

**Wiring rule:** If your task produces data consumed by another task, expose it through the SQLite schema or a typed Protocol (see `VectorBackend` in A3 for the pattern).

---

## Milestones

| Milestone | Target | Key Deliverables |
|---|---|---|
| M1 | Week 3 | Store & Retrieve: schema, BM25+dense fusion, FRQAD, wear batcher |
| M2 | Week 6 | Palace & Forget: onboarding, V(m), optical degradation, interference |
| M3 | Week 9 | Context & Twin Force: assembly, TFC autonomy, sleep consolidation |
| M4 | Week 12 | Network & Polish: P2P sharing, full CLI brand, zero-network sovereign mode |

---

## License

Apache-2.0
