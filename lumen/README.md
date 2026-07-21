# Lumen

**Twin-force memory and context framework for sovereign AI agents.**

Lumen gives local AI agents a structured, personal, bounded mind — using mnemonic palace architecture, biologically-grounded forgetting, and a twin-force controller that balances memory depth against context breadth.

- **No cloud.** No API calls. Your data never leaves your device.
- **No daemons.** Single in-process runtime. No Docker on the Pi.
- **Remembers what matters.** Biologically-grounded forgetting lets old memories dim naturally.
- **Learns what's important to you.** Per-user value model ranks memories by your goals and values.
- **Fits on a Raspberry Pi.** ~90 MB RAM. ~35 MB per 10,000 memories.

For full documentation, quick start, and deployment guides, see the [project README](../README.md).

---

## Install

```bash
pip install lumen
```

## Quick Start

```python
import lumen

# Store a memory
lumen.memory.store(
    content="User prefers dark mode and large fonts",
    room="preferences",
    source_type="user_input"
)

# Retrieve context
context = lumen.context.assemble(
    query="What UI settings does the user like?"
)
```

## Project Layout

```
lumen/
├── lumen/
│   ├── config.py              # Pydantic settings (device profiles)
│   ├── data/                  # SQLite schema and helpers
│   ├── brand/                 # LME-/LCX-/LLM- exception hierarchy
│   ├── force/
│   │   ├── mnemonic/          # Force A: palace, store, retrieve, forget
│   │   └── contextual/        # Force B: context assembly, embeddings
│   ├── lumen/                 # Unification: controller, search, CLI
│   ├── sovereign/             # Edge kernels (FRQAD, optical, wear)
│   ├── p2p/                   # Beam protocol
│   ├── api/                   # FastAPI server
│   ├── cli/                   # Typer + Rich commands
│   └── integrations/          # LangChain, MCP server
├── tests/                     # Full test suite
├── pyproject.toml             # Dependencies and tool config
└── README.md                  # This file
```

---

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Code quality
ruff check lumen/
mypy lumen/ --ignore-missing-imports

# Tests
pytest tests/ -q --tb=short --cov=lumen --cov-report=term-missing
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for full contributor guidelines.

---

## License

Apache-2.0
