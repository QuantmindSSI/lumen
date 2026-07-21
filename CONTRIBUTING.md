# Contributing to Lumen

Thank you for considering contributing to Lumen. Every contribution — code, documentation, design, testing, or community support — helps local AI agents develop minds of their own.

---

## Code of Conduct

This project adheres to a standard of respect, inclusivity, and intellectual honesty. We expect all contributors to:

- Be respectful in all interactions
- Welcome newcomers and help them learn
- Accept constructive criticism gracefully
- Focus on what is best for the community and the project

---

## How to Contribute

### 1. Reporting Bugs

Before opening an issue:

1. Search existing issues to avoid duplicates
2. Check `REVIEW_CHECKLIST.md` for known limitations
3. Test against the latest `main` branch

When reporting, include:

```markdown
**Device:** (e.g., Raspberry Pi 5 4GB, x86_64 desktop)
**Lumen version:** (e.g., 0.1.0)
**Python version:** (e.g., 3.12.3)
**What happened:**
**What you expected:**
**Steps to reproduce:**
**Relevant logs:** (with `[LUMEN:*]` prefix lines)
```

### 2. Suggesting Features

Feature requests should align with Lumen's core philosophy:

> Sovereign-first. Edge-native. Biologically-grounded. Palace-structured.

Open a GitHub Discussion first for major features. For small enhancements, an issue is fine.

### 3. Contributing Code

#### Development Setup

```bash
git clone https://github.com/QuantumindSSI/lumen.git
cd lumen/lumen
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

#### Code Standards

- **Lint:** `ruff check lumen/` must pass
- **Tests:** `pytest tests/` must pass with >50% coverage (target: >70%)
- **Types:** Add type hints for new public APIs
- **Docstrings:** Google-style for all public modules/classes/functions
- **Naming:** Follow the twin-force lexicon (room, locus, dim, illuminate — not delete, slot, fetch)

#### Branch Naming

```
feat/<short-description>       # New feature
fix/<short-description>        # Bug fix
docs/<short-description>       # Documentation
refactor/<short-description>   # Code restructuring
test/<short-description>       # Test-only changes
```

#### Commit Messages

Use conventional commits:

```
feat: add optical forgetting scheduler for RPi5
fix: resolve race condition in consolidation pipeline
docs: update deployment guide with Nginx example
test: add BEIR benchmark for retrieval_dense
refactor: simplify TFC state machine transitions
```

#### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run `ruff check lumen/` and `pytest tests/`
5. Update `CHANGELOG.md` under `[Unreleased]`
6. Open a PR with a clear description and reference any related issues
7. Address review feedback
8. A maintainer will merge when CI passes and review is approved

---

## Architecture Guidelines

### The Twin-Force Rule

Every change should respect the duality:

- **Memory (Force A)** = substantive, concrete, archival
- **Context (Force B)** = active, verbal, ephemeral
- **Lumen (Unification)** = light, balance, the intersection

### Adding a New Room

Rooms are top-level categories in the palace. To add one:

1. Define in `lumen/data/schema.sql` (if schema change needed)
2. Add retrieval path in `lumen/force/mnemonic/retrieval_*.py`
3. Register in `lumen/lumen/search.py` pipeline
4. Add test in `tests/test_retrieval_*.py`
5. Document in `docs/palace.md`

### Adding a Forgetting Level

The forgetting stack is L1-L4. If you need L5:

1. Create `lumen/force/mnemonic/forgetting_l5_*.py`
2. Hook into `lumen/lumen/sleep.py` consolidation scheduler
3. Add compliance audit logging
4. Write test in `tests/test_forgetting_*.py`

---

## Documentation Contributions

Docs live in:
- `README.md` — project overview and quickstart
- `BRAND_BIBLE.md` — visual/verbal identity
- `DEPLOYMENT.md` — production guide
- `docs/` (future) — full documentation site
- Inline docstrings — API reference

When documenting:
- Use the Lumen lexicon (see `BRAND_BIBLE.md` §5.1)
- Provide working code examples
- Include expected output where helpful
- Keep line length <= 100 characters

---

## Testing Guidelines

### Unit Tests

```python
def test_consolidation_creates_summary_chunk(db_conn, mock_embedder):
    """Consolidation should produce a summary chunk with V(m) > 0.5."""
    from lumen.force.mnemonic.consolidation import consolidate_room

    result = consolidate_room(db_conn, room_name="test_room")
    assert result > 0
```

### Integration Tests

Use `tests/conftest.py` fixtures:
- `db_conn` — in-memory SQLite with full schema
- `mock_embedder` — deterministic embedding generator
- `test_config` — generic device profile

### Benchmarks

Performance-critical changes should include benchmarks:

```bash
cd benchmarks
python benchmark_retrieval.py --backend sqlite-vec --dataset beir/scifact
```

---

## Recognition

Contributors are recognized as **Luminary** members. Significant contributions receive:

- Attribution in `CHANGELOG.md` and release notes
- Luminary badge in GitHub profile
- Optional: enamel pin for major architectural contributions

---

## Questions?

- GitHub Discussions: https://github.com/QuantumindSSI/lumen/discussions
- Matrix: #lumen:matrix.org
- Security: security@lumen.ai

---

*Thank you for helping local AI agents remember what matters and forget what doesn't.*
