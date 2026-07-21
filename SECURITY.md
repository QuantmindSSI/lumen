# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x (alpha) | Security patches only |
| < 0.1.0 | Not supported |

## Reporting a Vulnerability

Lumen takes security seriously. If you discover a vulnerability, please report it responsibly.

**Email:** security@lumen.ai

**PGP Key:** (coming soon — check this file for updates)

### What to Include

1. Description of the vulnerability
2. Steps to reproduce
3. Affected versions
4. Potential impact assessment
5. Suggested fix (if you have one)

### Response Timeline

| Phase | Timeframe |
|---|---|
| Acknowledgment | Within 48 hours |
| Initial assessment | Within 7 days |
| Patch or mitigation | Within 30 days (critical: 7 days) |
| Public disclosure | Coordinated with reporter |

## Security Design Principles

### Sovereign-by-Default

- Lumen **never** sends data to external APIs unless explicitly configured
- `LUMEN_SOVEREIGN=true` blocks all network calls from core memory operations
- Embedding models run locally via ONNX Runtime

### Forgetting as Security

- Biologically-grounded forgetting prevents indefinite data retention
- Compliance module supports safety-triggered deletion (PII, secrets)
- Provenance chains track data lineage for audit

### Storage Security

- SQLite WAL mode with configurable sync levels
- File permissions enforced: `~/.lumen` should be `700`
- Audit logs are append-only JSONL with structured entries

## Known Limitations (Alpha)

1. **No encryption at rest** — planned for v0.2.0 (SQLCipher or OS-level)
2. **No authentication on API** — bind to localhost or use reverse proxy with auth
3. **P2P sharing not audited** — Beam protocol is experimental; do not use across untrusted networks

## Dependency Security

We monitor dependencies for known vulnerabilities:

```bash
# Audit dependencies
pip install pip-audit
pip-audit --desc
```

Key dependencies and their security posture:

| Dependency | Criticality | Notes |
|---|---|---|
| SQLite | High | Built-in, audited by SQLite consortium |
| ONNX Runtime | Medium | No network code; models are local files |
| USearch | Medium | No network code; memory-mapped files |
| FastAPI / Uvicorn | Medium | Only exposed if API server enabled |

---

*Last updated: 2026-07-20*  
*Maintainer: Lumen Security Team*
