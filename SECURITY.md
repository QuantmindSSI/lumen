# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.2.x (beta) | Security patches and bug fixes |
| 0.1.x (alpha) | Security patches only |
| < 0.1.0 | Not supported |

## Reporting a Vulnerability

Lumen takes security seriously. If you discover a vulnerability, please report it via the GitHub Security Advisory page or email.

**Report via:** https://github.com/QuantumindSSI/lumen/security/advisories/new

**PGP Key:** For sensitive disclosures, encrypt emails to:
```
security@lumen.ai
```

Public key (ED25519):
```
-----BEGIN PGP PUBLIC KEY BLOCK-----

mDMEanMX+xYJKwYBBAHaRw8BAQdAiouSUIUCncVCqE4b1SGOPOhxSdLtEFpEyCf9
wIayqQi0J0x1bWVuIFNlY3VyaXR5IFRlYW0gPHNlY3VyaXR5QGx1bWVuLmFpPoiZ
BBMWCgBBFiEEu67vrpF7GiE9+8oCvCSdz9W9rjEFAmpzF/sCGwMFCQPCZwAFCwkI
BwICIgIGFQoJCAsCBBYCAwECHgcCF4AACgkQvCSdz9W9rjHYoAD+Kc+kZLioe+a/
r3x0gJAwcquZkvICU/HqbuH2NWmjo0gA/A9RSYJ3cm7EcskEzfwbVTJSD75N2xMB
WQYWQOYpjP4A
=iER2
-----END PGP PUBLIC KEY BLOCK-----
```

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
- `LUMEN_SOVEREIGN=true` (default) blocks P2P sharing, model downloads from HuggingFace, and clamps API CORS to localhost origins
- Embedding models run locally via ONNX Runtime

### Forgetting as Security

- Biologically-grounded forgetting prevents indefinite data retention
- Compliance module supports safety-triggered deletion (PII, secrets)
- Provenance chains track data lineage for audit

### Storage Security

- SQLite WAL mode with configurable sync levels
- File permissions enforced: `~/.lumen` should be `700`
- Audit logs are stored in the SQLite `audit_log` table with structured entries

## Known Limitations (Beta)

1. **No encryption at rest** — Lumen stores data in plaintext SQLite. Use OS-level disk encryption (FileVault, BitLocker, LUKS) as a stopgap. SQLCipher integration is planned for v0.3.0.
2. **API authentication is opt-in** — `X-API-Key` header validation is implemented but disabled when `LUMEN_API_KEY` is unset. Set it before exposing the server to any network.
3. **P2P sharing is plaintext** — Beam protocol transmits over TCP without transport encryption. It is intended for trusted household LANs only. Do not expose Beam to untrusted networks.
4. **Sovereign mode blocks P2P and downloads** — When `LUMEN_SOVEREIGN=true` (default), P2P Beam sharing and HuggingFace model downloads are blocked with `SovereignViolationError`. The API CORS policy is also clamped to localhost origins. Sovereign mode does not sandbox the host OS or prevent other non-Lumen processes from making network calls.

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

*Last updated: 2026-08-05*  
*Maintainer: Lumen Security Team*
