# Lumen Production Readiness Roadmap

This document tracks the remaining work to move Lumen from beta (`v0.1.x-alpha`) to production-ready (`v0.2.0-beta` and beyond).

---

## Phase 1 — Foundation (P0 Blockers)

### 1.1 Documentation Audit & Alignment
- [x] **P1.1 Re-access reinforcement** — Code landed in `search.py`. Docs incorrectly listed as "Open". Update README + whitepaper.
- [x] **P1.2 Intent router (LR classifier)** — Code landed in `intent.py`. Docs incorrectly listed as "Open". Update README + whitepaper.
- [ ] **API authentication** — `X-API-Key` middleware exists in `api/server.py`. Verify config support and document properly.
- [x] **Encryption-at-rest inconsistency** — Fixed. Config fields removed from code, all docs aligned to state "not yet implemented, use OS-level encryption."

### 1.2 SQLite Encryption-at-Rest
- [ ] Implement SQLCipher integration (`PRAGMA key = ...`) in `get_connection()`
- [ ] Support Fernet field-level encryption fallback for standard sqlite3
- [x] OS-level encryption guidance (FileVault, BitLocker, LUKS) documented in DEPLOYMENT.md and SECURITY.md
- [x] File permissions enforcement: `~/.lumen` is `700` by default

### 1.3 API Hardening
- [ ] Document `X-API-Key` and `X-Tenant-ID` headers in API docs
- [ ] Add rate-limiting verification tests
- [ ] Add request-size limit tests
- [ ] Security headers (HSTS, CSP for dashboard)

---

## Phase 2 — Quality & Scale Validation

### 2.1 Retrieval Benchmarks at Scale
- [ ] Run full BEIR subset evaluation (not sampled 3,000 docs)
- [ ] Submit or compare against MTEB leaderboard numbers
- [ ] Publish benchmark results in `docs/benchmarks/`

### 2.2 Optical Degradation Benchmarks
- [ ] Benchmark R@k / nDCG at FP16 precision level
- [ ] Benchmark R@k / nDCG at INT8 precision level
- [ ] Benchmark R@k / nDCG at BINARY precision level
- [ ] Document accuracy vs. storage trade-off curves

### 2.3 TFC Sensitivity Analysis
- [ ] Grid search or Bayesian optimization across TFC parameters
- [ ] Justify default `e=0.50`, `a=0.50`, `τ=7.0`, `r=3`
- [ ] Publish sensitivity report

### 2.4 V(m) Weight Calibration
- [ ] Run Nelder-Mead or gradient-based weight learning with ≥100 feedback samples
- [ ] Validate against human relevance judgments

---

## Phase 3 — Stress & Security Hardening

### 3.1 Load & Concurrency Stress Test
- [ ] 100k+ chunk ingestion test
- [ ] Multi-threaded query throughput test
- [ ] Memory-budget eviction under RSS pressure
- [ ] Verify L3 budget eviction triggers correctly at 85% RSS

### 3.2 P2P / Beam Security Audit
- [ ] Implement NaCl transport encryption (currently plaintext — documented as trusted-LAN only)
- [ ] Harden Beam protocol for adversarial networks
- [x] Document trust model and household-local limitation

### 3.3 Release Engineering
- [ ] PGP key for vulnerability reporting
- [ ] CI/CD: add `pip-audit`, fuzz tests, backend matrix tests
- [ ] Tag `v0.2.0-beta`
- [ ] Publish release notes with closed/open gaps

---

## Honest Claim Posture

As stated in the whitepaper:

> Lumen is a feature-complete beta framework for sovereign agentic memory with unique lifecycle management and native integrations; its retrieval quality is competitive in controlled settings but not yet validated at leaderboard scale.

This roadmap is about closing the gap between that honest claim and a production-ready system.
