# Lumen Production Readiness Roadmap

This document tracks the remaining work to move Lumen from beta (`v0.2.0-beta`) to production-ready (`v0.3.0-stable` and beyond).

---

## Phase 1 — Foundation (P0 Blockers) — CLOSED

### 1.1 Documentation Audit & Alignment
- [x] **P1.1 Re-access reinforcement** — Code landed in `search.py`. Docs incorrectly listed as "Open". Update README + whitepaper.
- [x] **P1.2 Intent router (LR classifier)** — Code landed in `intent.py`. Docs incorrectly listed as "Open". Update README + whitepaper.
- [x] **API authentication** — `X-API-Key` middleware verified in `api/server.py`. Documented in DEPLOYMENT.md §5.1.
- [x] **Encryption-at-rest inconsistency** — Fixed. Config fields removed from code, all docs aligned to state "not yet implemented, use OS-level encryption."

### 1.2 SQLite Encryption-at-Rest
- [ ] Implement SQLCipher integration (`PRAGMA key = ...`) in `get_connection()`
- [ ] Support Fernet field-level encryption fallback for standard sqlite3
- [x] OS-level encryption guidance (FileVault, BitLocker, LUKS) documented in DEPLOYMENT.md and SECURITY.md
- [x] File permissions enforcement: `~/.lumen` is `700` by default

### 1.3 API Hardening
- [x] Document `X-API-Key` and `X-Tenant-ID` headers in API docs
- [x] Add rate-limiting verification tests
- [x] Add request-size limit tests
- [x] Security headers (HSTS, CSP for dashboard)

---

## Phase 2 — Quality & Scale Validation — CLOSED

### 2.1 Retrieval Benchmarks at Scale
- [ ] Run full BEIR subset evaluation (not sampled 3,000 docs) — deferred to HPC
- [ ] Submit or compare against MTEB leaderboard numbers
- [x] Publish benchmark results in `benchmarks/*/results/`

### 2.2 Optical Degradation Benchmarks
- [x] Benchmark R@k / nDCG at FP16 precision level
- [x] Benchmark R@k / nDCG at INT8 precision level
- [x] Benchmark R@k / nDCG at BINARY precision level
- [x] Document accuracy vs. storage trade-off curves

### 2.3 TFC Sensitivity Analysis
- [x] Grid search across TFC parameters
- [x] Justify default `e=0.50`, `a=0.50`, `τ=7.0`, `r=3`
- [x] Publish sensitivity report to `benchmarks/tfc/results/tfc_sensitivity.json`

### 2.4 V(m) Weight Calibration
- [ ] Run Nelder-Mead or gradient-based weight learning with ≥100 feedback samples
- [ ] Validate against human relevance judgments

---

## Phase 3 — Stress & Security Hardening — CLOSED

### 3.1 Load & Concurrency Stress Test
- [x] 100k+ chunk ingestion test — 100k chunks in ~351s (~285 chunks/s)
- [x] Multi-threaded query throughput test — 1,000 queries in ~65s (~15 qps)
- [x] Memory-budget eviction under RSS pressure — peak 178MB
- [x] Verify L3 budget eviction triggers correctly at 85% RSS — verified at artificial 1MB limit

### 3.2 P2P / Beam Security Audit
- [ ] Implement NaCl transport encryption (currently plaintext — documented as trusted-LAN only) — target v0.3.0
- [ ] Harden Beam protocol for adversarial networks — target v0.3.0
- [x] Document trust model and household-local limitation
- [x] Remove misleading NaCl key generation from beam.py

### 3.3 Release Engineering
- [x] PGP key for vulnerability reporting — generated and published in SECURITY.md
- [ ] CI/CD: add `pip-audit`, fuzz tests, backend matrix tests
- [x] Tag `v0.2.0-beta`
- [x] Publish release notes with closed/open gaps

---

## Honest Claim Posture

As stated in the whitepaper:

> Lumen is a beta framework for sovereign agentic memory with unique lifecycle management and native integrations; its retrieval quality is competitive in controlled settings and validated by synthetic benchmarks, with full BEIR/MTEB leaderboard evaluation pending HPC resources.

This roadmap is about closing the gap between that honest claim and a production-ready system.
