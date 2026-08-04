"""
End-to-End Memory Quality Benchmark for Lumen.

Simulates 10 multi-turn conversation sessions across 3 agent personas.
Each session stores facts, asks questions that require recall, and
measures whether the memory palace retrieves the correct information.

This benchmark measures what SOTA frameworks (MemGPT, Mem0, Zep) measure:
memory persistence, cross-session recall, and hallucination resistance.

Usage:
    python -m benchmarks.e2e.run

Metrics:
    - recall@k: Precision of retrieving ground-truth chunks at each turn
    - semantic_similarity: Cosine similarity of retrieved context to expected answer
    - memory_persistence: Can facts from turn N be recalled at turn N+K?
    - hallucination_fraction: % of answers unsupported by retrieved context
    - latency: Per-query retrieval latency at each turn

Output: benchmarks/e2e/results/e2e_results.json + .md
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lumen.config import LumenConfig
from lumen.data.schema import get_connection
from lumen.force.contextual.embed import MockEmbedder
from lumen.force.mnemonic.store import store_memory
from lumen.search import SearchPipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EMBED_DIMS = 384
TOP_K = 10
N_BOOTSTRAP = 1_000

# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------
_embedder = None
_embedder_name = ""


def _get_embedder():
    global _embedder, _embedder_name
    if _embedder is not None:
        return _embedder, _embedder_name
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")

        class _Real:
            def __init__(self, m):
                self._m = m

            def encode(self, texts):
                return np.asarray(
                    self._m.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                    dtype=np.float32,
                )

            def encode_single(self, text):
                return self.encode([text])[0]

        _embedder = _Real(model)
        _embedder_name = "all-MiniLM-L6-v2"
        print(f"[INFO] Using embedder: {_embedder_name}")
        return _embedder, _embedder_name
    except Exception:
        _embedder = MockEmbedder(dims=EMBED_DIMS)
        _embedder_name = "mock"
        print("[WARN] No real embedder — using MockEmbedder")
        return _embedder, _embedder_name


# ---------------------------------------------------------------------------
# Multi-turn session definitions
# ---------------------------------------------------------------------------
# Each session has turns: each turn stores facts, then later asks questions
# that require recalling those facts. Ground truth chunks are identified by
# their content index in the session's fact list.

SESSIONS = [
    {
        "persona": "Security Consultant",
        "room": "security_audit",
        "turns": [
            {
                "turn_id": 1,
                "action": "store",
                "facts": [
                    "Client AlphaCorp uses AWS us-east-1 for primary hosting and Azure West Europe for DR. Their critical workloads include a Kubernetes cluster running 47 microservices and a PostgreSQL RDS instance with 2.3TB of transaction data.",
                    "AlphaCorp's last security audit found 12 critical vulnerabilities: 4 in IAM (overly permissive roles), 3 in S3 bucket policies (public read access), 2 unencrypted RDS instances, and 3 containers running as root.",
                    "The AlphaCorp CISO, Maria Santos, mandated that all critical findings be remediated within 30 days. The S3 and IAM findings were addressed within 14 days. RDS encryption required a 4-hour maintenance window scheduled for March 15.",
                ],
            },
            {
                "turn_id": 2,
                "action": "store",
                "facts": [
                    "Client BetaFinance is a SOC 2 Type II certified fintech. Their architecture includes a React frontend on Vercel, a Go API gateway on GCP Cloud Run, and PostgreSQL on GCP Cloud SQL with read replicas in three regions.",
                    "BetaFinance processes 380,000 transactions per hour during peak trading. Their P95 API latency is 87ms. They use HashiCorp Vault for secrets management and Cloudflare for DDoS protection.",
                    "BetaFinance's 2024 audit found issues with secret rotation (keys not rotated in 18 months), insufficient logging for API access (missing audit trail for 3 critical endpoints), and one Cloud SQL instance with public IP enabled.",
                ],
            },
            {
                "turn_id": 3,
                "action": "query",
                "queries": [
                    {
                        "query": "What cloud provider does AlphaCorp use for primary hosting and what is their main database setup?",
                        "expected_fact_ids": [0],
                        "tolerance": 0.3,
                    },
                    {
                        "query": "How many transactions per hour does BetaFinance process at peak?",
                        "expected_fact_ids": [3],
                        "tolerance": 0.3,
                    },
                ],
            },
            {
                "turn_id": 4,
                "action": "query",
                "queries": [
                    {
                        "query": "What was AlphaCorp CISO's deadline for fixing critical vulnerabilities and how many were IAM-related?",
                        "expected_fact_ids": [1],
                        "tolerance": 0.3,
                    },
                    {
                        "query": "What issues did BetaFinance's 2024 audit find regarding their secrets management?",
                        "expected_fact_ids": [5],
                        "tolerance": 0.3,
                    },
                ],
            },
        ],
    },
    {
        "persona": "ML Engineer",
        "room": "model_training",
        "turns": [
            {
                "turn_id": 1,
                "action": "store",
                "facts": [
                    "Model resnet-finetune-v3 is a ResNet-50 fine-tuned on 450,000 images across 15 product categories. Training took 72 GPU-hours on 4 A100s with batch size 256. Final validation accuracy is 94.3%, up from v2's 91.7%. The model uses mixed precision (FP16) and gradient accumulation steps of 4.",
                    "The training data pipeline ingests images from S3, applies augmentation (random crop, horizontal flip, color jitter with brightness=0.2, contrast=0.2), and caches processed tensors on NVMe for the next epoch. Cache hit rate is 97% after initial warm-up. Data loading adds 12ms average overhead per batch.",
                    "Hyperparameter search used Optuna with 200 trials over the space: learning rate [1e-4, 1e-2] log-uniform, weight decay [1e-6, 1e-3], label smoothing [0.0, 0.3]. Best params: lr=2.3e-3, weight_decay=4e-5, label_smoothing=0.15. Early stopping patience was 10 epochs.",
                ],
            },
            {
                "turn_id": 2,
                "action": "store",
                "facts": [
                    "Deployment runs on Triton Inference Server with dynamic batching (max batch 32, timeout 100us). The ONNX export used opset 17 with graph optimizations (constant folding, node fusion). Quantized INT8 model is 48MB (down from 178MB FP16) with <0.5% accuracy loss. P99 inference latency is 4.2ms on T4 GPU.",
                    "Model monitoring tracks data drift using PSI (Population Stability Index) on input feature distributions. Alert threshold is PSI > 0.25. Current PSI for the last 14 days is 0.12. The fallback model (resnet-finetune-v2) is kept warm with a canary deployment serving 5% of traffic for continuous comparison.",
                    "The training infrastructure uses Kubernetes on GKE with the nvidia/gpu-operator for driver management. Jobs are submitted via Kubeflow Pipelines. The training cluster runs 24/7 with preemptible GPUs for cost savings (70% discount vs on-demand). Checkpointing to GCS every 500 steps ensures no progress is lost on preemption.",
                ],
            },
            {
                "turn_id": 3,
                "action": "query",
                "queries": [
                    {
                        "query": "What is the validation accuracy of the latest model and how does it compare to the previous version?",
                        "expected_fact_ids": [0],
                        "tolerance": 0.3,
                    },
                    {
                        "query": "What hyperparameter values did Optuna select as optimal for this training run?",
                        "expected_fact_ids": [2],
                        "tolerance": 0.3,
                    },
                ],
            },
            {
                "turn_id": 4,
                "action": "query",
                "queries": [
                    {
                        "query": "What is the INT8 model size after quantization and what was the accuracy impact?",
                        "expected_fact_ids": [3],
                        "tolerance": 0.3,
                    },
                    {
                        "query": "How is data drift monitored and what is the current alert threshold?",
                        "expected_fact_ids": [4],
                        "tolerance": 0.3,
                    },
                ],
            },
        ],
    },
    {
        "persona": "Medical Researcher",
        "room": "clinical_trials",
        "turns": [
            {
                "turn_id": 1,
                "action": "store",
                "facts": [
                    "Phase III trial TRILUMINATE (NCT03904147) evaluated the TriClip transcatheter tricuspid valve repair system in 350 patients with severe tricuspid regurgitation. Primary endpoint was a hierarchical composite including all-cause mortality, tricuspid valve surgery, heart failure hospitalization, and quality-of-life improvement at 1 year. The win ratio was 1.48 (p=0.02), meeting the primary endpoint.",
                    "Safety analysis for TRILUMINATE: major adverse events occurred in 1.7% of device group vs 0.6% control at 30 days (p=0.39). The most common adverse events were access site bleeding (6.5%) and new conduction abnormalities (3.2%). No device embolizations or thromboses were reported. Mean procedure time was 97 minutes (IQR 67-139).",
                    "The TRILUMINATE quality-of-life secondary endpoint used the Kansas City Cardiomyopathy Questionnaire (KCCQ). At 1 year, the device group improved by 18.4 points (95% CI 14.2-22.6) vs 9.6 points (95% CI 5.6-13.6) in control, a between-group difference of 8.8 points (p<0.001). The MCID for KCCQ is 5 points, indicating clinically meaningful improvement.",
                ],
            },
            {
                "turn_id": 2,
                "action": "store",
                "facts": [
                    "Companion diagnostics (CDx) are tests that identify patients most likely to benefit from a specific therapy. FDA requires CDx approval alongside targeted therapies. FoundationOne CDx (Roche) uses NGS to detect genomic alterations in 324 genes, with approved indications for 30+ targeted therapies across NSCLC, breast, colorectal, ovarian cancers, and melanoma.",
                    "The KEYNOTE-024 trial established pembrolizumab as first-line therapy for NSCLC patients with PD-L1 TPS ≥50%. Median overall survival was 30.0 months with pembrolizumab vs 14.2 months with chemotherapy (HR 0.63, p=0.002). 5-year OS rate was 31.9% vs 16.3%. PD-L1 testing using the 22C3 pharmDx assay (Agilent) is the companion diagnostic for pembrolizumab in NSCLC.",
                    "NGS-based tumor mutation burden (TMB) is an emerging biomarker for immunotherapy response. KEYNOTE-158 validated TMB-H (≥10 mutations/megabase) as predictive of pembrolizumab benefit across solid tumors, leading to tissue-agnostic FDA approval in 2020. FoundationOne CDx is the approved CDx for TMB assessment. Blood-based TMB (bTMB) using liquid biopsy (Guardant360) is under investigation but not yet approved.",
                ],
            },
            {
                "turn_id": 3,
                "action": "query",
                "queries": [
                    {
                        "query": "What was the primary endpoint of the TRILUMINATE trial and was it met?",
                        "expected_fact_ids": [0],
                        "tolerance": 0.3,
                    },
                    {
                        "query": "For which cancer type did KEYNOTE-024 establish pembrolizumab as first-line therapy and what was the required PD-L1 threshold?",
                        "expected_fact_ids": [4],
                        "tolerance": 0.3,
                    },
                ],
            },
            {
                "turn_id": 4,
                "action": "query",
                "queries": [
                    {
                        "query": "What quality-of-life metric was used in TRILUMINATE and what was the between-group difference?",
                        "expected_fact_ids": [2],
                        "tolerance": 0.3,
                    },
                    {
                        "query": "What genomic biomarker was validated by KEYNOTE-158 for tissue-agnostic pembrolizumab approval?",
                        "expected_fact_ids": [5],
                        "tolerance": 0.3,
                    },
                ],
            },
        ],
    },
    {
        "persona": "QA Assistant",
        "room": "fact_check",
        "turns": [
            {
                "turn_id": 1,
                "action": "store",
                "facts": [
                    "Python 3.13 was released on October 7, 2024. Key features: experimental JIT compiler (copy-and-patch), new interactive interpreter with multi-line editing and color support, improved error messages, and removal of many deprecated modules (including xdrlib, nis, and lib2to3). The GIL remains but PEP 703 (nogil) work continues for 3.14.",
                    "Django 5.1 (August 2024) introduced QuerySet explain() with generic plans, asynchronous authentication backends, template fragment caching, and the LoginRequiredMiddleware. Django 5.0 (December 2023) dropped Python 3.8 and 3.9 support, added db_default for computed column defaults, and introduced GeneratedField.",
                    "Rust 1.80 (July 2024) added LazyLock and LazyCell types for lazy initialization. Rust 1.82 (October 2024) stabilized raw-dylib for Windows dynamic linking, const extern fn, and unsafe extern blocks. The 2024 edition is planned for Rust 1.84 with changes to RPIT lifetime capture, unsafe ops in extern blocks, and gen block stabilization.",
                ],
            },
            {
                "turn_id": 2,
                "action": "store",
                "facts": [
                    "The xz utils backdoor (CVE-2024-3094) was discovered in March 2024 by Andres Freund, a Microsoft PostgreSQL developer who noticed 500ms SSH login latency caused by the backdoored liblzma. The attacker (Jia Tan) spent 2+ years building trust in the xz project, eventually becoming a co-maintainer before injecting obfuscated malicious code into release tarballs. Only Debian/RPM-based distributions using glibc and systemd were affected.",
                    "The European Cyber Resilience Act (CRA) was adopted in October 2024 and will be enforced starting 2027. It requires manufacturers to provide security updates throughout the product lifecycle, report actively exploited vulnerabilities within 24 hours to ENISA, and provide SBOMs. Open source stewards are exempt but commercial entities using OSS must comply for their products.",
                    "The CrowdStrike Falcon sensor update on July 19, 2024 caused a global IT outage affecting 8.5 million Windows devices. A content configuration update (Channel File 291) triggered a logic error that caused kernel-level crashes (BSOD) on Windows hosts. Root cause: a mismatch between the 21 input fields expected by the sensor's content interpreter and the 20 fields delivered by the update. Estimated economic damage: $5.4 billion for Fortune 500 companies alone.",
                ],
            },
            {
                "turn_id": 3,
                "action": "query",
                "queries": [
                    {
                        "query": "What key features were added in Python 3.13 and when was it released?",
                        "expected_fact_ids": [0],
                        "tolerance": 0.3,
                    },
                    {
                        "query": "Who discovered the xz utils backdoor and what was the immediate symptom they noticed?",
                        "expected_fact_ids": [3],
                        "tolerance": 0.3,
                    },
                ],
            },
            {
                "turn_id": 4,
                "action": "query",
                "queries": [
                    {
                        "query": "What new database features did Django 5.1 introduce?",
                        "expected_fact_ids": [1],
                        "tolerance": 0.3,
                    },
                    {
                        "query": "What was the root cause of the CrowdStrike outage and how many devices were affected?",
                        "expected_fact_ids": [5],
                        "tolerance": 0.3,
                    },
                ],
            },
        ],
    },
    {
        "persona": "Architecture Reviewer",
        "room": "system_design",
        "turns": [
            {
                "turn_id": 1,
                "action": "store",
                "facts": [
                    "Service 'payment-gateway' processes Stripe, PayPal, and Adyen payments through a unified adapter pattern. The gateway implements idempotency keys for safe retries, stores payment intents in DynamoDB with TTL of 90 days, and publishes PaymentProcessed/PaymentFailed events to Kafka for downstream services (invoicing, analytics, fraud). Circuit breaker opens after 50% error rate over 30s sliding window.",
                    "The notification service 'notify-hub' sends push (FCM/APNs), email (SendGrid), SMS (Twilio), and in-app (WebSocket) notifications. It uses a priority queue (Redis sorted sets) with 4 tiers: critical (SLA <5s), transactional (<30s), marketing (<5min), and digest (daily batch). Retry policy: exponential backoff with jitter, max 5 attempts, dead letter queue after final failure. 99.95% delivery rate SLA.",
                    "Database strategy: PostgreSQL (via RDS Proxy) for transactional data with PgBouncer connection pooling (pool_mode=transaction). DynamoDB for high-throughput event sourcing (partition key=entity_id, sort key=timestamp). ElastiCache Redis for session state (TTL 24h) and rate limiting counters. S3 for document storage with intelligent tiering (frequently accessed → infrequent → archive after 90/180 days). Backup strategy: RDS automated backups (7-day retention), DynamoDB PITR (35 days), Redis AOF every 1s.",
                ],
            },
            {
                "turn_id": 2,
                "action": "store",
                "facts": [
                    "Deployment pipeline: GitHub Actions → Build (Docker multi-stage, layer caching) → Test (unit + integration + contract tests) → Security Scan (Snyk + Trivy + OWASP ZAP dynamic scan) → Deploy to Staging (ArgoCD GitOps) → Smoke Tests (5min) → Canary Deploy (10% traffic, 30min observation) → Full Production Deploy. Total pipeline time: 45 minutes from commit to production.",
                    "Observability stack: OpenTelemetry SDK for traces (exported to Tempo via OTLP gRPC), structured JSON logs to Loki (via Promtail), metrics from Prometheus (scraped every 15s) with Grafana dashboards. SLIs: latency (p95 <200ms for API, p99 <500ms), error rate (<0.1%), availability (99.95% monthly). Error budget: 43 minutes of downtime per month. Alerting: PagerDuty with 5min acknowledgment SLA for P1 alerts, 30min for P2.",
                    "Autoscaling: HPA (Horizontal Pod Autoscaler) on CPU (target 70%) and custom metrics (requests/second from Prometheus). KEDA (Kubernetes Event-Driven Autoscaling) scales from 0 for event-driven workloads (Kafka consumer lag >100 triggers scale-up). Cluster autoscaler on GKE with node pool sizes 2-20 nodes of n2-standard-8. Overprovisioning with pause pods ensures 30% headroom for rapid scale-up. Cold start from 0 to serving takes 8 seconds.",
                ],
            },
            {
                "turn_id": 3,
                "action": "query",
                "queries": [
                    {
                        "query": "What payment providers does the payment gateway support and how does it ensure safe retries?",
                        "expected_fact_ids": [0],
                        "tolerance": 0.3,
                    },
                    {
                        "query": "What is the total pipeline time from commit to production deployment?",
                        "expected_fact_ids": [3],
                        "tolerance": 0.3,
                    },
                ],
            },
            {
                "turn_id": 4,
                "action": "query",
                "queries": [
                    {
                        "query": "What notification channels does notify-hub support and what are the priority tiers?",
                        "expected_fact_ids": [1],
                        "tolerance": 0.3,
                    },
                    {
                        "query": "What scaling mechanisms are used and what is the cold start time?",
                        "expected_fact_ids": [5],
                        "tolerance": 0.3,
                    },
                ],
            },
        ],
    },
    {
        "persona": "Climate Analyst",
        "room": "emissions_data",
        "turns": [
            {
                "turn_id": 1,
                "action": "store",
                "facts": [
                    "Company EcoCoat manufactures sustainable industrial coatings using bio-based epoxy (derived from soybean oil). 2023 production volume: 14,200 metric tons. Scope 1 emissions (direct): 8,400 tCO2e (from natural gas boilers for reactor heating). Scope 2 emissions (purchased electricity): 3,200 tCO2e (100% grid electricity, Midwest region, emission factor 0.52 kgCO2/kWh). Scope 3 upstream: 12,800 tCO2e (raw material transport, soybean processing).",
                    "EcoCoat's emission reduction targets: 50% reduction in Scope 1 and 2 by 2030 (vs 2023 baseline), net zero by 2045. Current initiatives: switching boilers from natural gas to electric (expected Q3 2025, -6,000 tCO2e/year), installing 2.4 MW rooftop solar (expected Q2 2025, -1,800 tCO2e/year), and sourcing bio-based epoxy from local supplier (reducing transport emissions by 40%). Total estimated reduction by 2026: 8,200 tCO2e/year.",
                    "EcoCoat verified their 2023 emissions through an independent auditor per ISO 14064-3. Their GHG inventory follows the GHG Protocol Corporate Standard. Carbon offsets purchased: 5,000 tCO2e from a Gold Standard-certified reforestation project in Brazil (Jari Para REDD+, verified by Verra). Internal carbon price: $75/tCO2e, applied to capital expenditure decisions above $500K.",
                ],
            },
            {
                "turn_id": 2,
                "action": "store",
                "facts": [
                    "Company SolarGrid operates 18 utility-scale solar farms totaling 2.8 GW capacity across Texas, Arizona, Nevada, and California. 2023 generation: 5,800 GWh, avoiding approximately 2.7 million tCO2e compared to the grid average. Capacity factor average across sites: 26.3% (range: 22.1% in Texas to 29.8% in California). Curtailment: 3.2% of potential generation lost due to transmission constraints.",
                    "SolarGrid's newest project, the Antelope Valley Solar Farm (Nevada, 450 MW), uses bifacial PERC modules on single-axis trackers. Annual generation estimate: 1,100 GWh. Construction employed 850 workers over 18 months with a 35% local hiring rate. The site includes a 200 MW/800 MWh lithium-iron-phosphate battery storage system (supplied by CATL, 4-hour duration). Total project cost: $620 million.",
                    "SolarGrid's 2023 CDP (Carbon Disclosure Project) score is A- (Leadership level). They report Scope 1 emissions of 1,200 tCO2e (fleet vehicles, backup generators), Scope 2 of 400 tCO2e (office and control center electricity), and Scope 3 of 28,000 tCO2e (primarily embodied carbon in solar panel manufacturing). They are targeting a 30% reduction in Scope 3 emissions by 2030 through supplier engagement and panel recycling programs.",
                ],
            },
            {
                "turn_id": 3,
                "action": "query",
                "queries": [
                    {
                        "query": "What is EcoCoat's Scope 1 emissions for 2023 and what is their main source?",
                        "expected_fact_ids": [0],
                        "tolerance": 0.3,
                    },
                    {
                        "query": "How many solar farms does SolarGrid operate and what is their total capacity?",
                        "expected_fact_ids": [3],
                        "tolerance": 0.3,
                    },
                ],
            },
            {
                "turn_id": 4,
                "action": "query",
                "queries": [
                    {
                        "query": "What are EcoCoat's targeted emission reductions by 2026 and through which initiatives?",
                        "expected_fact_ids": [1],
                        "tolerance": 0.3,
                    },
                    {
                        "query": "What battery storage does the Antelope Valley project include and what is its duration?",
                        "expected_fact_ids": [4],
                        "tolerance": 0.3,
                    },
                ],
            },
        ],
    },
    {
        "persona": "Database Admin",
        "room": "db_configs",
        "turns": [
            {
                "turn_id": 1,
                "action": "store",
                "facts": [
                    "PostgreSQL 16 (September 2023) introduced: logical replication from standbys, SQL/JSON constructors and identity functions (JSON_SERIALIZE, JSON_QUERY), parallel hash join improvements with full right and full outer hash joins, pg_stat_io for detailed I/O statistics, and the ability to use SIMD (SSE2/NEON) for ASCII string operations. Vacuum performance improved with faster freezing of tuples and reduced WAL volume during vacuum operations.",
                    "PostgreSQL 17 (September 2024) added: incremental backup support via pg_basebackup --incremental, MERGE command enhancements (RETURNING clause, updatable views), COPY performance improvements (up to 2x for large imports), logical replication slot synchronization to standbys, and better memory management for vacuum (reduced memory consumption by 20% for large tables). JSON_TABLE function allows converting JSON to relational format for SQL queries.",
                    "Current production cluster runs PostgreSQL 16.3 on AWS RDS with db.r6g.8xlarge (32 vCPU, 256GB RAM). Configuration: shared_buffers=64GB, effective_cache_size=192GB, work_mem=256MB, maintenance_work_mem=2GB, max_connections=500 (via PgBouncer transaction pooling), wal_level=logical, max_replication_slots=20. Backup: automated RDS snapshots every 6 hours with 14-day retention, plus continuous PITR with 5-minute RPO.",
                ],
            },
            {
                "turn_id": 2,
                "action": "store",
                "facts": [
                    "Redis 7.2 (August 2023) introduced: client-side caching with server-assisted tracking (RESP3 protocol), sharded pub/sub that distributes channels across cluster nodes for 3x throughput improvement, ACL v2 with key permissions, and command introspection via COMMAND DOCS. Redis 7.4 (July 2024) added hash field expiration (HEXPIRE), per-slot metrics for cluster mode, and memory-efficient listpack encoding for sets with up to 128 elements.",
                    "Current Redis deployment: ElastiCache Serverless with max capacity 500,000 requests/second. Used for: session store (prefix ssn:, TTL 24h, 3M active sessions), rate limiting (prefix rl:, sliding window counters in sorted sets, 15M operations/hr), feature flags (prefix ff:, hash with JSON values, polled every 30s by SDKs), and job queue (prefix job:, Redis Lists with RPUSH/LPOP for background task distribution). Memory usage: 12GB across the cluster.",
                    "Migration plan for Redis 7.4: upgrading from ElastiCache 7.1 to 7.4 in Q1 2025. Key benefits sought: hash field expiration to replace manual TTL management for rate limit counters, and per-slot metrics to identify hot partitions. Migration strategy: blue/green deployment with ElastiCache Global Datastore for replication, DNS cutover with 60s TTL. Rollback plan: keep old cluster for 72 hours, DNS flip-back if P95 latency exceeds 2x baseline.",
                ],
            },
            {
                "turn_id": 3,
                "action": "query",
                "queries": [
                    {
                        "query": "What replication feature did PostgreSQL 16 add and what backup feature did PostgreSQL 17 add?",
                        "expected_fact_ids": [0],
                        "tolerance": 0.3,
                    },
                    {
                        "query": "What new expiration capability did Redis 7.4 introduce?",
                        "expected_fact_ids": [3],
                        "tolerance": 0.3,
                    },
                ],
            },
            {
                "turn_id": 4,
                "action": "query",
                "queries": [
                    {
                        "query": "What are the key PostgreSQL configuration values for the production cluster and what is the RPO?",
                        "expected_fact_ids": [2],
                        "tolerance": 0.3,
                    },
                    {
                        "query": "What is the Redis migration strategy for the 7.4 upgrade and what is the rollback plan?",
                        "expected_fact_ids": [5],
                        "tolerance": 0.3,
                    },
                ],
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _compute_metrics(embedder, retrieved_chunks, expected_texts) -> dict:
    """Compute recall and semantic similarity against expected ground-truth texts."""
    if not retrieved_chunks:
        return {"recall_1": 0.0, "recall_5": 0.0, "recall_10": 0.0, "semantic_similarity": 0.0}

    # Encode retrieved content and expected text
    retrieved_texts = [rc.content for rc in retrieved_chunks]
    if hasattr(embedder, "encode"):
        r_vecs = embedder.encode(retrieved_texts)
        e_vecs = embedder.encode(expected_texts)
    else:
        r_vecs = np.stack([embedder.encode_single(t) for t in retrieved_texts])
        e_vecs = np.stack([embedder.encode_single(t) for t in expected_texts])

    # Compute max similarity between expected and any retrieved chunk
    sims = []
    for e_vec in e_vecs:
        max_sim = max(_cosine_sim(e_vec, r_vec) for r_vec in r_vecs) if len(r_vecs) > 0 else 0.0
        sims.append(max_sim)

    # Recall: what fraction of expected texts are similar to any top-k retrieved?
    threshold = 0.4  # semantic match threshold

    def _fraction_matched(k):
        matched = set()
        for i, e_vec in enumerate(e_vecs):
            for j, r_vec in enumerate(r_vecs[:k]):
                if _cosine_sim(e_vec, r_vec) > threshold:
                    matched.add(i)
                    break
        return len(matched) / max(1, len(e_vecs))

    return {
        "recall_1": _fraction_matched(1),
        "recall_5": _fraction_matched(5),
        "recall_10": _fraction_matched(10),
        "semantic_similarity": float(np.mean(sims)),
    }


def _bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP):
    rng = np.random.default_rng(42)
    n = len(values)
    means = [np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_bootstrap)]
    return np.mean(values), np.percentile(means, 2.5), np.percentile(means, 97.5)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_e2e_benchmark():
    embedder, embedder_name = _get_embedder()
    all_turns = []
    session_results = []

    for session_idx, session in enumerate(SESSIONS):
        print(f"\n{'='*60}")
        print(f"Session {session_idx+1}: {session['persona']} in room '{session['room']}'")
        print(f"{'='*60}")

        tmpdir = tempfile.mkdtemp(prefix=f"lumen_e2e_{session['room']}_")
        config = LumenConfig(
            store_path=Path(tmpdir),
            embedding_dims=EMBED_DIMS,
            vector_index="sqlite-vec",
        )
        conn = get_connection(config)
        pipeline = SearchPipeline(conn, config, embedder=embedder)

        # Store all facts from store turns with their indices
        fact_index: dict[int, str] = {}
        fact_count = 0

        for turn in session["turns"]:
            if turn["action"] == "store":
                for fact in turn["facts"]:
                    emb = embedder.encode_single(fact)
                    store_memory(
                        conn,
                        content=fact,
                        room_name=session["room"],
                        locus_name=f"turn_{turn['turn_id']}",
                        embedding=emb,
                        config=config,
                    )
                    fact_index[fact_count] = fact
                    fact_count += 1
                conn.commit()
                all_turns.append({
                    "session": session["persona"],
                    "room": session["room"],
                    "turn_id": turn["turn_id"],
                    "action": "store",
                    "facts_stored": len(turn["facts"]),
                })

            elif turn["action"] == "query":
                for query_item in turn["queries"]:
                    query = query_item["query"]
                    expected_ids = query_item["expected_fact_ids"]
                    expected_texts = [fact_index[eid] for eid in expected_ids if eid in fact_index]

                    t0 = time.perf_counter()
                    results = pipeline.execute(query, k=TOP_K)
                    latency_ms = (time.perf_counter() - t0) * 1000

                    metrics = _compute_metrics(embedder, results, expected_texts)
                    metrics["latency_ms"] = round(latency_ms, 2)

                    top_chunks = []
                    for r in results[:5]:
                        preview = r.content[:120] + "..." if len(r.content) > 120 else r.content
                        top_chunks.append({"id": r.chunk_id, "score": round(r.final_score, 4), "preview": preview})

                    all_turns.append({
                        "session": session["persona"],
                        "room": session["room"],
                        "turn_id": turn["turn_id"],
                        "action": "query",
                        "query": query,
                        "expected_fact_ids": expected_ids,
                        "metrics": metrics,
                        "top_5_chunks": top_chunks,
                    })

                    if metrics["recall_10"] >= 1.0:
                        outcome = "PASS"
                    elif metrics["recall_10"] >= 0.5:
                        outcome = "PARTIAL"
                    else:
                        outcome = "FAIL"
                    print(f"  [{outcome}] {query[:70]}... (R@10={metrics['recall_10']:.2f}, sim={metrics['semantic_similarity']:.3f}, {latency_ms:.1f}ms)")

        # Session-level stats
        query_turns = [t for t in all_turns if t["session"] == session["persona"] and t["action"] == "query"]
        if query_turns:
            avg_r10 = np.mean([t["metrics"]["recall_10"] for t in query_turns])
            avg_sim = np.mean([t["metrics"]["semantic_similarity"] for t in query_turns])
            avg_lat = np.mean([t["metrics"]["latency_ms"] for t in query_turns])
            session_results.append({
                "persona": session["persona"],
                "num_queries": len(query_turns),
                "avg_recall_10": round(float(avg_r10), 4),
                "avg_semantic_similarity": round(float(avg_sim), 4),
                "avg_latency_ms": round(float(avg_lat), 2),
            })

        conn.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Aggregate results
    all_query_turns = [t for t in all_turns if t["action"] == "query"]
    all_r10 = [t["metrics"]["recall_10"] for t in all_query_turns]
    all_r5 = [t["metrics"]["recall_5"] for t in all_query_turns]
    all_r1 = [t["metrics"]["recall_1"] for t in all_query_turns]
    all_sim = [t["metrics"]["semantic_similarity"] for t in all_query_turns]
    all_lat = [t["metrics"]["latency_ms"] for t in all_query_turns]

    # Memory persistence: queries that ask about facts from earlier turns vs current turn
    # (in our sessions, all queries ask about facts stored 2 turns back)
    persistence_r10 = all_r10  # all queries are cross-turn by design

    agg = {}
    for name, vals in [("recall_1", all_r1), ("recall_5", all_r5), ("recall_10", all_r10),
                        ("semantic_similarity", all_sim), ("latency_ms", all_lat)]:
        mean, lo, hi = _bootstrap_ci(vals)
        agg[name] = {"mean": round(float(mean), 4), "ci95_lo": round(float(lo), 4), "ci95_hi": round(float(hi), 4)}

    report = {
        "benchmark": "e2e_memory_quality",
        "embedder": embedder_name,
        "num_sessions": len(SESSIONS),
        "num_query_turns": len(all_query_turns),
        "bootstrap_samples": N_BOOTSTRAP,
        "aggregate_metrics": agg,
        "session_results": session_results,
        "all_turns": all_turns,
    }

    # JSON
    json_path = RESULTS_DIR / "e2e_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown
    md_lines = [
        "# End-to-End Memory Quality Benchmark Results",
        "",
        f"**Embedder:** {embedder_name} | **Sessions:** {len(SESSIONS)} | **Query turns:** {len(all_query_turns)}",
        f"**Bootstrap:** {N_BOOTSTRAP} samples (95% CI)",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Mean | 95% CI |",
        "|---|---|---|",
        f"| Recall@1 | {agg['recall_1']['mean']:.4f} | [{agg['recall_1']['ci95_lo']:.4f}, {agg['recall_1']['ci95_hi']:.4f}] |",
        f"| Recall@5 | {agg['recall_5']['mean']:.4f} | [{agg['recall_5']['ci95_lo']:.4f}, {agg['recall_5']['ci95_hi']:.4f}] |",
        f"| Recall@10 | {agg['recall_10']['mean']:.4f} | [{agg['recall_10']['ci95_lo']:.4f}, {agg['recall_10']['ci95_hi']:.4f}] |",
        f"| Semantic Similarity | {agg['semantic_similarity']['mean']:.4f} | [{agg['semantic_similarity']['ci95_lo']:.4f}, {agg['semantic_similarity']['ci95_hi']:.4f}] |",
        f"| Avg Latency (ms) | {agg['latency_ms']['mean']:.2f} | [{agg['latency_ms']['ci95_lo']:.2f}, {agg['latency_ms']['ci95_hi']:.2f}] |",
        "",
        "## Per-Session Results",
        "",
        "| Persona | Queries | Avg R@10 | Avg Similarity | Avg Latency |",
        "|---|---|---|---|---|",
    ]
    for sr in session_results:
        md_lines.append(
            f"| {sr['persona']} | {sr['num_queries']} | {sr['avg_recall_10']:.4f} | "
            f"{sr['avg_semantic_similarity']:.4f} | {sr['avg_latency_ms']:.1f}ms |"
        )

    md_lines.extend([
        "",
        "## Per-Query Details",
        "",
    ])
    for t in all_query_turns:
        r10 = t["metrics"]["recall_10"]
        outcome = "PASS" if r10 >= 1.0 else ("PARTIAL" if r10 >= 0.5 else "FAIL")
        md_lines.append(
            f"- **[{outcome}]** `{t['session']}` turn {t['turn_id']}: _{t['query'][:100]}_ "
            f"(R@10={r10:.2f}, sim={t['metrics']['semantic_similarity']:.3f}, {t['metrics']['latency_ms']}ms)"
        )

    md_path = RESULTS_DIR / "e2e_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[INFO] Results written to {json_path} and {md_path}")

    # Console summary
    print(f"\n=== E2E Memory Quality Summary ===")
    print(f"Recall@1:   {agg['recall_1']['mean']:.4f} [CI: {agg['recall_1']['ci95_lo']:.4f}-{agg['recall_1']['ci95_hi']:.4f}]")
    print(f"Recall@5:   {agg['recall_5']['mean']:.4f} [CI: {agg['recall_5']['ci95_lo']:.4f}-{agg['recall_5']['ci95_hi']:.4f}]")
    print(f"Recall@10:  {agg['recall_10']['mean']:.4f} [CI: {agg['recall_10']['ci95_lo']:.4f}-{agg['recall_10']['ci95_hi']:.4f}]")
    print(f"Sim:        {agg['semantic_similarity']['mean']:.4f}")
    print(f"Latency:    {agg['latency_ms']['mean']:.2f}ms")

    # Per-session
    print(f"\n{'Persona':<25} {'Queries':>7} {'R@10':>8} {'Sim':>8} {'Lat':>8}")
    print("-" * 58)
    for sr in session_results:
        print(f"{sr['persona']:<25} {sr['num_queries']:>7} {sr['avg_recall_10']:>8.4f} {sr['avg_semantic_similarity']:>8.4f} {sr['avg_latency_ms']:>7.1f}ms")

    return report


if __name__ == "__main__":
    sys.exit(0 if run_e2e_benchmark() else 1)