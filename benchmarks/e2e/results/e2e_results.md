# End-to-End Memory Quality Benchmark Results

**Embedder:** all-MiniLM-L6-v2 | **Sessions:** 7 | **Query turns:** 28
**Bootstrap:** 1000 samples (95% CI)

## Aggregate Metrics

| Metric | Mean | 95% CI |
|---|---|---|
| Recall@1 | 0.9643 | [0.8929, 1.0000] |
| Recall@5 | 1.0000 | [1.0000, 1.0000] |
| Recall@10 | 1.0000 | [1.0000, 1.0000] |
| Semantic Similarity | 1.0000 | [1.0000, 1.0000] |
| Avg Latency (ms) | 56.22 | [35.40, 78.71] |

## Per-Session Results

| Persona | Queries | Avg R@10 | Avg Similarity | Avg Latency |
|---|---|---|---|---|
| Security Consultant | 4 | 1.0000 | 1.0000 | 80.0ms |
| ML Engineer | 4 | 1.0000 | 1.0000 | 38.5ms |
| Medical Researcher | 4 | 1.0000 | 1.0000 | 49.7ms |
| QA Assistant | 4 | 1.0000 | 1.0000 | 48.1ms |
| Architecture Reviewer | 4 | 1.0000 | 1.0000 | 56.8ms |
| Climate Analyst | 4 | 1.0000 | 1.0000 | 46.1ms |
| Database Admin | 4 | 1.0000 | 1.0000 | 74.3ms |

## Per-Query Details

- **[PASS]** `Security Consultant` turn 3: _What cloud provider does AlphaCorp use for primary hosting and what is their main database setup?_ (R@10=1.00, sim=1.000, 14.99ms)
- **[PASS]** `Security Consultant` turn 3: _How many transactions per hour does BetaFinance process at peak?_ (R@10=1.00, sim=1.000, 14.62ms)
- **[PASS]** `Security Consultant` turn 4: _What was AlphaCorp CISO's deadline for fixing critical vulnerabilities and how many were IAM-related_ (R@10=1.00, sim=1.000, 275.65ms)
- **[PASS]** `Security Consultant` turn 4: _What issues did BetaFinance's 2024 audit find regarding their secrets management?_ (R@10=1.00, sim=1.000, 14.71ms)
- **[PASS]** `ML Engineer` turn 3: _What is the validation accuracy of the latest model and how does it compare to the previous version?_ (R@10=1.00, sim=1.000, 30.1ms)
- **[PASS]** `ML Engineer` turn 3: _What hyperparameter values did Optuna select as optimal for this training run?_ (R@10=1.00, sim=1.000, 32.58ms)
- **[PASS]** `ML Engineer` turn 4: _What is the INT8 model size after quantization and what was the accuracy impact?_ (R@10=1.00, sim=1.000, 41.98ms)
- **[PASS]** `ML Engineer` turn 4: _How is data drift monitored and what is the current alert threshold?_ (R@10=1.00, sim=1.000, 49.4ms)
- **[PASS]** `Medical Researcher` turn 3: _What was the primary endpoint of the TRILUMINATE trial and was it met?_ (R@10=1.00, sim=1.000, 15.47ms)
- **[PASS]** `Medical Researcher` turn 3: _For which cancer type did KEYNOTE-024 establish pembrolizumab as first-line therapy and what was the_ (R@10=1.00, sim=1.000, 15.94ms)
- **[PASS]** `Medical Researcher` turn 4: _What quality-of-life metric was used in TRILUMINATE and what was the between-group difference?_ (R@10=1.00, sim=1.000, 125.31ms)
- **[PASS]** `Medical Researcher` turn 4: _What genomic biomarker was validated by KEYNOTE-158 for tissue-agnostic pembrolizumab approval?_ (R@10=1.00, sim=1.000, 42.22ms)
- **[PASS]** `QA Assistant` turn 3: _What key features were added in Python 3.13 and when was it released?_ (R@10=1.00, sim=1.000, 83.32ms)
- **[PASS]** `QA Assistant` turn 3: _Who discovered the xz utils backdoor and what was the immediate symptom they noticed?_ (R@10=1.00, sim=1.000, 16.13ms)
- **[PASS]** `QA Assistant` turn 4: _What new database features did Django 5.1 introduce?_ (R@10=1.00, sim=1.000, 11.61ms)
- **[PASS]** `QA Assistant` turn 4: _What was the root cause of the CrowdStrike outage and how many devices were affected?_ (R@10=1.00, sim=1.000, 81.25ms)
- **[PASS]** `Architecture Reviewer` turn 3: _What payment providers does the payment gateway support and how does it ensure safe retries?_ (R@10=1.00, sim=1.000, 20.8ms)
- **[PASS]** `Architecture Reviewer` turn 3: _What is the total pipeline time from commit to production deployment?_ (R@10=1.00, sim=1.000, 12.54ms)
- **[PASS]** `Architecture Reviewer` turn 4: _What notification channels does notify-hub support and what are the priority tiers?_ (R@10=1.00, sim=1.000, 61.97ms)
- **[PASS]** `Architecture Reviewer` turn 4: _What scaling mechanisms are used and what is the cold start time?_ (R@10=1.00, sim=1.000, 131.71ms)
- **[PASS]** `Climate Analyst` turn 3: _What is EcoCoat's Scope 1 emissions for 2023 and what is their main source?_ (R@10=1.00, sim=1.000, 139.87ms)
- **[PASS]** `Climate Analyst` turn 3: _How many solar farms does SolarGrid operate and what is their total capacity?_ (R@10=1.00, sim=1.000, 13.56ms)
- **[PASS]** `Climate Analyst` turn 4: _What are EcoCoat's targeted emission reductions by 2026 and through which initiatives?_ (R@10=1.00, sim=1.000, 14.35ms)
- **[PASS]** `Climate Analyst` turn 4: _What battery storage does the Antelope Valley project include and what is its duration?_ (R@10=1.00, sim=1.000, 16.63ms)
- **[PASS]** `Database Admin` turn 3: _What replication feature did PostgreSQL 16 add and what backup feature did PostgreSQL 17 add?_ (R@10=1.00, sim=1.000, 16.42ms)
- **[PASS]** `Database Admin` turn 3: _What new expiration capability did Redis 7.4 introduce?_ (R@10=1.00, sim=1.000, 76.56ms)
- **[PASS]** `Database Admin` turn 4: _What are the key PostgreSQL configuration values for the production cluster and what is the RPO?_ (R@10=1.00, sim=1.000, 189.36ms)
- **[PASS]** `Database Admin` turn 4: _What is the Redis migration strategy for the 7.4 upgrade and what is the rollback plan?_ (R@10=1.00, sim=1.000, 15.07ms)