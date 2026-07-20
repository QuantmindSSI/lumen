# Retrieval Benchmark Results

**Corpus:** synthetic | **Passages:** 1000 | **Queries:** 50 | **Top-K:** 10

| Configuration | Recall@10 (mean ± std) | nDCG@10 (mean ± std) | Latency p50 (ms) | Latency p95 (ms) | Latency p99 (ms) |
|---|---|---|---|---|---|
| bm25_only | 0.045 ± 0.0687 | 0.3 ± 0.4583 | 0.35 | 0.9 | 0.98 |
| dense_only | 0.0084 ± 0.01 | 0.0534 ± 0.0667 | 0.99 | 1.25 | 1.34 |
| hybrid | 0.009 ± 0.01 | 0.0681 ± 0.0859 | 6.16 | 9.61 | 15.22 |
