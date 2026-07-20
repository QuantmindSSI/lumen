# Performance Benchmark Results

## Query Latency & Footprint

| Corpus Size | Ingestion (s) | DB Size (MB) | RAM (MB) | Latency p50 (ms) | Latency p95 (ms) | Latency p99 (ms) |
|---|---|---|---|---|---|---|
| 1,000 | 1.44 | 2.91 | 68.14 | 16.07 | 26.08 | 35.3 |
| 10,000 | 27.1 | 27.92 | 128.65 | 156.79 | 178.75 | 180.68 |
| 50,000 | 417.97 | 139.76 | 397.73 | 850.76 | 873.55 | 877.02 |

## Write Amplification

| Corpus Size | Updates | Naive Time (ms) | Batch Time (ms) | Speedup |
|---|---|---|---|---|
| 1,000 | 1000 | 4.57 | 24.57 | 0.19x |
| 10,000 | 1000 | 5.64 | 33.48 | 0.17x |
| 50,000 | 1000 | 5.7 | 30.35 | 0.19x |
