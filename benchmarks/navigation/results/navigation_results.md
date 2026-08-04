# Palace Navigation Efficiency (PNE) Benchmark Results

**Embedder:** all-MiniLM-L6-v2 | **Total Chunks:** 2,400 | **Queries:** 324 | **Seeds:** 3
**Rooms:** 8 | **Cross-Topic Rate:** 0.15 | **Bootstrap:** 1000 samples (95% CI)

## Routing & Pruning

- **Intent Routing Accuracy:** 100.0%
- **Mean Pruning Ratio:** 79.9% of corpus excluded by room filter
- **Median Pruning Ratio:** 75.0%

## Latency Comparison (ms)

| Strategy | Mean | p50 | p95 | p99 |
|---|---|---|---|---|
| global | 46.86 | 21.56 | 144.07 | 202.39 |
| oracle | 48.87 | 25.09 | 145.08 | 183.42 |
| routed | 47.13 | 26.9 | 139.66 | 193.97 |

## Recall@k

| k | Global | Oracle | Routed |
|---|---|---|---|
| 1 | 0.0102 | 0.0102 | 0.0102 |
| 3 | 0.0302 | 0.0302 | 0.0302 |
| 5 | 0.0511 | 0.0501 | 0.0511 |
| 10 | 0.0983 | 0.0974 | 0.0986 |
| 20 | 0.1906 | 0.1901 | 0.1908 |

## Oracle Room-Constrained vs Global

| Metric | Mean | 95% CI |
|---|---|---|
| Recall Retention @ 1 | 1.0000 | [1.0000, 1.0000] |
| Recall Retention @ 3 | 1.0000 | [1.0000, 1.0000] |
| Recall Retention @ 5 | 0.9969 | [0.9907, 1.0000] |
| Recall Retention @ 10 | 0.9994 | [0.9954, 1.0027] |
| Recall Retention @ 20 | 0.9984 | [0.9957, 1.0008] |
| Latency Speedup | 1.43x | [1.24x, 1.63x] |

## Routed Room-Constrained vs Global

| Metric | Mean | 95% CI |
|---|---|---|
| Recall Retention @ 1 | 1.0000 | [1.0000, 1.0000] |
| Recall Retention @ 3 | 1.0000 | [1.0000, 1.0000] |
| Recall Retention @ 5 | 1.0000 | [1.0000, 1.0000] |
| Recall Retention @ 10 | 1.0036 | [1.0004, 1.0079] |
| Recall Retention @ 20 | 1.0030 | [0.9992, 1.0083] |
| Latency Speedup | 1.71x | [1.46x, 1.97x] |

## Intent Distribution

- **factual:** 324 queries