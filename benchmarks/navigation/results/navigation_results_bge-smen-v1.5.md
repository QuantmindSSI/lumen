# PNE Benchmark [BAAI/bge-small-en-v1.5]
**Chunks:** 1,280 | **Queries:** 324 | **Seeds:** 3
**Routing:** 100.0% | **Pruning:** 79.9%

## Latency (ms)
| Strategy | Mean | p50 | p95 | p99 |
|---|---|---|---|---|
| global | 33.24 | 30.25 | 49.05 | 71.81 |
| oracle | 27.0 | 23.97 | 44.5 | 58.4 |
| routed | 26.41 | 23.87 | 47.45 | 63.94 |

## Recall@k
| k | Global | Oracle | Routed |
|---|---|---|---|
| 1 | 0.0219 | 0.0188 | 0.0219 |
| 3 | 0.0580 | 0.0546 | 0.0577 |
| 5 | 0.0923 | 0.0920 | 0.0921 |
| 10 | 0.1730 | 0.1723 | 0.1724 |
| 20 | 0.3337 | 0.3323 | 0.3324 |

## Oracle vs Global
| Metric | Mean | 95% CI |
|---|---|---|
| R@1 Retention | 0.9938 | [0.9846, 1.0000] |
| R@3 Retention | 0.9938 | [0.9846, 1.0031] |
| R@5 Retention | 1.0015 | [0.9956, 1.0080] |
| R@10 Retention | 0.9993 | [0.9926, 1.0064] |
| R@20 Retention | 0.9959 | [0.9921, 0.9997] |
| Latency Speedup | 1.32x | [1.26, 1.39] |

## Routed vs Global
| Metric | Mean | 95% CI |
|---|---|---|
| R@1 Retention | 0.9969 | [0.9907, 1.0000] |
| R@3 Retention | 0.9969 | [0.9897, 1.0036] |
| R@5 Retention | 1.0040 | [0.9981, 1.0106] |
| R@10 Retention | 0.9998 | [0.9928, 1.0070] |
| R@20 Retention | 0.9964 | [0.9926, 1.0000] |
| Latency Speedup | 1.35x | [1.30, 1.41] |