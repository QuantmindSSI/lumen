# PNE Benchmark [all-MiniLM-L6-v2]
**Chunks:** 1,280 | **Queries:** 324 | **Seeds:** 3
**Routing:** 100.0% | **Pruning:** 79.9%

## Latency (ms)
| Strategy | Mean | p50 | p95 | p99 |
|---|---|---|---|---|
| global | 25.59 | 22.76 | 45.07 | 67.8 |
| oracle | 18.55 | 16.29 | 35.41 | 56.31 |
| routed | 18.98 | 16.47 | 35.99 | 49.39 |

## Recall@k
| k | Global | Oracle | Routed |
|---|---|---|---|
| 1 | 0.0219 | 0.0188 | 0.0219 |
| 3 | 0.0590 | 0.0590 | 0.0590 |
| 5 | 0.0963 | 0.0962 | 0.0963 |
| 10 | 0.1834 | 0.1830 | 0.1837 |
| 20 | 0.3490 | 0.3479 | 0.3487 |

## Oracle vs Global
| Metric | Mean | 95% CI |
|---|---|---|
| R@1 Retention | 0.9969 | [0.9907, 1.0000] |
| R@3 Retention | 1.0000 | [1.0000, 1.0000] |
| R@5 Retention | 0.9988 | [0.9963, 1.0000] |
| R@10 Retention | 0.9984 | [0.9961, 1.0003] |
| R@20 Retention | 0.9971 | [0.9948, 0.9993] |
| Latency Speedup | 1.50x | [1.42, 1.57] |

## Routed vs Global
| Metric | Mean | 95% CI |
|---|---|---|
| R@1 Retention | 1.0000 | [1.0000, 1.0000] |
| R@3 Retention | 1.0000 | [1.0000, 1.0000] |
| R@5 Retention | 0.9995 | [0.9963, 1.0023] |
| R@10 Retention | 1.0022 | [0.9988, 1.0067] |
| R@20 Retention | 0.9998 | [0.9975, 1.0025] |
| Latency Speedup | 1.50x | [1.41, 1.59] |