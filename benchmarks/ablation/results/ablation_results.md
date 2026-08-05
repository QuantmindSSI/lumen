# Component Ablation Results
**Embedder:** all-MiniLM-L6-v2 | **Queries:** 20 | **Corpus:** domain_corpus (111 chunks)

## Results

| Configuration | Room Acc | Locus Acc | KW Recall | Latency | Δ Room | Δ Locus | Δ KW |
|---|---|---|---|---|---|---|---|
| FULL | 1.0000 | 1.0000 | 0.9608 | 47.7ms | +0.0000 | +0.0000 | +0.0000 |
| NO_TFC | 1.0000 | 1.0000 | 0.9608 | 66.1ms | +0.0000 | +0.0000 | +0.0000 |
| NO_VM | 1.0000 | 1.0000 | 0.9608 | 142.6ms | +0.0000 | +0.0000 | +0.0000 |
| NO_GRAPH | 1.0000 | 1.0000 | 0.9608 | 88.0ms | +0.0000 | +0.0000 | +0.0000 |
| BM25_ONLY | 1.0000 | 1.0000 | 0.9483 | 2.2ms | +0.0000 | +0.0000 | -0.0125 |
| DENSE_ONLY | 1.0000 | 1.0000 | 0.9608 | 31.0ms | +0.0000 | +0.0000 | +0.0000 |