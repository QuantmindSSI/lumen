# Component Ablation Results
**Embedder:** all-MiniLM-L6-v2 | **Queries:** 20 | **Corpus:** domain_corpus (111 chunks)

## Results

| Configuration | Room Acc | Locus Acc | KW Recall | Latency | Δ Room | Δ Locus | Δ KW |
|---|---|---|---|---|---|---|---|
| FULL | 1.0000 | 1.0000 | 0.9608 | 28.6ms | +0.0000 | +0.0000 | +0.0000 |
| NO_TFC | 1.0000 | 1.0000 | 0.9608 | 19.7ms | +0.0000 | +0.0000 | +0.0000 |
| NO_VM | 1.0000 | 1.0000 | 0.9608 | 18.6ms | +0.0000 | +0.0000 | +0.0000 |
| NO_GRAPH | 1.0000 | 1.0000 | 0.9608 | 17.3ms | +0.0000 | +0.0000 | +0.0000 |
| BM25_ONLY | 1.0000 | 1.0000 | 0.9483 | 2.1ms | +0.0000 | +0.0000 | -0.0125 |
| DENSE_ONLY | 1.0000 | 1.0000 | 0.9608 | 14.9ms | +0.0000 | +0.0000 | +0.0000 |