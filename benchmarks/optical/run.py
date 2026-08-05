"""Optical degradation benchmark.

Measures retrieval quality (recall@k, nDCG@k) as embeddings are quantized
through the degradation chain: FP32 → FP16 → INT8 → BINARY.

This validates the accuracy-vs-storage trade-off at each precision level.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lumen.config import LumenConfig
from lumen.data.schema import get_connection
from lumen.force.mnemonic.retrieval_dense import VectorChannel
from lumen.force.mnemonic.retrieval_lexical import LexicalChannel
from lumen.force.mnemonic.store import store_memory
from lumen.search import SearchPipeline
from lumen.sovereign.optical import quantize_vector

RESULTS_DIR = Path(__file__).with_suffix("").parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMBED_DIMS = 384
N_PASSAGES = 1_000
N_QUERIES = 100
TOP_K_VALUES = [1, 3, 5, 10]


def _generate_synthetic_corpus(rng: np.random.Generator, n: int, dim: int):
    """Generate random normalized embeddings and text."""
    embeddings = rng.standard_normal((n, dim)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    texts = [f"Document {i} about {' '.join([f'topic{rng.integers(0, 100)}' for _ in range(20)])}" for i in range(n)]
    return texts, embeddings


def _generate_queries(rng: np.random.Generator, corpus_embeddings: np.ndarray, n: int, k_relevant: int = 5):
    """Generate queries by noising corpus embeddings and pick top-k cosine neighbors as relevant."""
    dim = corpus_embeddings.shape[1]
    indices = rng.choice(len(corpus_embeddings), size=n, replace=False)
    query_embs = []
    query_relevant = []
    for idx in indices:
        noise = rng.standard_normal(dim).astype(np.float32) * 0.1
        q_emb = corpus_embeddings[idx] + noise
        q_emb /= np.linalg.norm(q_emb) + 1e-12
        query_embs.append(q_emb)
        # Top-k cosine similarity neighbors
        sims = corpus_embeddings @ q_emb
        top_k = np.argpartition(-sims, k_relevant)[:k_relevant]
        query_relevant.append({int(i): float(sims[i]) for i in top_k})
    return query_embs, query_relevant


def _compute_metrics(retrieved_lists: list[list[int]], query_relevant: list[dict[int, float]], k_values: list[int]):
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    for retrieved, rel in zip(retrieved_lists, query_relevant, strict=False):
        rel_set = set(rel.keys())
        for k in k_values:
            topk = retrieved[:k]
            hits = len(set(topk) & rel_set)
            metrics[f"recall@{k}"].append(hits / max(1, len(rel_set)))
            # nDCG
            dcg = sum((1.0 / np.log2(i + 2)) for i, cid in enumerate(topk) if cid in rel_set)
            ideal_hits = min(len(rel_set), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
            metrics[f"ndcg@{k}"].append(dcg / max(idcg, 1e-12))
    return metrics


def run_degradation_level(level: str, docs: list[str], corpus_embeddings: np.ndarray, query_embs: list[np.ndarray], query_relevant: list[dict[int, float]]):
    """Run retrieval benchmark at a specific quantization level."""
    config = LumenConfig(embedding_dims=EMBED_DIMS, vector_index="sqlite-vec")
    tmpdir = tempfile.mkdtemp(prefix="lumen_optical_")
    config.store_path = Path(tmpdir) / "store"

    conn = get_connection(config)
    class _BenchEmbedder:
        def __init__(self, query_map: dict[int, np.ndarray]):
            self._query_map = query_map
        def encode_single(self, text: str) -> np.ndarray:
            # Extract index from "query {i}" pattern used below
            if text.startswith("query "):
                idx = int(text.split()[1])
                return self._query_map[idx]
            return np.zeros(EMBED_DIMS, dtype=np.float32)
        def encode(self, texts: list[str]) -> np.ndarray:
            return np.stack([self.encode_single(t) for t in texts])

    pid_to_chunk_id = {}
    for pid, (text, emb) in enumerate(zip(docs, corpus_embeddings, strict=False)):
        q_emb = quantize_vector(emb, level) if level != "FP32" else emb
        chunk_id = store_memory(
            conn,
            content=text,
            room_name="benchmark",
            locus_name="test",
            source_type="import",
            embedding=q_emb,
            config=config,
        )
        pid_to_chunk_id[pid] = chunk_id
    conn.commit()

    # Update query relevant mapping to chunk IDs
    query_relevant_chunks = [
        {pid_to_chunk_id[pid]: score for pid, score in rel.items() if pid in pid_to_chunk_id}
        for rel in query_relevant
    ]

    # Build query embedding map for the embedder
    query_emb_map = dict(enumerate(query_embs))
    embedder = _BenchEmbedder(query_emb_map)

    LexicalChannel(conn)
    vector = VectorChannel(config, conn)
    pipeline = SearchPipeline(conn, config, embedder=embedder)

    configs = {
        "dense": lambda q, i: [h.chunk_id for h in vector.search(quantize_vector(q, level) if level != "FP32" else q, k=50)],
        "hybrid": lambda q, i: [r.chunk_id for r in pipeline.execute(f"query {i}", k=50, max_repair_attempts=0)],
    }

    results_by_config = {}
    latencies_by_config = {}
    for cfg_name, fn in configs.items():
        retrieved_lists = []
        latencies = []
        for i, q in enumerate(query_embs):
            t0 = time.perf_counter()
            results = fn(q, i)
            latencies.append((time.perf_counter() - t0) * 1000)
            retrieved_lists.append(results)
        results_by_config[cfg_name] = retrieved_lists
        latencies_by_config[cfg_name] = latencies

    conn.close()
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    summary = {"level": level}
    for cfg_name, retrieved_lists in results_by_config.items():
        metrics = _compute_metrics(retrieved_lists, query_relevant_chunks, TOP_K_VALUES)
        summary[cfg_name] = {
            mk: round(float(np.mean(values)), 4)
            for mk, values in metrics.items()
        }
        summary[cfg_name]["latency_ms_mean"] = round(float(np.mean(latencies_by_config[cfg_name])), 2)
    return summary


def main():
    print("=== Optical Degradation Benchmark ===")
    rng = np.random.default_rng(42)
    docs, corpus_embeddings = _generate_synthetic_corpus(rng, N_PASSAGES, EMBED_DIMS)
    query_embs, query_relevant = _generate_queries(rng, corpus_embeddings, N_QUERIES)

    all_results = []
    for level in ["FP32", "FP16", "INT8", "BINARY"]:
        print(f"\n--- {level} ---")
        result = run_degradation_level(level, docs, corpus_embeddings, query_embs, query_relevant)
        all_results.append(result)
        for cfg_name in ["dense", "hybrid"]:
            m = result[cfg_name]
            print(f"  {cfg_name}: R@10={m['recall@10']:.3f}, nDCG@10={m['ndcg@10']:.3f}, latency={m['latency_ms_mean']:.1f}ms")

    out_path = RESULTS_DIR / "optical_degradation.json"
    out_path.write_text(json.dumps({
        "n_passages": N_PASSAGES,
        "n_queries": N_QUERIES,
        "embed_dims": EMBED_DIMS,
        "results": all_results,
    }, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
