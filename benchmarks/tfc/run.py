"""TFC sensitivity analysis.

Grid search across Twin-Force Controller parameters to understand how
each dimension affects retrieval quality and memory survival.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from itertools import product
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lumen.config import LumenConfig
from lumen.controller import TwinForceController
from lumen.data.schema import get_connection
from lumen.force.mnemonic.retrieval_dense import VectorChannel
from lumen.force.mnemonic.store import store_memory
from lumen.search import SearchPipeline

RESULTS_DIR = Path(__file__).with_suffix("").parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMBED_DIMS = 384
N_PASSAGES = 500
N_QUERIES = 50
TOP_K = 10

# Parameter grid
E_VALUES = [0.3, 0.5, 0.7]
A_VALUES = [0.3, 0.5, 0.7]
TAU_VALUES = [3.0, 7.0, 14.0]
R_VALUES = [1, 3, 5]


def _generate_corpus_and_queries(rng: np.random.Generator, n_passages: int, n_queries: int, dim: int):
    embeddings = rng.standard_normal((n_passages, dim)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    texts = [f"Doc {i} {' '.join([f'word{rng.integers(0, 200)}' for _ in range(15)])}" for i in range(n_passages)]

    q_indices = rng.choice(n_passages, size=n_queries, replace=False)
    query_embs = []
    query_relevant = []
    for idx in q_indices:
        noise = rng.standard_normal(dim).astype(np.float32) * 0.15
        q_emb = embeddings[idx] + noise
        q_emb /= np.linalg.norm(q_emb) + 1e-12
        query_embs.append(q_emb)
        sims = embeddings @ q_emb
        top5 = np.argpartition(-sims, 5)[:5]
        query_relevant.append({int(i): float(sims[i]) for i in top5})
    return texts, embeddings, query_embs, query_relevant


def _compute_recall(retrieved_lists: list[list[int]], query_relevant: list[dict[int, float]], k: int):
    recalls = []
    for retrieved, rel in zip(retrieved_lists, query_relevant, strict=False):
        hits = len(set(retrieved[:k]) & set(rel.keys()))
        recalls.append(hits / max(1, len(rel)))
    return float(np.mean(recalls))


def run_single_config(e: float, a: float, tau: float, r: int, texts: list[str], embeddings: np.ndarray, query_embs: list[np.ndarray], query_relevant: list[dict[int, float]]):
    config = LumenConfig(embedding_dims=EMBED_DIMS, vector_index="sqlite-vec")
    tmpdir = tempfile.mkdtemp(prefix="lumen_tfc_")
    config.store_path = Path(tmpdir) / "store"

    conn = get_connection(config)

    # Override TFC defaults
    config.tfc_default_tau = tau
    config.tfc_default_resolution = r

    pid_to_chunk_id = {}
    for pid, (text, emb) in enumerate(zip(texts, embeddings, strict=False)):
        chunk_id = store_memory(
            conn,
            content=text,
            room_name="benchmark",
            locus_name="test",
            source_type="import",
            embedding=emb,
            config=config,
        )
        pid_to_chunk_id[pid] = chunk_id
    conn.commit()

    # Build embedder that returns query vectors by pattern match
    query_emb_map = dict(enumerate(query_embs))
    class _BenchEmbedder:
        def encode_single(self, text: str) -> np.ndarray:
            if text.startswith("benchmark query"):
                idx = int(text.split("_")[-1]) if "_" in text else 0
                return query_emb_map.get(idx, np.zeros(EMBED_DIMS, dtype=np.float32))
            return np.zeros(EMBED_DIMS, dtype=np.float32)
        def encode(self, texts: list[str]) -> np.ndarray:
            return np.stack([self.encode_single(t) for t in texts])

    embedder = _BenchEmbedder()

    tfc = TwinForceController()
    tfc.state.e = e
    tfc.state.a = a
    tfc.state.tau = tau
    tfc.state.r = r

    pipeline = SearchPipeline(conn, config, embedder=embedder, tfc=tfc)
    VectorChannel(config, conn)

    retrieved_lists = []
    latencies = []
    for i, _q in enumerate(query_embs):
        t0 = time.perf_counter()
        results = [r.chunk_id for r in pipeline.execute(f"benchmark query_{i}", k=50, max_repair_attempts=0)]
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved_lists.append(results)

    recall = _compute_recall(retrieved_lists, [
        {pid_to_chunk_id[pid]: score for pid, score in rel.items() if pid in pid_to_chunk_id}
        for rel in query_relevant
    ], TOP_K)

    # Simulate decay after tau days to measure survival
    from lumen.force.mnemonic.forgetting_l1_decay import ebbinghaus_decay
    ebbinghaus_decay(conn, user_half_life_days=tau)
    active = conn.execute("SELECT COUNT(*) FROM chunk WHERE valid_to IS NULL").fetchone()[0]
    survival_rate = active / max(1, N_PASSAGES)

    conn.close()
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "e": e,
        "a": a,
        "tau": tau,
        "r": r,
        "recall@10": round(recall, 4),
        "latency_ms_mean": round(float(np.mean(latencies)), 2),
        "survival_rate": round(survival_rate, 4),
    }


def main():
    print("=== TFC Sensitivity Analysis ===")
    rng = np.random.default_rng(123)
    texts, embeddings, query_embs, query_relevant = _generate_corpus_and_queries(rng, N_PASSAGES, N_QUERIES, EMBED_DIMS)

    results = []
    total = len(E_VALUES) * len(A_VALUES) * len(TAU_VALUES) * len(R_VALUES)
    for i, (e, a, tau, r) in enumerate(product(E_VALUES, A_VALUES, TAU_VALUES, R_VALUES)):
        print(f"\n[{i+1}/{total}] e={e} a={a} tau={tau} r={r}")
        res = run_single_config(e, a, tau, r, texts, embeddings, query_embs, query_relevant)
        results.append(res)
        print(f"  recall@10={res['recall@10']:.3f}, survival={res['survival_rate']:.2%}, latency={res['latency_ms_mean']:.1f}ms")

    out_path = RESULTS_DIR / "tfc_sensitivity.json"
    out_path.write_text(json.dumps({
        "n_passages": N_PASSAGES,
        "n_queries": N_QUERIES,
        "embed_dims": EMBED_DIMS,
        "grid": {
            "e": E_VALUES,
            "a": A_VALUES,
            "tau": TAU_VALUES,
            "r": R_VALUES,
        },
        "results": results,
    }, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
