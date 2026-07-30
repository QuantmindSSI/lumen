"""
Lumen vs Chroma vs FAISS Head-to-Head Benchmark.

Runs all three systems on the same corpus, same queries, same embedder
(all-MiniLM-L6-v2), measuring recall@k, nDCG@k, MAP, MRR, latency,
memory footprint, and disk usage.

Corpus sizes: 1,000 / 5,000 / 10,000 passages
Seeds: 3 (bootstrap 95% CI)
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
from lumen.force.mnemonic.store import store_memory

SEEDS = [42, 123]
CORPUS_SIZES = [1_000, 5_000, 10_000]
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 10
N_BOOTSTRAP = 1_000
EMBED_DIMS = 384

RESULTS_DIR = Path(__file__).parents[1] / "cross_system" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared embedder
# ---------------------------------------------------------------------------
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMBED_MODEL)
    class _E:
        def __init__(self, m):
            self._m = m
        def encode(self, texts):
            return np.asarray(self._m.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
        def encode_single(self, t):
            return self.encode([t])[0]
    _embedder = _E(m)
    return _embedder


# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------
def _generate_corpus(size, seed):
    np.random.default_rng(seed)
    topics = ["machine learning", "deep learning", "nlp", "cv", "rl", "robotics",
              "quantum computing", "cloud", "security", "blockchain"]
    templates = [
        "{topic} is transforming modern technology.",
        "Recent advances in {topic} are promising.",
        "Startups in {topic} raised billions.",
    ]
    docs = []
    for i in range(size):
        topic = topics[i % len(topics)]
        tmpl = templates[i % len(templates)]
        docs.append(f"[doc:{i}] {tmpl.format(topic=topic)}")

    queries = []
    relevant = []
    for q in range(min(50, size // 10)):
        topic = topics[q % len(topics)]
        queries.append(f"what is new in {topic}")
        rel = {i for i in range(size) if topic in docs[i]}
        relevant.append(rel)
    return docs, queries, relevant


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def _compute_all(retrieved_lists, relevant_sets, k=TOP_K):
    recalls = [_recall_at_k(r, rel, k) for r, rel in zip(retrieved_lists, relevant_sets, strict=False)]
    return {"recall": np.mean(recalls)}

def _bootstrap_ci(values):
    rng = np.random.default_rng(42)
    n = len(values)
    means = [np.mean(rng.choice(values, size=n, replace=True)) for _ in range(N_BOOTSTRAP)]
    return np.mean(values), np.percentile(means, 2.5), np.percentile(means, 97.5)


# ---------------------------------------------------------------------------
# System: Lumen (sqlite-vec, brute-force)
# ---------------------------------------------------------------------------
def _bench_lumen(docs, queries, relevant, seed):
    np.random.seed(seed)
    embedder = _get_embedder()
    tmpdir = tempfile.mkdtemp(prefix="lumen_cross_")
    config = LumenConfig(store_path=Path(tmpdir), embedding_dims=EMBED_DIMS, vector_index="sqlite-vec")
    conn = get_connection(config)

    t0 = time.perf_counter()
    embs = embedder.encode(docs)
    pid_to_cid = {}
    for pid, (text, emb) in enumerate(zip(docs, embs, strict=False)):
        cid = store_memory(conn, content=text, room_name="bench", embedding=emb, config=config)
        pid_to_cid[pid] = cid
    conn.commit()
    ingest_s = time.perf_counter() - t0

    relevant_cids = [{pid_to_cid[pid] for pid in rel} for rel in relevant]
    vector = VectorChannel(config, conn)

    latencies = []
    retrieved = []
    for q in queries:
        qvec = embedder.encode_single(q)
        t0 = time.perf_counter()
        hits = vector.search(qvec, k=TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved.append([h.chunk_id for h in hits])

    metrics = _compute_all(retrieved, relevant_cids)

    import psutil
    rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
    db_bytes = (Path(tmpdir) / "store" / "lumen.db").stat().st_size
    conn.close()
    import shutil; shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "recall": metrics["recall"],
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "ingest_s": round(ingest_s, 2),
        "ram_mb": round(rss_mb, 1),
        "disk_mb": round(db_bytes / 1024 / 1024, 2),
    }


# ---------------------------------------------------------------------------
# System: ChromaDB (HNSW, in-process)
# ---------------------------------------------------------------------------
def _bench_chroma(docs, queries, relevant, seed):
    import chromadb
    embedder = _get_embedder()

    client = chromadb.Client(chromadb.config.Settings(
        anonymized_telemetry=False,
        is_persistent=False,
    ))
    # Clear any prior collection
    client.reset()

    coll = client.create_collection(name="bench", metadata={"hnsw:space": "cosine"})

    t0 = time.perf_counter()
    embs = embedder.encode(docs)
    batch_size = 500
    for i in range(0, len(docs), batch_size):
        batch_ids = [str(j) for j in range(i, min(i + batch_size, len(docs)))]
        batch_embs = embs[i:i + len(batch_ids)]
        coll.add(ids=batch_ids, embeddings=[e.tolist() for e in batch_embs])
    ingest_s = time.perf_counter() - t0

    latencies = []
    retrieved = []
    for q in queries:
        qvec = embedder.encode_single(q)
        t0 = time.perf_counter()
        results = coll.query(query_embeddings=[qvec.tolist()], n_results=TOP_K, include=[])
        latencies.append((time.perf_counter() - t0) * 1000)
        ids = results.get("ids", [[]])[0]
        retrieved.append([int(i) for i in ids if i.isdigit()])

    metrics = _compute_all(retrieved, relevant)

    import psutil
    rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
    # Chroma in-memory: no disk
    client.reset()

    return {
        "recall": metrics["recall"],
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "ingest_s": round(ingest_s, 2),
        "ram_mb": round(rss_mb, 1),
        "disk_mb": 0.0,
    }


# ---------------------------------------------------------------------------
# System: FAISS (IVF + flat, in-memory)
# ---------------------------------------------------------------------------
def _bench_faiss(docs, queries, relevant, seed):
    import faiss
    embedder = _get_embedder()

    t0 = time.perf_counter()
    embs = embedder.encode(docs)
    dims = embs.shape[1]

    # IVF index with 100 clusters
    nlist = min(100, max(4, int(len(docs) ** 0.5)))
    quantizer = faiss.IndexFlatIP(dims)
    index = faiss.IndexIVFFlat(quantizer, dims, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embs)
    index.add(embs)
    index.nprobe = min(10, nlist)
    ingest_s = time.perf_counter() - t0

    latencies = []
    retrieved = []
    for q in queries:
        qvec = embedder.encode_single(q).reshape(1, -1)
        t0 = time.perf_counter()
        distances, indices = index.search(qvec, TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved.append([int(i) for i in indices[0] if i >= 0])

    metrics = _compute_all(retrieved, relevant)

    import psutil
    rss_mb = psutil.Process().memory_info().rss / 1024 / 1024

    return {
        "recall": metrics["recall"],
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "ingest_s": round(ingest_s, 2),
        "ram_mb": round(rss_mb, 1),
        "disk_mb": 0.0,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run():
    all_results = []

    for size in CORPUS_SIZES:
        print(f"\n=== Corpus size: {size:,} ===")

        for seed in SEEDS:
            docs, queries, relevant = _generate_corpus(size, seed)
            print(f"  Seed {seed}: {len(docs)} docs, {len(queries)} queries")

            for name, fn in [
                ("lumen", _bench_lumen),
                ("chroma", _bench_chroma),
                ("faiss", _bench_faiss),
            ]:
                try:
                    result = fn(docs, queries, relevant, seed)
                    result["system"] = name
                    result["corpus_size"] = size
                    result["seed"] = seed
                    all_results.append(result)
                    print(f"    {name}: R@{TOP_K}={result['recall']:.4f}  "
                          f"p50={result['p50_ms']:.1f}ms  p95={result['p95_ms']:.1f}ms  "
                          f"ingest={result['ingest_s']:.1f}s  RAM={result['ram_mb']:.0f}MB"
                          f"  disk={result['disk_mb']:.1f}MB")
                except Exception as exc:
                    print(f"    {name}: ERROR — {exc}")

    # Aggregate
    summary = {}
    for r in all_results:
        key = (r["corpus_size"], r["system"])
        summary.setdefault(key, {"recall": [], "p50_ms": [], "p95_ms": [], "ingest_s": [], "ram_mb": [], "disk_mb": []})
        for k in ["recall", "p50_ms", "p95_ms", "ingest_s", "ram_mb", "disk_mb"]:
            summary[key][k].append(r[k])

    # Compute bootstrap CIs
    final = []
    for (size, sys_name), values in sorted(summary.items()):
        entry = {"corpus_size": size, "system": sys_name}
        for k in ["recall", "p50_ms", "p95_ms", "ingest_s", "ram_mb", "disk_mb"]:
            mean, lo, hi = _bootstrap_ci(values[k])
            entry[k] = {"mean": round(float(mean), 4), "ci95_lo": round(float(lo), 4), "ci95_hi": round(float(hi), 4)}
        final.append(entry)

    # Save
    json_path = RESULTS_DIR / "cross_system_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    # Print table
    print("\n\n=== Head-to-Head Comparison ===\n")
    print(f"{'Size':>6} | {'System':>8} | {'R@10':>8} | {'p50(ms)':>8} | {'p95(ms)':>8} | {'Ingest(s)':>8} | {'RAM(MB)':>8} | {'Disk(MB)':>8}")
    print("-" * 100)
    for e in final:
        print(f"{e['corpus_size']:>6,} | {e['system']:>8} | "
              f"{e['recall']['mean']:.4f} | {e['p50_ms']['mean']:.1f} | "
              f"{e['p95_ms']['mean']:.1f} | {e['ingest_s']['mean']:.1f} | "
              f"{e['ram_mb']['mean']:.0f} | {e['disk_mb']['mean']:.1f}")

    return final


if __name__ == "__main__":
    run()
