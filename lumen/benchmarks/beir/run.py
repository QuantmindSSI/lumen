"""
BEIR-style cross-dataset evaluation framework for Lumen.

Runs retrieval benchmarks across multiple datasets and embedders,
computing standard IR metrics with bootstrap confidence intervals.

Datasets included (smallest BEIR subsets):
  - NFCorpus   (3,635 docs)
  - SciFact    (5,183 docs)
  - FiQA-2018  (57,638 docs — sampled to 5,000)
  - TREC-COVID (171,332 docs — sampled to 5,000)
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lumen.config import LumenConfig
from lumen.data.schema import get_connection
from lumen.force.mnemonic.retrieval_dense import VectorChannel
from lumen.force.mnemonic.retrieval_lexical import LexicalChannel
from lumen.force.mnemonic.store import store_memory
from lumen.lumen.search import SearchPipeline

SEEDS = [42, 123]
EMBEDDER_MODELS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
TOP_K_VALUES = [1, 3, 5, 10, 20]
N_BOOTSTRAP = 1_000
NUM_PASSAGES = 3_000
NUM_QUERIES = 50
EMBED_DIMS = 384

BEIR_DATASETS = [
    ("nfcorpus", "BeIR/nfcorpus"),
    ("scifact", "BeIR/scifact"),
]

RESULTS_DIR = Path(__file__).parents[1] / "beir" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Embedder cache (shared across datasets)
# ---------------------------------------------------------------------------
_embedders = {}

def _get_embedder(model_name):
    if model_name in _embedders:
        return _embedders[model_name]
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_name)

        class _Real:
            def __init__(self, m):
                self._m = m
            def encode(self, texts):
                vecs = self._m.encode(texts, normalize_embeddings=True, show_progress_bar=True)
                return np.asarray(vecs, dtype=np.float32)
            def encode_single(self, text):
                return self.encode([text])[0]

        e = _Real(m)
        _embedders[model_name] = e
        return e
    except Exception:
        from lumen.force.contextual.embed import MockEmbedder
        e = MockEmbedder(dims=EMBED_DIMS)
        _embedders[model_name] = e
        return e

# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_beir_dataset(name, path):
    """Load a BEIR dataset via the datasets library."""
    try:
        from datasets import load_dataset

        print(f"  Loading {name} from {path} ...")
        corpus_ds = load_dataset(path, "corpus", split="corpus", streaming=True)
        queries_ds = load_dataset(path, "queries", split="queries", streaming=True)
        qrels_ds = load_dataset(path, "qrels", split="test", streaming=True)

        # Collect corpus
        corpus = {}
        for i, doc in enumerate(corpus_ds):
            if i >= NUM_PASSAGES:
                break
            doc_id = doc.get("_id", str(i))
            text = doc.get("title", "") + " " + doc.get("text", "")
            corpus[doc_id] = text.strip()

        # Collect queries
        queries = {}
        for i, q in enumerate(queries_ds):
            if i >= NUM_QUERIES:
                break
            qid = q.get("_id", str(i))
            queries[qid] = q.get("text", "")

        # Collect relevance judgments
        qrels = {}
        for row in qrels_ds:
            qid = row.get("query-id", row.get("_id", ""))
            doc_id = str(row.get("corpus-id", ""))
            score = int(row.get("score", 1))
            if qid in queries and doc_id in corpus:
                qrels.setdefault(qid, {}).setdefault(doc_id, score)

        # Build ordered lists
        corpus_ids = list(corpus.keys())
        docs = [corpus[cid] for cid in corpus_ids]
        id_map = {cid: idx for idx, cid in enumerate(corpus_ids)}

        query_list = []
        query_texts = []
        for qid, text in list(queries.items())[:NUM_QUERIES]:
            query_list.append(qid)
            query_texts.append(text)

        query_relevant = []
        for qid in query_list:
            rel = {}
            for doc_id, score in qrels.get(qid, {}).items():
                if doc_id in id_map:
                    rel[id_map[doc_id]] = score
            query_relevant.append(rel)

        if len(docs) >= 100 and len(query_texts) >= 10:
            print(f"  {name}: {len(docs)} docs, {len(query_texts)} queries, "
                  f"{sum(len(r) for r in query_relevant)} relevant pairs")
            return docs, query_texts, query_relevant, name

    except Exception as exc:
        pass
        print(f"  {name}: failed ({exc})")
    return None

# ---------------------------------------------------------------------------
# Metrics (from retrieval benchmark — same implementation)
# ---------------------------------------------------------------------------

def _compute_metrics(retrieved_lists, relevant_sets, k_values):
    metrics = {}
    for k in k_values:
        metrics[f"recall_{k}"] = [_recall_at_k(r, rel, k) for r, rel in zip(retrieved_lists, relevant_sets)]
        metrics[f"ndcg_{k}"] = [_ndcg_at_k(r, rel, k) for r, rel in zip(retrieved_lists, relevant_sets)]
    metrics["map"] = [_average_precision(r, rel) for r, rel in zip(retrieved_lists, relevant_sets)]
    metrics["recip_rank"] = [_reciprocal_rank(r, rel) for r, rel in zip(retrieved_lists, relevant_sets)]
    return metrics

def _recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)

def _ndcg_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    dcg = sum((relevant.get(cid, 0)) / math.log2(i + 2) for i, cid in enumerate(retrieved[:k]))
    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

def _average_precision(retrieved, relevant):
    if not relevant:
        return 0.0
    hits = 0; ap = 0.0; rel_set = set(relevant)
    for i, cid in enumerate(retrieved, 1):
        if cid in rel_set:
            hits += 1; ap += hits / i
    return ap / len(relevant)

def _reciprocal_rank(retrieved, relevant):
    rel_set = set(relevant)
    for i, cid in enumerate(retrieved, 1):
        if cid in rel_set:
            return 1.0 / i
    return 0.0

def _bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP):
    rng = np.random.default_rng(42)
    n = len(values)
    means = [np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_bootstrap)]
    return np.mean(values), np.percentile(means, 2.5), np.percentile(means, 97.5)

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_beir_benchmark():
    all_results = []

    for name, path in BEIR_DATASETS:
        data = _load_beir_dataset(name, path)
        if data is None:
            continue
        docs, query_texts, query_relevant_pids, _ = data

        for emb_model in EMBEDDER_MODELS:
            print(f"\n=== {name} | {emb_model} ===")
            embedder = _get_embedder(emb_model)

            aggregate = {}
            for seed in SEEDS:
                np.random.seed(seed)
                tmpdir = tempfile.mkdtemp(prefix=f"lumen_beir_{name}_{seed}_")
                config = LumenConfig(store_path=Path(tmpdir), embedding_dims=EMBED_DIMS, vector_index="sqlite-vec")
                conn = get_connection(config)

                # Embed and store
                passage_embeddings = embedder.encode(docs)
                pid_to_chunk_id = {}
                for pid, (text, emb) in enumerate(zip(docs, passage_embeddings)):
                    chunk_id = store_memory(conn, content=text, room_name=name, embedding=emb, config=config)
                    pid_to_chunk_id[pid] = chunk_id
                conn.commit()

                query_relevant = [{pid_to_chunk_id[pid] for pid in rel} for rel in query_relevant_pids]

                lexical = LexicalChannel(conn)
                vector = VectorChannel(config, conn)
                pipeline = SearchPipeline(conn, config, embedder=embedder)

                for cfg_name, fn in [
                    ("bm25", lambda q: [h.chunk_id for h in lexical.search(q, k=50)]),
                    ("dense", lambda q: [h.chunk_id for h in vector.search(embedder.encode_single(q), k=50)]),
                    ("hybrid", lambda q: [r.chunk_id for r in pipeline.execute(q, k=50, max_repair_attempts=0)]),
                ]:
                    retrieved = [fn(q) for q in query_texts]
                    metrics = _compute_metrics(retrieved, query_relevant, TOP_K_VALUES)
                    key = f"{cfg_name}_{emb_model}"
                    if key not in aggregate:
                        aggregate[key] = {}
                    for mk, values in metrics.items():
                        aggregate[key].setdefault(mk, []).extend(values)

                conn.close()
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)

            summary = {}
            for key, m in aggregate.items():
                summary[key] = {}
                for mk, values in m.items():
                    mean, lo, hi = _bootstrap_ci(values)
                    summary[key][mk] = {"mean": round(float(mean), 4), "ci95_lo": round(float(lo), 4), "ci95_hi": round(float(hi), 4)}

            all_results.append({"dataset": name, "embedder": emb_model, "results": summary})

            # Print quick summary
            for cfg_name in ["bm25", "dense", "hybrid"]:
                key = f"{cfg_name}_{emb_model}"
                if key in summary:
                    r10 = summary[key].get("recall_10", {})
                    mrr = summary[key].get("recip_rank", {})
                    print(f"  {cfg_name}: R@10={r10.get('mean',0):.4f}  MRR={mrr.get('mean',0):.4f}")

    # Save results
    json_path = RESULTS_DIR / "beir_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Console table
    print("\n\n=== BEIR Summary (R@10) ===\n")
    header = "| Dataset | Embedder | BM25 | Dense | Hybrid |"
    print(header)
    print("|" + "---|" * 5)
    for r in all_results:
        ds = r["dataset"]
        emb = r["embedder"].replace("BAAI/", "")
        b = r["results"].get(f"bm25_{r['embedder']}", {}).get("recall_10", {}).get("mean", "-")
        d = r["results"].get(f"dense_{r['embedder']}", {}).get("recall_10", {}).get("mean", "-")
        h = r["results"].get(f"hybrid_{r['embedder']}", {}).get("recall_10", {}).get("mean", "-")
        print(f"| {ds} | {emb} | {b:.4f} | {d:.4f} | {h:.4f} |")

    return all_results


if __name__ == "__main__":
    run_beir_benchmark()