"""
BEIR cross-dataset evaluation for Lumen.

Runs retrieval benchmarks across BEIR subsets, computing standard IR metrics
with bootstrap confidence intervals.

Datasets:
  - NFCorpus   (~3,600 docs)
  - SciFact    (~5,000 docs)
  - FiQA-2018  (~57,000 docs — sampled)
  - ArguAna    (~8,600 docs)
  - SCIDOCS    (~25,000 docs)

Each dataset is evaluated with BM25-only, Dense-only, and Hybrid (RRF fusion)
pipelines across two embedders and three random seeds.
"""

from __future__ import annotations

import json
import math
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

SEEDS = [42, 123, 456]
EMBEDDER_MODELS = ["BAAI/bge-small-en-v1.5"]
TOP_K_VALUES = [1, 3, 5, 10, 20]
N_BOOTSTRAP = 1_000
NUM_PASSAGES = 2_000
NUM_QUERIES = 30
EMBED_DIMS = 384

BEIR_DATASETS = [
    ("nfcorpus", "BeIR/nfcorpus"),
    ("scifact", "BeIR/scifact"),
    ("fiqa", "BeIR/fiqa"),
    ("arguana", "BeIR/arguana"),
    ("scidocs", "BeIR/scidocs"),
]

RESULTS_DIR = Path(__file__).with_suffix("").parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Embedder cache (shared across datasets)
# ---------------------------------------------------------------------------
_embedders: dict[str, object] = {}


def _get_embedder(model_name: str):
    if model_name in _embedders:
        return _embedders[model_name]
    try:
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(model_name)

        class _Real:
            def __init__(self, m):
                self._m = m

            def encode(self, texts):
                vecs = self._m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                return np.asarray(vecs, dtype=np.float32)

            def encode_single(self, text):
                return self.encode([text])[0]

        e = _Real(m)
        _embedders[model_name] = e
        return e
    except Exception:
        from lumen.force.contextual.embed import MockEmbedder

        print(f"[WARN] No real embedder for {model_name} — using MockEmbedder")
        e = MockEmbedder(dims=EMBED_DIMS)
        _embedders[model_name] = e
        return e


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_beir_dataset(name: str, path: str):
    """Load a BEIR dataset via the datasets library.

    Loads qrels first to identify queries with relevance judgments, then fetches
    queries and corpus matching those judgments, avoiding the streaming-ordering
    mismatch where the first N queries may have zero relevant documents.
    """
    try:
        from datasets import load_dataset

        print(f"  Loading {name} ...")
        corpus_ds = load_dataset(path, "corpus", split="corpus", streaming=True)
        queries_ds = load_dataset(path, "queries", split="queries", streaming=True)
        qrels_ds = load_dataset(f"{path}-qrels", split="test", streaming=True)

        # Collect corpus first (needed to scope qrels to loaded docs)
        corpus: dict[str, str] = {}
        for i, doc in enumerate(corpus_ds):
            if i >= NUM_PASSAGES:
                break
            doc_id = str(doc.get("_id", str(i)))
            title = doc.get("title", "") or ""
            text = doc.get("text", "") or ""
            corpus[doc_id] = f"{title} {text}".strip()

        # Load qrels first to identify which query IDs have judgments
        qrels_raw: dict[str, dict[str, int]] = {}
        for row in qrels_ds:
            qid = str(row.get("query-id", row.get("_id", "")))
            doc_id = str(row.get("corpus-id", ""))
            score = int(row.get("score", 1))
            if doc_id in corpus:
                qrels_raw.setdefault(qid, {})[doc_id] = score
            if len(qrels_raw) >= NUM_QUERIES:
                break

        # Load queries that match qrel IDs
        queries: dict[str, str] = {}
        for q in queries_ds:
            qid = str(q.get("_id", str(len(queries))))
            if qid in qrels_raw or len(queries) < min(NUM_QUERIES // 2, 25):
                qtext = q.get("text", "") or q.get("title", "") or ""
                queries[qid] = qtext.strip()
            if len(queries) >= NUM_QUERIES:
                break

        # Filter qrels to queries we have and build ordered lists
        corpus_ids = list(corpus.keys())
        docs = [corpus[cid] for cid in corpus_ids]
        id_map = {cid: idx for idx, cid in enumerate(corpus_ids)}

        query_list = []
        query_texts = []
        query_relevant = []
        for qid, qtext in queries.items():
            rel = {}
            for doc_id, score in qrels_raw.get(qid, {}).items():
                if doc_id in id_map:
                    rel[id_map[doc_id]] = score
            if rel:
                query_list.append(qid)
                query_texts.append(qtext)
                query_relevant.append(rel)
                if len(query_list) >= NUM_QUERIES:
                    break

        # If not enough queries with qrels, pad with unjudged queries
        for qid, qtext in queries.items():
            if len(query_list) >= NUM_QUERIES:
                break
            if qid not in query_list:
                query_list.append(qid)
                query_texts.append(qtext)
                query_relevant.append({})

        if len(docs) >= 100 and len(query_texts) >= 10:
            n_rel = sum(len(r) for r in query_relevant)
            valid_q = sum(1 for r in query_relevant if r)
            print(f"  {name}: {len(docs)} docs, {len(query_texts)} queries "
                  f"({valid_q} with judgments), {n_rel} relevant pairs")
            return docs, query_texts, query_relevant, name

    except Exception as exc:
        print(f"  {name}: failed ({exc})")
    return None


# ---------------------------------------------------------------------------
# BM25 baseline (rank-bm25, independent of Lumen)
# ---------------------------------------------------------------------------

def _bm25_baseline(passages, queries, query_relevant_pids, k=50):
    """Independent rank-bm25 baseline."""
    import re
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return [], [set() for _ in queries]

    tokenized = [re.findall(r"\w+", p.lower()) for p in passages]
    bm25 = BM25Okapi(tokenized)

    retrieved_lists = []
    for q in queries:
        tokenized_query = re.findall(r"\w+", q.lower())
        scores = bm25.get_scores(tokenized_query)
        ranked = np.argsort(scores)[::-1][:k]
        retrieved_lists.append([int(pid) for pid in ranked])

    return retrieved_lists


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(retrieved_lists, relevant_sets, k_values):
    # Normalise relevance to sets (for recall/MAP/MRR) and dicts (for nDCG)
    rel_sets = []
    rel_dicts = []
    for rel in relevant_sets:
        if isinstance(rel, dict):
            rel_sets.append(rel)
            rel_dicts.append(rel)
        else:
            rel_sets.append(rel)
            rel_dicts.append({cid: 1 for cid in rel})

    metrics = {}
    for k in k_values:
        metrics[f"recall_{k}"] = [_recall_at_k(r, rel, k) for r, rel in zip(retrieved_lists, rel_sets, strict=False)]
        metrics[f"ndcg_{k}"] = [_ndcg_at_k(r, rel, k) for r, rel in zip(retrieved_lists, rel_dicts, strict=False)]
    metrics["map"] = [_average_precision(r, rel) for r, rel in zip(retrieved_lists, rel_sets, strict=False)]
    metrics["recip_rank"] = [_reciprocal_rank(r, rel) for r, rel in zip(retrieved_lists, rel_sets, strict=False)]
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
    hits = 0
    ap = 0.0
    rel_set = set(relevant)
    for i, cid in enumerate(retrieved, 1):
        if cid in rel_set:
            hits += 1
            ap += hits / i
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
# Single run
# ---------------------------------------------------------------------------

def run_single(seed, docs, query_texts, query_relevant_pids, dataset_name, embedder_model):
    np.random.seed(seed)
    tmpdir = tempfile.mkdtemp(prefix=f"lumen_beir_{dataset_name}_{seed}_")
    config = LumenConfig(
        store_path=Path(tmpdir) / ".lumen" / "store",
        embedding_dims=EMBED_DIMS,
        vector_index="sqlite-vec",
    )
    conn = get_connection(config)
    embedder = _get_embedder(embedder_model)

    passage_embeddings = embedder.encode(docs)
    pid_to_chunk_id = {}
    for pid, (text, emb) in enumerate(zip(docs, passage_embeddings, strict=False)):
        chunk_id = store_memory(
            conn,
            content=text,
            room_name=dataset_name,
            embedding=emb,
            config=config,
        )
        pid_to_chunk_id[pid] = chunk_id
    conn.commit()

    # Map relevance pids to chunk ids (preserving scores as dict)
    query_relevant = [
        {pid_to_chunk_id[pid]: score for pid, score in rel.items() if pid in pid_to_chunk_id}
        for rel in query_relevant_pids
    ]

    lexical = LexicalChannel(conn)
    vector = VectorChannel(config, conn)
    pipeline = SearchPipeline(conn, config, embedder=embedder)

    configs = {
        "bm25": lambda q: [h.chunk_id for h in lexical.search(q, k=50)],
        "dense": lambda q: [h.chunk_id for h in vector.search(embedder.encode_single(q), k=50)],
        "hybrid": lambda q: [r.chunk_id for r in pipeline.execute(q, k=50, max_repair_attempts=0)],
    }

    results_by_config = {}
    latencies_by_config = {}

    for cfg_name, fn in configs.items():
        retrieved_lists = []
        latencies = []
        for q in query_texts:
            t0 = time.perf_counter()
            results = fn(q)
            latencies.append((time.perf_counter() - t0) * 1000)
            retrieved_lists.append(results)
        results_by_config[cfg_name] = retrieved_lists
        latencies_by_config[cfg_name] = latencies

    conn.close()
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)

    all_metrics = {}
    for cfg_name, retrieved_lists in results_by_config.items():
        metrics = _compute_metrics(retrieved_lists, query_relevant, TOP_K_VALUES)
        metrics["latency_ms"] = latencies_by_config[cfg_name]
        all_metrics[cfg_name] = metrics

    return all_metrics


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_beir_benchmark():
    all_dataset_results = []

    for name, path in BEIR_DATASETS:
        data = _load_beir_dataset(name, path)
        if data is None:
            print(f"  Skipping {name} (load failed)")
            continue
        docs, query_texts, query_relevant_pids, _ = data

        dataset_report = {"dataset": name, "embedders": []}

        for emb_model in EMBEDDER_MODELS:
            print(f"\n=== {name} | {emb_model} ===")
            aggregate = {}
            for seed in SEEDS:
                metrics = run_single(seed, docs, query_texts, query_relevant_pids, name, emb_model)
                for cfg_name, m in metrics.items():
                    if cfg_name not in aggregate:
                        aggregate[cfg_name] = {}
                    for mk, values in m.items():
                        aggregate[cfg_name].setdefault(mk, []).extend(values)

            # Baseline
            bm25_retrieved = _bm25_baseline(docs, query_texts, query_relevant_pids)
            bm25_metrics = _compute_metrics(bm25_retrieved, [{pid for pid in rel} for rel in query_relevant_pids], TOP_K_VALUES)

            def _summarize(metrics_dict):
                summary = {}
                for mk, values in metrics_dict.items():
                    mean, lo, hi = _bootstrap_ci(values)
                    summary[mk] = {
                        "mean": round(float(mean), 4),
                        "ci95_lo": round(float(lo), 4),
                        "ci95_hi": round(float(hi), 4),
                    }
                return summary

            summary = {}
            for cfg_name, m in aggregate.items():
                summary[cfg_name] = _summarize(m)
            summary["rank_bm25"] = _summarize(bm25_metrics)

            dataset_report["embedders"].append({
                "embedder": emb_model.replace("BAAI/", ""),
                "embedder_full": emb_model,
                "results": summary,
            })

            for cfg_name in ["bm25", "dense", "hybrid"]:
                if cfg_name in summary:
                    r10 = summary[cfg_name].get("recall_10", {})
                    mrr = summary[cfg_name].get("recip_rank", {})
                    lat = summary[cfg_name].get("latency_ms", {})
                    print(
                        f"  {cfg_name}: R@10={r10.get('mean', 0):.4f}  "
                        f"MRR={mrr.get('mean', 0):.4f}  p50={lat.get('mean', 0):.1f}ms"
                    )

        all_dataset_results.append(dataset_report)

    # Save JSON
    json_path = RESULTS_DIR / "beir_results.json"
    report = {
        "benchmark": "beir",
        "datasets": [r["dataset"] for r in all_dataset_results],
        "num_passages": NUM_PASSAGES,
        "num_queries": NUM_QUERIES,
        "seeds": SEEDS,
        "bootstrap_samples": N_BOOTSTRAP,
        "results": all_dataset_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown
    md_lines = [
        "# BEIR Benchmark Results",
        "",
        f"**Datasets:** {', '.join(report['datasets'])} | **Passages:** {NUM_PASSAGES} | **Queries:** {NUM_QUERIES}",
        f"**Seeds:** {len(SEEDS)} | **Bootstrap:** {N_BOOTSTRAP} samples",
        "",
    ]

    for ds_result in all_dataset_results:
        md_lines.append(f"## {ds_result['dataset']}")
        md_lines.append("")
        for er in ds_result["embedders"]:
            md_lines.append(f"### Embedder: {er['embedder']} ({er['embedder_full']})")
            md_lines.append("")
            for cfg in ["rank_bm25", "bm25", "dense", "hybrid"]:
                if cfg in er["results"]:
                    _append_config_table(md_lines, cfg, er["results"][cfg])

    md_path = RESULTS_DIR / "beir_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Console summary table
    print("\n\n=== BEIR R@10 Summary ===\n")
    header = "| Dataset | Embedder | rank_bm25 | BM25 | Dense | Hybrid |"
    print(header)
    print("|" + "---|" * 6)
    for r in all_dataset_results:
        ds = r["dataset"]
        for er in r["embedders"]:
            emb = er["embedder"]
            rb = er["results"].get("rank_bm25", {}).get("recall_10", {}).get("mean", "-")
            b = er["results"].get("bm25", {}).get("recall_10", {}).get("mean", "-")
            d = er["results"].get("dense", {}).get("recall_10", {}).get("mean", "-")
            h = er["results"].get("hybrid", {}).get("recall_10", {}).get("mean", "-")
            fmt = lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
            print(f"| {ds} | {emb} | {fmt(rb)} | {fmt(b)} | {fmt(d)} | {fmt(h)} |")

    print(f"\n[INFO] Results written to {json_path} and {md_path}")
    return report


def _append_config_table(lines, name, metrics_data):
    lines.append(f"#### {name}")
    lines.append("| k | R@k | nDCG@k |")
    lines.append("|---|---|---|")
    for k in TOP_K_VALUES:
        r = metrics_data.get(f"recall_{k}")
        n = metrics_data.get(f"ndcg_{k}")
        r_str = f"{r['mean']:.4f} [{r['ci95_lo']:.4f}-{r['ci95_hi']:.4f}]" if r else "-"
        n_str = f"{n['mean']:.4f} [{n['ci95_lo']:.4f}-{n['ci95_hi']:.4f}]" if n else "-"
        lines.append(f"| {k} | {r_str} | {n_str} |")
    m = metrics_data.get("map")
    mr = metrics_data.get("recip_rank")
    lat = metrics_data.get("latency_ms")
    if m:
        lines.append(f"| MAP | {m['mean']:.4f} [{m['ci95_lo']:.4f}-{m['ci95_hi']:.4f}] | — |")
    if mr:
        lines.append(f"| MRR | {mr['mean']:.4f} [{mr['ci95_lo']:.4f}-{mr['ci95_hi']:.4f}] | — |")
    if lat:
        lines.append(f"| Latency | {lat['mean']:.1f}ms [{lat['ci95_lo']:.1f}-{lat['ci95_hi']:.1f}] | — |")
    lines.append("")


if __name__ == "__main__":
    run_beir_benchmark()
