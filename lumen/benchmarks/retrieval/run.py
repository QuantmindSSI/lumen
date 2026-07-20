"""
Retrieval Benchmark for Lumen.

Evaluates Hybrid (SearchPipeline), BM25-only, and Dense-only retrieval
using either a MS MARCO slice or a synthetic corpus.
Outputs JSON + Markdown table to benchmarks/retrieval/results/.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

# Resolve project root (two levels up from benchmarks/<suite>/run.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------
try:
    import lumen
except ImportError as exc:
    print(f"[ERROR] Cannot import lumen: {exc}")
    print("Ensure you run this script from the project root or install the package.")
    sys.exit(1)

from lumen.config import LumenConfig
from lumen.data.schema import get_connection, init_db
from lumen.force.contextual.embed import FallbackEmbedder
from lumen.force.mnemonic.retrieval_dense import VectorChannel
from lumen.force.mnemonic.retrieval_lexical import LexicalChannel
from lumen.force.mnemonic.store import store_memory
from lumen.lumen.search import SearchPipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
NUM_PASSAGES = 1000
NUM_QUERIES = 50
TOP_K = 10
EMBED_DIMS = 384

np.random.seed(SEED)

RESULTS_DIR = Path(__file__).with_suffix("").parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------
def _generate_synthetic_corpus(n_passages: int = NUM_PASSAGES, n_queries: int = NUM_QUERIES):
    """Generate a deterministic synthetic corpus with labeled relevance."""
    topics = [
        "machine learning", "deep learning", "neural networks", "natural language processing",
        "computer vision", "reinforcement learning", "robotics", "quantum computing",
        "cloud infrastructure", "cybersecurity", "blockchain", "data privacy",
        "climate change", "renewable energy", "space exploration", "biotechnology",
        "mental health", "nutrition", "exercise science", "sleep hygiene",
    ]

    templates = [
        "{topic} is transforming the way we approach modern challenges.",
        "Recent advances in {topic} have shown promising results in clinical trials.",
        "Experts believe {topic} will become mainstream within the next decade.",
        "A comprehensive review of {topic} highlights both benefits and risks.",
        "Startups focusing on {topic} have attracted significant venture capital.",
        "Governments worldwide are drafting regulations around {topic}.",
        "The intersection of {topic} and ethics remains a hotly debated issue.",
        "Educational institutions are updating curricula to include {topic}.",
        "Public awareness of {topic} has grown exponentially since 2020.",
        "Open-source tools have democratised access to {topic} for hobbyists.",
        "Critics argue that {topic} is overhyped and lacks practical utility.",
        "Long-term studies on {topic} suggest cautious optimism is warranted.",
        "Industry leaders convened at a summit to discuss {topic} standards.",
        "The economic impact of {topic} is projected to reach billions annually.",
        "Researchers published a breakthrough paper on {topic} last month.",
        "Consumer adoption of {topic} varies significantly across demographics.",
        "Integrating {topic} into legacy systems poses non-trivial engineering challenges.",
        "Several Fortune 500 companies have established dedicated {topic} divisions.",
        "The history of {topic} dates back further than most people realise.",
        "Future work in {topic} aims to address scalability and fairness concerns.",
    ]

    passages: list[str] = []
    passage_topics: list[list[str]] = []
    rng = np.random.default_rng(SEED)
    for i in range(n_passages):
        topic = topics[i % len(topics)]
        alt_topic = topics[(i + 7) % len(topics)]
        tmpl = templates[i % len(templates)]
        # Mix in a second topic for ~30% of passages to create soft relevance
        if i % 3 == 0:
            text = tmpl.format(topic=topic) + f" It also relates to {alt_topic}."
            passage_topics.append([topic, alt_topic])
        else:
            text = tmpl.format(topic=topic)
            passage_topics.append([topic])
        # Ensure uniqueness to avoid dedup in store_memory
        text = f"[pid:{i}] {text}"
        passages.append(text)

    queries: list[str] = []
    query_relevant: list[set[int]] = []
    for q in range(n_queries):
        topic = topics[q % len(topics)]
        # Use concise keyword queries that BM25 can match reliably while
        # still exercising the full pipeline.
        modifiers = [
            "",
            "advances",
            "future",
            "risks",
            "importance",
            "trends",
            "applications",
            "challenges",
        ]
        mod = modifiers[q % len(modifiers)]
        query_text = f"{topic} {mod}".strip()
        queries.append(query_text)

        # Determine relevant passages by topic presence
        relevant = set()
        for pid, ptops in enumerate(passage_topics):
            if topic in passages[pid].lower():
                relevant.add(pid)
            # Soft relevance via keyword overlap for passages sharing 2+ words
            query_words = set(query_text.lower().split())
            passage_words = set(passages[pid].lower().split())
            if len(query_words & passage_words) >= 2:
                relevant.add(pid)
        query_relevant.append(relevant)

    return passages, queries, query_relevant


def load_corpus():
    """Attempt to load MS MARCO; fall back to synthetic corpus on any failure."""
    try:
        from datasets import load_dataset
        print("[INFO] Attempting to load MS MARCO passage dev subset via `datasets` ...")
        ds = load_dataset("ms_marco", "v1.1", split="validation", streaming=True)
        passages_dict: dict[str, str] = {}
        queries_dict: dict[str, str] = {}
        qrels: dict[str, set[str]] = {}

        count = 0
        for example in ds:
            qid = str(example["query_id"])
            queries_dict[qid] = example["query"]
            passages_list = example["passages"]
            for pid, is_selected in zip(passages_list["passage_id"], passages_list["is_selected"]):
                pid_str = str(pid)
                if pid_str not in passages_dict:
                    passages_dict[pid_str] = passages_list["passage_text"][
                        passages_list["passage_id"].index(pid)
                    ]
                if is_selected:
                    qrels.setdefault(qid, set()).add(pid_str)
            count += 1
            if count >= NUM_QUERIES:
                break

        # Take first NUM_PASSAGES unique passages
        passage_ids = list(passages_dict.keys())[:NUM_PASSAGES]
        passages = [passages_dict[pid] for pid in passage_ids]
        id_map = {old: new for new, old in enumerate(passage_ids)}

        queries = []
        query_relevant = []
        for qid, query_text in list(queries_dict.items())[:NUM_QUERIES]:
            queries.append(query_text)
            rel = {id_map[pid] for pid in qrels.get(qid, set()) if pid in id_map}
            query_relevant.append(rel)

        if len(passages) < 100 or len(queries) < 10:
            raise RuntimeError("MS MARCO slice too small; falling back to synthetic.")

        print(f"[INFO] Loaded MS MARCO slice: {len(passages)} passages, {len(queries)} queries.")
        return passages, queries, query_relevant, "ms_marco"
    except Exception as exc:
        print(f"[WARN] Could not load MS MARCO ({exc}). Using synthetic corpus.")
        p, q, r = _generate_synthetic_corpus()
        return p, q, r, "synthetic"


# ---------------------------------------------------------------------------
# FTS5 sanitization
# ---------------------------------------------------------------------------
import re

_FTS5_SPECIAL = re.compile(r'[^\w\s]')


def _sanitize_fts5(query: str) -> str:
    """Strip FTS5-special characters to avoid syntax errors."""
    cleaned = _FTS5_SPECIAL.sub(' ', query)
    return ' '.join(cleaned.split())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def recall_at_k(retrieved: list[int], relevant: set[int], k: int = TOP_K) -> float:
    if not relevant:
        return 0.0
    retrieved_set = set(retrieved[:k])
    return len(retrieved_set & relevant) / len(relevant)


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int = TOP_K) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for i, cid in enumerate(retrieved[:k], start=1):
        rel = 1.0 if cid in relevant else 0.0
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    # Ideal DCG
    ideal_rels = [1.0] * min(len(relevant), k)
    idcg = sum((2 ** r - 1) / math.log2(i + 1) for i, r in enumerate(ideal_rels, start=1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark() -> dict[str, Any]:
    passages, queries, query_relevant, corpus_name = load_corpus()

    # Setup temporary Lumen store
    tmpdir = tempfile.mkdtemp(prefix="lumen_benchmark_retrieval_")
    config = LumenConfig(
        store_path=Path(tmpdir),
        embedding_dims=EMBED_DIMS,
        vector_index="sqlite-vec",
    )
    conn = get_connection(config)
    embedder = FallbackEmbedder(dims=EMBED_DIMS)

    print("[INFO] Embedding and storing passages ...")
    passage_embeddings = embedder.encode(passages)
    pid_to_chunk_id: dict[int, int] = {}
    for pid, (text, emb) in enumerate(zip(passages, passage_embeddings)):
        chunk_id = store_memory(
            conn,
            content=text,
            room_name="benchmark_retrieval",
            locus_name=f"locus_{pid % 20}",
            embedding=emb,
            config=config,
        )
        pid_to_chunk_id[pid] = chunk_id
    conn.commit()

    # Convert pid-based relevance to chunk_id-based relevance for fair metrics
    query_relevant = [
        {pid_to_chunk_id[pid] for pid in rel_pids}
        for rel_pids in query_relevant
    ]

    # Build baselines
    lexical = LexicalChannel(conn)
    vector = VectorChannel(config, conn)
    pipeline = SearchPipeline(conn, config, embedder=embedder)

    def run_hybrid(query: str) -> list[int]:
        safe_query = _sanitize_fts5(query)
        results = pipeline.execute(safe_query, k=TOP_K)
        return [r.chunk_id for r in results]

    def run_bm25_only(query: str) -> list[int]:
        safe_query = _sanitize_fts5(query)
        hits = lexical.search(safe_query, k=TOP_K)
        return [h.chunk_id for h in hits]

    def run_dense_only(query: str) -> list[int]:
        qvec = embedder.encode_single(query)
        hits = vector.search(qvec, k=TOP_K)
        return [h.chunk_id for h in hits]

    configs = {
        "hybrid": run_hybrid,
        "bm25_only": run_bm25_only,
        "dense_only": run_dense_only,
    }

    results_by_config: dict[str, dict[str, list[float]]] = {
        name: {"recall@10": [], "ndcg@10": [], "latency_ms": []} for name in configs
    }

    print("[INFO] Running queries ...")
    for qid, (query, relevant) in enumerate(zip(queries, query_relevant)):
        for cfg_name, fn in configs.items():
            t0 = time.perf_counter()
            retrieved = fn(query)
            latency_ms = (time.perf_counter() - t0) * 1000
            results_by_config[cfg_name]["recall@10"].append(
                recall_at_k(retrieved, relevant, TOP_K)
            )
            results_by_config[cfg_name]["ndcg@10"].append(
                ndcg_at_k(retrieved, relevant, TOP_K)
            )
            results_by_config[cfg_name]["latency_ms"].append(latency_ms)

    # Aggregate
    summary: dict[str, Any] = {}
    for cfg_name, metrics in results_by_config.items():
        summary[cfg_name] = {
            "recall@10_mean": round(float(np.mean(metrics["recall@10"])), 4),
            "recall@10_std": round(float(np.std(metrics["recall@10"])), 4),
            "ndcg@10_mean": round(float(np.mean(metrics["ndcg@10"])), 4),
            "ndcg@10_std": round(float(np.std(metrics["ndcg@10"])), 4),
            "latency_p50_ms": round(float(np.percentile(metrics["latency_ms"], 50)), 2),
            "latency_p95_ms": round(float(np.percentile(metrics["latency_ms"], 95)), 2),
            "latency_p99_ms": round(float(np.percentile(metrics["latency_ms"], 99)), 2),
        }

    report = {
        "benchmark": "retrieval",
        "corpus": corpus_name,
        "num_passages": len(passages),
        "num_queries": len(queries),
        "top_k": TOP_K,
        "seed": SEED,
        "results": summary,
    }

    # Persist
    json_path = RESULTS_DIR / "retrieval_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = RESULTS_DIR / "retrieval_results.md"
    lines = [
        "# Retrieval Benchmark Results",
        "",
        f"**Corpus:** {report['corpus']} | **Passages:** {report['num_passages']} | **Queries:** {report['num_queries']} | **Top-K:** {report['top_k']}",
        "",
        "| Configuration | Recall@10 (mean ± std) | nDCG@10 (mean ± std) | Latency p50 (ms) | Latency p95 (ms) | Latency p99 (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for cfg_name in ["bm25_only", "dense_only", "hybrid"]:
        m = summary[cfg_name]
        lines.append(
            f"| {cfg_name} | {m['recall@10_mean']} ± {m['recall@10_std']} | "
            f"{m['ndcg@10_mean']} ± {m['ndcg@10_std']} | {m['latency_p50_ms']} | "
            f"{m['latency_p95_ms']} | {m['latency_p99_ms']} |"
        )
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Cleanup
    conn.close()
    try:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass

    print(f"[INFO] Results written to {json_path} and {md_path}")
    return report


if __name__ == "__main__":
    run_benchmark()
