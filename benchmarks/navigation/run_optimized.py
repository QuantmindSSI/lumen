"""
Palace Navigation Efficiency (PNE) Benchmark — Optimized Edition.

Identical methodology to the standard PNE benchmark but: (1) always uses
a real sentence-transformers embedder, (2) pre-caches all chunk embeddings
in memory to enable perfectly fair dense-search comparison (same brute-force
algorithm for both global and room-constrained, differing only in search space).

Run with:
    python -m benchmarks.navigation.run_optimized [--embedder all-MiniLM-L6-v2 | BAAI/bge-small-en-v1.5]

Outputs: benchmarks/navigation/results/navigation_results_real.json + .md
"""

from __future__ import annotations

import json
import math
import re
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
from lumen.force.mnemonic.retrieval_dense import DenseHit
from lumen.force.mnemonic.retrieval_lexical import LexicalChannel, LexicalHit
from lumen.force.mnemonic.store import store_memory
from lumen.fusion import fuse_and_rerank
from lumen.intent import IntentRouter

# ---------------------------------------------------------------------------
# Configuration — same as standard PNE but with real embedder enforced
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 456]
TOP_K_VALUES = [1, 3, 5, 10, 20]
N_BOOTSTRAP = 1_000
EMBED_DIMS = 384
RESULTS_DIR = Path(__file__).with_suffix("").parent / "results"

ROOM_CONFIG = [
    {"name": "machine_learning", "topics": ["machine learning", "deep learning", "neural network", "training pipelines", "gradient descent"], "loci": ["recent", "fundamentals", "applications", "challenges"]},
    {"name": "nlp", "topics": ["natural language processing", "transformer architecture", "tokenization methods", "bert model", "large language models"], "loci": ["recent", "fundamentals", "applications", "challenges"]},
    {"name": "computer_vision", "topics": ["computer vision", "convolutional neural network", "image recognition", "object detection", "semantic segmentation"], "loci": ["recent", "fundamentals", "applications", "challenges"]},
    {"name": "robotics", "topics": ["robotics", "automation systems", "kinematics", "manipulator design", "sensor fusion"], "loci": ["recent", "fundamentals", "applications", "challenges"]},
    {"name": "cybersecurity", "topics": ["cybersecurity", "threat detection", "encryption standards", "zero trust architecture", "penetration testing"], "loci": ["recent", "fundamentals", "applications", "challenges"]},
    {"name": "cloud", "topics": ["cloud computing", "serverless architecture", "kubernetes orchestration", "devops practices", "scalability engineering"], "loci": ["recent", "fundamentals", "applications", "challenges"]},
    {"name": "blockchain", "topics": ["blockchain technology", "consensus mechanisms", "smart contracts", "decentralization", "cryptocurrency markets"], "loci": ["recent", "fundamentals", "applications", "challenges"]},
    {"name": "healthcare", "topics": ["healthcare innovation", "clinical trials", "diagnosis assistance", "medical imaging", "patient data privacy"], "loci": ["recent", "fundamentals", "applications", "challenges"]},
]
ADJACENCY = {
    "machine_learning": ["nlp", "computer_vision", "healthcare"],
    "nlp": ["machine_learning", "healthcare"],
    "computer_vision": ["machine_learning", "robotics", "healthcare"],
    "robotics": ["computer_vision", "cybersecurity"],
    "cybersecurity": ["robotics", "cloud", "blockchain"],
    "cloud": ["cybersecurity", "blockchain", "healthcare"],
    "blockchain": ["cybersecurity", "cloud"],
    "healthcare": ["machine_learning", "nlp", "computer_vision", "cloud"],
}

PASSAGES_PER_LOCUS = 40  # Reduced from 75 for speed (1280 total)
CROSS_TOPIC_RATE = 0.15
PURE_QUERIES_PER_ROOM = 8
CROSS_QUERIES_PER_EDGE = 2

PURE_TEMPLATES = [
    "Recent advances in {topic} have shown promising results in {aspect}.",
    "Experts believe {topic} will transform the {aspect} landscape within five years.",
    "A comprehensive review of {topic} highlights both benefits and risks for {aspect}.",
    "Startups focusing on {topic} have attracted significant venture capital for {aspect}.",
    "Governments worldwide are drafting regulations around {topic} in {aspect}.",
    "The intersection of {topic} and ethics remains a hotly debated issue in {aspect}.",
    "Educational institutions are updating curricula to include {topic} for {aspect}.",
    "Public awareness of {topic} has grown exponentially since 2020 in {aspect}.",
    "Open-source tools have democratised access to {topic} for {aspect} practitioners.",
    "Industry leaders predict {topic} will dominate {aspect} by 2030.",
]
CROSS_TEMPLATES = [
    "The intersection of {topic} and {alt_topic} is generating significant research interest.",
    "Startups combining {topic} with {alt_topic} raised record funding last quarter.",
    "A novel framework merges {topic} and {alt_topic} to solve hard problems.",
    "Conferences now feature dedicated tracks on {topic} applied to {alt_topic}.",
    "Regulators are drafting guidelines for {topic} systems used in {alt_topic}.",
]


# ---------------------------------------------------------------------------
# Embedder — always real
# ---------------------------------------------------------------------------
_embedder = None
_embedder_name = ""


def _get_embedder(model_request: str = "all-MiniLM-L6-v2"):
    global _embedder, _embedder_name
    if _embedder is not None:
        return _embedder, _embedder_name
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_request)

    class _Real:
        def __init__(self, m):
            self._m = m
        def encode(self, texts):
            return np.asarray(self._m.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
        def encode_single(self, text):
            return self.encode([text])[0]

    _embedder = _Real(model)
    _embedder_name = model_request
    print(f"[INFO] Embedder: {_embedder_name} ({model.get_sentence_embedding_dimension()} dims)")
    return _embedder, _embedder_name


# ---------------------------------------------------------------------------
# Corpus + queries (same as standard)
# ---------------------------------------------------------------------------
def _generate_corpus(seed):
    rng = np.random.default_rng(seed)
    passages = []
    for room in ROOM_CONFIG:
        for locus in room["loci"]:
            for i in range(PASSAGES_PER_LOCUS):
                if rng.random() < CROSS_TOPIC_RATE:
                    alt_room_name = rng.choice(ADJACENCY[room["name"]])
                    alt_room = next(r for r in ROOM_CONFIG if r["name"] == alt_room_name)
                    topic = rng.choice(room["topics"])
                    alt_topic = rng.choice(alt_room["topics"])
                    text = rng.choice(CROSS_TEMPLATES).format(topic=topic, alt_topic=alt_topic)
                else:
                    topic = rng.choice(room["topics"])
                    text = rng.choice(PURE_TEMPLATES).format(topic=topic, aspect=locus)
                text = f"[{room['name']}/{locus}/{i}] {text}"
                passages.append({"text": text, "room": room["name"], "locus": locus})
    rng.shuffle(passages)
    return passages


def _generate_queries(passages, seed):
    rng = np.random.default_rng(seed)
    queries = []
    for room in ROOM_CONFIG:
        for _ in range(PURE_QUERIES_PER_ROOM):
            topic = rng.choice(room["topics"])
            modifier = rng.choice(["recent advances", "future trends", "best practices", "challenges"])
            queries.append({"query": f"{topic} {modifier}", "ground_truth_rooms": [room["name"]]})
    for room in ROOM_CONFIG:
        for alt in ADJACENCY[room["name"]]:
            for _ in range(CROSS_QUERIES_PER_EDGE):
                topic = rng.choice(room["topics"])
                alt_room = next(r for r in ROOM_CONFIG if r["name"] == alt)
                alt_topic = rng.choice(alt_room["topics"])
                queries.append({"query": f"{topic} {alt_topic}", "ground_truth_rooms": [room["name"], alt]})
    rng.shuffle(queries)
    return queries


def _compute_relevant_set(passages, query):
    q_words = set(re.findall(r"\w+", query.lower()))
    return {idx for idx, p in enumerate(passages) if len(set(re.findall(r"\w+", p["text"].lower())) & q_words) >= 2}


# ---------------------------------------------------------------------------
# Room Router
# ---------------------------------------------------------------------------
class RoomRouter:
    def __init__(self):
        self.room_keywords = {}
        for room in ROOM_CONFIG:
            words = set()
            for topic in room["topics"]:
                words.update(re.findall(r"\w+", topic.lower()))
            self.room_keywords[room["name"]] = words

    def predict(self, query):
        q_words = set(re.findall(r"\w+", query.lower()))
        scores = {room: len(q_words & kw) for room, kw in self.room_keywords.items()}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [r for r, s in ranked if s > 0][:2]


# ---------------------------------------------------------------------------
# Embedding cache + fair dense search (same algorithm for both scopes)
# ---------------------------------------------------------------------------
class EmbeddingCache:
    """Pre-loaded cache of all chunk embeddings for fair dense comparison."""
    def __init__(self):
        self._data: dict[int, tuple[np.ndarray, int]] = {}  # chunk_id -> (embedding, room_id)

    def load(self, conn, dims):
        rows = conn.execute(
            "SELECT vf.chunk_id, vf.embedding, ch.room_id FROM vec_fallback vf "
            "JOIN chunk ch ON ch.chunk_id = vf.chunk_id WHERE ch.valid_to IS NULL"
        ).fetchall()
        for cid, emb_blob, rid in rows:
            self._data[cid] = (np.frombuffer(emb_blob, dtype=np.float32), rid)

    def search(self, query_vec, k, room_ids=None):
        hits = []
        q = query_vec.astype(np.float32)
        qn = np.linalg.norm(q)
        for cid, (vec, rid) in self._data.items():
            if room_ids is not None and rid not in room_ids:
                continue
            vn = np.linalg.norm(vec)
            sim = 0.0 if vn == 0 or qn == 0 else float(np.dot(q, vec) / (qn * vn))
            hits.append(DenseHit(cid, sim, vec))
        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:k]

    def count_in_rooms(self, room_ids):
        return sum(1 for _, rid in self._data.values() if rid in room_ids)

    def total(self):
        return len(self._data)


# ---------------------------------------------------------------------------
# Room-constrained lexical (same as before)
# ---------------------------------------------------------------------------
class RoomConstrainedLexicalChannel:
    def __init__(self, conn, room_ids):
        self.conn = conn
        self.room_ids = list(room_ids)

    def search(self, query, k=20):
        safe_query = LexicalChannel._sanitize(query)
        if not safe_query or not self.room_ids:
            return []
        placeholders = ",".join("?" for _ in self.room_ids)
        rows = self.conn.execute(
            f"SELECT c.rowid, c.rank FROM chunk_fts c JOIN chunk ch ON ch.chunk_id = c.rowid "
            f"WHERE ch.room_id IN ({placeholders}) AND c.chunk_fts MATCH ? ORDER BY c.rank LIMIT ?",
            (*self.room_ids, safe_query, k),
        ).fetchall()
        return [LexicalHit(cid, rank, b"") for cid, rank in rows]


# ---------------------------------------------------------------------------
# Unified search runner
# ---------------------------------------------------------------------------
def _run_strategy(conn, config, embedder, embedding_cache, query, room_ids, k):
    t0 = time.perf_counter()

    if room_ids is None:
        lexical = LexicalChannel(conn)
    else:
        lexical = RoomConstrainedLexicalChannel(conn, room_ids)

    qvec = embedder.encode_single(query)
    lexical_hits = lexical.search(query, k=k)

    # Fair dense search: same brute-force algorithm, different scope
    room_ids_for_dense = room_ids if room_ids else None
    dense_hits = embedding_cache.search(qvec, k=k, room_ids=room_ids_for_dense)

    results = fuse_and_rerank(
        lexical_hits, dense_hits, goal_tree_keywords=[], conn=conn,
        budget_candidates=200, query_embedding=qvec, graph_hits=None,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return results[:k], elapsed_ms


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def _bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP):
    rng = np.random.default_rng(42)
    n = len(values)
    means = [np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_bootstrap)]
    return np.mean(values), np.percentile(means, 2.5), np.percentile(means, 97.5)


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------
def _run_single_seed(seed, embedder_model):
    passages = _generate_corpus(seed)
    queries = _generate_queries(passages, seed)
    for q in queries:
        q["relevant"] = _compute_relevant_set(passages, q["query"])

    embedder, _ = _get_embedder(embedder_model)
    texts = [p["text"] for p in passages]
    print(f"[INFO] Embedding {len(passages)} passages with {embedder_model} ...")
    embeddings = embedder.encode(texts)

    tmpdir = tempfile.mkdtemp(prefix=f"lumen_pne_real_{seed}_")
    config = LumenConfig(store_path=Path(tmpdir), embedding_dims=EMBED_DIMS, vector_index="sqlite-vec")
    conn = get_connection(config)

    room_name_to_id = {}
    for idx, (p, emb) in enumerate(zip(passages, embeddings, strict=False)):
        chunk_id = store_memory(conn, content=p["text"], room_name=p["room"], locus_name=p["locus"], embedding=emb, config=config)
        passages[idx]["chunk_id"] = chunk_id
        if p["room"] not in room_name_to_id:
            row = conn.execute("SELECT room_id FROM room WHERE name = ?", (p["room"],)).fetchone()
            room_name_to_id[p["room"]] = row[0]
        if idx % 500 == 0:
            conn.commit()
    conn.commit()

    for q in queries:
        q["relevant_chunk_ids"] = {passages[idx]["chunk_id"] for idx in q["relevant"] if idx < len(passages) and "chunk_id" in passages[idx]}

    # Pre-load embedding cache for fair dense comparison
    embedding_cache = EmbeddingCache()
    embedding_cache.load(conn, config.embedding_dims)

    total_chunks = embedding_cache.total()
    router = RoomRouter()
    intent_router = IntentRouter()

    per_query = []
    for q in queries:
        gt_room_ids = [room_name_to_id[r] for r in q["ground_truth_rooms"] if r in room_name_to_id]
        predicted_rooms = router.predict(q["query"])
        predicted_room_ids = [room_name_to_id[r] for r in predicted_rooms if r in room_name_to_id]
        routing_correct = len(set(predicted_rooms) & set(q["ground_truth_rooms"])) > 0

        if predicted_room_ids:
            pruned_count = embedding_cache.count_in_rooms(predicted_room_ids)
            pruning_ratio = 1.0 - (pruned_count / max(1, total_chunks))
        else:
            pruning_ratio = 0.0

        intent = intent_router.classify(q["query"])

        strategies = {}
        for name, rids in [
            ("global", None),
            ("oracle", gt_room_ids),
            ("routed", predicted_room_ids if predicted_room_ids else gt_room_ids),
        ]:
            results, latency_ms = _run_strategy(conn, config, embedder, embedding_cache, q["query"], rids, k=max(TOP_K_VALUES))
            retrieved_ids = [r.chunk_id for r in results]
            strategies[name] = {
                "latency_ms": latency_ms,
                **{f"recall_{k}": _recall_at_k(retrieved_ids, q["relevant_chunk_ids"], k) for k in TOP_K_VALUES},
            }

        per_query.append({
            "query": q["query"],
            "intent": intent,
            "ground_truth_rooms": q["ground_truth_rooms"],
            "predicted_rooms": predicted_rooms,
            "routing_correct": routing_correct,
            "pruning_ratio": pruning_ratio,
            "num_relevant": len(q["relevant_chunk_ids"]),
            "strategies": strategies,
        })

    conn.close()
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    return per_query, total_chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(embedder_model="all-MiniLM-L6-v2", short_label="MiniLM"):
    all_per_query = []
    total_chunks = 0

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"PNE Benchmark — seed {seed} — {embedder_model}")
        print(f"{'='*60}")
        per_query, tc = _run_single_seed(seed, embedder_model)
        all_per_query.extend(per_query)
        total_chunks = tc

    # Aggregates
    def _agg(vals):
        return _bootstrap_ci(vals)

    summary = {}
    for strategy in ["global", "oracle", "routed"]:
        summary[strategy] = {}
        latencies = [q["strategies"][strategy]["latency_ms"] for q in all_per_query]
        summary[strategy]["latency_ms"] = {
            "mean": round(float(np.mean(latencies)), 2),
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
        }
        for k in TOP_K_VALUES:
            vals = [q["strategies"][strategy][f"recall_{k}"] for q in all_per_query]
            mean, lo, hi = _agg(vals)
            summary[strategy][f"recall_{k}"] = {"mean": round(float(mean), 4), "ci95_lo": round(float(lo), 4), "ci95_hi": round(float(hi), 4)}

    for baseline, constrained in [("global", "oracle"), ("global", "routed")]:
        key = f"{constrained}_vs_{baseline}"
        summary[key] = {}
        for k in TOP_K_VALUES:
            ratios = []
            for q in all_per_query:
                base_r = q["strategies"][baseline][f"recall_{k}"]
                cons_r = q["strategies"][constrained][f"recall_{k}"]
                ratios.append(cons_r / base_r if base_r > 0 else (1.0 if cons_r == 0 else 0.0))
            mean, lo, hi = _agg(ratios)
            summary[key][f"recall_retention_{k}"] = {"mean": round(float(mean), 4), "ci95_lo": round(float(lo), 4), "ci95_hi": round(float(hi), 4)}
        speedups = [q["strategies"][baseline]["latency_ms"] / max(q["strategies"][constrained]["latency_ms"], 0.001) for q in all_per_query]
        mean, lo, hi = _agg(speedups)
        summary[key]["latency_speedup"] = {"mean": round(float(mean), 2), "ci95_lo": round(float(lo), 2), "ci95_hi": round(float(hi), 2)}

    routing_accuracy = np.mean([q["routing_correct"] for q in all_per_query])
    pruning_ratios = [q["pruning_ratio"] for q in all_per_query]
    summary["routing_and_pruning"] = {
        "routing_accuracy": round(float(routing_accuracy), 4),
        "mean_pruning_ratio": round(float(np.mean(pruning_ratios)), 4),
        "median_pruning_ratio": round(float(np.median(pruning_ratios)), 4),
    }
    summary["intent_distribution"] = {}
    for q in all_per_query:
        summary["intent_distribution"][q["intent"]] = summary["intent_distribution"].get(q["intent"], 0) + 1

    label = short_label.replace("/", "_")
    json_path = RESULTS_DIR / f"navigation_results_{label}.json"
    md_path = RESULTS_DIR / f"navigation_results_{label}.md"
    _write_output(summary, all_per_query, embedder_model, total_chunks, json_path, md_path)

    # Console
    print(f"\n=== PNE Summary [{embedder_model}] ===")
    print(f"Routing Accuracy: {routing_accuracy:.1%}  Pruning: {summary['routing_and_pruning']['mean_pruning_ratio']:.1%}")
    print(f"Global  p50={summary['global']['latency_ms']['p50']:.1f}ms  R@10={summary['global']['recall_10']['mean']:.4f}")
    print(f"Oracle  p50={summary['oracle']['latency_ms']['p50']:.1f}ms  R@10={summary['oracle']['recall_10']['mean']:.4f}  speedup={summary['oracle_vs_global']['latency_speedup']['mean']:.2f}x  retention={summary['oracle_vs_global']['recall_retention_10']['mean']:.4f}")
    print(f"Routed  p50={summary['routed']['latency_ms']['p50']:.1f}ms  R@10={summary['routed']['recall_10']['mean']:.4f}  speedup={summary['routed_vs_global']['latency_speedup']['mean']:.2f}x  retention={summary['routed_vs_global']['recall_retention_10']['mean']:.4f}")

    return summary


def _write_output(summary, all_per_query, embedder_name, total_chunks, json_path, md_path):
    report = {
        "benchmark": "navigation",
        "embedder": embedder_name,
        "num_rooms": len(ROOM_CONFIG),
        "total_chunks": total_chunks,
        "num_queries": len(all_per_query),
        "num_seeds": len(SEEDS),
        "summary": summary,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md = [
        f"# PNE Benchmark [{embedder_name}]",
        f"**Chunks:** {total_chunks:,} | **Queries:** {len(all_per_query)} | **Seeds:** {len(SEEDS)}",
        f"**Routing:** {summary['routing_and_pruning']['routing_accuracy']:.1%} | **Pruning:** {summary['routing_and_pruning']['mean_pruning_ratio']:.1%}",
        "",
        "## Latency (ms)",
        "| Strategy | Mean | p50 | p95 | p99 |",
        "|---|---|---|---|---|",
    ]
    for s in ["global", "oracle", "routed"]:
        lat = summary[s]["latency_ms"]
        md.append(f"| {s} | {lat['mean']} | {lat['p50']} | {lat['p95']} | {lat['p99']} |")
    md += ["", "## Recall@k", "| k | Global | Oracle | Routed |", "|---|---|---|---|"]
    for k in TOP_K_VALUES:
        md.append(f"| {k} | {summary['global'][f'recall_{k}']['mean']:.4f} | {summary['oracle'][f'recall_{k}']['mean']:.4f} | {summary['routed'][f'recall_{k}']['mean']:.4f} |")
    md += ["", "## Oracle vs Global", "| Metric | Mean | 95% CI |", "|---|---|---|"]
    for k in TOP_K_VALUES:
        rr = summary["oracle_vs_global"][f"recall_retention_{k}"]
        md.append(f"| R@{k} Retention | {rr['mean']:.4f} | [{rr['ci95_lo']:.4f}, {rr['ci95_hi']:.4f}] |")
    ls = summary["oracle_vs_global"]["latency_speedup"]
    md.append(f"| Latency Speedup | {ls['mean']:.2f}x | [{ls['ci95_lo']:.2f}, {ls['ci95_hi']:.2f}] |")
    md += ["", "## Routed vs Global", "| Metric | Mean | 95% CI |", "|---|---|---|"]
    for k in TOP_K_VALUES:
        rr = summary["routed_vs_global"][f"recall_retention_{k}"]
        md.append(f"| R@{k} Retention | {rr['mean']:.4f} | [{rr['ci95_lo']:.4f}, {rr['ci95_hi']:.4f}] |")
    ls = summary["routed_vs_global"]["latency_speedup"]
    md.append(f"| Latency Speedup | {ls['mean']:.2f}x | [{ls['ci95_lo']:.2f}, {ls['ci95_hi']:.2f}] |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    args = ap.parse_args()
    label = args.embedder.replace("BAAI/", "").replace("all-", "")
    run(args.embedder, label)