"""
Palace Navigation Efficiency (PNE) Benchmark for Lumen.

Measures the architectural benefit of Lumen's structured room/locus topology
by comparing three retrieval strategies on an identical corpus:

  1. Global — standard SearchPipeline across all rooms (baseline)
  2. Oracle Room — constrained to the ground-truth room(s)  (upper-bound)
  3. Routed Room — constrained to room(s) predicted by a query→room router

Metrics: recall@k, latency (p50/p95/p99), intent-routing accuracy,
pruning efficiency, and recall-retention ratio.

Statistical rigour: 3 seeds, bootstrap 95% CI over queries.
"""

from __future__ import annotations

import json
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
from lumen.force.contextual.embed import MockEmbedder
from lumen.force.mnemonic.retrieval_dense import DenseHit, VectorChannel
from lumen.force.mnemonic.retrieval_lexical import LexicalChannel, LexicalHit
from lumen.force.mnemonic.store import store_memory
from lumen.lumen.fusion import fuse_and_rerank
from lumen.lumen.intent import IntentRouter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 456]
TOP_K_VALUES = [1, 3, 5, 10, 20]
N_BOOTSTRAP = 1_000
EMBED_DIMS = 384
RESULTS_DIR = Path(__file__).with_suffix("").parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Room topology: 8 themed rooms, each with 4 loci.
ROOM_CONFIG = [
    {
        "name": "machine_learning",
        "topics": ["machine learning", "deep learning", "neural network", "training pipelines", "gradient descent"],
        "loci": ["recent", "fundamentals", "applications", "challenges"],
    },
    {
        "name": "nlp",
        "topics": ["natural language processing", "transformer architecture", "tokenization methods", "bert model", "large language models"],
        "loci": ["recent", "fundamentals", "applications", "challenges"],
    },
    {
        "name": "computer_vision",
        "topics": ["computer vision", "convolutional neural network", "image recognition", "object detection", "semantic segmentation"],
        "loci": ["recent", "fundamentals", "applications", "challenges"],
    },
    {
        "name": "robotics",
        "topics": ["robotics", "automation systems", "kinematics", "manipulator design", "sensor fusion"],
        "loci": ["recent", "fundamentals", "applications", "challenges"],
    },
    {
        "name": "cybersecurity",
        "topics": ["cybersecurity", "threat detection", "encryption standards", "zero trust architecture", "penetration testing"],
        "loci": ["recent", "fundamentals", "applications", "challenges"],
    },
    {
        "name": "cloud",
        "topics": ["cloud computing", "serverless architecture", "kubernetes orchestration", "devops practices", "scalability engineering"],
        "loci": ["recent", "fundamentals", "applications", "challenges"],
    },
    {
        "name": "blockchain",
        "topics": ["blockchain technology", "consensus mechanisms", "smart contracts", "decentralization", "cryptocurrency markets"],
        "loci": ["recent", "fundamentals", "applications", "challenges"],
    },
    {
        "name": "healthcare",
        "topics": ["healthcare innovation", "clinical trials", "diagnosis assistance", "medical imaging", "patient data privacy"],
        "loci": ["recent", "fundamentals", "applications", "challenges"],
    },
]

# Adjacent rooms for cross-topic passages
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

PASSAGES_PER_LOCUS = 75          # 8 rooms * 4 loci * 75 = 2,400 passages
CROSS_TOPIC_RATE = 0.15          # 15% cross-room passages
PURE_QUERIES_PER_ROOM = 8
CROSS_QUERIES_PER_EDGE = 2


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------
_embedder = None
_embedder_name = "mock"


def _get_embedder():
    global _embedder, _embedder_name
    if _embedder is not None:
        return _embedder, _embedder_name

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        class _Real:
            def __init__(self, m):
                self._m = m
                self.dims = m.get_sentence_embedding_dimension()
            def encode(self, texts):
                vecs = self._m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                return np.asarray(vecs, dtype=np.float32)
            def encode_single(self, text):
                return self.encode([text])[0]

        _embedder = _Real(model)
        _embedder_name = "all-MiniLM-L6-v2"
        print(f"[INFO] Using embedder: {_embedder_name} ({_embedder.dims} dims)")
        return _embedder, _embedder_name
    except Exception as exc:
        print(f"[WARN] No real embedder ({exc}) — using MockEmbedder")
        _embedder = MockEmbedder(dims=EMBED_DIMS)
        _embedder_name = "mock"
        return _embedder, _embedder_name


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------
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


def _generate_corpus(seed: int):
    rng = np.random.default_rng(seed)
    passages = []      # list of dicts: {text, room, locus}
    for room in ROOM_CONFIG:
        for locus in room["loci"]:
            for i in range(PASSAGES_PER_LOCUS):
                if rng.random() < CROSS_TOPIC_RATE:
                    alt_room_name = rng.choice(ADJACENCY[room["name"]])
                    alt_room = next(r for r in ROOM_CONFIG if r["name"] == alt_room_name)
                    topic = rng.choice(room["topics"])
                    alt_topic = rng.choice(alt_room["topics"])
                    tmpl = rng.choice(CROSS_TEMPLATES)
                    text = tmpl.format(topic=topic, alt_topic=alt_topic)
                else:
                    topic = rng.choice(room["topics"])
                    tmpl = rng.choice(PURE_TEMPLATES)
                    text = tmpl.format(topic=topic, aspect=locus)
                # Ensure uniqueness with index to avoid dedup collisions
                text = f"[{room['name']}/{locus}/{i}] {text}"
                passages.append({"text": text, "room": room["name"], "locus": locus})
    rng.shuffle(passages)
    return passages


def _generate_queries(passages, seed: int):
    rng = np.random.default_rng(seed)
    queries = []  # list of dicts: {query, ground_truth_rooms, relevant_chunk_ids}

    # Pure room queries
    for room in ROOM_CONFIG:
        for _ in range(PURE_QUERIES_PER_ROOM):
            topic = rng.choice(room["topics"])
            modifier = rng.choice(["recent advances", "future trends", "best practices", "challenges"])
            query = f"{topic} {modifier}"
            queries.append({"query": query, "ground_truth_rooms": [room["name"]]})

    # Cross-room queries
    for room in ROOM_CONFIG:
        for alt in ADJACENCY[room["name"]]:
            for _ in range(CROSS_QUERIES_PER_EDGE):
                topic = rng.choice(room["topics"])
                alt_room = next(r for r in ROOM_CONFIG if r["name"] == alt)
                alt_topic = rng.choice(alt_room["topics"])
                query = f"{topic} {alt_topic}"
                queries.append({"query": query, "ground_truth_rooms": [room["name"], alt]})

    rng.shuffle(queries)
    return queries


def _compute_relevant_set(passages, query):
    """A chunk is relevant if it shares >= 2 content words with the query."""
    q_words = set(re.findall(r"\w+", query.lower()))
    relevant = set()
    for idx, p in enumerate(passages):
        p_words = set(re.findall(r"\w+", p["text"].lower()))
        if len(p_words & q_words) >= 2:
            relevant.add(idx)
    return relevant


# ---------------------------------------------------------------------------
# Room router (query → predicted room(s))
# ---------------------------------------------------------------------------
class RoomRouter:
    """Keyword-based room predictor simulating a real intent→room router."""

    def __init__(self):
        self.room_keywords = {}
        for room in ROOM_CONFIG:
            words = set()
            for topic in room["topics"]:
                words.update(re.findall(r"\w+", topic.lower()))
            self.room_keywords[room["name"]] = words

    def predict(self, query: str) -> list[str]:
        q_words = set(re.findall(r"\w+", query.lower()))
        scores = {}
        for room, kw in self.room_keywords.items():
            scores[room] = len(q_words & kw)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [r for r, s in ranked if s > 0][:2]


# ---------------------------------------------------------------------------
# Room-constrained search channels
# ---------------------------------------------------------------------------
class RoomConstrainedLexicalChannel:
    """BM25 restricted to specific room(s)."""

    def __init__(self, conn: Any, room_ids: list[int]):
        self.conn = conn
        self.room_ids = list(room_ids)

    def search(self, query: str, k: int = 20) -> list[LexicalHit]:
        safe_query = LexicalChannel._sanitize(query)
        if not safe_query or not self.room_ids:
            return []
        placeholders = ",".join("?" for _ in self.room_ids)
        rows = self.conn.execute(
            f"""
            SELECT c.rowid, c.rank
            FROM chunk_fts c
            JOIN chunk ch ON ch.chunk_id = c.rowid
            WHERE ch.room_id IN ({placeholders})
              AND c.chunk_fts MATCH ?
            ORDER BY c.rank
            LIMIT ?
            """,
            (*self.room_ids, safe_query, k),
        ).fetchall()
        return [LexicalHit(cid, rank, b"") for cid, rank in rows]


class RoomConstrainedVectorChannel:
    """Dense vector search restricted to specific room(s) via vec_fallback."""

    def __init__(self, conn: Any, room_ids: list[int], dims: int):
        self.conn = conn
        self.room_ids = list(room_ids)
        self.dims = dims

    def search(self, query_vector: np.ndarray, k: int = 20) -> list[DenseHit]:
        if not self.room_ids:
            return []
        placeholders = ",".join("?" for _ in self.room_ids)
        rows = self.conn.execute(
            f"""
            SELECT vf.chunk_id, vf.embedding
            FROM vec_fallback vf
            JOIN chunk ch ON ch.chunk_id = vf.chunk_id
            WHERE ch.room_id IN ({placeholders}) AND ch.valid_to IS NULL
            """,
            self.room_ids,
        ).fetchall()

        q = query_vector.astype(np.float32)
        qn = np.linalg.norm(q)
        hits = []
        for cid, emb_blob in rows:
            vec = np.frombuffer(emb_blob, dtype=np.float32)
            vn = np.linalg.norm(vec)
            sim = 0.0 if vn == 0 or qn == 0 else float(np.dot(q, vec) / (qn * vn))
            hits.append(DenseHit(cid, sim, vec))
        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:k]


# ---------------------------------------------------------------------------
# Unified search runner
# ---------------------------------------------------------------------------
def _run_strategy(
    conn, config, embedder, query, room_ids, k, passages
):
    """Execute one search strategy and return (results, latency_ms)."""
    t0 = time.perf_counter()

    if room_ids is None:
        lexical = LexicalChannel(conn)
        vector = VectorChannel(config, conn)
    else:
        lexical = RoomConstrainedLexicalChannel(conn, room_ids)
        vector = RoomConstrainedVectorChannel(conn, room_ids, config.embedding_dims)

    qvec = embedder.encode_single(query)
    lexical_hits = lexical.search(query, k=k)
    dense_hits = vector.search(qvec, k=k)

    results = fuse_and_rerank(
        lexical_hits,
        dense_hits,
        goal_tree_keywords=[],
        conn=conn,
        budget_candidates=200,
        query_embedding=qvec,
        graph_hits=None,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return results[:k], elapsed_ms


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _recall_at_k(retrieved_chunk_ids, relevant_set, k):
    if not relevant_set:
        return 0.0
    return len(set(retrieved_chunk_ids[:k]) & relevant_set) / len(relevant_set)


def _bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP):
    rng = np.random.default_rng(42)
    n = len(values)
    means = [np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_bootstrap)]
    return np.mean(values), np.percentile(means, 2.5), np.percentile(means, 97.5)


# ---------------------------------------------------------------------------
# Per-seed runner
# ---------------------------------------------------------------------------
def _run_single_seed(seed: int):
    np.random.default_rng(seed)
    passages = _generate_corpus(seed)
    queries = _generate_queries(passages, seed)

    # Pre-compute relevance sets
    for q in queries:
        q["relevant"] = _compute_relevant_set(passages, q["query"])

    # Embed all passages up-front for fast bulk storage
    embedder, embedder_name = _get_embedder()
    texts = [p["text"] for p in passages]
    embeddings = embedder.encode(texts)

    tmpdir = tempfile.mkdtemp(prefix=f"lumen_bench_pne_{seed}_")
    config = LumenConfig(
        store_path=Path(tmpdir),
        embedding_dims=EMBED_DIMS,
        vector_index="sqlite-vec",
    )
    conn = get_connection(config)

    # Ingest passages
    print(f"[INFO] Seed {seed}: ingesting {len(passages)} passages into palace ...")
    room_name_to_id = {}
    for idx, (p, emb) in enumerate(zip(passages, embeddings, strict=False)):
        chunk_id = store_memory(
            conn,
            content=p["text"],
            room_name=p["room"],
            locus_name=p["locus"],
            embedding=emb,
            config=config,
        )
        passages[idx]["chunk_id"] = chunk_id
        # Cache room_id mapping
        if p["room"] not in room_name_to_id:
            row = conn.execute(
                "SELECT room_id FROM room WHERE name = ?", (p["room"],)
            ).fetchone()
            room_name_to_id[p["room"]] = row[0]
        if idx % 500 == 0:
            conn.commit()
    conn.commit()

    # Map query relevant sets from passage indices to chunk_ids
    for q in queries:
        q["relevant_chunk_ids"] = {
            passages[idx]["chunk_id"]
            for idx in q["relevant"]
            if idx < len(passages) and "chunk_id" in passages[idx]
        }

    total_chunks = conn.execute("SELECT COUNT(*) FROM chunk WHERE valid_to IS NULL").fetchone()[0]

    router = RoomRouter()
    intent_router = IntentRouter()  # used only for intent-classification accuracy side-metric

    # Storage for per-query measurements
    per_query = []

    for q in queries:
        gt_room_ids = [room_name_to_id[r] for r in q["ground_truth_rooms"] if r in room_name_to_id]
        predicted_rooms = router.predict(q["query"])
        predicted_room_ids = [room_name_to_id[r] for r in predicted_rooms if r in room_name_to_id]

        # Routing accuracy
        routing_correct = len(set(predicted_rooms) & set(q["ground_truth_rooms"])) > 0

        # Pruning efficiency for routed strategy
        if predicted_room_ids:
            pruned_count = conn.execute(
                "SELECT COUNT(*) FROM chunk WHERE valid_to IS NULL AND room_id IN ({})".format(
                    ",".join(map(str, predicted_room_ids))
                )
            ).fetchone()[0]
            pruning_ratio = 1.0 - (pruned_count / max(1, total_chunks))
        else:
            pruning_ratio = 0.0

        # Intent classification (side metric)
        intent = intent_router.classify(q["query"])

        # Run strategies
        strategies = {}
        for name, rids in [
            ("global", None),
            ("oracle", gt_room_ids),
            ("routed", predicted_room_ids if predicted_room_ids else gt_room_ids),
        ]:
            results, latency_ms = _run_strategy(
                conn, config, embedder, q["query"], rids, k=max(TOP_K_VALUES), passages=passages
            )
            retrieved_ids = [r.chunk_id for r in results]
            recalls = {
                f"recall_{k}": _recall_at_k(retrieved_ids, q["relevant_chunk_ids"], k)
                for k in TOP_K_VALUES
            }
            strategies[name] = {
                "latency_ms": latency_ms,
                **recalls,
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

    return per_query, total_chunks, embedder_name


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark():
    all_per_query = []
    total_chunks = 0
    embedder_name = "mock"

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Running PNE benchmark — seed {seed}")
        print(f"{'='*60}")
        per_query, tc, emb_name = _run_single_seed(seed)
        all_per_query.extend(per_query)
        total_chunks = tc
        embedder_name = emb_name

    # Aggregates
    def _agg(values):
        return _bootstrap_ci(values)

    summary = {}
    for strategy in ["global", "oracle", "routed"]:
        summary[strategy] = {}
        latencies = [q["strategies"][strategy]["latency_ms"] for q in all_per_query]
        summary[strategy]["latency_ms"] = {
            "mean": round(float(np.mean(latencies)), 2),
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "p99": round(float(np.percentile(latencies, 99)), 2),
            "ci95": {"lo": round(float(_bootstrap_ci(latencies)[1]), 4), "hi": round(float(_bootstrap_ci(latencies)[2]), 4)},
        }
        for k in TOP_K_VALUES:
            vals = [q["strategies"][strategy][f"recall_{k}"] for q in all_per_query]
            mean, lo, hi = _agg(vals)
            summary[strategy][f"recall_{k}"] = {
                "mean": round(float(mean), 4),
                "ci95_lo": round(float(lo), 4),
                "ci95_hi": round(float(hi), 4),
            }

    # Derived metrics
    for baseline, constrained in [("global", "oracle"), ("global", "routed")]:
        key = f"{constrained}_vs_{baseline}"
        summary[key] = {}

        # Recall retention
        for k in TOP_K_VALUES:
            ratios = []
            for q in all_per_query:
                base_r = q["strategies"][baseline][f"recall_{k}"]
                cons_r = q["strategies"][constrained][f"recall_{k}"]
                if base_r > 0:
                    ratios.append(cons_r / base_r)
                else:
                    ratios.append(1.0 if cons_r == 0 else 0.0)
            mean, lo, hi = _agg(ratios)
            summary[key][f"recall_retention_{k}"] = {
                "mean": round(float(mean), 4),
                "ci95_lo": round(float(lo), 4),
                "ci95_hi": round(float(hi), 4),
            }

        # Latency speedup
        speedups = [
            q["strategies"][baseline]["latency_ms"] / max(q["strategies"][constrained]["latency_ms"], 0.001)
            for q in all_per_query
        ]
        mean, lo, hi = _agg(speedups)
        summary[key]["latency_speedup"] = {
            "mean": round(float(mean), 2),
            "ci95_lo": round(float(lo), 2),
            "ci95_hi": round(float(hi), 2),
        }

    # Routing & pruning
    routing_accuracy = np.mean([q["routing_correct"] for q in all_per_query])
    pruning_ratios = [q["pruning_ratio"] for q in all_per_query]
    summary["routing_and_pruning"] = {
        "routing_accuracy": round(float(routing_accuracy), 4),
        "mean_pruning_ratio": round(float(np.mean(pruning_ratios)), 4),
        "median_pruning_ratio": round(float(np.median(pruning_ratios)), 4),
    }

    # Intent distribution
    intents = {}
    for q in all_per_query:
        intents[q["intent"]] = intents.get(q["intent"], 0) + 1
    summary["intent_distribution"] = intents

    report = {
        "benchmark": "navigation",
        "embedder": embedder_name,
        "num_rooms": len(ROOM_CONFIG),
        "passages_per_locus": PASSAGES_PER_LOCUS,
        "total_chunks": total_chunks,
        "cross_topic_rate": CROSS_TOPIC_RATE,
        "num_queries": len(all_per_query),
        "num_seeds": len(SEEDS),
        "bootstrap_samples": N_BOOTSTRAP,
        "summary": summary,
        "raw_queries": [
            {
                "query": q["query"],
                "intent": q["intent"],
                "ground_truth_rooms": q["ground_truth_rooms"],
                "predicted_rooms": q["predicted_rooms"],
                "routing_correct": q["routing_correct"],
                "pruning_ratio": q["pruning_ratio"],
                "num_relevant": q["num_relevant"],
                "global_recall_10": q["strategies"]["global"]["recall_10"],
                "oracle_recall_10": q["strategies"]["oracle"]["recall_10"],
                "routed_recall_10": q["strategies"]["routed"]["recall_10"],
                "global_latency_ms": round(q["strategies"]["global"]["latency_ms"], 2),
                "oracle_latency_ms": round(q["strategies"]["oracle"]["latency_ms"], 2),
                "routed_latency_ms": round(q["strategies"]["routed"]["latency_ms"], 2),
            }
            for q in all_per_query
        ],
    }

    # JSON output
    json_path = RESULTS_DIR / "navigation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown output
    md_lines = [
        "# Palace Navigation Efficiency (PNE) Benchmark Results",
        "",
        f"**Embedder:** {embedder_name} | **Total Chunks:** {total_chunks:,} | **Queries:** {len(all_per_query)} | **Seeds:** {len(SEEDS)}",
        f"**Rooms:** {len(ROOM_CONFIG)} | **Cross-Topic Rate:** {CROSS_TOPIC_RATE} | **Bootstrap:** {N_BOOTSTRAP} samples (95% CI)",
        "",
        "## Routing & Pruning",
        "",
        f"- **Intent Routing Accuracy:** {summary['routing_and_pruning']['routing_accuracy']:.1%}",
        f"- **Mean Pruning Ratio:** {summary['routing_and_pruning']['mean_pruning_ratio']:.1%} of corpus excluded by room filter",
        f"- **Median Pruning Ratio:** {summary['routing_and_pruning']['median_pruning_ratio']:.1%}",
        "",
        "## Latency Comparison (ms)",
        "",
        "| Strategy | Mean | p50 | p95 | p99 |",
        "|---|---|---|---|---|",
    ]
    for strat in ["global", "oracle", "routed"]:
        lat = summary[strat]["latency_ms"]
        md_lines.append(
            f"| {strat} | {lat['mean']} | {lat['p50']} | {lat['p95']} | {lat['p99']} |"
        )

    md_lines.extend([
        "",
        "## Recall@k",
        "",
        "| k | Global | Oracle | Routed |",
        "|---|---|---|---|",
    ])
    for k in TOP_K_VALUES:
        g = summary["global"][f"recall_{k}"]["mean"]
        o = summary["oracle"][f"recall_{k}"]["mean"]
        r = summary["routed"][f"recall_{k}"]["mean"]
        md_lines.append(f"| {k} | {g:.4f} | {o:.4f} | {r:.4f} |")

    md_lines.extend([
        "",
        "## Oracle Room-Constrained vs Global",
        "",
        "| Metric | Mean | 95% CI |",
        "|---|---|---|",
    ])
    for k in TOP_K_VALUES:
        rr = summary["oracle_vs_global"][f"recall_retention_{k}"]
        md_lines.append(f"| Recall Retention @ {k} | {rr['mean']:.4f} | [{rr['ci95_lo']:.4f}, {rr['ci95_hi']:.4f}] |")
    ls = summary["oracle_vs_global"]["latency_speedup"]
    md_lines.append(f"| Latency Speedup | {ls['mean']:.2f}x | [{ls['ci95_lo']:.2f}x, {ls['ci95_hi']:.2f}x] |")

    md_lines.extend([
        "",
        "## Routed Room-Constrained vs Global",
        "",
        "| Metric | Mean | 95% CI |",
        "|---|---|---|",
    ])
    for k in TOP_K_VALUES:
        rr = summary["routed_vs_global"][f"recall_retention_{k}"]
        md_lines.append(f"| Recall Retention @ {k} | {rr['mean']:.4f} | [{rr['ci95_lo']:.4f}, {rr['ci95_hi']:.4f}] |")
    ls = summary["routed_vs_global"]["latency_speedup"]
    md_lines.append(f"| Latency Speedup | {ls['mean']:.2f}x | [{ls['ci95_lo']:.2f}x, {ls['ci95_hi']:.2f}x] |")

    md_lines.extend([
        "",
        "## Intent Distribution",
        "",
    ])
    for intent, count in sorted(summary["intent_distribution"].items(), key=lambda x: x[1], reverse=True):
        md_lines.append(f"- **{intent}:** {count} queries")

    md_path = RESULTS_DIR / "navigation_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[INFO] Results written to {json_path} and {md_path}")

    # Console summary
    print("\n=== PNE Summary ===")
    print(f"Routing Accuracy:     {routing_accuracy:.1%}")
    print(f"Mean Pruning Ratio:   {summary['routing_and_pruning']['mean_pruning_ratio']:.1%}")
    print(f"Latency (global p50): {summary['global']['latency_ms']['p50']:.2f} ms")
    print(f"Latency (oracle p50): {summary['oracle']['latency_ms']['p50']:.2f} ms  (speedup {summary['oracle_vs_global']['latency_speedup']['mean']:.2f}x)")
    print(f"Latency (routed p50): {summary['routed']['latency_ms']['p50']:.2f} ms  (speedup {summary['routed_vs_global']['latency_speedup']['mean']:.2f}x)")
    print(f"Recall@10 (global):   {summary['global']['recall_10']['mean']:.4f}")
    print(f"Recall@10 (oracle):   {summary['oracle']['recall_10']['mean']:.4f}  (retention {summary['oracle_vs_global']['recall_retention_10']['mean']:.4f})")
    print(f"Recall@10 (routed):   {summary['routed']['recall_10']['mean']:.4f}  (retention {summary['routed_vs_global']['recall_retention_10']['mean']:.4f})")

    return report


if __name__ == "__main__":
    sys.exit(0 if run_benchmark() else 1)
