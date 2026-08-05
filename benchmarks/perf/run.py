"""
Performance Benchmark for Lumen.

Measures query latency (p50/p95/p99), RAM footprint, storage size,
and write amplification with/without WearAwareBatcher for 1k/10k/50k corpora.
Outputs JSON + Markdown table to benchmarks/perf/results/.
"""

from __future__ import annotations

import json
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

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

try:
    import lumen  # noqa: F401
except ImportError as exc:
    print(f"[ERROR] Cannot import lumen: {exc}")
    sys.exit(1)

import contextlib

from lumen.config import LumenConfig
from lumen.data.schema import get_connection
from lumen.force.contextual.embed import FallbackEmbedder
from lumen.force.mnemonic.store import store_memory
from lumen.search import SearchPipeline
from lumen.sovereign.wear import WearAwareBatcher

SEED = 42
EMBED_DIMS = 384
CORPUS_SIZES = [1_000, 10_000, 50_000]
RESULTS_DIR = Path(__file__).with_suffix("").parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)


def _current_ram_mb() -> float:
    if _HAS_PSUTIL:
        try:
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            pass
    return 0.0


def _db_size_mb(db_path: Path) -> float:
    try:
        return os.path.getsize(db_path) / (1024 * 1024)
    except Exception:
        return 0.0


def _measure_latency(pipeline: SearchPipeline, queries: list[str], k: int = 10) -> dict[str, float]:
    latencies: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        with contextlib.suppress(Exception):
            pipeline.execute(q, k=k)
        latencies.append((time.perf_counter() - t0) * 1000)
    arr = np.array(latencies)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
    }


def _run_write_amplification(conn: sqlite3.Connection, n_updates: int = 1000) -> dict[str, Any]:
    """Compare naive per-write vs WearAwareBatcher bulk flush."""
    # Prepare IDs (we'll create a temp table of ids)
    rows = conn.execute(
        "SELECT chunk_id FROM chunk WHERE valid_to IS NULL LIMIT ?", (n_updates,)
    ).fetchall()
    ids = [r[0] for r in rows]
    if len(ids) < n_updates:
        # Pad with duplicates if corpus is smaller than expected
        ids = (ids * ((n_updates // len(ids)) + 1))[:n_updates]

    # Naive: individual executes with implicit transaction per call
    t0 = time.perf_counter()
    for cid in ids:
        conn.execute("UPDATE chunk SET vm_score = vm_score * 0.99 WHERE chunk_id = ?", (cid,))
    conn.commit()
    naive_time_ms = (time.perf_counter() - t0) * 1000

    # Reset values
    for cid in ids:
        conn.execute("UPDATE chunk SET vm_score = 0.5 WHERE chunk_id = ?", (cid,))
    conn.commit()

    # Batched: WearAwareBatcher
    batcher = WearAwareBatcher(conn, max_batch_size=n_updates, max_latency_ms=1000)
    t0 = time.perf_counter()
    for cid in ids:
        batcher.queue.append((
            "UPDATE chunk SET vm_score = vm_score * 0.99 WHERE chunk_id = ?",
            (cid,),
        ))
    batcher.flush_sync(list(batcher.queue))
    batcher.queue.clear()
    batch_time_ms = (time.perf_counter() - t0) * 1000

    # Reset values
    for cid in ids:
        conn.execute("UPDATE chunk SET vm_score = 0.5 WHERE chunk_id = ?", (cid,))
    conn.commit()

    speedup = round(naive_time_ms / max(batch_time_ms, 0.001), 2)
    return {
        "naive_time_ms": round(naive_time_ms, 2),
        "batch_time_ms": round(batch_time_ms, 2),
        "speedup": speedup,
        "updates": n_updates,
    }


def run_benchmark() -> dict[str, Any]:
    queries = [
        "machine learning applications",
        "cloud infrastructure scaling",
        "data privacy regulations",
        "neural network architectures",
        "renewable energy trends",
        "cybersecurity best practices",
        "quantum computing basics",
        "biotechnology breakthroughs",
        "mental health awareness",
        "space exploration missions",
        "blockchain consensus mechanisms",
        "computer vision pipelines",
        "natural language understanding",
        "robotics automation",
        "climate change mitigation",
    ]

    per_size_results: list[dict] = []

    for size in CORPUS_SIZES:
        print(f"[INFO] Benchmarking corpus size {size:,} ...")
        tmpdir = tempfile.mkdtemp(prefix=f"lumen_benchmark_perf_{size}_")
        config = LumenConfig(
            store_path=Path(tmpdir),
            embedding_dims=EMBED_DIMS,
            vector_index="sqlite-vec",
        )
        conn = get_connection(config)
        embedder = FallbackEmbedder(dims=EMBED_DIMS)

        # Generate corpus
        rng = np.random.default_rng(SEED + size)
        texts = [
            f"perf_chunk_{i:06d} " + " ".join(
                [f"token{rng.integers(0, 2000)}" for _ in range(rng.integers(15, 40))]
            ) for i in range(size)
        ]
        embs = embedder.encode(texts)

        t0 = time.perf_counter()
        for i, (text, emb) in enumerate(zip(texts, embs, strict=False)):
            store_memory(
                conn,
                content=text,
                room_name="benchmark_perf",
                locus_name=f"locus_{i % 50}",
                embedding=emb,
                config=config,
            )
            if i % 1000 == 0:
                conn.commit()
        conn.commit()
        ingestion_time_s = time.perf_counter() - t0

        ram_mb = _current_ram_mb()
        db_mb = _db_size_mb(config.db_path)

        pipeline = SearchPipeline(conn, config, embedder=embedder)
        n_queries = 50 if size <= 1000 else (20 if size <= 10_000 else 10)
        query_subset = queries[:n_queries]
        latencies = _measure_latency(pipeline, query_subset, k=10)

        wa = _run_write_amplification(conn, n_updates=min(1000, size))

        per_size_results.append({
            "corpus_size": size,
            "ingestion_time_s": round(ingestion_time_s, 2),
            "ram_mb": round(ram_mb, 2),
            "db_size_mb": round(db_mb, 2),
            "query_latency_ms": latencies,
            "write_amplification": wa,
        })

        conn.close()
        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    report = {
        "benchmark": "perf",
        "seed": SEED,
        "corpus_sizes": CORPUS_SIZES,
        "per_size": per_size_results,
    }

    json_path = RESULTS_DIR / "perf_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = RESULTS_DIR / "perf_results.md"
    lines = [
        "# Performance Benchmark Results",
        "",
        "## Query Latency & Footprint",
        "",
        "| Corpus Size | Ingestion (s) | DB Size (MB) | RAM (MB) | Latency p50 (ms) | Latency p95 (ms) | Latency p99 (ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in per_size_results:
        lat = r["query_latency_ms"]
        lines.append(
            f"| {r['corpus_size']:,} | {r['ingestion_time_s']} | {r['db_size_mb']} | {r['ram_mb']} | "
            f"{lat['p50_ms']} | {lat['p95_ms']} | {lat['p99_ms']} |"
        )

    lines.extend([
        "",
        "## Write Amplification",
        "",
        "| Corpus Size | Updates | Naive Time (ms) | Batch Time (ms) | Speedup |",
        "|---|---|---|---|---|",
    ])
    for r in per_size_results:
        wa = r["write_amplification"]
        lines.append(
            f"| {r['corpus_size']:,} | {wa['updates']} | {wa['naive_time_ms']} | "
            f"{wa['batch_time_ms']} | {wa['speedup']}x |"
        )
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Results written to {json_path} and {md_path}")
    return report


if __name__ == "__main__":
    run_benchmark()
