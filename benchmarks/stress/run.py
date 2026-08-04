"""Stress test: 100k+ chunk ingestion, multi-threaded query throughput,
and L3 budget eviction under RSS pressure.

Run with:
    python -m benchmarks.stress.run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sqlite3
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

from lumen.config import LumenConfig
from lumen.data.schema import get_connection, init_db
from lumen.force.contextual.embed import MockEmbedder
from lumen.force.mnemonic.forgetting_l3_budget import budget_curated_eviction
from lumen.force.mnemonic.store import store_memory
from lumen.search import SearchPipeline


def _make_config(store_path: Path) -> LumenConfig:
    return LumenConfig(
        store_path=store_path,
        model_path=store_path / "models",
        vector_index="sqlite-vec",
        device="generic",
        memory_limit_mb=256,
        context_budget=2048,
    )


def _ingest_chunks(conn: sqlite3.Connection, config: LumenConfig, embedder, count: int):
    t0 = time.perf_counter()
    for i in range(count):
        content = f"stress_test_chunk_{i:06d} " + "word " * 20
        store_memory(
            conn,
            content=content,
            room_name="stress",
            config=config,
            embedding=embedder.encode_single(content),
        )
        if i % 10000 == 0 and i > 0:
            elapsed = time.perf_counter() - t0
            print(f"  Ingested {i} chunks in {elapsed:.1f}s ({i/elapsed:.0f} chunks/s)")
    total = time.perf_counter() - t0
    print(f"  Total ingestion: {count} chunks in {total:.1f}s ({count/total:.0f} chunks/s)")
    return total


def _query_throughput(conn: sqlite3.Connection, config: LumenConfig, embedder, queries: int, workers: int):
    queries_list = [f"stress_test_chunk_{i % 100:03d}" for i in range(queries)]

    def _run(q: str) -> float:
        # Each worker gets its own connection because sqlite3 is not thread-safe
        tconn = get_connection(config)
        pipeline = SearchPipeline(tconn, config, embedder=embedder)
        t0 = time.perf_counter()
        pipeline.execute(q, k=10)
        elapsed = time.perf_counter() - t0
        tconn.close()
        return elapsed

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        latencies = list(ex.map(_run, queries_list))
    total = time.perf_counter() - t0
    latencies_ms = [lat * 1000 for lat in latencies]
    latencies_ms.sort()
    p50 = latencies_ms[len(latencies_ms) // 2]
    p99 = latencies_ms[int(len(latencies_ms) * 0.99)]
    print(f"  Query throughput: {queries} queries in {total:.1f}s ({queries/total:.0f} qps)")
    print(f"  Latency p50={p50:.1f}ms p99={p99:.1f}ms")
    return total, p50, p99


def _trigger_l3_eviction(conn: sqlite3.Connection, config: LumenConfig):
    # Artificially lower the memory limit to force eviction
    print("  Triggering L3 eviction with artificial low limit...")
    evicted = budget_curated_eviction(conn, config, target_ram_mb=1.0)
    print(f"  L3 evicted {evicted} chunks")
    return evicted


def main() -> int:
    parser = argparse.ArgumentParser(description="Lumen stress benchmark")
    parser.add_argument("--chunks", type=int, default=100_000, help="Number of chunks to ingest")
    parser.add_argument("--queries", type=int, default=1_000, help="Number of query requests")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent query workers")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        store = Path(tmpdir) / "store"
        store.mkdir()
        config = _make_config(store)
        conn = get_connection(config)
        init_db(conn)
        embedder = MockEmbedder(dims=config.embedding_dims)

        print(f"--- Stress Test: {args.chunks} chunks, {args.queries} queries ---")

        tracemalloc.start()
        ingest_time = _ingest_chunks(conn, config, embedder, args.chunks)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"  Memory current={current/1024/1024:.1f}MB peak={peak/1024/1024:.1f}MB")

        chunk_count = conn.execute("SELECT COUNT(*) FROM chunk WHERE valid_to IS NULL").fetchone()[0]
        print(f"  Active chunks in DB: {chunk_count}")

        query_time, p50, p99 = _query_throughput(conn, config, embedder, args.queries, args.workers)

        evicted = _trigger_l3_eviction(conn, config)

        result = {
            "chunks_ingested": args.chunks,
            "ingest_time_s": round(ingest_time, 2),
            "ingest_rate_hz": round(args.chunks / ingest_time, 1),
            "query_count": args.queries,
            "query_time_s": round(query_time, 2),
            "query_rate_hz": round(args.queries / query_time, 1),
            "latency_p50_ms": round(p50, 2),
            "latency_p99_ms": round(p99, 2),
            "peak_memory_mb": round(peak / 1024 / 1024, 1),
            "l3_evicted": evicted,
        }
        print(f"\nResults: {result}")
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
