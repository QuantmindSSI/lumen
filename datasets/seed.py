"""
Palace seed script — populates Lumen rooms from the domain corpus JSON.

Usage:
    python -m datasets.seed [--corpus datasets/domain_corpus.json] [--embedder all-MiniLM-L6-v2]

Creates a fully-populated memory palace with 8 rooms, 32 loci, and ~400
domain-specific fact chunks.  Each chunk is embedded and indexed so the
palace is ready for search, assembly, and benchmarking.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lumen.config import LumenConfig
from lumen.data.schema import ensure_schema, get_connection
from lumen.force.mnemonic.store import store_memory


def _get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)

        class _Real:
            def __init__(self, m):
                self._m = m

            def encode(self, texts):
                return np.asarray(
                    self._m.encode(texts, normalize_embeddings=True, show_progress_bar=True),
                    dtype=np.float32,
                )

            def encode_single(self, text):
                return self.encode([text])[0]

        print(f"[INFO] Using embedder: {model_name}")
        return _Real(model)
    except Exception as exc:
        print(f"[WARN] Real embedder unavailable ({exc}). Using MockEmbedder.")
        from lumen.force.contextual.embed import MockEmbedder

        config = LumenConfig()
        return MockEmbedder(dims=config.embedding_dims)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Seed Lumen palace with domain corpus")
    parser.add_argument(
        "--corpus",
        default=str(Path(__file__).resolve().parent / "domain_corpus.json"),
        help="Path to corpus JSON file",
    )
    parser.add_argument("--embedder", default="all-MiniLM-L6-v2", help="Embedder model name")
    args = parser.parse_args()

    # Load corpus
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"[ERROR] Corpus file not found: {corpus_path}")
        return 1

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"[INFO] Loaded corpus: {corpus['name']} — {corpus['description']}")
    print(f"[INFO] Rooms: {len(corpus['rooms'])}")

    # Init Lumen
    config = LumenConfig(store_path=Path("./data/store"), model_path=Path("./data/models"))
    config.store_path.mkdir(parents=True, exist_ok=True)
    config.model_path.mkdir(parents=True, exist_ok=True)

    conn = get_connection(config)
    embedder = _get_embedder(args.embedder)

    total_chunks = 0
    total_rooms = 0
    total_loci = 0

    print(f"\n[INFO] Seeding palace with embedder: {args.embedder}")
    t0 = time.perf_counter()

    for room in corpus["rooms"]:
        room_name = room["name"]
        print(f"\n  Room: {room_name} ({room['room_type']})")

        all_room_chunks = room["chunks"]
        all_texts = [c["content"] for c in all_room_chunks]
        all_loci = [c["locus"] for c in all_room_chunks]

        # Batch embed for efficiency
        embeddings = embedder.encode(all_texts)

        for chunk, emb, locus_name in zip(all_room_chunks, embeddings, all_loci):
            store_memory(
                conn,
                content=f"[{room_name}/{locus_name}] {chunk['content']}",
                room_name=room_name,
                locus_name=locus_name,
                embedding=emb,
                source_type="import",
                source_ref=f"corpus_v1:{room_name}",
                config=config,
            )
            total_chunks += 1
            if total_chunks % 100 == 0:
                conn.commit()
                print(f"    ... {total_chunks} chunks stored")

        conn.commit()
        total_rooms += 1
        total_loci += len(set(all_loci))
        print(f"    {len(all_room_chunks)} chunks across {len(set(all_loci))} loci stored")

    elapsed = time.perf_counter() - t0

    # Verify
    room_count = conn.execute("SELECT COUNT(*) FROM room").fetchone()[0]
    locus_count = conn.execute("SELECT COUNT(*) FROM locus").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunk WHERE valid_to IS NULL").fetchone()[0]
    fts_count = conn.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0]
    vec_count = conn.execute("SELECT COUNT(*) FROM vec_fallback").fetchone()[0]

    print(f"\n{'='*60}")
    print(f"Seeding Complete")
    print(f"{'='*60}")
    print(f"  Elapsed:        {elapsed:.1f}s")
    print(f"  Rooms created:  {room_count}")
    print(f"  Loci created:   {locus_count}")
    print(f"  Chunks stored:  {chunk_count}")
    print(f"  FTS5 indexed:   {fts_count}")
    print(f"  Vectors indexed:{vec_count}")
    print(f"  DB location:    {config.db_path}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())