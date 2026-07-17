"""C8: Real SearchPipeline.

Orchestrates: intent classify → parallel BM25 + dense retrieval → fusion → budget enforcement → TFC update
"""

import time
from typing import List

import numpy as np
import sqlite3

from lumen.config import LumenConfig
from lumen.force.contextual.embed import FallbackEmbedder
from lumen.force.mnemonic.retrieval_dense import DenseHit, VectorChannel
from lumen.force.mnemonic.retrieval_lexical import LexicalChannel
from lumen.lumen.controller import TwinForceController
from lumen.lumen.fusion import RetrievedChunk, fuse_and_rerank
from lumen.lumen.intent import IntentRouter

logger = None
try:
    import structlog
    logger = structlog.get_logger()
except Exception:
    pass


class SearchPipeline:
    def __init__(self, conn: sqlite3.Connection, config: LumenConfig,
                 tfc: TwinForceController | None = None,
                 embedder=None):
        self.conn = conn
        self.config = config
        self.tfc = tfc or TwinForceController()
        self.intent_router = IntentRouter()
        self.lexical = LexicalChannel(conn)
        self.vector = VectorChannel(config, conn)
        self.embedder = embedder or FallbackEmbedder(dims=config.embedding_dims)

    def execute(self, query: str, goal_tree_keywords: List[str] | None = None) -> List[RetrievedChunk]:
        start = time.perf_counter()

        # Stage 1: Intent classification
        intent = self.intent_router.classify(query, self.tfc)

        # Stage 2: Parallel retrieval (sequential for simplicity, but real)
        query_vec = self.embedder.encode_single(query)
        lexical_hits = self.lexical.search(query, k=20)
        dense_hits = self.vector.search(query_vec, k=20)

        # Stage 3: Fusion & rerank
        results = fuse_and_rerank(
            lexical_hits, dense_hits,
            goal_tree_keywords or [],
            self.conn,
            budget_candidates=200,
            query_embedding=query_vec,
        )

        # Stage 4: TFC update
        self.tfc.update({
            "novelty": 0.5 if not results else 0.3,
            "repetition": 0.0,
            "context_pressure": 0.0,
        })

        elapsed_ms = (time.perf_counter() - start) * 1000
        if logger:
            logger.info("search_executed", query=query, intent=intent,
                        results=len(results), latency_ms=round(elapsed_ms, 2))

        return results
