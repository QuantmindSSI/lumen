"""C8: Real SearchPipeline.

Orchestrates: intent classify → parallel BM25 + dense retrieval → fusion → budget enforcement → TFC update
"""

import sqlite3
import time

from lumen.config import LumenConfig
from lumen.force.mnemonic.retrieval_dense import VectorChannel
from lumen.force.mnemonic.retrieval_graph import GraphChannel
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
                 embedder=None,
                 graph: GraphChannel | None = None):
        self.conn = conn
        self.config = config
        self.tfc = tfc or TwinForceController()
        self.intent_router = IntentRouter()
        self.lexical = LexicalChannel(conn)
        self.vector = VectorChannel(config, conn)
        if embedder is not None:
            self.embedder = embedder
        else:
            from lumen.force.contextual.embed import get_embedder
            self.embedder = get_embedder(config, allow_mock=False)
        self.graph = graph

    def execute(
        self,
        query: str,
        goal_tree_keywords: list[str] | None = None,
        k: int = 20,
        max_repair_attempts: int = 1,
    ) -> list[RetrievedChunk]:
        start = time.perf_counter()

        # Stage 1: Intent classification
        intent = self.intent_router.classify(query, self.tfc)

        # Stage 2: Parallel retrieval (sequential for simplicity, but real)
        query_vec = self.embedder.encode_single(query)
        lexical_hits = self.lexical.search(query, k=k)
        dense_hits = self.vector.search(query_vec, k=k)

        # Stage 2b: Graph retrieval
        graph_hits = []
        if self.graph is not None and dense_hits:
            seed_ids = [hit.chunk_id for hit in dense_hits[:3]]
            for seed in seed_ids:
                graph_hits.extend(self.graph.traverse_from_seed(seed, hops=2))

        # Stage 3: Fusion & rerank
        results = fuse_and_rerank(
            lexical_hits, dense_hits,
            goal_tree_keywords or [],
            self.conn,
            budget_candidates=200,
            query_embedding=query_vec,
            graph_hits=graph_hits or None,
        )

        # Stage 4: TFC update
        self.tfc.update({
            "novelty": 0.5 if not results else 0.3,
            "repetition": 0.0,
            "context_pressure": 0.0,
        })

        # Stage 5: Repair loop
        if max_repair_attempts > 0:
            reason = None
            if not results:
                reason = "empty_results"
            elif all(r.final_score < 0.01 for r in results):
                reason = "low_confidence"

            if reason:
                from lumen.lumen.repair import SearchRepair

                repair = SearchRepair(self.tfc, self)
                repaired = repair.attempt_repair(query, reason)
                if repaired:
                    results = repaired

        elapsed_ms = (time.perf_counter() - start) * 1000
        if logger:
            logger.info("search_executed", query=query, intent=intent,
                        results=len(results), latency_ms=round(elapsed_ms, 2))

        return results
