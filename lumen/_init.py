"""Shared palace initialization — used by both API server and MCP server."""

from __future__ import annotations

import sqlite3

from lumen.brand.errors import ModelNotAvailableError
from lumen.config import LumenConfig
from lumen.controller import TwinForceController
from lumen.conversation import ConversationMemory
from lumen.data.schema import ensure_schema, get_connection
from lumen.force.contextual.embed import MockEmbedder, get_embedder
from lumen.logging import get_console_logger
from lumen.search import SearchPipeline

logger = get_console_logger(__name__)


def initialize_palace(config: LumenConfig) -> dict:
    """Initialize the memory palace: schema, embedder, pipeline, conversation.

    Returns a dictionary with keys: conn, embedder, pipeline, conversation,
    tfc, embedding_model_available. The caller is responsible for closing
    conn when done.
    """
    conn = get_connection(config)
    ensure_schema(conn)

    embedder = None
    embedding_model_available = False
    try:
        embedder = get_embedder(config, allow_mock=False)
        embedding_model_available = True
        logger.info("embedder_ready", model=config.embedding_model)
    except ModelNotAvailableError as exc:
        logger.warning("embedder_fallback", reason=str(exc))
        embedder = MockEmbedder(dims=config.embedding_dims)

    pipeline = SearchPipeline(conn, config, embedder=embedder)
    conversation = ConversationMemory(config=config, conn=conn, embedder=embedder)
    tfc = TwinForceController()

    return {
        "conn": conn,
        "embedder": embedder,
        "pipeline": pipeline,
        "conversation": conversation,
        "tfc": tfc,
        "embedding_model_available": embedding_model_available,
    }