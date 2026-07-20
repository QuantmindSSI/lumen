"""FastAPI server exposing Lumen memory operations.

Endpoints:
  POST /search    — semantic + lexical hybrid search
  POST /store     — store a memory chunk
  POST /feedback  — log explicit or implicit feedback
  GET  /health    — liveness probe
  GET  /status    — palace overview
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from lumen.config import LumenConfig
from lumen.data.schema import ensure_schema, get_connection
from lumen.force.contextual.embed import MockEmbedder, get_embedder
from lumen.lumen.controller import TwinForceController
from lumen.lumen.conversation import ConversationMemory
from lumen.lumen.fusion import RetrievedChunk
from lumen.lumen.search import SearchPipeline

app = FastAPI(title="Lumen Memory API", version="0.1.0")

# Module-level state (single instance for pilot)
_state: dict = {}


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(5, ge=1, le=50)
    room_hint: str | None = Field(None, description="Optional room to bias search toward")


class SearchResponse(BaseModel):
    query: str
    results: list[dict]
    latency_ms: float


class StoreRequest(BaseModel):
    content: str = Field(..., min_length=1)
    room: str = Field(..., min_length=1)
    locus: str | None = None
    source_type: str = Field("user_input", pattern="^(user_input|agent_reasoning|consolidation|import|p2p_share)$")


class StoreResponse(BaseModel):
    chunk_id: int
    room: str


class FeedbackRequest(BaseModel):
    chunk_id: int = Field(..., ge=1)
    was_useful: bool
    user_id: str = "default"
    feedback_type: str = Field("explicit", pattern="^(implicit|explicit|repair)$")


class FeedbackResponse(BaseModel):
    success: bool


class StatusResponse(BaseModel):
    rooms: int
    active_chunks: int
    tfc: dict
    embedding_model_available: bool


def _get_config() -> LumenConfig:
    return _state.get("config", LumenConfig())


def _get_conn() -> sqlite3.Connection:
    return _state.get("conn")


def _get_pipeline() -> SearchPipeline:
    return _state.get("pipeline")


def _get_conversation_memory() -> ConversationMemory:
    return _state.get("conversation")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure schema and shared resources.

    If *_state* already contains a connection (e.g. tests injected it),
    skip re-initialisation.
    """
    if _state.get("conn") is not None:
        yield
        return

    config = LumenConfig()
    config.store_path.mkdir(parents=True, exist_ok=True)
    config.model_path.mkdir(parents=True, exist_ok=True)
    conn = get_connection(config)
    ensure_schema(conn)

    embedder = None
    try:
        embedder = get_embedder(config, allow_mock=False)
    except Exception:
        embedder = MockEmbedder(dims=config.embedding_dims)

    pipeline = SearchPipeline(conn, config, embedder=embedder)
    conversation = ConversationMemory(config=config, conn=conn, embedder=embedder)
    tfc = TwinForceController()

    _state.update({
        "config": config,
        "conn": conn,
        "pipeline": pipeline,
        "conversation": conversation,
        "tfc": tfc,
        "embedder": embedder,
    })
    yield
    conn.close()


app.router.lifespan_context = lifespan


@app.get("/health")
async def health() -> dict:
    conn = _get_conn()
    try:
        conn.execute("SELECT 1")
        return {"status": "ok"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unreachable: {exc}") from exc


@app.get("/status")
async def status() -> StatusResponse:
    conn = _get_conn()
    config = _get_config()
    room_count = conn.execute("SELECT COUNT(*) FROM room").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunk WHERE valid_to IS NULL").fetchone()[0]
    tfc = _state.get("tfc") or TwinForceController()
    model_dir = Path(config.model_path) / config.embedding_model
    return StatusResponse(
        rooms=room_count,
        active_chunks=chunk_count,
        tfc=tfc.to_env(),
        embedding_model_available=model_dir.exists(),
    )


@app.post("/search")
async def search(req: SearchRequest) -> SearchResponse:
    import time
    t0 = time.perf_counter()
    pipeline = _get_pipeline()
    results = pipeline.execute(req.query, k=req.top_k)
    latency_ms = (time.perf_counter() - t0) * 1000

    serialised = []
    for rc in results[: req.top_k]:
        serialised.append({
            "chunk_id": rc.chunk_id,
            "room": rc.room_name,
            "locus": rc.locus_name,
            "content": rc.content,
            "score": round(rc.final_score, 4),
            "vm_score": round(rc.vm_score, 4),
            "provenance_id": rc.provenance_id,
        })

    return SearchResponse(query=req.query, results=serialised, latency_ms=round(latency_ms, 2))


@app.post("/store")
async def store(req: StoreRequest) -> StoreResponse:
    from lumen.force.mnemonic.store import store_memory

    config = _get_config()
    conn = _get_conn()
    embedder = _state.get("embedder")
    embedding = embedder.encode_single(req.content) if embedder is not None else None

    chunk_id = store_memory(
        conn,
        content=req.content,
        room_name=req.room,
        locus_name=req.locus,
        source_type=req.source_type,
        embedding=embedding,
        config=config,
    )
    return StoreResponse(chunk_id=chunk_id, room=req.room)


@app.post("/feedback")
async def feedback(req: FeedbackRequest) -> FeedbackResponse:
    conversation = _get_conversation_memory()
    conversation.log_explicit_feedback(
        chunk_id=req.chunk_id,
        was_useful=req.was_useful,
        user_id=req.user_id,
        feedback_type=req.feedback_type,
    )
    return FeedbackResponse(success=True)


@app.post("/assemble")
async def assemble(req: SearchRequest) -> dict:
    """Retrieve + assemble context in one call."""
    import time
    t0 = time.perf_counter()
    conversation = _get_conversation_memory()
    turn = conversation.retrieve_and_assemble(req.query, top_k=req.top_k)
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "query": req.query,
        "assembled_context": turn.assembled_context,
        "retrieved_count": len(turn.retrieved_chunks),
        "latency_ms": round(latency_ms, 2),
    }


@app.post("/turn")
async def turn(req: dict) -> dict:
    """Store a full conversation turn.

    Expected body::

        {
          "user_msg": "...",
          "assistant_msg": "...",
          "room": "conversations",
          "retrieved_chunk_ids": [1, 2, 3]
        }
    """
    user_msg = req.get("user_msg", "")
    assistant_msg = req.get("assistant_msg", "")
    room = req.get("room", "conversations")
    chunk_ids = req.get("retrieved_chunk_ids", [])

    if not user_msg or not assistant_msg:
        raise HTTPException(status_code=422, detail="user_msg and assistant_msg are required")

    conversation = _get_conversation_memory()
    # Reconstruct RetrievedChunk stubs for feedback logging
    stubs = []
    conn = _get_conn()
    for cid in chunk_ids:
        row = conn.execute(
            "SELECT room_id, locus_id, content FROM chunk WHERE chunk_id = ?", (cid,)
        ).fetchone()
        if row:
            stubs.append(
                RetrievedChunk(
                    chunk_id=cid,
                    room_name="unknown",
                    locus_name="none",
                    content=row["content"] or "",
                    provenance_id=None,
                    rrf_score=0.0,
                    vm_score=0.0,
                    frqad_score=0.0,
                    recency_hours=0.0,
                    final_score=0.0,
                )
            )

    user_id, assistant_id = conversation.store_turn(
        user_msg=user_msg,
        assistant_msg=assistant_msg,
        retrieved_chunks=stubs,
        room_name=room,
    )
    return {
        "user_chunk_id": user_id,
        "assistant_chunk_id": assistant_id,
        "room": room,
        "feedback_logged_for": chunk_ids,
    }


def main() -> None:
    """CLI entrypoint for ``uvicorn lumen.api.server:app``."""
    import uvicorn
    uvicorn.run("lumen.api.server:app", host="0.0.0.0", port=8848, log_level="info")


if __name__ == "__main__":
    main()
