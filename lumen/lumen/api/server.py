"""FastAPI server exposing Lumen memory operations.

Endpoints:
  POST /search    — semantic + lexical hybrid search
  POST /store     — store a memory chunk
  POST /feedback  — log explicit or implicit feedback
  POST /assemble  — retrieve + assemble context in one call
  POST /turn      — store full conversation turn
  GET  /health    — liveness probe
  GET  /status    — palace overview
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from lumen.brand.errors import ModelNotAvailableError
from lumen.config import LumenConfig
from lumen.data.schema import ensure_schema, get_connection
from lumen.force.contextual.embed import MockEmbedder, get_embedder
from lumen.lumen.controller import TwinForceController
from lumen.lumen.conversation import ConversationMemory
from lumen.lumen.fusion import RetrievedChunk
from lumen.lumen.search import SearchPipeline

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_config = LumenConfig()
_config.store_path.mkdir(parents=True, exist_ok=True)
_config.model_path.mkdir(parents=True, exist_ok=True)
_config.resolve_device_defaults()

limiter = Limiter(key_func=get_remote_address, default_limits=[_config.api_rate_limit])

app = FastAPI(
    title="Lumen Memory API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _config.allowed_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level state (single instance for pilot)
_state: dict = {}


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------
async def _auth_middleware(request: Request, call_next):
    if _config.api_key and request.url.path not in ("/health", "/docs", "/redoc", "/openapi.json"):
        provided = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if provided != _config.api_key:
            return JSONResponse(
                status_code=401, content={"detail": "Invalid or missing API key"}
            )
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.middleware("http")(_auth_middleware)


class _SizeLimitMiddleware:
    def __init__(self, app, max_bytes: int = 1_048_576):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            content_length = 0
            for name, value in scope.get("headers", []):
                if name == b"content-length":
                    content_length = int(value)
                    break
            if content_length > self.max_bytes:
                async def _send_413(msg):
                    if msg["type"] == "http.response.start":
                        msg["status"] = 413
                        await send(msg)
                await send({
                    "type": "http.response.start",
                    "status": 413,
                    "headers": [(b"content-type", b"application/json")],
                })
                await send({
                    "type": "http.response.body",
                    "body": b'{"detail":"Request body too large"}',
                })
                return
        await self.app(scope, receive, send)


app.add_middleware(_SizeLimitMiddleware, max_bytes=_config.request_max_size_bytes)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Search query string")
    top_k: int = Field(5, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    results: list[dict]
    latency_ms: float


class StoreRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=50000)
    room: str = Field(..., min_length=1, max_length=128)
    locus: str | None = Field(None, max_length=128)
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


class TurnRequest(BaseModel):
    user_msg: str = Field(..., min_length=1, max_length=50000)
    assistant_msg: str = Field(..., min_length=1, max_length=50000)
    room: str = Field("conversations", max_length=128)
    retrieved_chunk_ids: list[int] = Field(default_factory=list, max_length=100)


class TurnResponse(BaseModel):
    user_chunk_id: int
    assistant_chunk_id: int
    room: str
    feedback_logged_for: list[int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_config() -> LumenConfig:
    return _state.get("config", _config)


def _get_conn() -> sqlite3.Connection | None:
    return _state.get("conn")


def _get_pipeline() -> SearchPipeline | None:
    return _state.get("pipeline")


def _get_conversation_memory() -> ConversationMemory | None:
    return _state.get("conversation")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if _state.get("conn") is not None:
        yield
        return

    conn = get_connection(_config)
    ensure_schema(conn)

    embedder = None
    embedding_model_available = False
    try:
        embedder = get_embedder(_config, allow_mock=False)
        embedding_model_available = True
        logger.info("embedder_ready", model=_config.embedding_model)
    except ModelNotAvailableError as exc:
        logger.warning("embedder_fallback", reason=str(exc))
        embedder = MockEmbedder(dims=_config.embedding_dims)

    pipeline = SearchPipeline(conn, _config, embedder=embedder)
    conversation = ConversationMemory(config=_config, conn=conn, embedder=embedder)
    tfc = TwinForceController()

    _state.update({
        "config": _config,
        "conn": conn,
        "pipeline": pipeline,
        "conversation": conversation,
        "tfc": tfc,
        "embedder": embedder,
        "embedding_model_available": embedding_model_available,
    })
    logger.info("server_started", device=_config.device, model_available=embedding_model_available)
    yield
    conn.close()
    logger.info("server_stopped")


app.router.lifespan_context = lifespan


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error("unhandled_error", request_id=request_id, error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    conn = _get_conn()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database not initialised")
    try:
        conn.execute("SELECT 1")
        return {"status": "ok"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database unreachable") from exc


@app.get("/status")
async def status() -> StatusResponse:
    conn = _get_conn()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database not initialised")
    room_count = conn.execute("SELECT COUNT(*) FROM room").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunk WHERE valid_to IS NULL").fetchone()[0]
    tfc = _state.get("tfc") or TwinForceController()
    return StatusResponse(
        rooms=room_count,
        active_chunks=chunk_count,
        tfc=tfc.to_env(),
        embedding_model_available=_state.get("embedding_model_available", False),
    )


@app.post("/search")
@limiter.limit(_config.api_rate_limit)
async def search(request: Request, req: SearchRequest) -> SearchResponse:
    t0 = time.perf_counter()
    pipeline = _get_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Search pipeline not initialised")
    results = pipeline.execute(req.query, k=req.top_k)
    latency_ms = (time.perf_counter() - t0) * 1000

    serialised = [
        {
            "chunk_id": rc.chunk_id,
            "room": rc.room_name,
            "locus": rc.locus_name,
            "content": rc.content,
            "score": round(rc.final_score, 4),
            "vm_score": round(rc.vm_score, 4),
            "provenance_id": rc.provenance_id,
        }
        for rc in results[:req.top_k]
    ]
    logger.info("search_completed", query=req.query[:100], results=len(serialised), latency_ms=round(latency_ms, 2))
    return SearchResponse(query=req.query, results=serialised, latency_ms=round(latency_ms, 2))


@app.post("/store")
@limiter.limit(_config.api_rate_limit)
async def store(request: Request, req: StoreRequest) -> StoreResponse:
    from lumen.force.mnemonic.store import store_memory

    config = _get_config()
    conn = _get_conn()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database not initialised")
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
@limiter.limit(_config.api_rate_limit)
async def feedback(request: Request, req: FeedbackRequest) -> FeedbackResponse:
    conversation = _get_conversation_memory()
    if conversation is None:
        raise HTTPException(status_code=503, detail="Conversation memory not initialised")
    conversation.log_explicit_feedback(
        chunk_id=req.chunk_id,
        was_useful=req.was_useful,
        user_id=req.user_id,
        feedback_type=req.feedback_type,
    )
    return FeedbackResponse(success=True)


@app.post("/assemble")
@limiter.limit(_config.api_rate_limit)
async def assemble(request: Request, req: SearchRequest) -> dict:
    t0 = time.perf_counter()
    conversation = _get_conversation_memory()
    if conversation is None:
        raise HTTPException(status_code=503, detail="Conversation memory not initialised")
    turn = conversation.retrieve_and_assemble(req.query, top_k=req.top_k)
    latency_ms = (time.perf_counter() - t0) * 1000
    return {
        "query": req.query,
        "assembled_context": turn.assembled_context,
        "retrieved_count": len(turn.retrieved_chunks),
        "latency_ms": round(latency_ms, 2),
    }


@app.post("/turn")
@limiter.limit(_config.api_rate_limit)
async def turn(request: Request, req: TurnRequest) -> TurnResponse:
    conversation = _get_conversation_memory()
    if conversation is None:
        raise HTTPException(status_code=503, detail="Conversation memory not initialised")
    conn = _get_conn()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database not initialised")

    stubs = []
    for cid in req.retrieved_chunk_ids:
        row = conn.execute(
            "SELECT content FROM chunk WHERE chunk_id = ?", (cid,)
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
        user_msg=req.user_msg,
        assistant_msg=req.assistant_msg,
        retrieved_chunks=stubs,
        room_name=req.room,
    )
    return TurnResponse(
        user_chunk_id=user_id,
        assistant_chunk_id=assistant_id,
        room=req.room,
        feedback_logged_for=req.retrieved_chunk_ids,
    )


def main() -> None:
    """CLI entrypoint for production server."""
    import uvicorn

    host = os.environ.get("LUMEN_API_HOST", _config.api_host)
    port = int(os.environ.get("LUMEN_API_PORT", str(_config.api_port)))
    log_level = os.environ.get("LUMEN_LOG_LEVEL", _config.log_level)
    uvicorn.run("lumen.api.server:app", host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()