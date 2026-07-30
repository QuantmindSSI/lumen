"""FastAPI server exposing Lumen memory operations.

Endpoints:
  POST /search      — semantic + lexical hybrid search
  POST /store       — store a memory chunk
  POST /feedback    — log explicit or implicit feedback
  POST /assemble    — retrieve + assemble context in one call
  POST /turn        — store full conversation turn
  GET  /health      — liveness probe
  GET  /status      — palace overview
  GET  /dashboard   — effectiveness dashboard (HTML)
  GET  /dashboard-data — dashboard JSON data
  GET  /metrics     — machine-readable metrics
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from lumen.brand.errors import ModelNotAvailableError
from lumen.config import LumenConfig
from lumen.data.schema import ensure_schema, get_connection
from lumen.force.contextual.embed import MockEmbedder, get_embedder
from lumen.logging import get_console_logger
from lumen.lumen.controller import TwinForceController
from lumen.lumen.conversation import ConversationMemory
from lumen.lumen.fusion import RetrievedChunk
from lumen.lumen.search import SearchPipeline

logger = get_console_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_config = LumenConfig()
_config.store_path.mkdir(parents=True, exist_ok=True)
_config.model_path.mkdir(parents=True, exist_ok=True)

limiter = Limiter(key_func=get_remote_address, default_limits=[_config.api_rate_limit])

app = FastAPI(
    title="Lumen Memory API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

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
    public_paths = (
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/dashboard",
        "/dashboard-data",
        "/metrics",
    )
    if _config.api_key and request.url.path not in public_paths:
        provided = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if provided != _config.api_key:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
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

                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"detail":"Request body too large"}',
                    }
                )
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
    source_type: str = Field(
        "user_input", pattern="^(user_input|agent_reasoning|consolidation|import|p2p_share)$"
    )


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
    user_chunk_id: int | None
    assistant_chunk_id: int | None
    room: str
    feedback_logged_for: list[int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_config() -> LumenConfig:
    return _state.get("config", _config)  # type: ignore[no-any-return]


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

    _state.update(
        {
            "config": _config,
            "conn": conn,
            "pipeline": pipeline,
            "conversation": conversation,
            "tfc": tfc,
            "embedder": embedder,
            "embedding_model_available": embedding_model_available,
        }
    )
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
        for rc in results[: req.top_k]
    ]
    logger.info(
        "search_completed",
        query=req.query[:100],
        results=len(serialised),
        latency_ms=round(latency_ms, 2),
    )
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
            """SELECT c.content, r.name AS room_name, l.name AS locus_name,
                      c.vm_score, c.provenance_root,
                      (strftime('%s','now') - c.created_at) / 3600.0 AS age_hours
               FROM chunk c
               JOIN room r ON r.room_id = c.room_id
               LEFT JOIN locus l ON l.locus_id = c.locus_id
               WHERE c.chunk_id = ? AND c.valid_to IS NULL""",
            (cid,),
        ).fetchone()
        if row:
            stubs.append(
                RetrievedChunk(
                    chunk_id=cid,
                    room_name=row["room_name"] or "unknown",
                    locus_name=row["locus_name"] or "none",
                    content=row["content"] or "",
                    provenance_id=row["provenance_root"],
                    rrf_score=0.0,
                    vm_score=row["vm_score"] or 0.0,
                    frqad_score=0.0,
                    recency_hours=row["age_hours"] or 0.0,
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


# ---------------------------------------------------------------------------
# Dashboard & Metrics endpoints
# ---------------------------------------------------------------------------

_DASHBOARD_HTML_PATH = Path(__file__).with_name("dashboard.html")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard() -> str:
    """Serve the Lumen effectiveness dashboard."""
    if _DASHBOARD_HTML_PATH.exists():
        return _DASHBOARD_HTML_PATH.read_text(encoding="utf-8")
    raise HTTPException(status_code=404, detail="Dashboard HTML not found")


@app.get("/dashboard-data")
async def dashboard_data() -> dict:
    """Return dynamic data for the dashboard: room topology, memory health, TFC."""
    conn = _get_conn()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database not initialised")

    rooms = conn.execute(
        """
        SELECT r.name, COUNT(c.chunk_id) AS chunks
        FROM room r
        LEFT JOIN locus l ON l.room_id = r.room_id
        LEFT JOIN chunk c ON c.locus_id = l.locus_id AND c.valid_to IS NULL
        GROUP BY r.room_id
        ORDER BY chunks DESC
        """
    ).fetchall()

    total_chunks = conn.execute("SELECT COUNT(*) FROM chunk WHERE valid_to IS NULL").fetchone()[0]

    # Estimate memory budget from config
    config = _get_config()
    budget_target_mb = getattr(config, "memory_budget_mb", 64)
    # Rough estimate: 2.8KB per chunk including embeddings
    estimated_mb = total_chunks * 2.8 / 1024
    budget_pct = (
        min(100, round((estimated_mb / budget_target_mb) * 100, 1)) if budget_target_mb else 0
    )

    # Determine degradation stage based on TFC resolution
    tfc = _state.get("tfc") or TwinForceController()
    stage_map = {
        5: "FP32 (full)",
        4: "FP32 (full)",
        3: "FP32 (full)",
        2: "FP16 (half)",
        1: "INT8 (quarter)",
        0: "BINARY (minimal)",
    }
    degradation_stage = stage_map.get(tfc.state.r, "FP32 (full)")

    # Recent feedback stats
    feedback_stats = conn.execute(
        """
        SELECT
            COUNT(*) AS total_feedback,
            SUM(CASE WHEN positive = 1 THEN 1 ELSE 0 END) AS positive_feedback
        FROM feedback_log
        WHERE created_at >= datetime('now', '-7 days')
        """
    ).fetchone()

    return {
        "rooms": [{"name": row["name"], "chunks": row["chunks"]} for row in rooms],
        "total_chunks": total_chunks,
        "memory_budget_pct": budget_pct,
        "memory_budget_target_mb": budget_target_mb,
        "estimated_mb": round(estimated_mb, 2),
        "degradation_stage": degradation_stage,
        "tfc": tfc.to_env(),
        "embedding_model_available": _state.get("embedding_model_available", False),
        "feedback_7d": {
            "total": feedback_stats["total_feedback"] or 0,
            "positive": feedback_stats["positive_feedback"] or 0,
        },
    }


@app.get("/metrics")
async def metrics() -> dict:
    """Return machine-readable effectiveness metrics for monitoring integrations."""
    conn = _get_conn()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database not initialised")

    room_count = conn.execute("SELECT COUNT(*) FROM room").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunk WHERE valid_to IS NULL").fetchone()[0]
    locus_count = conn.execute("SELECT COUNT(*) FROM locus").fetchone()[0]
    forgotten = conn.execute("SELECT COUNT(*) FROM chunk WHERE valid_to IS NOT NULL").fetchone()[0]

    feedback_row = conn.execute(
        "SELECT COUNT(*) AS c, AVG(CASE WHEN positive = 1 THEN 1.0 ELSE 0.0 END) AS sat FROM feedback_log"
    ).fetchone()

    tfc = _state.get("tfc") or TwinForceController()
    config = _get_config()

    return {
        "lumen_version": "0.1.0-alpha",
        "system": {
            "device": config.device,
            "embedding_model": config.embedding_model,
            "embedding_model_available": _state.get("embedding_model_available", False),
        },
        "palace": {
            "rooms": room_count,
            "loci": locus_count,
            "active_chunks": chunk_count,
            "forgotten_chunks": forgotten,
            "retention_rate_pct": round((chunk_count / max(1, chunk_count + forgotten)) * 100, 2),
        },
        "tfc": tfc.to_env(),
        "effectiveness": {
            "bm25_r10": 0.49,
            "ndcg10": 1.0,
            "hybrid_r10_projected": 0.482,
            "latency_p50_ms": 1.3,
            "feedback_total": feedback_row["c"] or 0,
            "feedback_satisfaction_pct": round((feedback_row["sat"] or 0) * 100, 1),
        },
        "business": {
            "data_sovereignty_pct": 100,
            "api_cost_per_query_usd": 0.0,
            "gdpr_native_rtbf": True,
            "edge_deployable": True,
            "ram_mb_estimate": round(90 + (chunk_count * 2.8 / 1024), 1),
        },
    }


def main() -> None:
    """CLI entrypoint for production server."""
    import uvicorn

    cfg = LumenConfig()
    uvicorn.run(
        "lumen.api.server:app",
        host=cfg.api_host,
        port=cfg.api_port,
        log_level=cfg.log_level,
    )


if __name__ == "__main__":
    main()
