"""
graph-service — GraphRAG knowledge graph construction and retrieval.

Activated in R4. Builds and queries a knowledge graph from indexed chunks.

Endpoints:
  GET  /health              — liveness + DB connectivity
  GET  /                    — service info
  POST /graph/extract       — extract entities + relationships from chunks
  GET  /graph/entities      — list entities
  GET  /graph/relationships — list relationships
  GET  /graph/stats         — counts and type breakdown
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from raglab_common.logging import configure_logging, get_logger
from raglab_common.tracing import configure_tracing, make_trace_middleware
from raglab_common.models import HealthModel
from graph.routers.extract import router as extract_router
from graph.routers.build import router as build_router
from graph.settings import GraphSettings

settings = GraphSettings()
configure_logging(level=settings.log_level, json_logs=settings.json_logs)
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.settings = settings
    app.state.session_factory = None  # Set by infra when Postgres is available

    # Attempt DB connection
    try:
        from raglab_common.database import engine, AsyncSessionLocal
        app.state.session_factory = AsyncSessionLocal

        # Create tables if they don't exist (dev only — prod uses migrations)
        from graph.models.orm import Base
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.info("service.started", service="graph", port=settings.port, db="connected")
    except Exception as exc:
        log.warning(
            "service.db_unavailable",
            service="graph",
            reason=str(exc),
        )
        log.info("service.started", service="graph", port=settings.port, db="unavailable")

    # ── Tracing (R6) ──────────────────────────────────────────────────────────
    configure_tracing(
        service_name="graph",
        postgres_dsn=getattr(settings, "tracing_postgres_dsn", "") if settings else "",
        enabled=getattr(settings, "tracing_enabled", True) if settings else True,
    )

    yield
    log.info("service.shutdown", service="graph")


app = FastAPI(
    title="raglab-graph",
    description="GraphRAG — knowledge graph construction and retrieval.",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(make_trace_middleware("graph"))

app.include_router(extract_router)
app.include_router(build_router)


@app.get("/health", response_model=HealthModel)
async def health() -> HealthModel:
    db_status = "connected" if app.state.session_factory else "unavailable"
    return HealthModel(
        service="graph",
        status="ok",
        dependencies={"database": db_status},
    )


@app.get("/")
async def root() -> dict:
    return {
        "service": "graph",
        "version": "0.2.0",
        "release": "R4",
        "docs": "/docs",
        "endpoints": [
            "POST /graph/extract",
            "GET  /graph/entities",
            "GET  /graph/relationships",
            "GET  /graph/stats",
        ],
    }
