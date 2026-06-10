"""
observability-service — R6 activated.

Exposes read-only views over raglab_events (written by tracing.py in raglab-common).
Principle: observes, never mutates.

Endpoints:
  GET /health              — liveness + DB status
  GET /                    — service info
  GET /obs/traces          — list recent traces
  GET /obs/traces/{id}     — all spans for a trace
  GET /obs/traces/{id}/timeline — D3-ready waterfall structure
  GET /obs/services/stats  — per-service latency + error rates
  GET /obs/viewer          — D3.js trace viewer HTML page
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from raglab_common.logging import configure_logging, get_logger
from raglab_common.models import HealthModel
from raglab_common.tracing import configure_tracing, make_trace_middleware
from observability.routers.traces import router as traces_router
from observability.routers.chunks import router as chunks_router
from observability.routers.retrieval import router as retrieval_router
from observability.routers.cost import router as cost_router
from observability.routers.health import router as health_router

log = get_logger(__name__)


class ObservabilitySettings:
    service_name: str = "observability"
    port: int = 8011
    log_level: str = "info"
    json_logs: bool = False
    tracing_enabled: bool = True
    tracing_postgres_dsn: str = ""
    database_url: str = ""


settings = ObservabilitySettings()
configure_logging(level=settings.log_level, json_logs=settings.json_logs)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.settings = settings
    app.state.session_factory = None

    configure_tracing(
        service_name="observability",
        postgres_dsn=settings.tracing_postgres_dsn,
        enabled=settings.tracing_enabled,
    )

    try:
        from raglab_common.database import AsyncSessionLocal
        app.state.session_factory = AsyncSessionLocal
        log.info("service.started", service="observability", db="connected")
    except Exception as exc:
        log.warning("service.db_unavailable", service="observability", reason=str(exc))
        log.info("service.started", service="observability", db="unavailable")

    yield
    log.info("service.shutdown", service="observability")


app = FastAPI(
    title="raglab-observability",
    description="RAGLab observability — distributed tracing, cost dashboard, inspector",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(make_trace_middleware("observability"))
app.include_router(traces_router)
app.include_router(chunks_router)
app.include_router(retrieval_router)
app.include_router(cost_router)
app.include_router(health_router)


@app.get("/health", response_model=HealthModel)
async def health() -> HealthModel:
    db_status = "connected" if app.state.session_factory else "unavailable"
    return HealthModel(
        service="observability",
        status="ok",
        dependencies={"database": db_status},
    )


@app.get("/")
async def root() -> dict:
    return {
        "service": "observability",
        "version": "0.2.0",
        "release": "R6",
        "docs": "/docs",
        "endpoints": [
            "GET /obs/traces",
            "GET /obs/traces/{trace_id}",
            "GET /obs/traces/{trace_id}/timeline",
            "GET /obs/services/stats",
            "GET /obs/viewer",
            "GET /obs/chunks/docs",
            "GET /obs/chunks/{doc_id}",
            "GET /obs/inspector",
        ],
    }
