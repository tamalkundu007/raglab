"""
retrieval-service — vector retrieval execution.

Lifespan: initialises Qdrant client stored in app.state.

Endpoints:
  GET  /health      — liveness + qdrant status
  GET  /            — service info
  POST /retrieve    — execute retrieval
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from raglab_common.logging import configure_logging, get_logger
from raglab_common.models import HealthModel
from retrieval.routers.retrieve import router as retrieve_router
from retrieval.settings import RetrievalSettings

settings = RetrievalSettings()
configure_logging(level=settings.log_level, json_logs=settings.json_logs)
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.settings = settings
    app.state.qdrant_client = None
    try:
        from qdrant_client import QdrantClient
        app.state.qdrant_client = QdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port
        )
        log.info("qdrant.ready", host=settings.qdrant_host)
    except Exception as exc:
        log.warning("qdrant.unavailable", reason=str(exc))
    log.info("service.started", service="retrieval", port=settings.port)
    yield
    log.info("service.shutdown", service="retrieval")


app = FastAPI(
    title="raglab-retrieval",
    description="Retriever execution — dense, BM25, hybrid, MMR, re-ranker.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.include_router(retrieve_router)


@app.get("/health", response_model=HealthModel)
async def health() -> HealthModel:
    qdrant_ok = getattr(app.state, "qdrant_client", None) is not None
    deps = {"qdrant": "ok" if qdrant_ok else "unavailable"}
    return HealthModel(service="retrieval", status="ok" if qdrant_ok else "degraded", dependencies=deps)


@app.get("/")
async def root() -> dict:
    return {"service": "retrieval", "version": "0.1.0", "release": "R1", "docs": "/docs"}
