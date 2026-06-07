"""
indexing-service — Qdrant vector indexing and PostgreSQL metadata persistence.

Lifespan:
  - Creates QdrantIndexer (connect to Qdrant, verify reachability)
  - Initialises async Postgres engine via raglab_common.db
  - Ensures default collection exists at startup

Endpoints:
  GET  /health                          — liveness + dependency status
  GET  /                                — service info
  POST /index                           — index pre-embedded chunks
  GET  /collections/{name}              — collection stats
  POST /collections/{name}/ensure       — idempotent collection creation
  DELETE /collections/{name}            — delete collection
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from raglab_common.db import close_db, create_tables, init_db
from raglab_common.tracing import configure_tracing, make_trace_middleware
from raglab_common.logging import configure_logging, get_logger
from raglab_common.models import HealthModel
from indexing.qdrant_client import QdrantIndexer
from indexing.routers.index import router as index_router
from indexing.settings import IndexingSettings

settings = IndexingSettings()
configure_logging(level=settings.log_level, json_logs=settings.json_logs)
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Init Qdrant + Postgres; dispose on shutdown."""
    # Qdrant
    app.state.qdrant = None
    try:
        app.state.qdrant = QdrantIndexer(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            vector_size=settings.qdrant_vector_size,
            distance=settings.qdrant_distance,
            hnsw_m=settings.qdrant_hnsw_m,
            hnsw_ef_construct=settings.qdrant_hnsw_ef_construct,
            on_disk_payload=settings.qdrant_on_disk_payload,
        )
        app.state.qdrant.ensure_collection(settings.qdrant_default_collection)
        log.info("qdrant.ready", collection=settings.qdrant_default_collection)
    except Exception as exc:
        log.warning("qdrant.unavailable", reason=str(exc))

    # Postgres
    app.state.session_factory = None
    try:
        from raglab_common.db import _session_factory
        await init_db(settings.postgres_dsn)
        await create_tables()
        from raglab_common.db import _session_factory as sf
        app.state.session_factory = sf
        log.info("postgres.ready")
    except Exception as exc:
        log.warning("postgres.unavailable", reason=str(exc))

    log.info("service.started", service="indexing", port=settings.port)
    # ── Tracing (R6) ──────────────────────────────────────────────────────────
    configure_tracing(
        service_name="indexing",
        postgres_dsn=getattr(settings, "tracing_postgres_dsn", "") if settings else "",
        enabled=getattr(settings, "tracing_enabled", True) if settings else True,
    )

    yield

    await close_db()
    log.info("service.shutdown", service="indexing")


app = FastAPI(
    title="raglab-indexing",
    description="Qdrant vector indexing and PostgreSQL metadata persistence.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(make_trace_middleware("indexing"))

app.include_router(index_router)


@app.get("/health", response_model=HealthModel)
async def health() -> HealthModel:
    """Liveness + dependency status."""
    deps: dict[str, str] = {}
    deps["qdrant"] = "ok" if getattr(app.state, "qdrant", None) is not None else "unavailable"
    deps["postgres"] = "ok" if getattr(app.state, "session_factory", None) is not None else "unavailable"
    status = "ok" if all(v == "ok" for v in deps.values()) else "degraded"
    return HealthModel(service="indexing", status=status, dependencies=deps)


@app.get("/")
async def root() -> dict:
    return {
        "service": "indexing",
        "version": "0.1.0",
        "release": "R1",
        "docs": "/docs",
    }
