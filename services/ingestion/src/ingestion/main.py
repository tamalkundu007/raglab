"""
ingestion-service — document intake, idempotency, and async queue publishing.

Lifespan:
  - Connects RabbitMQPublisher (declares exchange + queues)
  - Initialises Postgres session factory (for duplicate detection + status tracking)

Endpoints:
  GET  /health          — liveness + dependency status
  GET  /                — service info
  POST /ingest          — submit document for async ingestion
  GET  /ingest/{doc_id} — check ingestion status
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from raglab_common.db import close_db, create_tables, init_db, _session_factory
from raglab_common.tracing import configure_tracing, make_trace_middleware
from raglab_common.logging import configure_logging, get_logger
from raglab_common.models import HealthModel
from ingestion.queue.publisher import RabbitMQPublisher
from ingestion.routers.ingest import router as ingest_router
from ingestion.settings import IngestionSettings

settings = IngestionSettings()
configure_logging(level=settings.log_level, json_logs=settings.json_logs)
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # RabbitMQ publisher
    app.state.publisher = None
    publisher = RabbitMQPublisher(
        url=settings.rabbitmq_url,
        confirm_delivery=settings.rabbitmq_confirm_delivery,
    )
    try:
        await publisher.connect()
        app.state.publisher = publisher
        log.info("rabbitmq.ready")
    except Exception as exc:
        log.warning("rabbitmq.unavailable", reason=str(exc))

    # Postgres
    app.state.session_factory = None
    try:
        await init_db(settings.postgres_dsn)
        await create_tables()
        from raglab_common.db import _session_factory as sf
        app.state.session_factory = sf
        log.info("postgres.ready")
    except Exception as exc:
        log.warning("postgres.unavailable", reason=str(exc))

    log.info("service.started", service="ingestion", port=settings.port)
    # ── Tracing (R6) ──────────────────────────────────────────────────────────
    configure_tracing(
        service_name="ingestion",
        postgres_dsn=getattr(settings, "tracing_postgres_dsn", "") if settings else "",
        enabled=getattr(settings, "tracing_enabled", True) if settings else True,
    )

    yield

    if app.state.publisher:
        await app.state.publisher.close()
    await close_db()
    log.info("service.shutdown", service="ingestion")


app = FastAPI(
    title="raglab-ingestion",
    description="Document intake, idempotency, and async queue publishing.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(make_trace_middleware("ingestion"))

app.include_router(ingest_router)


@app.get("/health", response_model=HealthModel)
async def health() -> HealthModel:
    deps = {
        "rabbitmq": "ok" if getattr(app.state, "publisher", None) and app.state.publisher.is_connected else "unavailable",
        "postgres": "ok" if getattr(app.state, "session_factory", None) is not None else "unavailable",
    }
    status = "ok" if all(v == "ok" for v in deps.values()) else "degraded"
    return HealthModel(service="ingestion", status=status, dependencies=deps)


@app.get("/")
async def root() -> dict:
    return {"service": "ingestion", "version": "0.1.0", "release": "R1", "docs": "/docs"}
