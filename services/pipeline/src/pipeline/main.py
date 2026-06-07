"""
pipeline-service — end-to-end RAG pipeline orchestration.

Lifespan:
  - Starts RabbitMQConsumer as background task
  - Initialises Postgres session factory (for idempotency checks + status updates)
  - Stores settings in app.state for pipeline runner access

Endpoints:
  GET  /health           — liveness + consumer status
  GET  /                 — service info
  POST /pipeline/run     — direct pipeline invocation (dev/test)
  GET  /pipeline/status  — consumer health
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from raglab_common.db import close_db, create_tables, init_db
from raglab_common.tracing import configure_tracing, make_trace_middleware
from raglab_common.logging import configure_logging, get_logger
from raglab_common.models import HealthModel
from pipeline.queue.consumer import RabbitMQConsumer
from pipeline.routers.pipeline import router as pipeline_router
from pipeline.runner import run_pipeline
from pipeline.settings import PipelineSettings

settings = PipelineSettings()
configure_logging(level=settings.log_level, json_logs=settings.json_logs)
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.settings = settings
    app.state.consumer = None
    app.state.consumer_running = False
    app.state.session_factory = None

    # Postgres
    try:
        await init_db(settings.postgres_dsn)
        await create_tables()
        from raglab_common.db import _session_factory as sf
        app.state.session_factory = sf
        log.info("postgres.ready")
    except Exception as exc:
        log.warning("postgres.unavailable", reason=str(exc))

    # RabbitMQ consumer — background task
    consumer = RabbitMQConsumer(
        url=settings.rabbitmq_url,
        pipeline_runner=run_pipeline,
        prefetch_count=settings.rabbitmq_prefetch_count,
    )
    app.state.consumer = consumer

    async def _run_consumer():
        try:
            await consumer.start(app.state)
            app.state.consumer_running = True
        except Exception as exc:
            log.warning("consumer.start_failed", reason=str(exc))
        finally:
            app.state.consumer_running = False

    consumer_task = asyncio.create_task(_run_consumer())
    log.info("service.started", service="pipeline", port=settings.port)

    # ── Tracing (R6) ──────────────────────────────────────────────────────────
    configure_tracing(
        service_name="pipeline",
        postgres_dsn=getattr(settings, "tracing_postgres_dsn", "") if settings else "",
        enabled=getattr(settings, "tracing_enabled", True) if settings else True,
    )

    yield

    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass
    await consumer.close()
    await close_db()
    log.info("service.shutdown", service="pipeline")


app = FastAPI(
    title="raglab-pipeline",
    description="End-to-end RAG pipeline orchestration.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(make_trace_middleware("pipeline"))

app.include_router(pipeline_router)


@app.get("/health", response_model=HealthModel)
async def health() -> HealthModel:
    consumer = getattr(app.state, "consumer", None)
    deps = {
        "consumer": "ok" if getattr(app.state, "consumer_running", False) else "starting",
        "postgres": "ok" if getattr(app.state, "session_factory", None) is not None else "unavailable",
    }
    return HealthModel(service="pipeline", status="ok", dependencies=deps)


@app.get("/")
async def root() -> dict:
    return {"service": "pipeline", "version": "0.1.0", "release": "R1", "docs": "/docs"}
