"""
storage-service — file storage backend abstraction.

Lifespan: initialises the configured storage backend (local | s3 | azure_blob).
All credentials come from env vars via pydantic-settings.

Endpoints:
  GET  /health                          — liveness + active backend
  GET  /                                — service info
  POST /storage/upload/{key}            — upload bytes
  GET  /storage/download/{key}          — download bytes
  DELETE /storage/{key}                 — delete object
  GET  /storage/exists/{key}            — existence check
  GET  /storage/backends                — list available backends
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from raglab_common.logging import configure_logging, get_logger
from raglab_common.tracing import configure_tracing, make_trace_middleware
from raglab_common.models import HealthModel
from storage.factory import StorageFactory
from storage.routers.storage import router as storage_router
from storage.settings import StorageSettings

settings = StorageSettings()
configure_logging(level=settings.log_level, json_logs=settings.json_logs)
log = get_logger(__name__)


def _build_backend_config(s: StorageSettings) -> dict:
    provider = s.storage_provider
    if provider == "local":
        return {"root": s.local_storage_root}
    if provider == "s3":
        cfg: dict = {"bucket": s.s3_bucket, "region": s.s3_region, "prefix": s.s3_prefix}
        return cfg
    if provider == "azure_blob":
        return {
            "container": s.azure_blob_container,
            "connection_string": s.azure_storage_connection_string,
            "account_name": s.azure_storage_account_name,
            "account_key": s.azure_storage_account_key,
            "prefix": s.azure_blob_prefix,
        }
    return {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.backend = None
    try:
        cfg = _build_backend_config(settings)
        app.state.backend = StorageFactory.create(settings.storage_provider, config=cfg)
        log.info(
            "service.started",
            service="storage",
            port=settings.port,
            backend=settings.storage_provider,
        )
    except Exception as exc:
        log.warning("storage.backend_init_failed", reason=str(exc))
    # ── Tracing (R6) ──────────────────────────────────────────────────────────
    configure_tracing(
        service_name="storage",
        postgres_dsn=getattr(settings, "tracing_postgres_dsn", "") if settings else "",
        enabled=getattr(settings, "tracing_enabled", True) if settings else True,
    )

    yield
    log.info("service.shutdown", service="storage")


app = FastAPI(
    title="raglab-storage",
    description="File storage backend abstraction — local, S3, Azure Blob.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(make_trace_middleware("storage"))

app.include_router(storage_router)


@app.get("/health", response_model=HealthModel)
async def health() -> HealthModel:
    backend = getattr(app.state, "backend", None)
    provider = backend.backend_type if backend else "none"
    status = "ok" if backend else "degraded"
    return HealthModel(
        service="storage",
        status=status,
        dependencies={"backend": provider},
    )


@app.get("/")
async def root() -> dict:
    backend = getattr(app.state, "backend", None)
    return {
        "service": "storage",
        "version": "0.1.0",
        "release": "R2",
        "docs": "/docs",
        "active_backend": backend.backend_type if backend else None,
    }
