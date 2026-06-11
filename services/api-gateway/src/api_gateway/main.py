"""
api-gateway — single entry point and health-aware router for RAGLab.

Lifespan:
  - Initialises HealthRegistry with all downstream service URLs.
  - Starts background health polling task.
  - Performs one immediate health poll before accepting traffic.

Endpoints:
  GET  /health                  — gateway liveness + aggregate service status
  GET  /                        — gateway info + quick service summary
  GET  /api/v1/health/services  — detailed per-service health snapshot
  POST /api/v1/ingest           — proxy → ingestion-service
  GET  /api/v1/ingest/{doc_id}  — proxy → ingestion-service
  POST /api/v1/retrieve         — proxy → retrieval-service
  POST /api/v1/generate         — proxy → llm-service
  GET  /api/v1/providers        — proxy → llm-service
  POST /api/v1/pipeline/run     — proxy → pipeline-service
  GET  /api/v1/pipeline/status  — proxy → pipeline-service
  GET  /api/v1/collections/{n}  — proxy → indexing-service
  POST /api/v1/collections/{n}/ensure → indexing-service
  DELETE /api/v1/collections/{n}     → indexing-service
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from raglab_common.logging import configure_logging, get_logger
from raglab_common.tracing import configure_tracing, make_trace_middleware
from raglab_common.models import HealthModel
from api_gateway.health_registry import DOWNSTREAM_SERVICES, HealthRegistry
from api_gateway.routers.gateway import router as gateway_router
from api_gateway.settings import GatewaySettings
# R7: JWT validation middleware — auth-service
try:
    from auth.middleware.jwt_validator import JWTValidatorMiddleware
    _AUTH_AVAILABLE = True
except ImportError:
    _AUTH_AVAILABLE = False


settings = GatewaySettings()
configure_logging(level=settings.log_level, json_logs=settings.json_logs)
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.settings = settings

    # Build URL map from settings (allows env-var override in production)
    url_map = {
        "ingestion":     settings.ingestion_url,
        "embedding":     settings.embedding_url,
        "indexing":      settings.indexing_url,
        "retrieval":     settings.retrieval_url,
        "llm":           settings.llm_url,
        "pipeline":      settings.pipeline_url,
        "config":        settings.config_url,
        "storage":       settings.storage_url,
        "ui":            settings.ui_url,
        "graph":         "http://graph:8010",
        "observability": "http://observability:8011",
        "auth":          "http://auth:8012",
    }

    registry = HealthRegistry(
        timeout=settings.health_check_timeout,
        ttl=settings.health_cache_ttl,
    )
    registry.configure_urls(url_map)
    app.state.registry = registry

    # Immediate first poll (best-effort — don't block startup on failure)
    try:
        await registry._poll_all()
        log.info("health_registry.initial_poll_done")
    except Exception as exc:
        log.warning("health_registry.initial_poll_failed", reason=str(exc))

    # Background polling task
    poll_task = asyncio.create_task(registry.run())
    log.info("service.started", service="api-gateway", port=settings.port)

    # ── Tracing (R6) ──────────────────────────────────────────────────────────
    configure_tracing(
        service_name="api-gateway",
        postgres_dsn=getattr(settings, "tracing_postgres_dsn", "") if settings else "",
        enabled=getattr(settings, "tracing_enabled", True) if settings else True,
    )

    yield

    poll_task.cancel()
    try:
        await poll_task
    except asyncio.CancelledError:
        pass
    log.info("service.shutdown", service="api-gateway")


app = FastAPI(
    title="RAGLab API Gateway",
    description="Single entry point and health-aware router for all RAGLab services.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(make_trace_middleware("api-gateway"))

# R7: JWT validation (bypass_auth=True when no providers configured — dev mode)
if _AUTH_AVAILABLE:
    _bypass = not bool(getattr(settings, "auth_enabled", False))
    app.add_middleware(
        JWTValidatorMiddleware,
        providers={},        # populated in lifespan once auth-service loads providers
        bypass_auth=_bypass,
    )

app.include_router(gateway_router)


@app.get("/health", response_model=HealthModel)
async def health() -> HealthModel:
    """Gateway liveness with aggregate downstream status."""
    registry = getattr(app.state, "registry", None)
    agg = registry.aggregate_status() if registry else "unknown"
    deps = {}
    if registry:
        for svc in registry.all_statuses():
            deps[svc["name"]] = svc["status"]
    return HealthModel(service="api-gateway", status=agg, dependencies=deps)


@app.get("/")
async def root() -> dict:
    registry = getattr(app.state, "registry", None)
    summary = {}
    if registry:
        for svc in registry.all_statuses():
            summary[svc["name"]] = svc["status"]
    return {
        "service": "api-gateway",
        "version": "0.1.0",
        "release": "R1",
        "docs": "/docs",
        "api_base": "/api/v1",
        "downstream": summary,
    }
