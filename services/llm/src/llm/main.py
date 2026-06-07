"""
llm-service — multi-provider LLM generation for RAGLab.

Lifespan: loads all configured providers at startup, skips on missing creds.

Endpoints:
  GET  /health      — liveness + loaded provider list
  GET  /            — service info
  POST /generate    — RAG generation
  GET  /providers   — available providers
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from raglab_common.logging import configure_logging, get_logger
from raglab_common.tracing import configure_tracing, make_trace_middleware
from raglab_common.models import HealthModel, LLMProvider
from llm.providers import get_llm_provider
from llm.routers.generate import router as generate_router
from llm.settings import LLMSettings

settings = LLMSettings()
configure_logging(level=settings.log_level, json_logs=settings.json_logs)
log = get_logger(__name__)

ACTIVE_PROVIDERS = [
    LLMProvider.AZURE_OPENAI,
    LLMProvider.OPENAI,
    LLMProvider.ANTHROPIC,
    LLMProvider.OLLAMA,
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.settings = settings
    app.state.providers = {}
    for provider in ACTIVE_PROVIDERS:
        try:
            p = get_llm_provider(provider, settings)
            app.state.providers[provider.value] = p
            log.info("llm.provider_loaded", provider=provider.value)
        except Exception as exc:
            log.warning("llm.provider_skipped", provider=provider.value, reason=str(exc))
    log.info("service.started", service="llm", port=settings.port,
             loaded=list(app.state.providers.keys()))
    # ── Tracing (R6) ──────────────────────────────────────────────────────────
    configure_tracing(
        service_name="llm",
        postgres_dsn=getattr(settings, "tracing_postgres_dsn", "") if settings else "",
        enabled=getattr(settings, "tracing_enabled", True) if settings else True,
    )

    yield
    log.info("service.shutdown", service="llm")


app = FastAPI(
    title="raglab-llm",
    description="Multi-provider LLM generation for RAGLab.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(make_trace_middleware("llm"))

app.include_router(generate_router)


@app.get("/health", response_model=HealthModel)
async def health() -> HealthModel:
    loaded = getattr(app.state, "providers", {})
    return HealthModel(
        service="llm",
        status="ok" if loaded else "degraded",
        dependencies={p: "ok" for p in loaded},
    )


@app.get("/")
async def root() -> dict:
    return {"service": "llm", "version": "0.1.0", "release": "R1", "docs": "/docs"}
