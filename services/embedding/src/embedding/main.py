"""
embedding-service — vector embedding generation via configurable model providers.

Lifespan:
  - Initialises all configured embedders at startup (one per available API key).
  - Embedders stored in app.state.embedders dict keyed by provider string.

Endpoints:
  GET  /health         — liveness
  GET  /               — service info
  POST /embed          — embed single text
  POST /embed/batch    — embed list of texts
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from raglab_common.logging import configure_logging, get_logger
from raglab_common.models import HealthModel, LLMProvider
from embedding.embedder import get_embedder
from embedding.routers.embed import router as embed_router
from embedding.settings import EmbeddingSettings

settings = EmbeddingSettings()
configure_logging(level=settings.log_level, json_logs=settings.json_logs)
log = get_logger(__name__)

ACTIVE_PROVIDERS = [
    LLMProvider.AZURE_OPENAI,
    LLMProvider.OPENAI,
    LLMProvider.OLLAMA,
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise embedders; dispose on shutdown."""
    app.state.embedders = {}
    for provider in ACTIVE_PROVIDERS:
        try:
            embedder = get_embedder(provider, settings)
            app.state.embedders[provider.value] = embedder
            log.info("embedder.loaded", provider=provider.value)
        except Exception as exc:
            log.warning(
                "embedder.skipped",
                provider=provider.value,
                reason=str(exc),
            )
    log.info(
        "service.started",
        service="embedding",
        port=settings.port,
        loaded_providers=list(app.state.embedders.keys()),
    )
    yield
    log.info("service.shutdown", service="embedding")


app = FastAPI(
    title="raglab-embedding",
    description="Vector embedding generation via configurable model providers.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.include_router(embed_router)


@app.get("/health", response_model=HealthModel)
async def health() -> HealthModel:
    """Liveness check. Reports which embedders are loaded."""
    loaded = list(getattr(app.state, "embedders", {}).keys())
    return HealthModel(
        service="embedding",
        dependencies={p: "ok" for p in loaded},
    )


@app.get("/")
async def root() -> dict:
    return {
        "service": "embedding",
        "version": "0.1.0",
        "release": "R1",
        "docs": "/docs",
        "providers_loaded": list(getattr(app.state, "embedders", {}).keys()),
    }
