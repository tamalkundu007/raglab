"""
llm — LLM provider abstraction — Azure OpenAI, OpenAI, Anthropic, Ollama.

Exposes:
  GET  /health  — liveness + dependency status
  GET  /        — service info

Full implementation: see service-specific routers (added per phase).
"""

from fastapi import FastAPI
from raglab_common.logging import configure_logging, get_logger
from raglab_common.models import HealthModel
from raglab_common.settings import BaseServiceSettings

settings = BaseServiceSettings()
configure_logging(level=settings.log_level, json_logs=settings.json_logs)
log = get_logger(__name__)

app = FastAPI(
    title="raglab-llm",
    description="LLM provider abstraction — Azure OpenAI, OpenAI, Anthropic, Ollama",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.on_event("startup")  # noqa: deprecated
async def _startup() -> None:
    log.info("service.started", service="llm", port=8005)


@app.get("/health", response_model=HealthModel)
async def health() -> HealthModel:
    """Liveness check. Extended dependency checks added per phase."""
    return HealthModel(service="llm")


@app.get("/")
async def root() -> dict:
    return {
        "service": "llm",
        "version": "0.1.0",
        "release": "R1",
        "docs": "/docs",
    }
