"""
ui-service — RAGLab Control Panel.

Serves the Jinja2 Control Panel template. All API calls from the UI
go to the api-gateway — the ui-service has no business logic.

Endpoints:
  GET  /         — Control Panel HTML
  GET  /health   — liveness
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from raglab_common.logging import configure_logging, get_logger
from raglab_common.models import HealthModel
from ui.routers.pages import router as pages_router
from ui.settings import UISettings

settings = UISettings()
configure_logging(level=settings.log_level, json_logs=settings.json_logs)
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.settings = settings
    log.info("service.started", service="ui", port=settings.port, gateway=settings.gateway_url)
    yield
    log.info("service.shutdown", service="ui")


app = FastAPI(
    title="RAGLab UI",
    description="RAGLab Control Panel — configurable RAG platform.",
    version="0.1.0",
    lifespan=lifespan,
    # Docs disabled — this service only serves the UI
    docs_url=None,
    redoc_url=None,
)

app.include_router(pages_router)


@app.get("/health", response_model=HealthModel)
async def health() -> HealthModel:
    return HealthModel(service="ui", status="ok")
