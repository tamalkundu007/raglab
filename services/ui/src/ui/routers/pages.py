"""UI-service page router — serves the Control Panel HTML."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter(tags=["ui"])
templates = Jinja2Templates(
    directory=str(Path(__file__).parent.parent / "templates")
)


def _ctx(request: Request) -> dict:
    """Build shared template context from app settings."""
    settings = getattr(request.app.state, "settings", None)
    return {
        "gateway_url": getattr(settings, "gateway_url", "http://localhost:8000"),
        "api_base": getattr(settings, "api_base", "/api/v1"),
        "app_title": getattr(settings, "app_title", "RAGLab"),
        "app_version": getattr(settings, "app_version", "R1"),
        "control_panel_url": "/",
        "comparison_url": "/compare",
    }


@router.get("/", response_class=HTMLResponse)
async def control_panel(request: Request) -> HTMLResponse:
    """Serve the RAGLab Control Panel."""
    return templates.TemplateResponse(
        request=request,
        name="control_panel.html",
        context=_ctx(request),
    )


@router.get("/compare", response_class=HTMLResponse)
async def comparison(request: Request) -> HTMLResponse:
    """Serve the Retrieval Comparison page (R3)."""
    return templates.TemplateResponse(
        request=request,
        name="comparison.html",
        context=_ctx(request),
    )
