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
        "graph_url": "/graph",
        "graph_service_url": getattr(settings, "graph_service_url", "http://graph:8010"),
        "healing_trace_url": "/healing-trace",
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


@router.get("/graph", response_class=HTMLResponse)
async def graph_explorer(request: Request) -> HTMLResponse:
    """Serve the Graph Explorer page (R4)."""
    return templates.TemplateResponse(
        request=request,
        name="graph.html",
        context=_ctx(request),
    )


@router.get("/healing-trace", response_class=HTMLResponse)
async def healing_trace(request: Request) -> HTMLResponse:
    """Serve the Self-Healing Trace page (R5)."""
    return templates.TemplateResponse(
        request=request,
        name="healing_trace.html",
        context=_ctx(request),
    )
