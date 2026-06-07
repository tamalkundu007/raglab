"""
Observability traces router — R6.

Endpoints:
  GET /obs/traces                    — list recent traces
  GET /obs/traces/{trace_id}         — get all spans for a trace (JSON)
  GET /obs/traces/{trace_id}/timeline — D3-ready timeline structure
  GET /obs/services/stats            — per-service span counts and error rates
  GET /obs/viewer                    — serve trace viewer HTML page
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from raglab_common.logging import get_logger
from observability.db.queries import (
    get_service_stats,
    get_trace,
    get_trace_timeline,
    list_recent_traces,
)

log = get_logger(__name__)
router = APIRouter(prefix="/obs", tags=["observability"])


async def get_session(request: Request) -> AsyncSession:
    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        raise HTTPException(status_code=503, detail="Database not initialised.")
    async with session_factory() as session:
        yield session


# ── Trace list ─────────────────────────────────────────────────────────────────

@router.get("/traces")
async def list_traces(
    limit: int = 50,
    service: str | None = None,
    status: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """List recent traces grouped by trace_id."""
    return await list_recent_traces(
        session, limit=limit, service_name=service, status=status
    )


@router.get("/traces/{trace_id}")
async def get_trace_spans(
    trace_id: str,
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """Return all spans for a trace_id."""
    spans = await get_trace(session, trace_id)
    if not spans:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found.")
    return spans


@router.get("/traces/{trace_id}/timeline")
async def get_timeline(
    trace_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Return D3-ready timeline structure for a trace."""
    timeline = await get_trace_timeline(session, trace_id)
    if not timeline["spans"]:
        raise HTTPException(status_code=404, detail=f"Trace '{trace_id}' not found.")
    return timeline


# ── Service stats ──────────────────────────────────────────────────────────────

@router.get("/services/stats")
async def service_stats(
    hours: int = 24,
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """Per-service span counts, error rates, and latency over the last N hours."""
    return await get_service_stats(session, hours=hours)


# ── Viewer page ────────────────────────────────────────────────────────────────

@router.get("/viewer", response_class=HTMLResponse)
async def trace_viewer(request: Request) -> HTMLResponse:
    """Serve the D3.js distributed trace viewer page."""
    from fastapi.templating import Jinja2Templates
    import os
    templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
    templates = Jinja2Templates(directory=os.path.abspath(templates_dir))
    settings = getattr(request.app.state, "settings", None)
    return templates.TemplateResponse(
        request=request,
        name="trace_viewer.html",
        context={
            "control_panel_url": "/",
            "obs_api_base": "/obs",
        },
    )
