"""Pipeline health + self-healing trace router — R6 Phase 6."""
from __future__ import annotations
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from raglab_common.logging import get_logger
from observability.db.health_queries import (
    get_pipeline_health, get_failed_jobs, get_heal_gate_summary,
)
from observability.routers.traces import get_session

log = get_logger(__name__)
router = APIRouter(prefix="/obs", tags=["observability"])


@router.get("/health/pipeline")
async def pipeline_health(hours: int = 24, session: AsyncSession = Depends(get_session)) -> dict:
    h = await get_pipeline_health(session, hours=hours)
    if not h:
        return {"total_jobs": 0, "successful": 0, "failed": 0, "avg_duration_ms": None}
    return h


@router.get("/health/failed-jobs")
async def failed_jobs(limit: int = 20, session: AsyncSession = Depends(get_session)) -> list[dict]:
    return await get_failed_jobs(session, limit=limit)


@router.get("/health/gates")
async def gate_summary(hours: int = 24, session: AsyncSession = Depends(get_session)) -> list[dict]:
    return await get_heal_gate_summary(session, hours=hours)


@router.get("/health/dashboard", response_class=HTMLResponse)
async def health_dashboard_page(request: Request) -> HTMLResponse:
    import os
    from fastapi.templating import Jinja2Templates
    templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
    templates = Jinja2Templates(directory=os.path.abspath(templates_dir))
    return templates.TemplateResponse(
        request=request, name="pipeline_health.html",
        context={"control_panel_url": "/", "obs_api_base": "/obs"},
    )
