"""Cost dashboard router — R6 Phase 5."""
from __future__ import annotations
from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from raglab_common.logging import get_logger
from observability.db.cost_queries import (
    get_token_summary, get_tokens_by_provider,
    get_daily_token_trend, get_cache_stats_summary,
)
from observability.routers.traces import get_session

log = get_logger(__name__)
router = APIRouter(prefix="/obs", tags=["observability"])


@router.get("/cost/summary")
async def cost_summary(hours: int = 24, session: AsyncSession = Depends(get_session)) -> dict:
    return await get_token_summary(session, hours=hours)


@router.get("/cost/by-provider")
async def cost_by_provider(hours: int = 24, session: AsyncSession = Depends(get_session)) -> list[dict]:
    return await get_tokens_by_provider(session, hours=hours)


@router.get("/cost/trend")
async def cost_trend(days: int = 7, session: AsyncSession = Depends(get_session)) -> list[dict]:
    return await get_daily_token_trend(session, days=days)


@router.get("/cost/cache")
async def cache_stats(hours: int = 24, session: AsyncSession = Depends(get_session)) -> dict:
    s = await get_cache_stats_summary(session, hours=hours)
    if not s:
        return {"total_hits": 0, "total_misses": 0, "avg_hit_rate_pct": 0.0, "total_embed_requests": 0}
    return s


@router.get("/cost/dashboard", response_class=HTMLResponse)
async def cost_dashboard_page(request: Request) -> HTMLResponse:
    import os
    from fastapi.templating import Jinja2Templates
    templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
    templates = Jinja2Templates(directory=os.path.abspath(templates_dir))
    return templates.TemplateResponse(
        request=request, name="cost_dashboard.html",
        context={"control_panel_url": "/", "obs_api_base": "/obs"},
    )
