"""
Retrieval scorer router — R6 Phase 4.

Endpoints:
  GET /obs/retrieval/queries         — list recent retrieval queries
  GET /obs/retrieval/queries/{tid}   — detail for one query trace
  GET /obs/retrieval/distribution    — score distribution buckets
  GET /obs/retrieval/healing         — healing stats summary
  GET /obs/retrieval/scorer          — serve retrieval scorer HTML
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from raglab_common.logging import get_logger
from observability.db.retrieval_queries import (
    get_healing_stats,
    get_query_detail,
    get_score_distribution,
    list_recent_queries,
)
from observability.routers.traces import get_session

log = get_logger(__name__)
router = APIRouter(prefix="/obs", tags=["observability"])


@router.get("/retrieval/queries")
async def list_queries(
    limit: int = 50,
    collection: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    return await list_recent_queries(session, limit=limit, collection=collection)


@router.get("/retrieval/queries/{trace_id}")
async def get_query(
    trace_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    detail = await get_query_detail(session, trace_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"No retrieval data for trace '{trace_id}'.")
    return detail


@router.get("/retrieval/distribution")
async def score_distribution(
    hours: int = 24,
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    return await get_score_distribution(session, hours=hours)


@router.get("/retrieval/healing")
async def healing_stats(
    hours: int = 24,
    session: AsyncSession = Depends(get_session),
) -> dict:
    stats = await get_healing_stats(session, hours=hours)
    if not stats:
        return {"total_queries": 0, "healed_count": 0, "avg_top_score": None}
    return stats


@router.get("/retrieval/scorer", response_class=HTMLResponse)
async def retrieval_scorer_page(request: Request) -> HTMLResponse:
    import os
    from fastapi.templating import Jinja2Templates
    templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
    templates = Jinja2Templates(directory=os.path.abspath(templates_dir))
    return templates.TemplateResponse(
        request=request,
        name="retrieval_scorer.html",
        context={"control_panel_url": "/", "obs_api_base": "/obs"},
    )
