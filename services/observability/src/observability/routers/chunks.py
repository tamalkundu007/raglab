"""
Chunk inspector router — R6 Phase 3.

Endpoints:
  GET /obs/chunks/docs              — list recently indexed documents
  GET /obs/chunks/{doc_id}          — all chunks for a document
  GET /obs/chunks/{doc_id}/summary  — quality gate summary for a document
  GET /obs/inspector                — serve chunk inspector HTML page
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from raglab_common.logging import get_logger
from observability.db.chunk_queries import (
    get_chunks_for_doc,
    get_doc_quality_summary,
    list_recent_docs,
)
from observability.routers.traces import get_session

log = get_logger(__name__)
router = APIRouter(prefix="/obs", tags=["observability"])


@router.get("/chunks/docs")
async def list_docs(
    collection: str = "raglab",
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """List recently indexed documents."""
    return await list_recent_docs(session, collection=collection, limit=limit)


@router.get("/chunks/{doc_id}")
async def get_chunks(
    doc_id: str,
    collection: str = "raglab",
    include_excluded: bool = True,
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """Return all chunks for a document with quality scores."""
    chunks = await get_chunks_for_doc(
        session, doc_id=doc_id, collection=collection,
        include_excluded=include_excluded,
    )
    if not chunks:
        raise HTTPException(status_code=404, detail=f"No chunks found for doc '{doc_id}'.")
    return chunks


@router.get("/chunks/{doc_id}/summary")
async def get_quality_summary(
    doc_id: str,
    collection: str = "raglab",
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Return quality gate summary for a document."""
    summary = await get_doc_quality_summary(session, doc_id=doc_id, collection=collection)
    if not summary:
        raise HTTPException(status_code=404, detail=f"No quality data for doc '{doc_id}'.")
    return summary


@router.get("/inspector", response_class=HTMLResponse)
async def chunk_inspector(request: Request) -> HTMLResponse:
    """Serve the chunk inspector HTML page."""
    import os
    from fastapi.templating import Jinja2Templates
    templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
    templates = Jinja2Templates(directory=os.path.abspath(templates_dir))
    return templates.TemplateResponse(
        request=request,
        name="chunk_inspector.html",
        context={"control_panel_url": "/", "obs_api_base": "/obs"},
    )
