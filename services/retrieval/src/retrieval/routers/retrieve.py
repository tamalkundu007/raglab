"""
Retrieval-service HTTP router.

Endpoints:
  POST /retrieve    — execute vector retrieval for a query
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from raglab_common.exceptions import RetrieverError
from raglab_common.logging import get_logger
from raglab_common.models import ChunkModel, LLMProvider, QueryModel, RetrieverType

log = get_logger(__name__)
router = APIRouter(tags=["retrieval"])


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection: str = Field(default="raglab")
    top_k: int = Field(default=5, ge=1, le=50)
    retriever_type: str = Field(default=RetrieverType.DENSE.value)
    llm_provider: str = Field(default=LLMProvider.AZURE_OPENAI.value)
    metadata_filter: dict[str, Any] = Field(default_factory=dict)
    retriever_config: dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    query: str
    collection: str
    retriever_type: str
    results: list[ChunkModel]
    result_count: int


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(body: RetrieveRequest, request: Request) -> RetrieveResponse:
    """
    Execute vector retrieval for a query.

    Flow:
      1. Embed query via embedding-service HTTP call.
      2. Run DenseRetriever against Qdrant (wired in app.state).
      3. Return ranked ChunkModel list.
    """
    qdrant_client = getattr(request.app.state, "qdrant_client", None)
    settings = getattr(request.app.state, "settings", None)

    if qdrant_client is None:
        raise HTTPException(status_code=503, detail="Vector store not available.")

    # Embed the query
    embedding_url = settings.embedding_url if settings else "http://embedding:8002"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{embedding_url}/embed",
                json={"text": body.query, "provider": body.llm_provider},
            )
            resp.raise_for_status()
            query_vector: list[float] = resp.json()["vector"]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Embedding-service call failed: {exc}")

    # Build QueryModel and run retriever
    from raglab_retrievers import RetrieverFactory
    query_model = QueryModel(
        text=body.query,
        collection=body.collection,
        top_k=body.top_k,
        retriever_type=RetrieverType(body.retriever_type),
        llm_provider=LLMProvider(body.llm_provider),
        metadata_filter=body.metadata_filter,
    )

    try:
        retriever = RetrieverFactory.create(
            body.retriever_type,
            config=body.retriever_config,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Provide embedder as a simple callable wrapping the pre-computed vector
    def _cached_embedder(text: str) -> list[float]:
        return query_vector

    results = retriever.retrieve(query_model, qdrant_client, embedder=_cached_embedder)

    return RetrieveResponse(
        query=body.query,
        collection=body.collection,
        retriever_type=body.retriever_type,
        results=results,
        result_count=len(results),
    )
