"""Embedding-service HTTP router."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from raglab_common.exceptions import EmbeddingError, NotImplementedFeatureError
from raglab_common.models import LLMProvider

router = APIRouter(prefix="/embed", tags=["embedding"])


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1)
    provider: str = Field(default=LLMProvider.AZURE_OPENAI.value)


class EmbedResponse(BaseModel):
    text_preview: str
    vector: list[float]
    dimensions: int
    provider: str


class EmbedBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=128)
    provider: str = Field(default=LLMProvider.AZURE_OPENAI.value)


class EmbedBatchResponse(BaseModel):
    count: int
    vectors: list[list[float]]
    dimensions: int
    provider: str


def _get_embedder(provider: str, request: Request):
    embedders: dict = getattr(request.app.state, "embedders", {})
    embedder = embedders.get(provider)
    if embedder is None:
        raise HTTPException(status_code=503, detail=f"Embedder for provider '{provider}' not available.")
    return embedder


@router.post("", response_model=EmbedResponse)
async def embed_text(body: EmbedRequest, request: Request) -> EmbedResponse:
    embedder = _get_embedder(body.provider, request)
    try:
        vector = embedder.embed(body.text)
    except NotImplementedFeatureError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return EmbedResponse(text_preview=body.text[:100], vector=vector, dimensions=len(vector), provider=body.provider)


@router.post("/batch", response_model=EmbedBatchResponse)
async def embed_batch(body: EmbedBatchRequest, request: Request) -> EmbedBatchResponse:
    embedder = _get_embedder(body.provider, request)
    try:
        vectors = embedder.embed_batch(body.texts)
    except NotImplementedFeatureError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    dims = len(vectors[0]) if vectors else 0
    return EmbedBatchResponse(count=len(vectors), vectors=vectors, dimensions=dims, provider=body.provider)
