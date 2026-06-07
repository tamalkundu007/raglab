"""
Embedding-service HTTP router.

R5: Cache-aware. Single and batch embed endpoints check EmbeddingCache
before calling the provider. Cache stats exposed at GET /embed/cache/stats.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from raglab_common.exceptions import EmbeddingError, NotImplementedFeatureError
from raglab_common.logging import get_logger
from raglab_common.models import LLMProvider

log = get_logger(__name__)
router = APIRouter(prefix="/embed", tags=["embedding"])


# ── Request / Response schemas ─────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1)
    provider: str = Field(default=LLMProvider.AZURE_OPENAI.value)


class EmbedResponse(BaseModel):
    text_preview: str
    vector: list[float]
    dimensions: int
    provider: str
    cache_hit: bool = False


class EmbedBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=128)
    provider: str = Field(default=LLMProvider.AZURE_OPENAI.value)


class EmbedBatchResponse(BaseModel):
    count: int
    vectors: list[list[float]]
    dimensions: int
    provider: str
    cache_hits: int = 0
    cache_misses: int = 0


class CacheStatsResponse(BaseModel):
    enabled: bool
    connected: bool
    hits: int
    misses: int
    errors: int
    total_requests: int
    hit_rate: float
    hit_rate_pct: float
    ttl_seconds: int
    redis_url: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_embedder(provider: str, request: Request):
    embedders: dict = getattr(request.app.state, "embedders", {})
    embedder = embedders.get(provider)
    if embedder is None:
        raise HTTPException(
            status_code=503,
            detail=f"Embedder for provider '{provider}' not available.",
        )
    return embedder


def _get_cache(request: Request):
    """Return the EmbeddingCache from app.state, or None if not configured."""
    return getattr(request.app.state, "embedding_cache", None)


def _model_name(embedder) -> str:
    """Extract model name from embedder for cache key scoping."""
    try:
        return embedder.model_name
    except AttributeError:
        try:
            return embedder._model or "unknown"
        except AttributeError:
            return "unknown"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("", response_model=EmbedResponse)
async def embed_text(body: EmbedRequest, request: Request) -> EmbedResponse:
    """
    Embed a single text string.

    R5: Checks Redis cache before calling the provider.
    Returns cache_hit=True if the vector was served from cache.
    """
    embedder = _get_embedder(body.provider, request)
    cache = _get_cache(request)
    model = _model_name(embedder)

    # Cache lookup
    if cache is not None:
        cached = cache.get(body.text, body.provider, model)
        if cached is not None:
            log.info("embed.cache_hit", provider=body.provider, model=model)
            return EmbedResponse(
                text_preview=body.text[:100],
                vector=cached,
                dimensions=len(cached),
                provider=body.provider,
                cache_hit=True,
            )

    # Provider call
    try:
        vector = embedder.embed(body.text)
    except NotImplementedFeatureError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # Cache write-through
    if cache is not None:
        cache.set(body.text, body.provider, model, vector)

    log.info("embed.computed", provider=body.provider, model=model, dims=len(vector))
    return EmbedResponse(
        text_preview=body.text[:100],
        vector=vector,
        dimensions=len(vector),
        provider=body.provider,
        cache_hit=False,
    )


@router.post("/batch", response_model=EmbedBatchResponse)
async def embed_batch(body: EmbedBatchRequest, request: Request) -> EmbedBatchResponse:
    """
    Embed a batch of texts.

    R5: Each text is checked individually in cache.
    Only uncached texts are sent to the provider (minimises API cost).
    """
    embedder = _get_embedder(body.provider, request)
    cache = _get_cache(request)
    model = _model_name(embedder)

    vectors: list[list[float]] = [[] for _ in body.texts]
    uncached_indices: list[int] = []
    hits = 0
    misses = 0

    # Phase 1: cache lookup for each text
    for i, text in enumerate(body.texts):
        if cache is not None:
            cached = cache.get(text, body.provider, model)
            if cached is not None:
                vectors[i] = cached
                hits += 1
                continue
        uncached_indices.append(i)
        misses += 1

    # Phase 2: batch-embed only uncached texts
    if uncached_indices:
        uncached_texts = [body.texts[i] for i in uncached_indices]
        try:
            new_vectors = embedder.embed_batch(uncached_texts)
        except NotImplementedFeatureError as exc:
            raise HTTPException(status_code=501, detail=str(exc))
        except EmbeddingError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        for idx, vec in zip(uncached_indices, new_vectors):
            vectors[idx] = vec
            if cache is not None:
                cache.set(body.texts[idx], body.provider, model, vec)

    log.info(
        "embed_batch.complete",
        count=len(body.texts),
        cache_hits=hits,
        cache_misses=misses,
        provider=body.provider,
    )

    dims = len(vectors[0]) if vectors else 0
    return EmbedBatchResponse(
        count=len(vectors),
        vectors=vectors,
        dimensions=dims,
        provider=body.provider,
        cache_hits=hits,
        cache_misses=misses,
    )


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(request: Request) -> CacheStatsResponse:
    """
    Return embedding cache statistics.

    hit_rate_pct is the key ROI metric:
    "80% cache hit rate = 80% fewer embedding API calls on re-ingestion."
    """
    cache = _get_cache(request)
    if cache is None:
        return CacheStatsResponse(
            enabled=False, connected=False,
            hits=0, misses=0, errors=0, total_requests=0,
            hit_rate=0.0, hit_rate_pct=0.0,
            ttl_seconds=0, redis_url="not configured",
        )
    return CacheStatsResponse(**cache.stats())


@router.delete("/cache/flush")
async def flush_cache(request: Request) -> dict:
    """
    Flush all cached embedding vectors.
    Dev/test only — use with care in production.
    """
    cache = _get_cache(request)
    if cache is None:
        return {"deleted": 0, "message": "Cache not configured."}
    deleted = cache.flush()
    cache.reset_stats()
    log.info("embed.cache_flushed", deleted=deleted)
    return {"deleted": deleted}
