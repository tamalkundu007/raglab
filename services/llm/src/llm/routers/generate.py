"""
LLM-service HTTP router.

Endpoints:
  POST /generate    — RAG generation: chunks + query → answer
  GET  /providers   — list available (loaded) providers
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from raglab_common.exceptions import LLMError, NotImplementedFeatureError
from raglab_common.logging import get_logger
from raglab_common.models import ChunkModel, LLMProvider, ResponseModel

log = get_logger(__name__)
router = APIRouter(tags=["llm"])


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question.")
    query_id: str = Field(default="", description="QueryModel ID for tracing.")
    chunks: list[ChunkModel] = Field(..., description="Retrieved context chunks.")
    provider: str = Field(default=LLMProvider.AZURE_OPENAI.value)
    system_prompt: str = Field(default="")
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class ProviderInfo(BaseModel):
    provider: str
    active: bool
    model: str | None = None


@router.post("/generate", response_model=ResponseModel)
async def generate(body: GenerateRequest, request: Request) -> ResponseModel:
    """
    Generate a RAG response from context chunks and a query.

    Assembles the RAG prompt (system + context + question), calls the
    selected LLM provider, and returns a ResponseModel with answer,
    sources, model name, and latency.
    """
    providers: dict = getattr(request.app.state, "providers", {})
    provider = providers.get(body.provider)

    if provider is None:
        raise HTTPException(
            status_code=503,
            detail=f"Provider '{body.provider}' not available. Check API key configuration.",
        )

    settings = getattr(request.app.state, "settings", None)
    system_prompt = body.system_prompt or (settings.rag_system_prompt if settings else "")

    try:
        response = provider.generate(
            query=body.query,
            chunks=body.chunks,
            system_prompt=system_prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
        )
        # Inject the query_id from the request
        response.query_id = body.query_id
        return response
    except NotImplementedFeatureError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")


@router.get("/providers", response_model=list[ProviderInfo])
async def list_providers(request: Request) -> list[ProviderInfo]:
    """List all LLM providers and their load status."""
    providers: dict = getattr(request.app.state, "providers", {})
    all_providers = [p.value for p in LLMProvider]
    return [
        ProviderInfo(
            provider=p,
            active=p in providers,
            model=providers[p]._model_name() if p in providers else None,
        )
        for p in all_providers
    ]
