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


# ── Caption endpoint (R4 — multimodal image captioning) ───────────────────────

class CaptionRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image bytes.")
    image_ext: str = Field(default="png", description="Image file extension (png, jpg, etc.).")
    caption_prompt: str = Field(
        default="Describe this image concisely for a RAG retrieval system. "
                "Focus on text, diagrams, tables, charts, or key visual elements. "
                "Be specific and factual.",
        description="Prompt sent to the multimodal LLM.",
    )
    provider: str = Field(default=LLMProvider.AZURE_OPENAI.value)
    max_tokens: int = Field(default=256, ge=1, le=1024)
    doc_id: str = Field(default="")
    page_number: int | None = Field(default=None)
    image_index: int = Field(default=0)


class CaptionResponse(BaseModel):
    caption: str
    provider: str
    model: str
    doc_id: str
    page_number: int | None = None
    image_index: int = 0
    captioned: bool = True


@router.post("/caption", response_model=CaptionResponse)
async def caption_image(body: CaptionRequest, request: Request) -> CaptionResponse:
    """
    Caption an image using a multimodal LLM.

    Accepts a base64-encoded image and returns a text caption.
    Used by PDFImageChunker when image_handling='caption' or 'both'.

    Supported providers: azure_openai (GPT-4V), anthropic (claude-3-*).
    Other providers return a graceful fallback description.
    """
    providers: dict = getattr(request.app.state, "providers", {})

    provider = providers.get(body.provider)
    if provider is None:
        raise HTTPException(
            status_code=503,
            detail=f"Provider '{body.provider}' not available.",
        )

    try:
        caption = provider.caption_image(
            image_b64=body.image_b64,
            image_ext=body.image_ext,
            prompt=body.caption_prompt,
            max_tokens=body.max_tokens,
        )
        return CaptionResponse(
            caption=caption,
            provider=body.provider,
            model=provider._model_name(),
            doc_id=body.doc_id,
            page_number=body.page_number,
            image_index=body.image_index,
            captioned=True,
        )
    except NotImplementedFeatureError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Caption error: {exc}")
