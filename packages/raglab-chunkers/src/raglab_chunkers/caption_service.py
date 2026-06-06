"""
CaptionService — HTTP client for the llm-service /caption endpoint.

Used by PDFImageChunker when image_handling='caption' or 'both'.
Sends extracted image regions to the llm-service and returns text captions.

Architecture note:
    Captioning is intentionally decoupled from PDFImageChunker — the chunker
    extracts images, CaptionService sends them to the llm-service for vision
    inference. This keeps the chunker layer stateless and the LLM dependency
    isolated to the llm-service.

    In production the pipeline-service calls PDFImageChunker and provides
    the llm_service_url via config. In tests, CaptionService is fully mockable.

Config:
    llm_service_url   : str   = "http://llm:8005"
    caption_provider  : str   = "azure_openai"
    caption_prompt    : str   = (default descriptive prompt)
    caption_max_tokens: int   = 256
    timeout_seconds   : float = 30.0
    on_failure        : str   = "placeholder" | "raise"
        "placeholder" — return a descriptive text on HTTP failure (default)
        "raise"       — raise CaptionError on any failure
"""

from __future__ import annotations

import base64
from typing import Any

from raglab_common.exceptions import ChunkerError
from raglab_common.logging import get_logger
from raglab_common.models import ChunkModel

# Module-level import for test patchability
try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore[assignment]

log = get_logger(__name__)

_DEFAULT_CAPTION_PROMPT = (
    "Describe this image concisely for a RAG retrieval system. "
    "Focus on text, diagrams, tables, charts, or key visual elements. "
    "Be specific and factual. If the image contains text, transcribe it."
)


class CaptionService:
    """
    HTTP client for the llm-service /caption endpoint.

    Sends base64-encoded image chunks to the llm-service and returns
    ChunkModel instances with captions replacing the placeholder text.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.llm_service_url: str = cfg.get("llm_service_url", "http://llm:8005").rstrip("/")
        self.caption_provider: str = cfg.get("caption_provider", "azure_openai")
        self.caption_prompt: str = cfg.get("caption_prompt", _DEFAULT_CAPTION_PROMPT)
        self.caption_max_tokens: int = int(cfg.get("caption_max_tokens", 256))
        self.timeout_seconds: float = float(cfg.get("timeout_seconds", 30.0))
        self.on_failure: str = cfg.get("on_failure", "placeholder")

        if self.on_failure not in ("placeholder", "raise"):
            raise ValueError(
                f"on_failure must be 'placeholder' or 'raise', got {self.on_failure!r}"
            )

    def caption_chunks(self, image_chunks: list[ChunkModel]) -> list[ChunkModel]:
        """
        Send image chunks to llm-service for captioning.

        Args:
            image_chunks: ChunkModel list where chunk_type='image' and
                          metadata['image_bytes'] contains base64 PNG data.

        Returns:
            New ChunkModel list with caption text and captioned=True in metadata.
            Chunks without image_bytes are returned unchanged.
        """
        result = []
        for chunk in image_chunks:
            if chunk.metadata.get("chunk_type") != "image" or "image_bytes" not in chunk.metadata:
                result.append(chunk)
                continue

            captioned = self._caption_one(chunk)
            result.append(captioned)

        return result

    def _caption_one(self, chunk: ChunkModel) -> ChunkModel:
        """Send one image chunk to /caption and return updated ChunkModel."""
        try:
            caption = self._call_caption_api(
                image_b64=chunk.metadata["image_bytes"],
                image_ext=chunk.metadata.get("image_ext", "png"),
                doc_id=chunk.doc_id,
                page_number=chunk.metadata.get("page_number"),
                image_index=chunk.metadata.get("image_index", 0),
            )
        except Exception as exc:
            log.warning(
                "caption_service.request_failed",
                chunk_id=chunk.chunk_id,
                error=str(exc),
            )
            if self.on_failure == "raise":
                raise ChunkerError(f"Caption request failed: {exc}") from exc
            # Placeholder fallback
            caption = (
                f"[Image captioning failed — {self.caption_provider} unavailable. "
                f"Page {chunk.metadata.get('page_number', '?')}, "
                f"region {chunk.metadata.get('image_index', 0) + 1}.]"
            )

        # Return new ChunkModel with caption text
        return ChunkModel(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            text=caption,
            chunk_index=chunk.chunk_index,
            token_count=len(caption.split()),
            metadata={
                **chunk.metadata,
                "captioned": True,
                "caption_provider": self.caption_provider,
            },
        )

    def _call_caption_api(
        self,
        image_b64: str,
        image_ext: str,
        doc_id: str,
        page_number: int | None,
        image_index: int,
    ) -> str:
        """
        POST to llm-service /caption endpoint.

        Uses the standard `requests` library (synchronous) because
        PDFImageChunker runs in the pipeline-service worker loop
        which is already async — we call this synchronously and let
        the worker thread handle I/O.
        """
        if _requests is None:
            raise ChunkerError("requests not installed. Run: pip install requests")

        payload = {
            "image_b64": image_b64,
            "image_ext": image_ext,
            "caption_prompt": self.caption_prompt,
            "provider": self.caption_provider,
            "max_tokens": self.caption_max_tokens,
            "doc_id": doc_id,
            "page_number": page_number,
            "image_index": image_index,
        }

        url = f"{self.llm_service_url}/caption"
        resp = _requests.post(url, json=payload, timeout=self.timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
        return data.get("caption", "[No caption returned]")
