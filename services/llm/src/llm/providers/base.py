"""
BaseLLMProvider — abstract interface for all RAGLab LLM providers.

Every provider must implement `generate()` which takes a prompt and
context chunks and returns a generated string.

The RAG prompt assembly (system prompt + context + question) is done
in the base class so every provider gets consistent formatting.
All provider-specific logic lives in `_call_api()`.
"""

from __future__ import annotations

import abc
import time
from typing import Any

from raglab_common.exceptions import LLMError
from raglab_common.logging import get_logger
from raglab_common.models import ChunkModel, ResponseModel

log = get_logger(__name__)

# Maximum context chars to include (prevents token overflow)
_MAX_CONTEXT_CHARS = 12_000


class BaseLLMProvider(abc.ABC):
    """Abstract base class for RAGLab LLM providers."""

    provider: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._log = get_logger(self.__class__.__name__)

    def generate(
        self,
        query: str,
        chunks: list[ChunkModel],
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ResponseModel:
        """
        Assemble RAG prompt from chunks and generate a response.

        Args:
            query:         User question.
            chunks:        Retrieved context chunks (ordered by relevance).
            system_prompt: System instruction for the LLM.
            max_tokens:    Maximum tokens to generate.
            temperature:   Sampling temperature (0.0 = deterministic).

        Returns:
            ResponseModel with answer, sources, latency, and model name.
        """
        context = self._build_context(chunks)
        prompt = self._build_prompt(query, context)

        self._log.info(
            "llm.generate",
            provider=self.provider,
            query_len=len(query),
            context_chunks=len(chunks),
        )

        start = time.monotonic()
        try:
            answer = self._call_api(
                system_prompt=system_prompt,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"{self.provider} generation failed: {exc}") from exc

        latency_ms = (time.monotonic() - start) * 1000
        self._log.info("llm.done", provider=self.provider, latency_ms=round(latency_ms, 1))

        return ResponseModel(
            query_id="",          # filled in by router from QueryModel
            answer=answer,
            sources=chunks,
            model=self._model_name(),
            latency_ms=latency_ms,
        )

    @abc.abstractmethod
    def _call_api(
        self,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Provider-specific API call. Returns the generated text string."""

    @abc.abstractmethod
    def _model_name(self) -> str:
        """Return the model identifier string for ResponseModel.model."""

    @staticmethod
    def _build_context(chunks: list[ChunkModel]) -> str:
        """
        Concatenate chunk texts into a numbered context block.

        Truncates at _MAX_CONTEXT_CHARS to stay within token budgets.
        Chunk index is 1-based for human readability in the prompt.
        """
        parts = []
        total = 0
        for i, chunk in enumerate(chunks, start=1):
            entry = f"[{i}] {chunk.text}"
            if total + len(entry) > _MAX_CONTEXT_CHARS:
                break
            parts.append(entry)
            total += len(entry)
        return "\n\n".join(parts)

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        """Assemble the user-turn prompt from query and context."""
        return (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

    def caption_image(
        self,
        image_b64: str,
        image_ext: str = "png",
        prompt: str = "Describe this image concisely for a RAG retrieval system.",
        max_tokens: int = 256,
    ) -> str:
        """
        Caption an image using multimodal LLM capabilities.

        Default implementation returns a graceful fallback for providers that
        don't support vision. Providers with vision support (Azure OpenAI GPT-4V,
        Anthropic claude-3-*) override this method.

        Args:
            image_b64: Base64-encoded image bytes.
            image_ext: Image file extension (png, jpg, etc.).
            prompt:    Instruction for the vision model.
            max_tokens: Maximum tokens for the caption.

        Returns:
            Caption string. Falls back to a placeholder if vision not supported.
        """
        # Default: providers without vision support return a descriptive placeholder
        return f"[Image — captioning not supported by provider '{self.provider}'. " \
               f"Use azure_openai or anthropic for multimodal captioning.]"
