"""
Multi-provider embedding abstraction for the embedding-service.

All providers expose the same interface: embed(text) -> list[float].
Imports are at module level so they can be patched in tests.

Active providers in R1: AzureOpenAIEmbedder, OpenAIEmbedder, OllamaEmbedder
Stub (R2+): VertexEmbedder
"""

from __future__ import annotations

import abc
from typing import Any

import httpx

from raglab_common.exceptions import EmbeddingError, NotImplementedFeatureError
from raglab_common.logging import get_logger
from raglab_common.models import LLMProvider

# Optional — graceful import for environments without openai installed
try:
    from openai import AzureOpenAI, OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    AzureOpenAI = None  # type: ignore[assignment,misc]
    OpenAI = None       # type: ignore[assignment,misc]
    _OPENAI_AVAILABLE = False

log = get_logger(__name__)


class BaseEmbedder(abc.ABC):
    provider: str = ""

    @abc.abstractmethod
    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class AzureOpenAIEmbedder(BaseEmbedder):
    provider = "azure_openai"

    def __init__(self, api_key: str, endpoint: str, deployment: str) -> None:
        if not _OPENAI_AVAILABLE or AzureOpenAI is None:
            raise EmbeddingError("openai package not installed.")
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2024-02-01",
        )
        self._deployment = deployment
        log.info("embedder.init", provider=self.provider, deployment=deployment)

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text.")
        try:
            response = self._client.embeddings.create(input=[text], model=self._deployment)
            return response.data[0].embedding
        except Exception as exc:
            raise EmbeddingError(f"Azure OpenAI embedding failed: {exc}") from exc

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = self._client.embeddings.create(input=texts, model=self._deployment)
            return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        except Exception as exc:
            raise EmbeddingError(f"Azure OpenAI batch embedding failed: {exc}") from exc


class OpenAIEmbedder(BaseEmbedder):
    provider = "openai"

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        if not _OPENAI_AVAILABLE or OpenAI is None:
            raise EmbeddingError("openai package not installed.")
        self._client = OpenAI(api_key=api_key)
        self._model = model
        log.info("embedder.init", provider=self.provider, model=model)

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text.")
        try:
            response = self._client.embeddings.create(input=[text], model=self._model)
            return response.data[0].embedding
        except Exception as exc:
            raise EmbeddingError(f"OpenAI embedding failed: {exc}") from exc


class OllamaEmbedder(BaseEmbedder):
    provider = "ollama"

    def __init__(self, base_url: str, model: str = "nomic-embed-text") -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.Client(timeout=30.0)
        log.info("embedder.init", provider=self.provider, model=model)

    def embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text.")
        try:
            response = self._client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as exc:
            raise EmbeddingError(f"Ollama embedding failed: {exc}") from exc


class VertexEmbedder(BaseEmbedder):
    provider = "vertex"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedFeatureError("VertexEmbedder", available_in="R2")

    def embed(self, text: str) -> list[float]:
        raise NotImplementedFeatureError("VertexEmbedder", available_in="R2")


def get_embedder(provider: str | LLMProvider, settings: Any) -> BaseEmbedder:
    key = provider.value if isinstance(provider, LLMProvider) else str(provider)

    if key == LLMProvider.AZURE_OPENAI.value:
        if not settings.azure_openai_api_key:
            raise EmbeddingError("RAGLAB_AZURE_OPENAI_API_KEY not set.")
        if not settings.azure_openai_endpoint:
            raise EmbeddingError("RAGLAB_AZURE_OPENAI_ENDPOINT not set.")
        deployment = getattr(settings, "azure_openai_embedding_deployment", "") or settings.azure_openai_deployment
        if not deployment:
            raise EmbeddingError("RAGLAB_AZURE_OPENAI_EMBEDDING_DEPLOYMENT not set.")
        return AzureOpenAIEmbedder(api_key=settings.azure_openai_api_key, endpoint=settings.azure_openai_endpoint, deployment=deployment)

    if key == LLMProvider.OPENAI.value:
        if not settings.openai_api_key:
            raise EmbeddingError("RAGLAB_OPENAI_API_KEY not set.")
        return OpenAIEmbedder(api_key=settings.openai_api_key, model=getattr(settings, "openai_embedding_model", "text-embedding-3-small"))

    if key == LLMProvider.OLLAMA.value:
        return OllamaEmbedder(base_url=settings.ollama_base_url, model=getattr(settings, "ollama_embedding_model", "nomic-embed-text"))

    if key == LLMProvider.VERTEX.value:
        raise NotImplementedFeatureError("VertexEmbedder", available_in="R2")

    raise EmbeddingError(f"Unknown embedding provider: {key!r}")
