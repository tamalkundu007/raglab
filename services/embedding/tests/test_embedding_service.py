"""
Tests for the embedding-service.

Covers:
- EmbeddingSettings defaults
- AzureOpenAIEmbedder, OpenAIEmbedder, OllamaEmbedder (all mocked)
- get_embedder factory routing
- /health and / endpoints
- /embed and /embed/batch endpoints
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from fastapi.testclient import TestClient

from raglab_common.exceptions import EmbeddingError, NotImplementedFeatureError
from raglab_common.models import LLMProvider
from embedding.embedder import (
    AzureOpenAIEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    VertexEmbedder,
    get_embedder,
)
from embedding.settings import EmbeddingSettings


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestEmbeddingSettings:
    def test_defaults(self):
        s = EmbeddingSettings()
        assert s.service_name == "embedding"
        assert s.port == 8002
        assert s.embedding_dimensions == 1536
        assert s.embedding_batch_size == 32


# ---------------------------------------------------------------------------
# AzureOpenAIEmbedder
# ---------------------------------------------------------------------------


class TestAzureOpenAIEmbedder:
    def _make(self):
        with patch("embedding.embedder.AzureOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            embedder = AzureOpenAIEmbedder(
                api_key="test-key",
                endpoint="https://test.openai.azure.com",
                deployment="text-embedding-3-small",
            )
            return embedder, mock_client

    def test_embed_calls_client(self):
        embedder, client = self._make()
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        client.embeddings.create.return_value = mock_resp
        result = embedder.embed("hello world")
        assert result == [0.1, 0.2, 0.3]

    def test_embed_empty_text_raises(self):
        embedder, _ = self._make()
        with pytest.raises(EmbeddingError):
            embedder.embed("")

    def test_embed_client_error_raises_embedding_error(self):
        embedder, client = self._make()
        client.embeddings.create.side_effect = Exception("API error")
        with pytest.raises(EmbeddingError, match="Azure OpenAI embedding failed"):
            embedder.embed("some text")

    def test_embed_batch_sorted_by_index(self):
        embedder, client = self._make()
        # Return results in reverse order to test sorting
        mock_resp = MagicMock()
        mock_resp.data = [
            MagicMock(index=1, embedding=[0.4, 0.5]),
            MagicMock(index=0, embedding=[0.1, 0.2]),
        ]
        client.embeddings.create.return_value = mock_resp
        result = embedder.embed_batch(["first", "second"])
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.4, 0.5]

    def test_embed_batch_empty_returns_empty(self):
        embedder, _ = self._make()
        assert embedder.embed_batch([]) == []


# ---------------------------------------------------------------------------
# OpenAIEmbedder
# ---------------------------------------------------------------------------


class TestOpenAIEmbedder:
    def _make(self):
        with patch("embedding.embedder.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            embedder = OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-small")
            return embedder, mock_client

    def test_embed_returns_vector(self):
        embedder, client = self._make()
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=[0.9, 0.8, 0.7])]
        client.embeddings.create.return_value = mock_resp
        assert embedder.embed("test") == [0.9, 0.8, 0.7]

    def test_embed_empty_raises(self):
        embedder, _ = self._make()
        with pytest.raises(EmbeddingError):
            embedder.embed("   ")


# ---------------------------------------------------------------------------
# OllamaEmbedder
# ---------------------------------------------------------------------------


class TestOllamaEmbedder:
    def _make(self):
        with patch("embedding.embedder.httpx") as mock_httpx:
            mock_client = MagicMock()
            mock_httpx.Client.return_value = mock_client
            embedder = OllamaEmbedder(
                base_url="http://localhost:11434",
                model="nomic-embed-text",
            )
            embedder._client = mock_client
            return embedder, mock_client

    def test_embed_returns_vector(self):
        embedder, client = self._make()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        client.post.return_value = mock_resp
        assert embedder.embed("ollama test") == [0.1, 0.2, 0.3]

    def test_embed_empty_raises(self):
        embedder, _ = self._make()
        with pytest.raises(EmbeddingError):
            embedder.embed("")

    def test_embed_connection_error_raises_embedding_error(self):
        embedder, client = self._make()
        client.post.side_effect = Exception("connection refused")
        with pytest.raises(EmbeddingError, match="Ollama embedding failed"):
            embedder.embed("test")


# ---------------------------------------------------------------------------
# VertexEmbedder stub
# ---------------------------------------------------------------------------


class TestVertexEmbedderStub:
    def test_instantiation_raises_not_implemented(self):
        with pytest.raises(NotImplementedFeatureError):
            VertexEmbedder()


# ---------------------------------------------------------------------------
# get_embedder factory
# ---------------------------------------------------------------------------


class TestGetEmbedder:
    def _settings(self, **overrides):
        s = EmbeddingSettings()
        for k, v in overrides.items():
            object.__setattr__(s, k, v)
        return s

    def test_missing_azure_key_raises(self):
        s = self._settings(azure_openai_api_key="", azure_openai_endpoint="")
        with pytest.raises(EmbeddingError, match="RAGLAB_AZURE_OPENAI_API_KEY"):
            get_embedder("azure_openai", s)

    def test_missing_azure_endpoint_raises(self):
        s = self._settings(azure_openai_api_key="key", azure_openai_endpoint="")
        with pytest.raises(EmbeddingError, match="RAGLAB_AZURE_OPENAI_ENDPOINT"):
            get_embedder("azure_openai", s)

    def test_missing_openai_key_raises(self):
        s = self._settings(openai_api_key="")
        with pytest.raises(EmbeddingError, match="RAGLAB_OPENAI_API_KEY"):
            get_embedder("openai", s)

    def test_vertex_raises_not_implemented(self):
        s = self._settings()
        with pytest.raises(NotImplementedFeatureError):
            get_embedder("vertex", s)

    def test_unknown_provider_raises(self):
        s = self._settings()
        with pytest.raises(EmbeddingError, match="Unknown embedding provider"):
            get_embedder("gpt5_embedder", s)

    def test_ollama_returns_embedder(self):
        with patch("embedding.embedder.httpx"):
            s = self._settings(ollama_base_url="http://localhost:11434")
            result = get_embedder("ollama", s)
            assert isinstance(result, OllamaEmbedder)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def client_with_mock_embedder():
    """TestClient with a pre-loaded mock embedder in app.state."""
    from embedding.main import app

    mock_embedder = MagicMock()
    mock_embedder.embed.return_value = [0.1] * 1536
    mock_embedder.embed_batch.return_value = [[0.1] * 1536, [0.2] * 1536]

    app.state.embedders = {"azure_openai": mock_embedder}
    return TestClient(app)


class TestEmbeddingEndpoints:
    def test_health_ok(self, client_with_mock_embedder):
        r = client_with_mock_embedder.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_root_ok(self, client_with_mock_embedder):
        r = client_with_mock_embedder.get("/")
        assert r.status_code == 200
        assert r.json()["service"] == "embedding"

    def test_embed_returns_vector(self, client_with_mock_embedder):
        r = client_with_mock_embedder.post(
            "/embed",
            json={"text": "What is RAG?", "provider": "azure_openai"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["dimensions"] == 1536
        assert len(body["vector"]) == 1536

    def test_embed_batch_returns_vectors(self, client_with_mock_embedder):
        r = client_with_mock_embedder.post(
            "/embed/batch",
            json={"texts": ["text one", "text two"], "provider": "azure_openai"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 2
        assert len(body["vectors"]) == 2

    def test_embed_missing_provider_returns_503(self, client_with_mock_embedder):
        r = client_with_mock_embedder.post(
            "/embed",
            json={"text": "test", "provider": "openai"},
        )
        assert r.status_code == 503
