"""
Tests for the llm-service.

Covers:
- BaseLLMProvider: context building, prompt building, context truncation
- AzureOpenAIProvider, OpenAIProvider, AnthropicProvider, OllamaProvider (all mocked)
- VertexProvider stub
- get_llm_provider factory: routing, missing creds, unknown provider
- /generate endpoint: success, missing provider, LLMError
- /providers endpoint: lists all with active flag
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from raglab_common.exceptions import LLMError, NotImplementedFeatureError
from raglab_common.models import ChunkModel, LLMProvider
from llm.providers.base import BaseLLMProvider, _MAX_CONTEXT_CHARS
from llm.settings import LLMSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_chunk(text: str = "Sample chunk text about RAG.", index: int = 0) -> ChunkModel:
    return ChunkModel(
        chunk_id=str(uuid.uuid4()),
        doc_id="doc-001",
        text=text,
        chunk_index=index,
        token_count=len(text.split()),
    )


def make_settings(**overrides) -> LLMSettings:
    s = LLMSettings()
    for k, v in overrides.items():
        object.__setattr__(s, k, v)
    return s


# ---------------------------------------------------------------------------
# BaseLLMProvider helpers (via concrete stub)
# ---------------------------------------------------------------------------


class _StubProvider(BaseLLMProvider):
    provider = "stub"
    def __init__(self, answer: str = "42"):
        super().__init__()
        self._answer = answer
    def _call_api(self, system_prompt, prompt, max_tokens, temperature):
        return self._answer
    def _model_name(self):
        return "stub/test"


class TestBaseLLMProvider:
    def test_generate_returns_response_model(self):
        p = _StubProvider(answer="This is the answer.")
        chunks = [make_chunk("Context about RAG systems.")]
        resp = p.generate("What is RAG?", chunks)
        assert resp.answer == "This is the answer."
        assert resp.model == "stub/test"
        assert resp.latency_ms >= 0

    def test_sources_propagated(self):
        p = _StubProvider()
        chunks = [make_chunk(f"Chunk {i}.") for i in range(3)]
        resp = p.generate("q", chunks)
        assert len(resp.sources) == 3

    def test_empty_chunks_still_generates(self):
        p = _StubProvider(answer="No context available.")
        resp = p.generate("q", [])
        assert "No context" in resp.answer

    def test_context_truncated_at_max_chars(self):
        long_text = "x" * (_MAX_CONTEXT_CHARS + 1000)
        chunks = [make_chunk(long_text)]
        context = BaseLLMProvider._build_context(chunks)
        assert len(context) <= _MAX_CONTEXT_CHARS + 200  # allow numbering overhead

    def test_multiple_chunks_numbered(self):
        chunks = [make_chunk(f"Chunk text {i}.") for i in range(3)]
        context = BaseLLMProvider._build_context(chunks)
        assert "[1]" in context
        assert "[2]" in context
        assert "[3]" in context

    def test_prompt_contains_query_and_context(self):
        context = "Some context here."
        prompt = BaseLLMProvider._build_prompt("What is RAG?", context)
        assert "What is RAG?" in prompt
        assert "Some context here." in prompt

    def test_api_exception_wrapped_as_llm_error(self):
        class _BrokenProvider(_StubProvider):
            def _call_api(self, *args, **kwargs):
                raise ValueError("API down")
        p = _BrokenProvider()
        with pytest.raises(LLMError, match="generation failed"):
            p.generate("q", [])


# ---------------------------------------------------------------------------
# AzureOpenAIProvider
# ---------------------------------------------------------------------------


class TestAzureOpenAIProvider:
    def _make(self, answer: str = "Azure answer"):
        with patch("llm.providers.azure_openai.AzureOpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            from llm.providers.azure_openai import AzureOpenAIProvider
            p = AzureOpenAIProvider("key", "https://ep.azure.com", "gpt-4o")
            # Wire mock response
            mock_choice = MagicMock()
            mock_choice.message.content = answer
            mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
            return p, mock_client

    def test_generate_returns_answer(self):
        p, _ = self._make("Answer from Azure.")
        resp = p.generate("q", [make_chunk()])
        assert resp.answer == "Answer from Azure."

    def test_model_name(self):
        p, _ = self._make()
        assert p._model_name() == "azure_openai/gpt-4o"

    def test_api_failure_raises_llm_error(self):
        p, client = self._make()
        client.chat.completions.create.side_effect = Exception("rate limit")
        with pytest.raises(LLMError):
            p.generate("q", [make_chunk()])


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    def test_generate_returns_answer(self):
        with patch("llm.providers.openai_provider.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            from llm.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider("sk-test", "gpt-4o-mini")
            mock_choice = MagicMock()
            mock_choice.message.content = "OpenAI answer"
            mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
            resp = p.generate("q", [make_chunk()])
            assert resp.answer == "OpenAI answer"

    def test_model_name(self):
        with patch("llm.providers.openai_provider.OpenAI"):
            from llm.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider("sk-test", "gpt-4o-mini")
            assert p._model_name() == "openai/gpt-4o-mini"


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    def test_generate_returns_answer(self):
        with patch("llm.providers.anthropic_provider.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client
            from llm.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider("ant-key", "claude-sonnet-4-20250514")
            mock_block = MagicMock()
            mock_block.text = "Anthropic answer"
            mock_client.messages.create.return_value = MagicMock(content=[mock_block])
            resp = p.generate("q", [make_chunk()])
            assert resp.answer == "Anthropic answer"

    def test_model_name(self):
        with patch("llm.providers.anthropic_provider.anthropic"):
            from llm.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider("key", "claude-sonnet-4-20250514")
            assert "anthropic" in p._model_name()


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------


class TestOllamaProvider:
    def test_generate_returns_answer(self):
        with patch("llm.providers.ollama_provider.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"message": {"content": "Ollama answer"}}
            mock_resp.raise_for_status = MagicMock()
            mock_httpx.post.return_value = mock_resp
            from llm.providers.ollama_provider import OllamaProvider
            p = OllamaProvider("http://localhost:11434", "llama3.2")
            resp = p.generate("q", [make_chunk()])
            assert resp.answer == "Ollama answer"

    def test_model_name(self):
        with patch("llm.providers.ollama_provider.httpx"):
            from llm.providers.ollama_provider import OllamaProvider
            p = OllamaProvider("http://localhost:11434", "llama3.2")
            assert p._model_name() == "ollama/llama3.2"

    def test_api_failure_raises_llm_error(self):
        with patch("llm.providers.ollama_provider.httpx") as mock_httpx:
            mock_httpx.post.side_effect = Exception("connection refused")
            from llm.providers.ollama_provider import OllamaProvider
            p = OllamaProvider("http://localhost:11434", "llama3.2")
            with pytest.raises(LLMError):
                p.generate("q", [make_chunk()])


# ---------------------------------------------------------------------------
# VertexProvider stub
# ---------------------------------------------------------------------------


class TestVertexProviderStub:
    def test_instantiation_raises_not_implemented(self):
        from llm.providers.vertex_provider import VertexProvider
        with pytest.raises(NotImplementedFeatureError):
            VertexProvider()


# ---------------------------------------------------------------------------
# get_llm_provider factory
# ---------------------------------------------------------------------------


class TestGetLLMProvider:
    def test_missing_azure_key_raises(self):
        from llm.providers import get_llm_provider
        s = make_settings(azure_openai_api_key="")
        with pytest.raises(LLMError, match="RAGLAB_AZURE_OPENAI_API_KEY"):
            get_llm_provider("azure_openai", s)

    def test_missing_openai_key_raises(self):
        from llm.providers import get_llm_provider
        s = make_settings(openai_api_key="")
        with pytest.raises(LLMError, match="RAGLAB_OPENAI_API_KEY"):
            get_llm_provider("openai", s)

    def test_missing_anthropic_key_raises(self):
        from llm.providers import get_llm_provider
        s = make_settings(anthropic_api_key="")
        with pytest.raises(LLMError, match="RAGLAB_ANTHROPIC_API_KEY"):
            get_llm_provider("anthropic", s)

    def test_vertex_raises_not_implemented(self):
        from llm.providers import get_llm_provider
        with pytest.raises(NotImplementedFeatureError):
            get_llm_provider("vertex", make_settings())

    def test_unknown_provider_raises_llm_error(self):
        from llm.providers import get_llm_provider
        with pytest.raises(LLMError, match="Unknown LLM provider"):
            get_llm_provider("gpt5", make_settings())

    def test_ollama_returns_provider(self):
        from llm.providers import get_llm_provider
        from llm.providers.ollama_provider import OllamaProvider
        with patch("llm.providers.ollama_provider.httpx"):
            s = make_settings(ollama_base_url="http://localhost:11434", ollama_chat_model="llama3.2")
            p = get_llm_provider("ollama", s)
            assert isinstance(p, OllamaProvider)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def llm_client():
    from llm.main import app
    mock_provider = MagicMock()
    from raglab_common.models import ResponseModel
    import datetime
    mock_provider.generate.return_value = ResponseModel(
        query_id="",
        answer="Mocked answer from provider.",
        sources=[make_chunk()],
        model="mock/test",
        latency_ms=42.0,
    )
    mock_provider._model_name.return_value = "mock/test"
    app.state.providers = {"azure_openai": mock_provider}
    app.state.settings = make_settings()
    return TestClient(app)


class TestLLMEndpoints:
    def test_health_ok(self, llm_client):
        r = llm_client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_root_ok(self, llm_client):
        r = llm_client.get("/")
        assert r.status_code == 200

    def test_generate_success(self, llm_client):
        chunk = make_chunk()
        r = llm_client.post("/generate", json={
            "query": "What is RAG?",
            "chunks": [chunk.model_dump(mode="json")],
            "provider": "azure_openai",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "Mocked answer from provider."
        assert body["model"] == "mock/test"

    def test_generate_missing_provider_returns_503(self, llm_client):
        r = llm_client.post("/generate", json={
            "query": "q",
            "chunks": [],
            "provider": "openai",
        })
        assert r.status_code == 503

    def test_generate_llm_error_returns_502(self, llm_client):
        from llm.main import app
        app.state.providers["azure_openai"].generate.side_effect = LLMError("API down")
        r = llm_client.post("/generate", json={
            "query": "q",
            "chunks": [],
            "provider": "azure_openai",
        })
        assert r.status_code == 502

    def test_providers_endpoint(self, llm_client):
        r = llm_client.get("/providers")
        assert r.status_code == 200
        providers = r.json()
        types = {p["provider"] for p in providers}
        assert "azure_openai" in types
        azure = next(p for p in providers if p["provider"] == "azure_openai")
        assert azure["active"] is True
        openai = next(p for p in providers if p["provider"] == "openai")
        assert openai["active"] is False
