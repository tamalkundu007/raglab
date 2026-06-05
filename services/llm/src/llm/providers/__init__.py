"""
LLM provider factory for the llm-service.

get_llm_provider() instantiates the correct provider from settings.
All callers go through this — no direct provider class imports outside this module.
"""

from __future__ import annotations

from typing import Any

from raglab_common.exceptions import LLMError, NotImplementedFeatureError
from raglab_common.models import LLMProvider
from llm.providers.base import BaseLLMProvider


def get_llm_provider(provider: str | LLMProvider, settings: Any) -> BaseLLMProvider:
    """
    Instantiate and return an LLM provider.

    Args:
        provider: LLMProvider enum value or string key.
        settings: LLMSettings instance supplying credentials.

    Returns:
        A BaseLLMProvider instance.

    Raises:
        LLMError: Unknown provider or missing credentials.
        NotImplementedFeatureError: Vertex (stub).
    """
    key = provider.value if isinstance(provider, LLMProvider) else str(provider)

    if key == LLMProvider.AZURE_OPENAI.value:
        from llm.providers.azure_openai import AzureOpenAIProvider
        if not settings.azure_openai_api_key:
            raise LLMError("RAGLAB_AZURE_OPENAI_API_KEY not set.")
        deployment = getattr(settings, "azure_openai_chat_deployment", "") or settings.azure_openai_deployment
        if not deployment:
            raise LLMError("RAGLAB_AZURE_OPENAI_CHAT_DEPLOYMENT not set.")
        return AzureOpenAIProvider(
            api_key=settings.azure_openai_api_key,
            endpoint=settings.azure_openai_endpoint,
            deployment=deployment,
        )

    if key == LLMProvider.OPENAI.value:
        from llm.providers.openai_provider import OpenAIProvider
        if not settings.openai_api_key:
            raise LLMError("RAGLAB_OPENAI_API_KEY not set.")
        return OpenAIProvider(api_key=settings.openai_api_key, model=settings.openai_chat_model)

    if key == LLMProvider.ANTHROPIC.value:
        from llm.providers.anthropic_provider import AnthropicProvider
        if not settings.anthropic_api_key:
            raise LLMError("RAGLAB_ANTHROPIC_API_KEY not set.")
        return AnthropicProvider(api_key=settings.anthropic_api_key, model=settings.anthropic_model)

    if key == LLMProvider.OLLAMA.value:
        from llm.providers.ollama_provider import OllamaProvider
        return OllamaProvider(base_url=settings.ollama_base_url, model=settings.ollama_chat_model)

    if key == LLMProvider.VERTEX.value:
        raise NotImplementedFeatureError("VertexProvider", available_in="R2")

    raise LLMError(f"Unknown LLM provider: {key!r}")
