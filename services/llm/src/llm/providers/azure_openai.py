"""Azure OpenAI chat completion provider."""

from __future__ import annotations

from typing import Any

from openai import AzureOpenAI

from raglab_common.exceptions import LLMError
from llm.providers.base import BaseLLMProvider


class AzureOpenAIProvider(BaseLLMProvider):
    """
    Azure OpenAI chat completion via the openai SDK.

    Requires:
        RAGLAB_AZURE_OPENAI_API_KEY
        RAGLAB_AZURE_OPENAI_ENDPOINT
        RAGLAB_AZURE_OPENAI_CHAT_DEPLOYMENT
    """

    provider = "azure_openai"

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2024-02-01",
        )
        self._deployment = deployment

    def _call_api(self, system_prompt: str, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            return content or ""
        except Exception as exc:
            raise LLMError(f"Azure OpenAI call failed: {exc}") from exc

    def _model_name(self) -> str:
        return f"azure_openai/{self._deployment}"
