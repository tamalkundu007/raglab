"""OpenAI chat completion provider."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from raglab_common.exceptions import LLMError
from llm.providers.base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """OpenAI chat completion. Requires RAGLAB_OPENAI_API_KEY."""

    provider = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def _call_api(self, system_prompt: str, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMError(f"OpenAI call failed: {exc}") from exc

    def _model_name(self) -> str:
        return f"openai/{self._model}"
