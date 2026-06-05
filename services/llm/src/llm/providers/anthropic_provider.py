"""Anthropic (Claude) chat completion provider."""

from __future__ import annotations

from typing import Any

import anthropic

from raglab_common.exceptions import LLMError
from llm.providers.base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude via the anthropic SDK.
    Requires RAGLAB_ANTHROPIC_API_KEY.
    """

    provider = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def _call_api(self, system_prompt: str, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            block = message.content[0]
            return block.text if hasattr(block, "text") else str(block)
        except Exception as exc:
            raise LLMError(f"Anthropic call failed: {exc}") from exc

    def _model_name(self) -> str:
        return f"anthropic/{self._model}"
