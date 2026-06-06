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

    def caption_image(
        self,
        image_b64: str,
        image_ext: str = "png",
        prompt: str = "Describe this image concisely for a RAG retrieval system.",
        max_tokens: int = 256,
    ) -> str:
        """
        Caption an image using Anthropic Claude vision (claude-3-* models).

        Sends the image as a base64 source block in the messages content array.
        Requires claude-3-haiku, claude-3-sonnet, claude-3-opus, or claude-3-5-*.
        """
        media_type = f"image/{image_ext.lower().replace('jpg', 'jpeg')}"

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            content = response.content[0].text if response.content else ""
            return content.strip() if content else "[No caption returned]"
        except Exception as exc:
            from raglab_common.exceptions import LLMError
            raise LLMError(f"Anthropic vision caption failed: {exc}") from exc
