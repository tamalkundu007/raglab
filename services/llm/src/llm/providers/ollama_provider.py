"""Ollama local chat completion provider."""

from __future__ import annotations

from typing import Any

import httpx

from raglab_common.exceptions import LLMError
from llm.providers.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """
    Ollama local model via REST API /api/chat.
    Requires Ollama running at RAGLAB_OLLAMA_BASE_URL.
    """

    provider = "ollama"

    def __init__(self, base_url: str, model: str = "llama3.2", config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._base_url = base_url.rstrip("/")
        self._model = model

    def _call_api(self, system_prompt: str, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            response = httpx.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as exc:
            raise LLMError(f"Ollama call failed: {exc}") from exc

    def _model_name(self) -> str:
        return f"ollama/{self._model}"
