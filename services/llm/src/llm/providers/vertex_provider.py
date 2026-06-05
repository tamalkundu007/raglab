"""Vertex AI provider — stub until R2."""

from __future__ import annotations

from typing import Any

from raglab_common.exceptions import NotImplementedFeatureError
from llm.providers.base import BaseLLMProvider


class VertexProvider(BaseLLMProvider):
    """Google Vertex AI — stub. Activates in R2."""

    provider = "vertex"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedFeatureError("VertexProvider", available_in="R2")

    def _call_api(self, system_prompt: str, prompt: str, max_tokens: int, temperature: float) -> str:
        raise NotImplementedFeatureError("VertexProvider", available_in="R2")

    def _model_name(self) -> str:
        return "vertex/stub"
