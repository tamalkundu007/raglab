"""
ChunkerFactory — registry-based factory for all RAGLab chunkers.

Usage:
    from raglab_chunkers.factory import ChunkerFactory

    chunker = ChunkerFactory.create("text", config={"chunk_size": 300})
    chunks  = chunker.chunk(text, doc_id="doc-001")

Registration:
    Chunkers are registered at import time via the `_REGISTRY` dict.
    R2+ chunkers are registered as stubs that raise NotImplementedFeatureError
    so the UI can show them as "Coming Soon" without breaking the factory.

Design:
    - Callers never import concrete chunker classes directly.
    - `ChunkerFactory.available()` drives the UI dropdown and config validation.
    - `ChunkerFactory.schema(chunker_type)` returns the UI parameter schema.
"""

from __future__ import annotations

from typing import Any

from raglab_common.exceptions import NotImplementedFeatureError
from raglab_common.logging import get_logger
from raglab_common.models import ChunkerType

from raglab_chunkers.base import BaseChunker
from raglab_chunkers.text_chunker import TextChunker

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Stub class for R2+ chunkers — visible in UI, raises on use
# ---------------------------------------------------------------------------


def _make_stub(name: str, available_in: str) -> type[BaseChunker]:
    """Create a stub chunker class that raises NotImplementedFeatureError."""

    class _StubChunker(BaseChunker):
        chunker_type = name

        def __init__(self, config: dict[str, Any] | None = None) -> None:
            raise NotImplementedFeatureError(
                feature=f"{name.upper()}Chunker",
                available_in=available_in,
            )

        def _chunk(self, text: str, doc_id: str, metadata: dict[str, Any]):
            raise NotImplementedFeatureError(feature=name, available_in=available_in)

        @classmethod
        def config_schema(cls) -> dict[str, Any]:
            return {
                "_stub": {
                    "type": "str",
                    "default": available_in,
                    "description": f"Available in {available_in}.",
                }
            }

    _StubChunker.__name__ = f"{name.title().replace('_', '')}ChunkerStub"
    return _StubChunker


# ---------------------------------------------------------------------------
# Registry — active chunkers + R2+ stubs
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseChunker]] = {
    # R1 — active
    ChunkerType.TEXT: TextChunker,

    # R2 — stubs (UI shows these as Coming Soon)
    ChunkerType.PDF:         _make_stub("pdf",         "R2"),
    ChunkerType.DOCX:        _make_stub("docx",        "R2"),
    ChunkerType.MARKDOWN:    _make_stub("markdown",    "R2"),
    ChunkerType.HTML:        _make_stub("html",        "R2"),
    ChunkerType.EXCEL:       _make_stub("excel",       "R2"),
    ChunkerType.PDF_IMAGES:  _make_stub("pdf_images",  "R2"),
    ChunkerType.TABLE_STITCH:_make_stub("table_stitch","R2"),
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class ChunkerFactory:
    """Registry-based factory for RAGLab chunkers."""

    @classmethod
    def create(
        cls,
        chunker_type: str | ChunkerType,
        config: dict[str, Any] | None = None,
    ) -> BaseChunker:
        """
        Instantiate and return a chunker for the given type.

        Args:
            chunker_type: String or ChunkerType enum value.
            config:       Optional parameter dict forwarded to the chunker.

        Returns:
            A BaseChunker instance.

        Raises:
            ValueError: If chunker_type is not in the registry.
            NotImplementedFeatureError: If the chunker is a stub (R2+ feature).
        """
        key = chunker_type.value if isinstance(chunker_type, ChunkerType) else str(chunker_type)
        cls_ref = _REGISTRY.get(key)
        if cls_ref is None:
            available = list(_REGISTRY.keys())
            raise ValueError(
                f"Unknown chunker type {key!r}. Available: {available}"
            )
        log.info("factory.create_chunker", chunker_type=key)
        return cls_ref(config=config)

    @classmethod
    def available(cls) -> list[dict[str, Any]]:
        """
        Return metadata for all registered chunkers.

        Used by the UI to populate the chunker dropdown with active/stub status.

        Returns:
            List of dicts: {type, active, available_in (if stub)}
        """
        result = []
        active_types = {ChunkerType.TEXT}
        for key, cls_ref in _REGISTRY.items():
            is_active = ChunkerType(key) in active_types
            entry: dict[str, Any] = {
                "type": key,
                "active": is_active,
            }
            if not is_active:
                # Determine release from stub class name
                try:
                    schema = cls_ref.config_schema()
                    entry["available_in"] = schema.get("_stub", {}).get("default", "future")
                except Exception:  # noqa: BLE001
                    entry["available_in"] = "future"
            result.append(entry)
        return result

    @classmethod
    def schema(cls, chunker_type: str | ChunkerType) -> dict[str, Any]:
        """
        Return the config schema for a given chunker type.

        Args:
            chunker_type: String or ChunkerType enum value.

        Returns:
            Config schema dict (see BaseChunker.config_schema docstring).

        Raises:
            ValueError: If chunker_type is unknown.
        """
        key = chunker_type.value if isinstance(chunker_type, ChunkerType) else str(chunker_type)
        cls_ref = _REGISTRY.get(key)
        if cls_ref is None:
            raise ValueError(f"Unknown chunker type {key!r}")
        return cls_ref.config_schema()
