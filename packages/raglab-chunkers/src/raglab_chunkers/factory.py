"""
ChunkerFactory — registry-based factory for all RAGLab chunkers.

R1 active:  TextChunker
R2 active:  PDFChunker, DOCXChunker, MarkdownChunker, HTMLChunker, ExcelChunker
R2 stubs:   PDFImagesChunker, TableStitchChunker
"""

from __future__ import annotations

from typing import Any

from raglab_common.exceptions import NotImplementedFeatureError
from raglab_common.logging import get_logger
from raglab_common.models import ChunkerType

from raglab_chunkers.base import BaseChunker
from raglab_chunkers.text_chunker import TextChunker
from raglab_chunkers.pdf_chunker import PDFChunker
from raglab_chunkers.docx_chunker import DOCXChunker
from raglab_chunkers.markdown_chunker import MarkdownChunker
from raglab_chunkers.html_chunker import HTMLChunker
from raglab_chunkers.excel_chunker import ExcelChunker
from raglab_chunkers.hybrid_chunker import HybridChunker
from raglab_chunkers.pdf_image_chunker import PDFImageChunker

log = get_logger(__name__)


def _make_stub(name: str, available_in: str) -> type[BaseChunker]:
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
            return {"_stub": {"type": "str", "default": available_in, "description": f"Available in {available_in}."}}

    _StubChunker.__name__ = f"{name.title().replace('_', '')}ChunkerStub"
    return _StubChunker


_REGISTRY: dict[str, type[BaseChunker]] = {
    # R1 — active
    ChunkerType.TEXT:         TextChunker,
    # R2 — active
    ChunkerType.PDF:          PDFChunker,
    ChunkerType.DOCX:         DOCXChunker,
    ChunkerType.MARKDOWN:     MarkdownChunker,
    ChunkerType.HTML:         HTMLChunker,
    ChunkerType.EXCEL:        ExcelChunker,
    # R2 meta-strategy
    "hybrid":            HybridChunker,
    # R2 stubs
    ChunkerType.PDF_IMAGES:   PDFImageChunker,
    ChunkerType.TABLE_STITCH: _make_stub("table_stitch", "R2-extended"),
}

_ACTIVE_TYPES = {
    ChunkerType.TEXT, ChunkerType.PDF, ChunkerType.DOCX,
    ChunkerType.MARKDOWN, ChunkerType.HTML, ChunkerType.EXCEL,
    ChunkerType.PDF_IMAGES,
}
_ACTIVE_STRINGS = {"hybrid"}  # meta-types not in ChunkerType enum


class ChunkerFactory:
    @classmethod
    def create(cls, chunker_type: str | ChunkerType, config: dict[str, Any] | None = None) -> BaseChunker:
        key = chunker_type.value if isinstance(chunker_type, ChunkerType) else str(chunker_type)
        cls_ref = _REGISTRY.get(key)
        if cls_ref is None:
            raise ValueError(f"Unknown chunker type {key!r}. Available: {list(_REGISTRY.keys())}")
        log.info("factory.create_chunker", chunker_type=key)
        return cls_ref(config=config)

    @classmethod
    def available(cls) -> list[dict[str, Any]]:
        result = []
        for key, cls_ref in _REGISTRY.items():
            try:
                is_active = ChunkerType(key) in _ACTIVE_TYPES
            except ValueError:
                is_active = key in _ACTIVE_STRINGS
            entry: dict[str, Any] = {"type": key, "active": is_active}
            if not is_active:
                try:
                    schema = cls_ref.config_schema()
                    entry["available_in"] = schema.get("_stub", {}).get("default", "future")
                except Exception:
                    entry["available_in"] = "future"
            result.append(entry)
        return result

    @classmethod
    def schema(cls, chunker_type: str | ChunkerType) -> dict[str, Any]:
        key = chunker_type.value if isinstance(chunker_type, ChunkerType) else str(chunker_type)
        cls_ref = _REGISTRY.get(key)
        if cls_ref is None:
            raise ValueError(f"Unknown chunker type {key!r}")
        return cls_ref.config_schema()
