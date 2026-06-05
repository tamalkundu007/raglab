"""Tests for DOCXChunker — python-docx mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from raglab_chunkers.docx_chunker import DOCXChunker, _is_heading, _heading_level
from raglab_common.models import ChunkModel

SAMPLE_TEXT = (
    "RAG systems retrieve documents from an external corpus. "
    "These documents are used as context for generation. "
    "The model grounds its answers in retrieved facts reducing hallucinations."
)


def make_para(text: str, style_name: str = "Normal") -> MagicMock:
    para = MagicMock()
    para.text = text
    para.style = MagicMock()
    para.style.name = style_name
    return para


def make_docx_mock(paragraphs: list[tuple[str, str]]) -> MagicMock:
    """paragraphs: list of (text, style_name)"""
    doc = MagicMock()
    doc.paragraphs = [make_para(t, s) for t, s in paragraphs]
    return doc


class TestParagraphHelpers:
    def test_is_heading_true_for_heading1(self):
        para = make_para("Introduction", "Heading 1")
        assert _is_heading(para) is True

    def test_is_heading_true_for_title(self):
        para = make_para("Document Title", "Title")
        assert _is_heading(para) is True

    def test_is_heading_false_for_normal(self):
        para = make_para("Body text", "Normal")
        assert _is_heading(para) is False

    def test_heading_level_extracts_number(self):
        para = make_para("Section", "Heading 3")
        assert _heading_level(para) == 3

    def test_heading_level_bare_heading(self):
        para = make_para("Top", "Heading")
        assert _heading_level(para) == 1

    def test_heading_level_zero_for_normal(self):
        para = make_para("Body", "Normal")
        assert _heading_level(para) == 0


class TestDOCXChunkerConfig:
    def test_defaults(self):
        c = DOCXChunker()
        assert c.chunk_size == 500
        assert c.preserve_headings is True
        assert c.include_heading_in_chunk is True
        assert c.tokenizer == "tiktoken"

    def test_custom_config(self):
        c = DOCXChunker(config={"chunk_size": 200, "preserve_headings": False, "tokenizer": "word_count"})
        assert c.chunk_size == 200
        assert c.preserve_headings is False

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError):
            DOCXChunker(config={"chunk_size": 0})

    def test_invalid_overlap(self):
        with pytest.raises(ValueError):
            DOCXChunker(config={"chunk_size": 100, "chunk_overlap": 100})


class TestDOCXChunkerPlainText:
    def test_plain_text_produces_chunks(self):
        c = DOCXChunker(config={"tokenizer": "word_count", "chunk_size": 20, "chunk_overlap": 3})
        chunks = c.chunk(SAMPLE_TEXT, "doc-001")
        assert len(chunks) >= 1
        assert all(isinstance(ch, ChunkModel) for ch in chunks)

    def test_doc_id_propagated(self):
        c = DOCXChunker(config={"tokenizer": "word_count"})
        chunks = c.chunk(SAMPLE_TEXT, "docx-doc")
        assert all(ch.doc_id == "docx-doc" for ch in chunks)

    def test_empty_input_returns_empty(self):
        c = DOCXChunker(config={"tokenizer": "word_count"})
        assert c.chunk("", "doc-001") == []


class TestDOCXChunkerHeadingAware:
    def _chunker(self, **kwargs) -> DOCXChunker:
        cfg = {"tokenizer": "word_count", "chunk_size": 30, "chunk_overlap": 3}
        cfg.update(kwargs)
        return DOCXChunker(config=cfg)

    def _chunk_doc(self, paragraphs: list[tuple[str, str]], **kwargs) -> list[ChunkModel]:
        c = self._chunker(**kwargs)
        mock_doc = make_docx_mock(paragraphs)
        return c._process_document(mock_doc, "doc-001", {})

    def test_heading_injected_into_metadata(self):
        paras = [
            ("Introduction", "Heading 1"),
            ("RAG retrieves documents for context.", "Normal"),
            ("It reduces hallucinations significantly.", "Normal"),
        ]
        chunks = self._chunk_doc(paras)
        assert any(ch.metadata.get("heading") == "Introduction" for ch in chunks)

    def test_heading_level_in_metadata(self):
        paras = [
            ("Methods", "Heading 2"),
            ("Dense retrieval is used for vector search.", "Normal"),
        ]
        chunks = self._chunk_doc(paras)
        headings = [ch for ch in chunks if ch.metadata.get("heading")]
        assert any(ch.metadata.get("heading_level") == 2 for ch in headings)

    def test_include_heading_in_chunk_text(self):
        paras = [
            ("Key Findings", "Heading 1"),
            ("RAG reduces hallucinations by grounding in evidence.", "Normal"),
        ]
        chunks = self._chunk_doc(paras, include_heading_in_chunk=True)
        assert any("Key Findings" in ch.text for ch in chunks)

    def test_no_headings_falls_back_to_full_text(self):
        paras = [
            ("First body paragraph with content.", "Normal"),
            ("Second body paragraph with more content.", "Normal"),
        ]
        chunks = self._chunk_doc(paras)
        assert len(chunks) >= 1

    def test_preserve_headings_false_single_stream(self):
        paras = [
            ("Introduction", "Heading 1"),
            ("Body content here with multiple words.", "Normal"),
            ("More Methods", "Heading 2"),
            ("Additional body content with details.", "Normal"),
        ]
        c = DOCXChunker(config={"tokenizer": "word_count", "chunk_size": 100, "chunk_overlap": 10, "preserve_headings": False})
        mock_doc = make_docx_mock(paras)
        chunks = c._process_document(mock_doc, "doc-001", {})
        # No heading metadata when preserve_headings=False
        assert not any("heading" in ch.metadata for ch in chunks)

    def test_multiple_sections_produce_chunks_per_section(self):
        body = "Word " * 20
        paras = [
            ("Section A", "Heading 1"),
            (body, "Normal"),
            ("Section B", "Heading 1"),
            (body, "Normal"),
        ]
        chunks = self._chunk_doc(paras)
        headings_seen = {ch.metadata.get("heading") for ch in chunks if ch.metadata.get("heading")}
        assert "Section A" in headings_seen
        assert "Section B" in headings_seen


class TestDOCXChunkerSchema:
    def test_schema_has_required_keys(self):
        schema = DOCXChunker.config_schema()
        for key in ["chunk_size", "chunk_overlap", "preserve_headings", "include_heading_in_chunk"]:
            assert key in schema
