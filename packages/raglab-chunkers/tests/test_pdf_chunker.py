"""Tests for PDFChunker — PyMuPDF mocked, all paths covered."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from raglab_chunkers.pdf_chunker import PDFChunker
from raglab_common.models import ChunkModel

SAMPLE_TEXT = (
    "Retrieval-Augmented Generation enhances language models with external knowledge. "
    "The retrieval step uses dense vector search to find relevant documents. "
    "These documents are passed as context to the language model at inference time."
)


def make_page_mock(text: str) -> MagicMock:
    page = MagicMock()
    page.get_text.return_value = text
    return page


class TestPDFChunkerConfig:
    def test_defaults(self):
        c = PDFChunker()
        assert c.chunk_size == 500
        assert c.chunk_overlap == 50
        assert c.respect_page_boundary is True
        assert c.page_metadata is True
        assert c.boundary_enforcement is True
        assert c.tokenizer == "tiktoken"

    def test_custom_config(self):
        c = PDFChunker(config={"chunk_size": 200, "tokenizer": "word_count", "respect_page_boundary": False})
        assert c.chunk_size == 200
        assert c.respect_page_boundary is False

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size"):
            PDFChunker(config={"chunk_size": 0})

    def test_invalid_overlap_gte_chunk_size(self):
        with pytest.raises(ValueError):
            PDFChunker(config={"chunk_size": 100, "chunk_overlap": 100})

    def test_invalid_tokenizer(self):
        with pytest.raises(ValueError, match="tokenizer"):
            PDFChunker(config={"tokenizer": "gpt5"})


class TestPDFChunkerPlainText:
    """chunk() with plain text exercises the _chunk() fallback (no PyMuPDF)."""

    def test_plain_text_produces_chunks(self):
        c = PDFChunker(config={"tokenizer": "word_count", "chunk_size": 20, "chunk_overlap": 3})
        chunks = c.chunk(SAMPLE_TEXT, "doc-pdf-001")
        assert len(chunks) >= 1
        assert all(isinstance(ch, ChunkModel) for ch in chunks)

    def test_sequential_indices(self):
        c = PDFChunker(config={"tokenizer": "word_count", "chunk_size": 20, "chunk_overlap": 3})
        chunks = c.chunk(SAMPLE_TEXT, "doc-001")
        assert [ch.chunk_index for ch in chunks] == list(range(len(chunks)))

    def test_doc_id_propagated(self):
        c = PDFChunker(config={"tokenizer": "word_count"})
        chunks = c.chunk(SAMPLE_TEXT, "my-doc")
        assert all(ch.doc_id == "my-doc" for ch in chunks)

    def test_empty_text_returns_empty(self):
        c = PDFChunker(config={"tokenizer": "word_count"})
        assert c.chunk("", "doc-001") == []

    def test_chunker_type_in_metadata(self):
        c = PDFChunker(config={"tokenizer": "word_count", "chunk_size": 20, "chunk_overlap": 3})
        chunks = c.chunk(SAMPLE_TEXT, "doc-001")
        assert all(ch.metadata.get("chunker") == "pdf" for ch in chunks)


class TestPDFChunkerBytes:
    """chunk_pdf_bytes() with mocked PyMuPDF."""

    def _make_fitz_mock(self, pages: list[str]) -> MagicMock:
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = len(pages)
        mock_doc.__iter__ = MagicMock(side_effect=lambda: iter([]))
        mock_doc.close = MagicMock()
        page_mocks = [make_page_mock(p) for p in pages]
        mock_doc.__getitem__ = lambda self, i: page_mocks[i]
        return mock_doc

    def test_single_page_produces_chunks(self):
        c = PDFChunker(config={"tokenizer": "word_count", "chunk_size": 20, "chunk_overlap": 3})
        mock_doc = self._make_fitz_mock([SAMPLE_TEXT])
        with patch("raglab_chunkers.pdf_chunker.fitz") as mock_fitz:
            mock_fitz.open.return_value = mock_doc
            chunks = c.chunk_pdf_bytes(b"fake-pdf", "doc-001")
        assert len(chunks) >= 1

    def test_multi_page_respects_boundary(self):
        pages = [f"Page {i} content. " * 5 for i in range(3)]
        c = PDFChunker(config={
            "tokenizer": "word_count", "chunk_size": 15, "chunk_overlap": 2,
            "respect_page_boundary": True, "page_metadata": True,
        })
        mock_doc = self._make_fitz_mock(pages)
        with patch("raglab_chunkers.pdf_chunker.fitz") as mock_fitz:
            mock_fitz.open.return_value = mock_doc
            chunks = c.chunk_pdf_bytes(b"fake-pdf", "doc-001")
        # Each chunk should have page_number in metadata
        assert all("page_number" in ch.metadata for ch in chunks)
        page_numbers = {ch.metadata["page_number"] for ch in chunks}
        assert page_numbers == {1, 2, 3}

    def test_no_page_boundary_concatenates(self):
        pages = ["First page content here. " * 3, "Second page content here. " * 3]
        c = PDFChunker(config={
            "tokenizer": "word_count", "chunk_size": 20, "chunk_overlap": 2,
            "respect_page_boundary": False,
        })
        mock_doc = self._make_fitz_mock(pages)
        with patch("raglab_chunkers.pdf_chunker.fitz") as mock_fitz:
            mock_fitz.open.return_value = mock_doc
            chunks = c.chunk_pdf_bytes(b"fake-pdf", "doc-001")
        # Without page boundary, no page_number metadata expected
        assert all("page_number" not in ch.metadata for ch in chunks)

    def test_empty_pages_skipped(self):
        pages = ["   ", "Real content here. " * 4, ""]
        c = PDFChunker(config={"tokenizer": "word_count", "chunk_size": 15, "chunk_overlap": 2})
        mock_doc = self._make_fitz_mock(pages)
        with patch("raglab_chunkers.pdf_chunker.fitz") as mock_fitz:
            mock_fitz.open.return_value = mock_doc
            chunks = c.chunk_pdf_bytes(b"fake-pdf", "doc-001")
        assert len(chunks) >= 1


class TestPDFChunkerSchema:
    def test_schema_has_required_keys(self):
        schema = PDFChunker.config_schema()
        for key in ["chunk_size", "chunk_overlap", "boundary_enforcement",
                    "tokenizer", "min_chunk_size", "respect_page_boundary", "page_metadata"]:
            assert key in schema

    def test_schema_defaults_match_class(self):
        schema = PDFChunker.config_schema()
        assert schema["chunk_size"]["default"] == 500
        assert schema["respect_page_boundary"]["default"] is True
        assert schema["page_metadata"]["default"] is True
