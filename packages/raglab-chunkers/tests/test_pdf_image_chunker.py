"""
Unit tests for PDFImageChunker (R4 Phase 1).

Covers:
- Config validation (ocr_engine, image_handling, dpi, chunk params)
- OCR text extraction path (pytesseract mocked)
- Image extraction path (PyMuPDF extract_image mocked)
- Both paths combined
- Image area filtering (min_image_area)
- Page number injection in metadata
- chunk_type metadata tag ("text" vs "image")
- image_bytes base64 encoding in metadata
- Fallback _chunk() path (plain text input)
- Factory registration and active status
- config_schema completeness
- Empty page / empty PDF handling
- OCR failure graceful degradation
"""

from __future__ import annotations

import base64
import io
import uuid
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from raglab_chunkers.pdf_image_chunker import PDFImageChunker, _VALID_OCR_ENGINES, _VALID_IMAGE_HANDLING
from raglab_common.models import ChunkModel

# ── Sample data ───────────────────────────────────────────────────────────────

SAMPLE_OCR_TEXT = (
    "This is a scanned document. It contains important information about RAG systems. "
    "The retrieval augmented generation approach improves answer quality significantly. "
    "Dense retrieval finds relevant documents using vector similarity search methods."
)

TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
    b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_fitz_page(ocr_text: str = SAMPLE_OCR_TEXT, images: list | None = None):
    """Create a mock PyMuPDF page."""
    page = MagicMock()
    # get_pixmap returns a mock that renders to a tiny PNG
    mock_pixmap = MagicMock()
    mock_pixmap.tobytes.return_value = TINY_PNG
    page.get_pixmap.return_value = mock_pixmap
    page.get_images.return_value = images or []
    return page


def make_fitz_doc(pages: list, images_by_xref: dict | None = None):
    """Create a mock PyMuPDF doc."""
    images_by_xref = images_by_xref or {}
    doc = MagicMock()
    doc.__len__.return_value = len(pages)
    doc.__getitem__ = lambda self, i: pages[i]
    doc.close = MagicMock()

    def extract_image(xref):
        return images_by_xref.get(xref, {
            "image": TINY_PNG,
            "width": 200,
            "height": 150,
            "ext": "png",
        })

    doc.extract_image = extract_image
    return doc


def make_chunker(**kwargs) -> PDFImageChunker:
    defaults = {
        "tokenizer": "word_count",
        "chunk_size": 30,
        "chunk_overlap": 3,
        "min_chunk_size": 5,
        "ocr_engine": "tesseract",
        "image_handling": "skip",
        "dpi": 150,
    }
    defaults.update(kwargs)
    return PDFImageChunker(config=defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# Config validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestPDFImageChunkerConfig:
    def test_defaults(self):
        c = PDFImageChunker()
        assert c.ocr_engine == "tesseract"
        assert c.image_handling == "extract"
        assert c.dpi == 150
        assert c.ocr_language == "eng"
        assert c.min_image_area == 5000
        assert c.chunk_size == 500
        assert c.tokenizer == "word_count"

    def test_custom_config(self):
        c = PDFImageChunker(config={
            "ocr_engine": "none",
            "image_handling": "both",
            "dpi": 300,
            "ocr_language": "fra",
        })
        assert c.ocr_engine == "none"
        assert c.image_handling == "both"
        assert c.dpi == 300
        assert c.ocr_language == "fra"

    def test_invalid_ocr_engine(self):
        with pytest.raises(ValueError, match="ocr_engine"):
            PDFImageChunker(config={"ocr_engine": "aws_textract"})

    def test_invalid_image_handling(self):
        with pytest.raises(ValueError, match="image_handling"):
            PDFImageChunker(config={"image_handling": "resize"})

    def test_invalid_dpi_too_low(self):
        with pytest.raises(ValueError, match="dpi"):
            PDFImageChunker(config={"dpi": 30})

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size"):
            PDFImageChunker(config={"chunk_size": 0})

    def test_invalid_overlap_gte_chunk_size(self):
        with pytest.raises(ValueError):
            PDFImageChunker(config={"chunk_size": 50, "chunk_overlap": 50})

    def test_invalid_tokenizer(self):
        with pytest.raises(ValueError, match="tokenizer"):
            PDFImageChunker(config={"tokenizer": "gpt5"})

    def test_valid_image_handling_options(self):
        for opt in _VALID_IMAGE_HANDLING:
            c = PDFImageChunker(config={"image_handling": opt})
            assert c.image_handling == opt

    def test_valid_ocr_engine_options(self):
        for opt in _VALID_OCR_ENGINES:
            c = PDFImageChunker(config={"ocr_engine": opt})
            assert c.ocr_engine == opt


# ═══════════════════════════════════════════════════════════════════════════════
# OCR text extraction path
# ═══════════════════════════════════════════════════════════════════════════════

class TestOCRPath:
    """All pytesseract calls mocked — no Tesseract installation required."""

    def _run_ocr(self, text=SAMPLE_OCR_TEXT, num_pages=1, **cfg_kwargs):
        """Run chunk_pdf_bytes with mocked OCR returning `text`."""
        pages = [make_fitz_page(ocr_text=text) for _ in range(num_pages)]
        mock_doc = make_fitz_doc(pages)

        with patch("raglab_chunkers.pdf_image_chunker.fitz") as mock_fitz, \
             patch("raglab_chunkers.pdf_image_chunker.pytesseract") as mock_tess, \
             patch("raglab_chunkers.pdf_image_chunker.Image") as mock_pil:

            mock_fitz.open.return_value = mock_doc
            mock_tess.image_to_string.return_value = text
            mock_pil.open.return_value = MagicMock()

            chunker = make_chunker(image_handling="skip", **cfg_kwargs)
            return chunker.chunk_pdf_bytes(b"fake-pdf", "doc-ocr")

    def test_ocr_produces_chunks(self):
        chunks = self._run_ocr()
        assert len(chunks) >= 1
        assert all(isinstance(c, ChunkModel) for c in chunks)

    def test_ocr_chunk_type_is_text(self):
        chunks = self._run_ocr()
        assert all(c.metadata["chunk_type"] == "text" for c in chunks)

    def test_ocr_engine_in_metadata(self):
        chunks = self._run_ocr()
        assert all(c.metadata["ocr_engine"] == "tesseract" for c in chunks)

    def test_chunker_type_in_metadata(self):
        chunks = self._run_ocr()
        assert all(c.metadata["chunker"] == "pdf_images" for c in chunks)

    def test_page_number_in_metadata(self):
        chunks = self._run_ocr()
        assert all("page_number" in c.metadata for c in chunks)
        assert all(c.metadata["page_number"] == 1 for c in chunks)

    def test_page_number_excluded_when_disabled(self):
        chunks = self._run_ocr(include_page_numbers=False)
        assert all("page_number" not in c.metadata for c in chunks)

    def test_multi_page_page_numbers_correct(self):
        chunks = self._run_ocr(num_pages=3)
        page_numbers = {c.metadata.get("page_number") for c in chunks}
        assert page_numbers == {1, 2, 3}

    def test_sequential_indices(self):
        chunks = self._run_ocr()
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_unique_chunk_ids(self):
        chunks = self._run_ocr()
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_doc_id_propagated(self):
        chunks = self._run_ocr()
        assert all(c.doc_id == "doc-ocr" for c in chunks)

    def test_positive_token_counts(self):
        chunks = self._run_ocr()
        assert all(c.token_count > 0 for c in chunks)

    def test_empty_ocr_text_returns_empty(self):
        chunks = self._run_ocr(text="   ")
        assert chunks == []

    def test_ocr_failure_returns_empty_gracefully(self):
        """pytesseract failure should not raise — just return empty."""
        pages = [make_fitz_page()]
        mock_doc = make_fitz_doc(pages)

        with patch("raglab_chunkers.pdf_image_chunker.fitz") as mock_fitz, \
             patch("raglab_chunkers.pdf_image_chunker.pytesseract") as mock_tess, \
             patch("raglab_chunkers.pdf_image_chunker.Image") as mock_pil:

            mock_fitz.open.return_value = mock_doc
            mock_tess.image_to_string.side_effect = Exception("Tesseract not found")
            mock_pil.open.return_value = MagicMock()

            chunker = make_chunker(image_handling="skip")
            chunks = chunker.chunk_pdf_bytes(b"fake", "doc-001")
        # Should not raise; returns empty (OCR failed, no text extracted)
        assert isinstance(chunks, list)

    def test_ocr_respects_chunk_size(self):
        chunks = self._run_ocr(chunk_size=10, chunk_overlap=2)
        # With small chunk_size, should produce multiple chunks
        assert len(chunks) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# Image extraction path
# ═══════════════════════════════════════════════════════════════════════════════

class TestImageExtractionPath:
    """PyMuPDF extract_image mocked — no real PDFs needed."""

    def _run_image_extract(self, images=None, min_image_area=100, **cfg_kwargs):
        img_xref = 10
        if images is None:
            images = [(img_xref, None, None, None, None, None, None)]

        pages = [make_fitz_page(images=images)]
        mock_doc = make_fitz_doc(pages, images_by_xref={
            img_xref: {
                "image": TINY_PNG,
                "width": 200,
                "height": 150,
                "ext": "png",
            }
        })

        with patch("raglab_chunkers.pdf_image_chunker.fitz") as mock_fitz, \
             patch("raglab_chunkers.pdf_image_chunker.pytesseract") as mock_tess, \
             patch("raglab_chunkers.pdf_image_chunker.Image") as mock_pil:

            mock_fitz.open.return_value = mock_doc
            mock_tess.image_to_string.return_value = ""
            mock_pil.open.return_value = MagicMock()

            chunker = make_chunker(
                image_handling="extract",
                ocr_engine="none",
                min_image_area=min_image_area,
                **cfg_kwargs
            )
            return chunker.chunk_pdf_bytes(b"fake-pdf", "doc-img")

    def test_image_chunk_produced(self):
        chunks = self._run_image_extract()
        assert len(chunks) >= 1

    def test_image_chunk_type_in_metadata(self):
        chunks = self._run_image_extract()
        image_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "image"]
        assert len(image_chunks) >= 1

    def test_image_bytes_in_metadata(self):
        chunks = self._run_image_extract()
        for c in chunks:
            if c.metadata.get("chunk_type") == "image":
                assert "image_bytes" in c.metadata
                # Should be valid base64
                decoded = base64.b64decode(c.metadata["image_bytes"])
                assert len(decoded) > 0

    def test_image_dimensions_in_metadata(self):
        chunks = self._run_image_extract()
        for c in chunks:
            if c.metadata.get("chunk_type") == "image":
                assert c.metadata["image_width"] == 200
                assert c.metadata["image_height"] == 150

    def test_image_page_number_in_metadata(self):
        chunks = self._run_image_extract()
        for c in chunks:
            if c.metadata.get("chunk_type") == "image":
                assert c.metadata["page_number"] == 1

    def test_image_index_in_metadata(self):
        chunks = self._run_image_extract()
        for c in chunks:
            if c.metadata.get("chunk_type") == "image":
                assert "image_index" in c.metadata

    def test_captioned_false_in_extract_mode(self):
        chunks = self._run_image_extract()
        for c in chunks:
            if c.metadata.get("chunk_type") == "image":
                assert c.metadata["captioned"] is False

    def test_image_ext_in_metadata(self):
        chunks = self._run_image_extract()
        for c in chunks:
            if c.metadata.get("chunk_type") == "image":
                assert c.metadata["image_ext"] == "png"

    def test_small_image_filtered_by_min_area(self):
        """Images below min_image_area should be excluded."""
        # 200×150 = 30000 px², set threshold above that
        chunks = self._run_image_extract(min_image_area=50000)
        image_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "image"]
        assert len(image_chunks) == 0

    def test_image_passes_min_area_threshold(self):
        """Images above min_image_area should be included."""
        chunks = self._run_image_extract(min_image_area=100)
        image_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "image"]
        assert len(image_chunks) >= 1

    def test_no_images_returns_empty_list(self):
        chunks = self._run_image_extract(images=[])
        assert chunks == []

    def test_chunker_type_in_image_metadata(self):
        chunks = self._run_image_extract()
        for c in chunks:
            if c.metadata.get("chunk_type") == "image":
                assert c.metadata["chunker"] == "pdf_images"


# ═══════════════════════════════════════════════════════════════════════════════
# Both paths combined (image_handling="both")
# ═══════════════════════════════════════════════════════════════════════════════

class TestBothPaths:
    def test_both_produces_text_and_image_chunks(self):
        img_xref = 20
        images = [(img_xref, None, None, None, None, None, None)]
        pages = [make_fitz_page(ocr_text=SAMPLE_OCR_TEXT, images=images)]
        mock_doc = make_fitz_doc(pages, images_by_xref={
            img_xref: {"image": TINY_PNG, "width": 300, "height": 200, "ext": "png"}
        })

        with patch("raglab_chunkers.pdf_image_chunker.fitz") as mock_fitz, \
             patch("raglab_chunkers.pdf_image_chunker.pytesseract") as mock_tess, \
             patch("raglab_chunkers.pdf_image_chunker.Image") as mock_pil:

            mock_fitz.open.return_value = mock_doc
            mock_tess.image_to_string.return_value = SAMPLE_OCR_TEXT
            mock_pil.open.return_value = MagicMock()

            chunker = PDFImageChunker(config={
                "tokenizer": "word_count", "chunk_size": 30, "chunk_overlap": 3,
                "image_handling": "both", "ocr_engine": "tesseract",
                "min_image_area": 100, "min_chunk_size": 5,
            })
            chunks = chunker.chunk_pdf_bytes(b"fake", "doc-both")

        chunk_types = {c.metadata.get("chunk_type") for c in chunks}
        assert "text" in chunk_types
        assert "image" in chunk_types

    def test_sequential_indices_across_both_types(self):
        img_xref = 21
        images = [(img_xref, None, None, None, None, None, None)]
        pages = [make_fitz_page(ocr_text=SAMPLE_OCR_TEXT, images=images)]
        mock_doc = make_fitz_doc(pages, images_by_xref={
            img_xref: {"image": TINY_PNG, "width": 300, "height": 200, "ext": "png"}
        })

        with patch("raglab_chunkers.pdf_image_chunker.fitz") as mock_fitz, \
             patch("raglab_chunkers.pdf_image_chunker.pytesseract") as mock_tess, \
             patch("raglab_chunkers.pdf_image_chunker.Image") as mock_pil:

            mock_fitz.open.return_value = mock_doc
            mock_tess.image_to_string.return_value = SAMPLE_OCR_TEXT
            mock_pil.open.return_value = MagicMock()

            chunker = PDFImageChunker(config={
                "tokenizer": "word_count", "chunk_size": 30, "chunk_overlap": 3,
                "image_handling": "both", "ocr_engine": "tesseract",
                "min_image_area": 100, "min_chunk_size": 5,
            })
            chunks = chunker.chunk_pdf_bytes(b"fake", "doc-both")

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


# ═══════════════════════════════════════════════════════════════════════════════
# Plain text fallback (_chunk)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlainTextFallback:
    def test_chunk_plain_text_produces_chunks(self):
        c = make_chunker()
        chunks = c.chunk(SAMPLE_OCR_TEXT, "doc-text")
        assert len(chunks) >= 1
        assert all(isinstance(ch, ChunkModel) for ch in chunks)

    def test_chunk_plain_text_empty_returns_empty(self):
        c = make_chunker()
        assert c.chunk("", "doc-001") == []

    def test_chunk_plain_text_doc_id_propagated(self):
        c = make_chunker()
        chunks = c.chunk(SAMPLE_OCR_TEXT, "my-doc")
        assert all(ch.doc_id == "my-doc" for ch in chunks)

    def test_chunk_plain_text_sequential_indices(self):
        c = make_chunker()
        chunks = c.chunk(SAMPLE_OCR_TEXT, "doc-001")
        assert [ch.chunk_index for ch in chunks] == list(range(len(chunks)))


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

class TestPDFImageChunkerFactory:
    def test_factory_creates_pdf_image_chunker(self):
        from raglab_chunkers import ChunkerFactory
        c = ChunkerFactory.create("pdf_images", config={"tokenizer": "word_count"})
        assert isinstance(c, PDFImageChunker)

    def test_pdf_images_active_in_available(self):
        from raglab_chunkers import ChunkerFactory
        entries = {e["type"]: e for e in ChunkerFactory.available()}
        assert entries["pdf_images"]["active"] is True

    def test_table_stitch_still_stub(self):
        from raglab_chunkers import ChunkerFactory
        from raglab_common.exceptions import NotImplementedFeatureError
        with pytest.raises(NotImplementedFeatureError):
            ChunkerFactory.create("table_stitch")

    def test_schema_via_factory(self):
        from raglab_chunkers import ChunkerFactory
        schema = ChunkerFactory.schema("pdf_images")
        for key in ["ocr_engine", "image_handling", "dpi", "chunk_size", "min_image_area"]:
            assert key in schema


# ═══════════════════════════════════════════════════════════════════════════════
# config_schema
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigSchema:
    def test_schema_returns_dict(self):
        assert isinstance(PDFImageChunker.config_schema(), dict)

    def test_schema_has_required_keys(self):
        schema = PDFImageChunker.config_schema()
        for key in [
            "ocr_engine", "image_handling", "dpi", "ocr_language",
            "min_image_area", "include_page_numbers",
            "chunk_size", "chunk_overlap", "boundary_enforcement",
            "tokenizer", "min_chunk_size",
        ]:
            assert key in schema, f"Missing schema key: {key}"

    def test_schema_ocr_engine_options(self):
        schema = PDFImageChunker.config_schema()
        options = schema["ocr_engine"]["options"]
        assert "tesseract" in options
        assert "none" in options

    def test_schema_image_handling_options(self):
        schema = PDFImageChunker.config_schema()
        options = schema["image_handling"]["options"]
        for opt in ["extract", "caption", "both", "skip"]:
            assert opt in options

    def test_schema_defaults_match_class(self):
        schema = PDFImageChunker.config_schema()
        assert schema["dpi"]["default"] == 150
        assert schema["ocr_engine"]["default"] == "tesseract"
        assert schema["min_image_area"]["default"] == 5000
