"""
PDFImageChunker — OCR text extraction + image region handling for RAG.

Strategy (R4):
    Standard PDFChunker uses PyMuPDF's text layer — only works on digitally
    created PDFs. PDFImageChunker handles the harder cases:

    1. **OCR path** — rasterise each PDF page to PIL image, run pytesseract
       to extract text, then chunk that text via split_into_windows(). Handles
       scanned PDFs, image-only PDFs, and PDFs with poor text layers.

    2. **Image extraction path** — detect image regions in the PDF page,
       extract each as a PIL image, and either:
         a. Store raw image bytes in ChunkModel.metadata["image_bytes"] (base64).
         b. Send to llm-service for multimodal captioning (Phase 2 path).

    3. **Both** — OCR text chunks + image region chunks in the same output list.

OCR engine options (R4 active):
    "tesseract" — pytesseract (local, no API key). Handles basic scanned docs.
    "none"      — skip OCR, rasterise and emit image blocks only.

Image handling options:
    "extract"  — extract images, store as base64 in metadata (no caption).
    "caption"  — extract images + caption via llm-service (Phase 2).
    "both"     — OCR text + image extraction.
    "skip"     — ignore images entirely (OCR text only).

Parameters:
    ocr_engine          : str   = "tesseract"
    image_handling      : str   = "extract"
    dpi                 : int   = 150     — render DPI for rasterisation
    ocr_language        : str   = "eng"   — tesseract language code
    min_image_area      : int   = 5000    — px² minimum to keep an image region
    include_page_numbers: bool  = True
    chunk_size          : int   = 500
    chunk_overlap       : int   = 50
    boundary_enforcement: bool  = True
    tokenizer           : str   = "word_count"
    min_chunk_size      : int   = 20

Reuse rule: text from OCR is chunked via split_into_windows() — same
boundary-backtracking algorithm as every other chunker.
"""

from __future__ import annotations

import base64
import io
import uuid
from typing import Any

from raglab_common.exceptions import ChunkerError
from raglab_common.models import ChunkModel

from raglab_chunkers._boundary import count_tokens, split_into_windows
from raglab_chunkers.base import BaseChunker

# Top-level optional imports — patchable in tests
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore[assignment]

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]

_VALID_OCR_ENGINES = ("tesseract", "none")
_VALID_IMAGE_HANDLING = ("extract", "caption", "both", "skip")


class PDFImageChunker(BaseChunker):
    """
    OCR-capable PDF chunker for scanned/image PDFs. Activates in R4.

    Handles both text extraction (via OCR) and image region extraction
    from PDF pages. Standard PDFChunker handles text-layer PDFs.
    """

    chunker_type: str = "pdf_images"

    _DEFAULT_OCR_ENGINE: str = "tesseract"
    _DEFAULT_IMAGE_HANDLING: str = "extract"
    _DEFAULT_DPI: int = 150
    _DEFAULT_OCR_LANGUAGE: str = "eng"
    _DEFAULT_MIN_IMAGE_AREA: int = 5000
    _DEFAULT_INCLUDE_PAGE_NUMBERS: bool = True
    _DEFAULT_CHUNK_SIZE: int = 500
    _DEFAULT_CHUNK_OVERLAP: int = 50
    _DEFAULT_BOUNDARY_ENFORCEMENT: bool = True
    _DEFAULT_TOKENIZER: str = "word_count"
    _DEFAULT_MIN_CHUNK_SIZE: int = 20

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = config or {}

        self.ocr_engine: str = cfg.get("ocr_engine", self._DEFAULT_OCR_ENGINE)
        self.image_handling: str = cfg.get("image_handling", self._DEFAULT_IMAGE_HANDLING)
        self.dpi: int = int(cfg.get("dpi", self._DEFAULT_DPI))
        self.ocr_language: str = cfg.get("ocr_language", self._DEFAULT_OCR_LANGUAGE)
        self.min_image_area: int = int(cfg.get("min_image_area", self._DEFAULT_MIN_IMAGE_AREA))
        self.include_page_numbers: bool = bool(
            cfg.get("include_page_numbers", self._DEFAULT_INCLUDE_PAGE_NUMBERS)
        )
        self.chunk_size: int = int(cfg.get("chunk_size", self._DEFAULT_CHUNK_SIZE))
        self.chunk_overlap: int = int(cfg.get("chunk_overlap", self._DEFAULT_CHUNK_OVERLAP))
        self.boundary_enforcement: bool = bool(
            cfg.get("boundary_enforcement", self._DEFAULT_BOUNDARY_ENFORCEMENT)
        )
        self.tokenizer: str = cfg.get("tokenizer", self._DEFAULT_TOKENIZER)
        self.min_chunk_size: int = int(cfg.get("min_chunk_size", self._DEFAULT_MIN_CHUNK_SIZE))

        # Validation
        if self.ocr_engine not in _VALID_OCR_ENGINES:
            raise ValueError(
                f"ocr_engine must be one of {_VALID_OCR_ENGINES}, got {self.ocr_engine!r}"
            )
        if self.image_handling not in _VALID_IMAGE_HANDLING:
            raise ValueError(
                f"image_handling must be one of {_VALID_IMAGE_HANDLING}, "
                f"got {self.image_handling!r}"
            )
        if self.dpi < 72:
            raise ValueError(f"dpi must be >= 72, got {self.dpi}")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        if self.tokenizer not in ("tiktoken", "word_count"):
            raise ValueError(
                f"tokenizer must be 'tiktoken' or 'word_count', got {self.tokenizer!r}"
            )

    # ── Public entry points ────────────────────────────────────────────────────

    def chunk_pdf_bytes(
        self,
        pdf_bytes: bytes,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[ChunkModel]:
        """
        Chunk a PDF from raw bytes using OCR and/or image extraction.

        Args:
            pdf_bytes: Raw PDF bytes.
            doc_id:    Document identifier.
            metadata:  Optional base metadata merged into each ChunkModel.

        Returns:
            List of ChunkModel — OCR text chunks and/or image region chunks.
        """
        metadata = metadata or {}
        if fitz is None:
            raise ChunkerError("PyMuPDF not installed. Run: pip install pymupdf")

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            raise ChunkerError(f"Failed to open PDF: {exc}") from exc

        chunks: list[ChunkModel] = []
        chunk_index = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_label = page_num + 1

            # ── OCR path ──────────────────────────────────────────────────────
            if self.image_handling in ("skip", "extract", "both") and self.ocr_engine != "none":
                ocr_text = self._ocr_page(page, page_label)
                if ocr_text.strip():
                    text_chunks = self._text_to_chunks(
                        ocr_text, doc_id, metadata, page_label, chunk_index
                    )
                    chunks.extend(text_chunks)
                    chunk_index += len(text_chunks)

            # ── Image extraction path ─────────────────────────────────────────
            if self.image_handling in ("extract", "both"):
                image_chunks = self._extract_images(page, doc, doc_id, metadata, page_label, chunk_index)
                chunks.extend(image_chunks)
                chunk_index += len(image_chunks)

        doc.close()
        return chunks

    def _chunk(self, text: str, doc_id: str, metadata: dict[str, Any]) -> list[ChunkModel]:
        """
        Fallback: chunk pre-extracted text (plain text input).

        When called via the standard chunk() API with plain text,
        applies OCR-style chunking without page rasterisation.
        """
        return self._text_to_chunks(text, doc_id, metadata, page_number=None, start_index=0)

    # ── OCR helpers ────────────────────────────────────────────────────────────

    def _ocr_page(self, page: Any, page_number: int) -> str:
        """Rasterise a PDF page and run OCR. Returns extracted text."""
        if pytesseract is None or Image is None:
            raise ChunkerError(
                "pytesseract + Pillow not installed. Run: pip install pytesseract pillow"
            )

        # Render page to pixmap at configured DPI
        mat = page.get_pixmap(dpi=self.dpi)
        img_bytes = mat.tobytes("png")

        pil_img = Image.open(io.BytesIO(img_bytes))

        try:
            text = pytesseract.image_to_string(pil_img, lang=self.ocr_language)
        except Exception as exc:
            self._log.warning(
                "ocr.page_failed",
                page=page_number,
                error=str(exc),
            )
            text = ""

        return text

    def _text_to_chunks(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any],
        page_number: int | None,
        start_index: int,
    ) -> list[ChunkModel]:
        """Split OCR text into ChunkModel list via split_into_windows()."""
        if not text.strip():
            return []

        raw_chunks = split_into_windows(
            text=text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            boundary_enforcement=self.boundary_enforcement,
            tokenizer=self.tokenizer,
            min_chunk_size=self.min_chunk_size,
        )

        result = []
        for i, chunk_text in enumerate(raw_chunks):
            chunk_meta: dict[str, Any] = {
                **metadata,
                "chunker": self.chunker_type,
                "chunk_type": "text",
                "ocr_engine": self.ocr_engine,
                "tokenizer": self.tokenizer,
            }
            if self.include_page_numbers and page_number is not None:
                chunk_meta["page_number"] = page_number

            result.append(ChunkModel(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=chunk_text,
                chunk_index=start_index + i,
                token_count=count_tokens(chunk_text, mode=self.tokenizer),
                metadata=chunk_meta,
            ))
        return result

    # ── Image extraction helpers ───────────────────────────────────────────────

    def _extract_images(
        self,
        page: Any,
        doc: Any,
        doc_id: str,
        metadata: dict[str, Any],
        page_number: int,
        start_index: int,
    ) -> list[ChunkModel]:
        """
        Extract image regions from a PDF page.

        Each image region becomes a ChunkModel with:
          - text: a placeholder description ("Image on page N, region M")
          - metadata["image_bytes"]: base64-encoded PNG of the image region
          - metadata["chunk_type"]: "image"
          - metadata["image_index"]: region index on the page
        """
        image_chunks: list[ChunkModel] = []
        try:
            image_list = page.get_images(full=True)
        except Exception:
            return []

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image.get("image", b"")
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Skip images below minimum area threshold
                if width * height < self.min_image_area:
                    continue

                # Base64-encode for storage
                img_b64 = base64.b64encode(img_bytes).decode("ascii")
                placeholder_text = (
                    f"[Image on page {page_number}, region {img_idx + 1}] "
                    f"Width: {width}px, Height: {height}px."
                )

                chunk_meta: dict[str, Any] = {
                    **metadata,
                    "chunker": self.chunker_type,
                    "chunk_type": "image",
                    "page_number": page_number,
                    "image_index": img_idx,
                    "image_width": width,
                    "image_height": height,
                    "image_bytes": img_b64,
                    "image_ext": base_image.get("ext", "png"),
                    "ocr_engine": self.ocr_engine,
                    "captioned": False,
                }

                image_chunks.append(ChunkModel(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=placeholder_text,
                    chunk_index=start_index + len(image_chunks),
                    token_count=count_tokens(placeholder_text, mode=self.tokenizer),
                    metadata=chunk_meta,
                ))

            except Exception as exc:
                self._log.warning(
                    "image_extraction.failed",
                    page=page_number,
                    img_index=img_idx,
                    error=str(exc),
                )
                continue

        return image_chunks

    # ── Schema ─────────────────────────────────────────────────────────────────

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return {
            "ocr_engine": {
                "type": "str", "default": cls._DEFAULT_OCR_ENGINE,
                "options": ["tesseract", "none"],
                "description": "OCR engine for text extraction from rasterised pages.",
            },
            "image_handling": {
                "type": "str", "default": cls._DEFAULT_IMAGE_HANDLING,
                "options": ["extract", "caption", "both", "skip"],
                "description": (
                    "extract: store image bytes in metadata. "
                    "caption: send to llm-service for multimodal caption (R4 Phase 2). "
                    "both: OCR text + image extraction. "
                    "skip: OCR text only."
                ),
            },
            "dpi": {
                "type": "int", "default": cls._DEFAULT_DPI,
                "min": 72, "max": 600,
                "description": "Page rasterisation DPI. Higher = better OCR, slower.",
            },
            "ocr_language": {
                "type": "str", "default": cls._DEFAULT_OCR_LANGUAGE,
                "description": "Tesseract language code (e.g. 'eng', 'deu', 'fra+eng').",
            },
            "min_image_area": {
                "type": "int", "default": cls._DEFAULT_MIN_IMAGE_AREA,
                "min": 0, "max": 100000,
                "description": "Minimum pixel area (width×height) to keep an image region.",
            },
            "include_page_numbers": {
                "type": "bool", "default": cls._DEFAULT_INCLUDE_PAGE_NUMBERS,
                "description": "Inject page_number into chunk metadata.",
            },
            "chunk_size": {
                "type": "int", "default": cls._DEFAULT_CHUNK_SIZE,
                "min": 20, "max": 4000,
                "description": "Target token count per OCR text chunk.",
            },
            "chunk_overlap": {
                "type": "int", "default": cls._DEFAULT_CHUNK_OVERLAP,
                "min": 0, "max": 500,
                "description": "Overlap tokens between consecutive OCR text chunks.",
            },
            "boundary_enforcement": {
                "type": "bool", "default": cls._DEFAULT_BOUNDARY_ENFORCEMENT,
                "description": "Backtrack to sentence boundary in OCR text chunking.",
            },
            "tokenizer": {
                "type": "str", "default": cls._DEFAULT_TOKENIZER,
                "options": ["tiktoken", "word_count"],
                "description": "Token counting mode.",
            },
            "min_chunk_size": {
                "type": "int", "default": cls._DEFAULT_MIN_CHUNK_SIZE,
                "min": 1, "max": 200,
                "description": "Minimum tokens per OCR text chunk.",
            },
        }
