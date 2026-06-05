"""
PDFChunker — text-only PDF chunking with page awareness and boundary backtracking.

Strategy (R2 FRS spec):
    1. Extract text page-by-page using PyMuPDF (fitz).
    2. If `respect_page_boundary=True`, treat each page as a structural unit
       and run `split_into_windows()` within it — chunks never cross page breaks.
    3. If `respect_page_boundary=False`, concatenate all pages into a single
       text stream then chunk normally (same as TextChunker on the extracted text).
    4. `page_metadata=True` injects `page_number` into each ChunkModel.metadata.

Parameters:
    chunk_size           : int   = 500    — target tokens per chunk
    chunk_overlap        : int   = 50     — overlap tokens between chunks
    boundary_enforcement : bool  = True   — sentence boundary backtracking
    boundary_chars       : list  = ['.','!','?']
    tokenizer            : str   = "tiktoken" | "word_count"
    min_chunk_size       : int   = 50     — minimum tokens, no backtrack below
    respect_page_boundary: bool  = True   — never cross PDF page boundaries
    page_metadata        : bool  = True   — inject page_number into metadata

Reuse rule: token+boundary splitting delegates entirely to
`_boundary.split_into_windows()` — no reimplementation here.
"""

from __future__ import annotations

import uuid
from typing import Any

from raglab_common.exceptions import ChunkerError, NotImplementedFeatureError
from raglab_common.models import ChunkModel

from raglab_chunkers._boundary import count_tokens, split_into_windows

# Top-level import so tests can patch raglab_chunkers.pdf_chunker.fitz
try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    fitz = None  # type: ignore[assignment]
    _FITZ_AVAILABLE = False
from raglab_chunkers.base import BaseChunker


class PDFChunker(BaseChunker):
    """
    Text-only PDF chunker with page awareness.

    Requires PyMuPDF (fitz). Activates in R2.
    For PDFs with scanned images, use PDFImagesChunker (R2 extended scope).
    """

    chunker_type: str = "pdf"

    _DEFAULT_CHUNK_SIZE: int = 500
    _DEFAULT_CHUNK_OVERLAP: int = 50
    _DEFAULT_BOUNDARY_ENFORCEMENT: bool = True
    _DEFAULT_BOUNDARY_CHARS: list[str] = [".", "!", "?"]
    _DEFAULT_TOKENIZER: str = "tiktoken"
    _DEFAULT_MIN_CHUNK_SIZE: int = 50
    _DEFAULT_RESPECT_PAGE_BOUNDARY: bool = True
    _DEFAULT_PAGE_METADATA: bool = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = config or {}
        self.chunk_size: int = int(cfg.get("chunk_size", self._DEFAULT_CHUNK_SIZE))
        self.chunk_overlap: int = int(cfg.get("chunk_overlap", self._DEFAULT_CHUNK_OVERLAP))
        self.boundary_enforcement: bool = bool(
            cfg.get("boundary_enforcement", self._DEFAULT_BOUNDARY_ENFORCEMENT)
        )
        self.boundary_chars: frozenset[str] = frozenset(
            cfg.get("boundary_chars", self._DEFAULT_BOUNDARY_CHARS)
        )
        self.tokenizer: str = cfg.get("tokenizer", self._DEFAULT_TOKENIZER)
        self.min_chunk_size: int = int(cfg.get("min_chunk_size", self._DEFAULT_MIN_CHUNK_SIZE))
        self.respect_page_boundary: bool = bool(
            cfg.get("respect_page_boundary", self._DEFAULT_RESPECT_PAGE_BOUNDARY)
        )
        self.page_metadata: bool = bool(cfg.get("page_metadata", self._DEFAULT_PAGE_METADATA))

        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        if self.tokenizer not in ("tiktoken", "word_count"):
            raise ValueError(f"tokenizer must be 'tiktoken' or 'word_count', got {self.tokenizer!r}")

    def chunk_pdf_bytes(
        self, pdf_bytes: bytes, doc_id: str, metadata: dict[str, Any] | None = None
    ) -> list[ChunkModel]:
        """
        Chunk from raw PDF bytes.

        Args:
            pdf_bytes: Raw PDF file bytes.
            doc_id:    Document ID.
            metadata:  Optional metadata dict.

        Returns:
            List of ChunkModel instances.
        """
        metadata = metadata or {}
        if not _FITZ_AVAILABLE or fitz is None:
            raise ChunkerError("PyMuPDF not installed. Run: pip install pymupdf")

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            raise ChunkerError(f"Failed to open PDF: {exc}") from exc

        pages: list[tuple[int, str]] = []  # (1-based page_number, text)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text and text.strip():
                pages.append((page_num + 1, text.strip()))
        doc.close()

        return self._pages_to_chunks(pages, doc_id, metadata)

    def _chunk(self, text: str, doc_id: str, metadata: dict[str, Any]) -> list[ChunkModel]:
        """
        Chunk from pre-extracted text string (plain text input path).

        When called via the standard chunk() API with plain text, treats the
        entire text as a single page (page_number=None).
        """
        pages = [(None, text)]
        return self._pages_to_chunks(pages, doc_id, metadata)

    def _pages_to_chunks(
        self,
        pages: list[tuple[int | None, str]],
        doc_id: str,
        metadata: dict[str, Any],
    ) -> list[ChunkModel]:
        """Convert page texts to ChunkModel list, respecting page boundaries if configured."""
        chunks: list[ChunkModel] = []
        chunk_index = 0

        if self.respect_page_boundary:
            # Structural unit = page → split_into_windows within each page
            for page_number, page_text in pages:
                raw_chunks = split_into_windows(
                    text=page_text,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    boundary_enforcement=self.boundary_enforcement,
                    boundary_chars=self.boundary_chars,
                    tokenizer=self.tokenizer,
                    min_chunk_size=self.min_chunk_size,
                )
                for chunk_text in raw_chunks:
                    chunk_meta = {
                        **metadata,
                        "chunker": self.chunker_type,
                        "chunk_size_config": self.chunk_size,
                        "tokenizer": self.tokenizer,
                        "boundary_enforcement": self.boundary_enforcement,
                    }
                    if self.page_metadata and page_number is not None:
                        chunk_meta["page_number"] = page_number

                    chunks.append(ChunkModel(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        token_count=count_tokens(chunk_text, mode=self.tokenizer),
                        metadata=chunk_meta,
                    ))
                    chunk_index += 1
        else:
            # Concatenate all pages, chunk as single stream
            full_text = "\n\n".join(text for _, text in pages)
            raw_chunks = split_into_windows(
                text=full_text,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                boundary_enforcement=self.boundary_enforcement,
                boundary_chars=self.boundary_chars,
                tokenizer=self.tokenizer,
                min_chunk_size=self.min_chunk_size,
            )
            for chunk_text in raw_chunks:
                chunks.append(ChunkModel(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    token_count=count_tokens(chunk_text, mode=self.tokenizer),
                    metadata={
                        **metadata,
                        "chunker": self.chunker_type,
                        "chunk_size_config": self.chunk_size,
                        "tokenizer": self.tokenizer,
                        "boundary_enforcement": self.boundary_enforcement,
                    },
                ))
                chunk_index += 1

        return chunks

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return {
            "chunk_size": {
                "type": "int", "default": cls._DEFAULT_CHUNK_SIZE,
                "min": 50, "max": 4000,
                "description": "Target token count per chunk.",
            },
            "chunk_overlap": {
                "type": "int", "default": cls._DEFAULT_CHUNK_OVERLAP,
                "min": 0, "max": 500,
                "description": "Overlap token count between consecutive chunks.",
            },
            "boundary_enforcement": {
                "type": "bool", "default": cls._DEFAULT_BOUNDARY_ENFORCEMENT,
                "description": "Backtrack to nearest sentence boundary.",
            },
            "boundary_chars": {
                "type": "list", "default": cls._DEFAULT_BOUNDARY_CHARS,
                "description": "Characters treated as sentence boundaries.",
            },
            "tokenizer": {
                "type": "str", "default": cls._DEFAULT_TOKENIZER,
                "options": ["tiktoken", "word_count"],
                "description": "Token counting mode.",
            },
            "min_chunk_size": {
                "type": "int", "default": cls._DEFAULT_MIN_CHUNK_SIZE,
                "min": 1, "max": 200,
                "description": "Minimum tokens per chunk.",
            },
            "respect_page_boundary": {
                "type": "bool", "default": cls._DEFAULT_RESPECT_PAGE_BOUNDARY,
                "description": "Never split chunks across PDF page boundaries.",
            },
            "page_metadata": {
                "type": "bool", "default": cls._DEFAULT_PAGE_METADATA,
                "description": "Inject page_number into each chunk's metadata.",
            },
        }
