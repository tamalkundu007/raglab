"""
TextChunker — fixed-token chunking with sentence boundary backtracking.

The only active chunker in R1. All other chunkers are activated in R2+
and reuse `_boundary.split_into_windows()` within their structural units.

Algorithm (from FRS):
    1. Tokenise input with tiktoken (cl100k_base) or word-count.
    2. Slide a window of `chunk_size` tokens across the text with `chunk_overlap`.
    3. If a window ends mid-sentence, walk backward word-by-word until a
       boundary character (`.`, `!`, `?`) is found.
    4. If no boundary is found within `min_chunk_size` tokens of the window
       end, return the original window unchanged.

Parameters (all configurable, shown in UI Control Panel):
    chunk_size          : int   = 500    — target tokens per chunk
    chunk_overlap       : int   = 50     — overlap tokens between chunks
    boundary_enforcement: bool  = True   — enable sentence backtracking
    boundary_chars      : list  = ['.','!','?']
    tokenizer           : str   = "tiktoken" | "word_count"
    min_chunk_size      : int   = 50     — minimum tokens; no backtrack below
"""

from __future__ import annotations

import uuid
from typing import Any

from raglab_common.models import ChunkModel

from raglab_chunkers._boundary import split_into_windows, count_tokens
from raglab_chunkers.base import BaseChunker


class TextChunker(BaseChunker):
    """
    Fixed-token chunker with sentence boundary backtracking.

    Active in R1. The simplest chunker — works on plain text strings.
    For structured documents (PDF, DOCX, MD, etc.) use the R2+ chunkers
    which add structural awareness before delegating to the same algorithm.
    """

    chunker_type: str = "text"

    # Parameter defaults — match FRS spec exactly
    _DEFAULT_CHUNK_SIZE: int = 500
    _DEFAULT_CHUNK_OVERLAP: int = 50
    _DEFAULT_BOUNDARY_ENFORCEMENT: bool = True
    _DEFAULT_BOUNDARY_CHARS: list[str] = [".", "!", "?"]
    _DEFAULT_TOKENIZER: str = "tiktoken"
    _DEFAULT_MIN_CHUNK_SIZE: int = 50

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

        # Validate
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        if self.min_chunk_size < 1:
            raise ValueError(f"min_chunk_size must be >= 1, got {self.min_chunk_size}")
        if self.tokenizer not in ("tiktoken", "word_count"):
            raise ValueError(f"tokenizer must be 'tiktoken' or 'word_count', got {self.tokenizer!r}")

    def _chunk(self, text: str, doc_id: str, metadata: dict[str, Any]) -> list[ChunkModel]:
        """Split plain text into ChunkModel instances."""
        raw_chunks = split_into_windows(
            text=text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            boundary_enforcement=self.boundary_enforcement,
            boundary_chars=self.boundary_chars,
            tokenizer=self.tokenizer,
            min_chunk_size=self.min_chunk_size,
        )

        return [
            ChunkModel(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=chunk_text,
                chunk_index=i,
                token_count=count_tokens(chunk_text, mode=self.tokenizer),
                metadata={
                    **metadata,
                    "chunker": self.chunker_type,
                    "chunk_size_config": self.chunk_size,
                    "chunk_overlap_config": self.chunk_overlap,
                    "boundary_enforcement": self.boundary_enforcement,
                    "tokenizer": self.tokenizer,
                },
            )
            for i, chunk_text in enumerate(raw_chunks)
        ]

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        """UI-renderable parameter schema for the Control Panel."""
        return {
            "chunk_size": {
                "type": "int",
                "default": cls._DEFAULT_CHUNK_SIZE,
                "min": 50,
                "max": 4000,
                "description": "Target token count per chunk.",
            },
            "chunk_overlap": {
                "type": "int",
                "default": cls._DEFAULT_CHUNK_OVERLAP,
                "min": 0,
                "max": 500,
                "description": "Overlap token count between consecutive chunks.",
            },
            "boundary_enforcement": {
                "type": "bool",
                "default": cls._DEFAULT_BOUNDARY_ENFORCEMENT,
                "description": (
                    "If True, backtrack to the nearest sentence boundary "
                    "rather than cutting mid-sentence."
                ),
            },
            "boundary_chars": {
                "type": "list",
                "default": cls._DEFAULT_BOUNDARY_CHARS,
                "description": "Characters treated as sentence boundaries.",
            },
            "tokenizer": {
                "type": "str",
                "default": cls._DEFAULT_TOKENIZER,
                "options": ["tiktoken", "word_count"],
                "description": (
                    "tiktoken uses cl100k_base (GPT-4 compatible). "
                    "word_count is faster with no external dependency."
                ),
            },
            "min_chunk_size": {
                "type": "int",
                "default": cls._DEFAULT_MIN_CHUNK_SIZE,
                "min": 1,
                "max": 200,
                "description": (
                    "Minimum token count per chunk. "
                    "Boundary backtracking never produces a chunk smaller than this."
                ),
            },
        }
