"""
HybridChunker — meta-chunking strategy for RAGLab R2.

This is NOT a document-type chunker (it doesn't know about PDFs or DOCX).
It is a META-STRATEGY that wraps any structural chunker and applies
token+boundary backtracking WITHIN each structural unit.

Why this exists:
    A DOCXChunker splits at headings → gives you sections.
    A MarkdownChunker splits at headers → gives you sections.
    But what if a section is 3,000 tokens? The structural chunker alone
    won't subdivide it further.
    HybridChunker: structure first → then token-window within each unit.

Configuration:
    structural_first   : bool = True
        If True (default): use the source chunker's structural awareness
        to get units, then apply split_into_windows() within each.
        If False: skip structural splitting, run split_into_windows() directly
        on the full text (equivalent to TextChunker — useful for A/B testing).

    max_unit_tokens    : int = 1000
        Units larger than this are split with split_into_windows().
        Units smaller than this are kept as-is (already within budget).

    source_chunker     : str = "text"
        Which chunker to use for structural splitting. Supported:
        "text", "markdown", "html" (any chunker whose _chunk() returns
        ChunkModel list where each chunk represents a structural unit).
        For PDF/DOCX the structural split is already embedded in those
        chunkers — set source_chunker="text" and use those chunkers directly.

    chunk_size         : int = 500    — token window for intra-unit splits
    chunk_overlap      : int = 50     — overlap for intra-unit splits
    boundary_enforcement: bool = True — sentence backtracking within units
    boundary_chars     : list = ['.','!','?']
    tokenizer          : str = "tiktoken" | "word_count"
    min_chunk_size     : int = 50
    source_config      : dict = {}    — forwarded to source chunker constructor

Naming distinction (critical — do NOT confuse with hybrid RETRIEVAL):
    HybridChunker   = structural chunking + token windowing (THIS FILE)
    HybridRetriever = dense + sparse retrieval fusion (raglab-retrievers, R3)
    These are completely separate concepts. The UI Control Panel must label
    them unambiguously: "Hybrid Chunking" vs "Hybrid Retrieval".
"""

from __future__ import annotations

import uuid
from typing import Any

from raglab_common.models import ChunkModel

from raglab_chunkers._boundary import count_tokens, split_into_windows
from raglab_chunkers.base import BaseChunker


class HybridChunker(BaseChunker):
    """
    Meta-chunking strategy: structural splitting + token+boundary backtracking.

    The source chunker provides structural units (headings, pages, tags, rows).
    HybridChunker then applies split_into_windows() within any unit that
    exceeds max_unit_tokens — keeping large sections within token budget
    while respecting the document's own structural boundaries.

    Activates in R2. Registered in ChunkerFactory as "hybrid".
    """

    chunker_type: str = "hybrid"

    _DEFAULT_STRUCTURAL_FIRST: bool = True
    _DEFAULT_MAX_UNIT_TOKENS: int = 1000
    _DEFAULT_SOURCE_CHUNKER: str = "markdown"
    _DEFAULT_CHUNK_SIZE: int = 500
    _DEFAULT_CHUNK_OVERLAP: int = 50
    _DEFAULT_BOUNDARY_ENFORCEMENT: bool = True
    _DEFAULT_BOUNDARY_CHARS: list[str] = [".", "!", "?"]
    _DEFAULT_TOKENIZER: str = "tiktoken"
    _DEFAULT_MIN_CHUNK_SIZE: int = 50

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = config or {}

        self.structural_first: bool = bool(
            cfg.get("structural_first", self._DEFAULT_STRUCTURAL_FIRST)
        )
        self.max_unit_tokens: int = int(
            cfg.get("max_unit_tokens", self._DEFAULT_MAX_UNIT_TOKENS)
        )
        self.source_chunker_type: str = cfg.get("source_chunker", self._DEFAULT_SOURCE_CHUNKER)
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
        self.source_config: dict[str, Any] = cfg.get("source_config", {})

        # Validation
        if self.max_unit_tokens < 1:
            raise ValueError(f"max_unit_tokens must be >= 1, got {self.max_unit_tokens}")
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
            raise ValueError(
                f"tokenizer must be 'tiktoken' or 'word_count', got {self.tokenizer!r}"
            )

        # Build source chunker (lazy — avoid circular import at module load)
        self._source_chunker: BaseChunker | None = None

    def _get_source_chunker(self) -> BaseChunker:
        """
        Lazy-init the source chunker via ChunkerFactory.

        Import is deferred to avoid circular imports at module load time
        (factory.py imports HybridChunker; HybridChunker would import factory).
        """
        if self._source_chunker is None:
            from raglab_chunkers.factory import ChunkerFactory
            # Forward tokenizer and structural params to source chunker
            source_cfg = {
                "tokenizer": self.tokenizer,
                "chunk_size": self.max_unit_tokens,  # source gets max_unit_tokens as its chunk_size
                "chunk_overlap": 0,                   # no overlap at structural level
                "boundary_enforcement": False,        # no backtracking at structural level
                **self.source_config,
            }
            self._source_chunker = ChunkerFactory.create(
                self.source_chunker_type, config=source_cfg
            )
        return self._source_chunker

    def _chunk(self, text: str, doc_id: str, metadata: dict[str, Any]) -> list[ChunkModel]:
        """
        Two-pass hybrid chunking:
          Pass 1: Use source chunker to get structural units.
          Pass 2: For each unit exceeding max_unit_tokens, apply split_into_windows().
                  For units within budget, keep as-is.

        If structural_first=False, skip Pass 1 and apply split_into_windows()
        directly — useful for comparison / A/B testing.
        """
        if not self.structural_first:
            # Bypass structural pass — pure token windowing
            return self._window_text(text, doc_id, metadata, start_index=0, unit_meta={})

        # Pass 1: structural units from source chunker
        source_chunker = self._get_source_chunker()
        structural_units = source_chunker.chunk(text, doc_id=doc_id, metadata=metadata)

        if not structural_units:
            # Source chunker produced nothing — fall back to direct windowing
            return self._window_text(text, doc_id, metadata, start_index=0, unit_meta={})

        # Pass 2: subdivide oversized units
        final_chunks: list[ChunkModel] = []
        chunk_index = 0

        for unit in structural_units:
            unit_tokens = count_tokens(unit.text, mode=self.tokenizer)

            if unit_tokens <= self.max_unit_tokens and unit_tokens <= self.chunk_size:
                # Unit is within both budgets — emit as-is, reindex
                final_chunks.append(ChunkModel(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=unit.text,
                    chunk_index=chunk_index,
                    token_count=unit_tokens,
                    metadata={
                        **unit.metadata,
                        "hybrid_source": self.source_chunker_type,
                        "hybrid_pass": "structural",
                        "chunker": self.chunker_type,
                    },
                ))
                chunk_index += 1
            else:
                # Unit exceeds budget — subdivide with split_into_windows()
                sub_chunks = self._window_text(
                    text=unit.text,
                    doc_id=doc_id,
                    metadata={
                        **unit.metadata,
                        "hybrid_source": self.source_chunker_type,
                        "hybrid_pass": "token_window",
                        "chunker": self.chunker_type,
                    },
                    start_index=chunk_index,
                    unit_meta=unit.metadata,
                )
                final_chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)

        return final_chunks

    def _window_text(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any],
        start_index: int,
        unit_meta: dict[str, Any],
    ) -> list[ChunkModel]:
        """Apply split_into_windows() to `text` and return ChunkModel list."""
        raw_chunks = split_into_windows(
            text=text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            boundary_enforcement=self.boundary_enforcement,
            boundary_chars=self.boundary_chars,
            tokenizer=self.tokenizer,
            min_chunk_size=self.min_chunk_size,
        )
        result = []
        for i, chunk_text in enumerate(raw_chunks):
            result.append(ChunkModel(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                text=chunk_text,
                chunk_index=start_index + i,
                token_count=count_tokens(chunk_text, mode=self.tokenizer),
                metadata={
                    **metadata,
                    "chunker": self.chunker_type,
                    "tokenizer": self.tokenizer,
                    "boundary_enforcement": self.boundary_enforcement,
                },
            ))
        return result

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return {
            "structural_first": {
                "type": "bool",
                "default": cls._DEFAULT_STRUCTURAL_FIRST,
                "description": (
                    "If True: use source chunker for structure, then token-window within "
                    "each unit. If False: skip structure, apply token windowing directly."
                ),
            },
            "max_unit_tokens": {
                "type": "int",
                "default": cls._DEFAULT_MAX_UNIT_TOKENS,
                "min": 50,
                "max": 8000,
                "description": (
                    "Maximum tokens per structural unit before intra-unit "
                    "token windowing is applied."
                ),
            },
            "source_chunker": {
                "type": "str",
                "default": cls._DEFAULT_SOURCE_CHUNKER,
                "options": ["text", "markdown", "html"],
                "description": (
                    "Which chunker to use for structural splitting (Pass 1). "
                    "For PDF/DOCX, use those chunkers directly — they embed "
                    "structural logic internally."
                ),
            },
            "chunk_size": {
                "type": "int",
                "default": cls._DEFAULT_CHUNK_SIZE,
                "min": 50,
                "max": 4000,
                "description": "Token window size for intra-unit splits (Pass 2).",
            },
            "chunk_overlap": {
                "type": "int",
                "default": cls._DEFAULT_CHUNK_OVERLAP,
                "min": 0,
                "max": 500,
                "description": "Overlap tokens between intra-unit sub-chunks.",
            },
            "boundary_enforcement": {
                "type": "bool",
                "default": cls._DEFAULT_BOUNDARY_ENFORCEMENT,
                "description": "Apply sentence boundary backtracking within token windows.",
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
                "description": "Token counting mode.",
            },
            "min_chunk_size": {
                "type": "int",
                "default": cls._DEFAULT_MIN_CHUNK_SIZE,
                "min": 1,
                "max": 200,
                "description": "Minimum tokens per chunk (no backtracking past this).",
            },
            "source_config": {
                "type": "dict",
                "default": {},
                "description": "Additional config forwarded to the source chunker.",
            },
        }
