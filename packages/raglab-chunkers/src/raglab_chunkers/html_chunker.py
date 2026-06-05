"""
HTMLChunker — semantic tag-aware HTML chunking with boundary backtracking.

Strategy (R2 FRS spec):
    1. Parse HTML with BeautifulSoup, extract text from semantic tags
       (p, article, section, div, li, td, etc.).
    2. Each semantic node becomes a structural unit.
    3. If a node's text exceeds `chunk_size_fallback` tokens, apply
       `split_into_windows()` within it.
    4. Small nodes are concatenated greedily until the token budget fills,
       then `split_into_windows()` runs on the concatenated block.
    5. Scripts and styles stripped by default.

Parameters:
    split_tags          : list  = ['p','article','section','div','li','td','th','blockquote','h1'..'h6']
    include_tag_attrs   : bool  = False — include tag name in chunk metadata
    strip_scripts_styles: bool  = True  — remove script/style elements
    chunk_size_fallback : int   = 500   — max tokens before intra-node split
    overlap_fallback    : int   = 50    — overlap for intra-node splits
    boundary_enforcement: bool  = True
    boundary_chars      : list  = ['.','!','?']
    tokenizer           : str   = "tiktoken" | "word_count"
    min_chunk_size      : int   = 50
"""

from __future__ import annotations

import uuid
from typing import Any

from raglab_common.models import ChunkModel

from raglab_chunkers._boundary import count_tokens, split_into_windows
from raglab_chunkers.base import BaseChunker

_DEFAULT_SPLIT_TAGS = [
    "p", "article", "section", "div", "li", "td", "th",
    "blockquote", "h1", "h2", "h3", "h4", "h5", "h6",
    "pre", "code",
]


class HTMLChunker(BaseChunker):
    """
    Semantic tag-aware HTML chunker using BeautifulSoup. Activates in R2.
    """

    chunker_type: str = "html"

    _DEFAULT_CHUNK_SIZE_FALLBACK: int = 500
    _DEFAULT_OVERLAP_FALLBACK: int = 50
    _DEFAULT_BOUNDARY_ENFORCEMENT: bool = True
    _DEFAULT_BOUNDARY_CHARS: list[str] = [".", "!", "?"]
    _DEFAULT_TOKENIZER: str = "tiktoken"
    _DEFAULT_MIN_CHUNK_SIZE: int = 50
    _DEFAULT_STRIP_SCRIPTS_STYLES: bool = True
    _DEFAULT_INCLUDE_TAG_ATTRS: bool = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = config or {}
        self.split_tags: list[str] = cfg.get("split_tags", _DEFAULT_SPLIT_TAGS)
        self.include_tag_attrs: bool = bool(cfg.get("include_tag_attrs", self._DEFAULT_INCLUDE_TAG_ATTRS))
        self.strip_scripts_styles: bool = bool(cfg.get("strip_scripts_styles", self._DEFAULT_STRIP_SCRIPTS_STYLES))
        self.chunk_size_fallback: int = int(cfg.get("chunk_size_fallback", self._DEFAULT_CHUNK_SIZE_FALLBACK))
        self.overlap_fallback: int = int(cfg.get("overlap_fallback", self._DEFAULT_OVERLAP_FALLBACK))
        self.boundary_enforcement: bool = bool(cfg.get("boundary_enforcement", self._DEFAULT_BOUNDARY_ENFORCEMENT))
        self.boundary_chars: frozenset[str] = frozenset(cfg.get("boundary_chars", self._DEFAULT_BOUNDARY_CHARS))
        self.tokenizer: str = cfg.get("tokenizer", self._DEFAULT_TOKENIZER)
        self.min_chunk_size: int = int(cfg.get("min_chunk_size", self._DEFAULT_MIN_CHUNK_SIZE))

        if self.chunk_size_fallback < 1:
            raise ValueError(f"chunk_size_fallback must be >= 1, got {self.chunk_size_fallback}")
        if self.overlap_fallback < 0:
            raise ValueError(f"overlap_fallback must be >= 0, got {self.overlap_fallback}")
        if self.overlap_fallback >= self.chunk_size_fallback:
            raise ValueError(
                f"overlap_fallback ({self.overlap_fallback}) must be < chunk_size_fallback ({self.chunk_size_fallback})"
            )
        if self.tokenizer not in ("tiktoken", "word_count"):
            raise ValueError(f"tokenizer must be 'tiktoken' or 'word_count', got {self.tokenizer!r}")

    def _chunk(self, text: str, doc_id: str, metadata: dict[str, Any]) -> list[ChunkModel]:
        """
        Parse HTML from `text`, extract semantic nodes, and chunk.

        Falls back gracefully to plain-text chunking if BeautifulSoup
        finds no matching tags.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            from raglab_common.exceptions import ChunkerError
            raise ChunkerError("beautifulsoup4 not installed. Run: pip install beautifulsoup4") from exc

        soup = BeautifulSoup(text, "html.parser")

        # Strip scripts and styles
        if self.strip_scripts_styles:
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

        # Extract text from semantic nodes
        node_texts: list[dict[str, Any]] = []
        for tag_name in self.split_tags:
            for element in soup.find_all(tag_name):
                node_text = element.get_text(separator=" ", strip=True)
                if node_text:
                    node_texts.append({
                        "text": node_text,
                        "tag": tag_name,
                    })

        # Fallback: if no nodes matched, chunk the full stripped text
        if not node_texts:
            plain = soup.get_text(separator=" ", strip=True)
            if not plain:
                return []
            raw_chunks = split_into_windows(
                text=plain,
                chunk_size=self.chunk_size_fallback,
                chunk_overlap=self.overlap_fallback,
                boundary_enforcement=self.boundary_enforcement,
                boundary_chars=self.boundary_chars,
                tokenizer=self.tokenizer,
                min_chunk_size=self.min_chunk_size,
            )
            return [
                ChunkModel(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=c,
                    chunk_index=i,
                    token_count=count_tokens(c, mode=self.tokenizer),
                    metadata={**metadata, "chunker": self.chunker_type, "tokenizer": self.tokenizer},
                )
                for i, c in enumerate(raw_chunks)
            ]

        # Greedily bin nodes into chunks
        chunks: list[ChunkModel] = []
        chunk_index = 0
        bin_texts: list[str] = []
        bin_tokens: int = 0
        bin_tags: list[str] = []

        def flush_bin() -> None:
            nonlocal bin_texts, bin_tokens, bin_tags, chunk_index
            if not bin_texts:
                return
            combined = " ".join(bin_texts)
            # If combined is still oversized, split it further
            raw = split_into_windows(
                text=combined,
                chunk_size=self.chunk_size_fallback,
                chunk_overlap=self.overlap_fallback,
                boundary_enforcement=self.boundary_enforcement,
                boundary_chars=self.boundary_chars,
                tokenizer=self.tokenizer,
                min_chunk_size=self.min_chunk_size,
            )
            for c in raw:
                chunk_meta = {**metadata, "chunker": self.chunker_type, "tokenizer": self.tokenizer}
                if self.include_tag_attrs:
                    chunk_meta["tags"] = list(set(bin_tags))
                chunks.append(ChunkModel(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    text=c,
                    chunk_index=chunk_index,
                    token_count=count_tokens(c, mode=self.tokenizer),
                    metadata=chunk_meta,
                ))
                chunk_index += 1
            bin_texts = []
            bin_tokens = 0
            bin_tags = []

        for node in node_texts:
            node_text = node["text"]
            node_tokens = count_tokens(node_text, mode=self.tokenizer)

            # Oversized single node — flush first, then split node itself
            if node_tokens > self.chunk_size_fallback:
                flush_bin()
                raw = split_into_windows(
                    text=node_text,
                    chunk_size=self.chunk_size_fallback,
                    chunk_overlap=self.overlap_fallback,
                    boundary_enforcement=self.boundary_enforcement,
                    boundary_chars=self.boundary_chars,
                    tokenizer=self.tokenizer,
                    min_chunk_size=self.min_chunk_size,
                )
                for c in raw:
                    chunk_meta = {**metadata, "chunker": self.chunker_type, "tokenizer": self.tokenizer}
                    if self.include_tag_attrs:
                        chunk_meta["tags"] = [node["tag"]]
                    chunks.append(ChunkModel(
                        chunk_id=str(uuid.uuid4()),
                        doc_id=doc_id,
                        text=c,
                        chunk_index=chunk_index,
                        token_count=count_tokens(c, mode=self.tokenizer),
                        metadata=chunk_meta,
                    ))
                    chunk_index += 1
                continue

            # Would overflow bin — flush and start new bin
            if bin_tokens + node_tokens > self.chunk_size_fallback and bin_texts:
                flush_bin()

            bin_texts.append(node_text)
            bin_tokens += node_tokens
            bin_tags.append(node["tag"])

        flush_bin()
        return chunks

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return {
            "split_tags": {"type": "list", "default": _DEFAULT_SPLIT_TAGS, "description": "HTML tags treated as structural splitting units."},
            "include_tag_attrs": {"type": "bool", "default": cls._DEFAULT_INCLUDE_TAG_ATTRS, "description": "Include source tag name(s) in chunk metadata."},
            "strip_scripts_styles": {"type": "bool", "default": cls._DEFAULT_STRIP_SCRIPTS_STYLES, "description": "Remove script and style elements before chunking."},
            "chunk_size_fallback": {"type": "int", "default": cls._DEFAULT_CHUNK_SIZE_FALLBACK, "min": 50, "max": 4000, "description": "Max tokens per structural node before intra-node split."},
            "overlap_fallback": {"type": "int", "default": cls._DEFAULT_OVERLAP_FALLBACK, "min": 0, "max": 500, "description": "Overlap for intra-node splits."},
            "boundary_enforcement": {"type": "bool", "default": cls._DEFAULT_BOUNDARY_ENFORCEMENT, "description": "Backtrack to nearest sentence boundary."},
            "boundary_chars": {"type": "list", "default": cls._DEFAULT_BOUNDARY_CHARS, "description": "Characters treated as sentence boundaries."},
            "tokenizer": {"type": "str", "default": cls._DEFAULT_TOKENIZER, "options": ["tiktoken", "word_count"], "description": "Token counting mode."},
            "min_chunk_size": {"type": "int", "default": cls._DEFAULT_MIN_CHUNK_SIZE, "min": 1, "max": 200, "description": "Minimum tokens per chunk."},
        }
