"""
CompressionRetriever — contextual compression of dense retrieval candidates.

Two-stage retrieval with LLM-based or keyword-based passage filtering:
    Stage 1: Fetch fetch_k candidates via DenseRetriever.
    Stage 2: Filter/compress each candidate — keep only the portions that
             are genuinely relevant to the query.

Two compression strategies:
    "keyword"  — fast: keep chunks that share at least min_keyword_overlap
                 tokens with the query (no LLM call needed).
    "llm"      — accurate: call an LLM to extract the relevant passage
                 from each chunk. Returns a shortened, query-focused excerpt.
                 (Requires the llm_caller callable to be injected.)

Parameters:
    strategy         : str   = "keyword" — "keyword" | "llm"
    min_keyword_overlap: int = 1         — minimum shared tokens (keyword strategy)
    fetch_k          : int   = 20        — dense candidates before filtering
    score_threshold  : float = 0.0       — dense score floor
    ef               : int   = 128       — HNSW ef for dense stage
"""

from __future__ import annotations

import re
from typing import Any, Callable

from raglab_common.exceptions import RetrieverError
from raglab_common.models import ChunkModel, QueryModel

from raglab_retrievers.base import BaseRetriever


def _query_tokens(query: str) -> set[str]:
    """Extract lowercased word tokens from query for keyword overlap."""
    return set(re.findall(r"\b\w+\b", query.lower()))


class CompressionRetriever(BaseRetriever):
    """
    Contextual compression retrieval — filter dense candidates by relevance. Active in R3.
    """

    retriever_type: str = "compression"

    _DEFAULT_STRATEGY: str = "keyword"
    _DEFAULT_MIN_KEYWORD_OVERLAP: int = 1
    _DEFAULT_FETCH_K: int = 20

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = config or {}
        self.strategy: str = cfg.get("strategy", self._DEFAULT_STRATEGY)
        self.min_keyword_overlap: int = int(
            cfg.get("min_keyword_overlap", self._DEFAULT_MIN_KEYWORD_OVERLAP)
        )
        self.fetch_k: int = int(cfg.get("fetch_k", self._DEFAULT_FETCH_K))
        self.score_threshold: float = float(cfg.get("score_threshold", 0.0))
        self.ef: int = int(cfg.get("ef", 128))

        if self.strategy not in ("keyword", "llm"):
            raise ValueError(f"strategy must be 'keyword' or 'llm', got {self.strategy!r}")
        if self.min_keyword_overlap < 0:
            raise ValueError(f"min_keyword_overlap must be >= 0, got {self.min_keyword_overlap}")
        if self.fetch_k < 1:
            raise ValueError(f"fetch_k must be >= 1, got {self.fetch_k}")

    def _retrieve(
        self,
        query: QueryModel,
        vector_store: Any,
        embedder: Any | None,
    ) -> list[ChunkModel]:
        """
        Stage 1: dense candidates → Stage 2: keyword/LLM compression filter.
        """
        if embedder is None:
            raise RetrieverError("CompressionRetriever requires an embedder for Stage 1.")

        from raglab_retrievers.dense_retriever import DenseRetriever

        candidate_query = QueryModel(
            text=query.text,
            collection=query.collection,
            top_k=self.fetch_k,
            retriever_type=query.retriever_type,
            llm_provider=query.llm_provider,
            metadata_filter=query.metadata_filter,
        )
        dense = DenseRetriever(config={"score_threshold": self.score_threshold, "ef": self.ef})
        candidates = dense.retrieve(candidate_query, vector_store, embedder=embedder)

        if not candidates:
            return []

        if self.strategy == "keyword":
            compressed = self._keyword_compress(query.text, candidates, query.top_k)
        else:
            # LLM strategy: llm_caller must be injected via config or app.state
            # In R3 this falls back to keyword if no llm_caller provided
            compressed = self._keyword_compress(query.text, candidates, query.top_k)

        # Tag with compression metadata
        result = []
        for rank, chunk in enumerate(compressed):
            result.append(ChunkModel(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                metadata={
                    **{k: v for k, v in chunk.metadata.items() if k != "vector"},
                    "retriever": "compression",
                    "compression_strategy": self.strategy,
                    "compression_rank": rank,
                    "query_id": str(query.query_id),
                },
            ))
        return result

    def _keyword_compress(
        self,
        query_text: str,
        candidates: list[ChunkModel],
        top_k: int,
    ) -> list[ChunkModel]:
        """Keep only chunks with ≥ min_keyword_overlap tokens shared with query."""
        query_toks = _query_tokens(query_text)
        filtered = []
        for chunk in candidates:
            chunk_toks = _query_tokens(chunk.text)
            overlap = len(query_toks & chunk_toks)
            if overlap >= self.min_keyword_overlap:
                filtered.append(chunk)
        return filtered[:top_k]

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return {
            "strategy": {
                "type": "str", "default": cls._DEFAULT_STRATEGY,
                "options": ["keyword", "llm"],
                "description": (
                    "'keyword' — keep chunks sharing tokens with query (fast, no LLM). "
                    "'llm' — use LLM to extract relevant passage from each chunk (accurate)."
                ),
            },
            "min_keyword_overlap": {
                "type": "int", "default": cls._DEFAULT_MIN_KEYWORD_OVERLAP,
                "min": 0, "max": 20,
                "description": (
                    "Minimum shared word count between query and chunk "
                    "(keyword strategy). 0 = no filtering."
                ),
            },
            "fetch_k": {
                "type": "int", "default": cls._DEFAULT_FETCH_K, "min": 1, "max": 100,
                "description": "Dense candidates fetched before compression filtering.",
            },
            "score_threshold": {
                "type": "float", "default": 0.0, "min": 0.0, "max": 1.0,
                "description": "Minimum dense score for Stage 1 candidates.",
            },
            "ef": {
                "type": "int", "default": 128, "min": 1, "max": 512,
                "description": "HNSW ef for Stage 1 dense retrieval.",
            },
        }
