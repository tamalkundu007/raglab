"""
HybridRetriever — dense + sparse retrieval fusion via Reciprocal Rank Fusion.

Combines DenseRetriever (semantic) and BM25Retriever (keyword) results using
Reciprocal Rank Fusion (RRF). RRF is parameter-light and consistently
outperforms simple score interpolation in practice.

RRF formula per document d:
    RRF(d) = Σ_r  1 / (k + rank_r(d))
    where k=60 (default), r iterates over each ranked list.

Alpha parameter controls the blend:
    alpha=1.0 → pure dense
    alpha=0.0 → pure sparse (BM25)
    alpha=0.5 → equal weight (default)

NAMING DISTINCTION (critical — tested):
    HybridRetriever = dense + sparse retrieval fusion (THIS FILE, R3)
    HybridChunker   = structural + token chunking meta-strategy (raglab-chunkers, R2)
    Completely separate concepts.

Parameters:
    alpha       : float = 0.5   — weight of dense results in RRF fusion
    rrf_k       : int   = 60    — RRF constant (higher = flatter score distribution)
    dense_top_k : int   = 20    — candidates fetched from dense retriever (≥ top_k)
    bm25_top_k  : int   = 20    — candidates fetched from BM25 (≥ top_k)
    score_threshold: float = 0.0 — passed to dense retriever
    ef          : int   = 128   — HNSW ef passed to dense retriever
"""

from __future__ import annotations

from typing import Any

from raglab_common.exceptions import RetrieverError
from raglab_common.models import ChunkModel, QueryModel, RetrieverType

from raglab_retrievers.base import BaseRetriever


class HybridRetriever(BaseRetriever):
    """
    Dense + BM25 fusion via Reciprocal Rank Fusion. Active in R3.

    Requires BOTH a Qdrant-compatible vector_store AND a BM25Corpus
    in `vector_store` (passed as a tuple or a HybridStore object).
    Requires an embedder callable for the dense leg.

    vector_store interface:
        Pass a HybridStore(qdrant_client, bm25_corpus) or
        a tuple (qdrant_client, bm25_corpus).
    """

    retriever_type: str = "hybrid"

    _DEFAULT_ALPHA: float = 0.5
    _DEFAULT_RRF_K: int = 60
    _DEFAULT_DENSE_TOP_K: int = 20
    _DEFAULT_BM25_TOP_K: int = 20

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = config or {}
        self.alpha: float = float(cfg.get("alpha", self._DEFAULT_ALPHA))
        self.rrf_k: int = int(cfg.get("rrf_k", self._DEFAULT_RRF_K))
        self.dense_top_k: int = int(cfg.get("dense_top_k", self._DEFAULT_DENSE_TOP_K))
        self.bm25_top_k: int = int(cfg.get("bm25_top_k", self._DEFAULT_BM25_TOP_K))
        self.score_threshold: float = float(cfg.get("score_threshold", 0.0))
        self.ef: int = int(cfg.get("ef", 128))

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0.0, 1.0], got {self.alpha}")
        if self.rrf_k < 1:
            raise ValueError(f"rrf_k must be >= 1, got {self.rrf_k}")
        if self.dense_top_k < 1:
            raise ValueError(f"dense_top_k must be >= 1, got {self.dense_top_k}")
        if self.bm25_top_k < 1:
            raise ValueError(f"bm25_top_k must be >= 1, got {self.bm25_top_k}")

    def _retrieve(
        self,
        query: QueryModel,
        vector_store: Any,
        embedder: Any | None,
    ) -> list[ChunkModel]:
        """
        Run dense + BM25 retrieval and fuse results via RRF.
        """
        from raglab_retrievers.bm25_retriever import BM25Corpus
        from raglab_retrievers.dense_retriever import DenseRetriever

        # Unpack hybrid vector store
        qdrant_client, bm25_corpus = self._unpack_store(vector_store)

        # Expand query.top_k for candidate retrieval
        expanded_query_dense = QueryModel(
            text=query.text,
            collection=query.collection,
            top_k=self.dense_top_k,
            retriever_type=query.retriever_type,
            llm_provider=query.llm_provider,
            metadata_filter=query.metadata_filter,
        )
        expanded_query_bm25 = QueryModel(
            text=query.text,
            collection=query.collection,
            top_k=self.bm25_top_k,
            retriever_type=query.retriever_type,
            llm_provider=query.llm_provider,
            metadata_filter=query.metadata_filter,
        )

        # Dense leg
        dense_chunks: list[ChunkModel] = []
        if self.alpha > 0.0 and embedder is not None:
            dense = DenseRetriever(config={
                "score_threshold": self.score_threshold,
                "ef": self.ef,
            })
            dense_chunks = dense.retrieve(expanded_query_dense, qdrant_client, embedder=embedder)

        # BM25 leg
        bm25_chunks: list[ChunkModel] = []
        if self.alpha < 1.0 and bm25_corpus is not None:
            from raglab_retrievers.bm25_retriever import BM25Retriever
            bm25 = BM25Retriever()
            bm25_chunks = bm25.retrieve(expanded_query_bm25, bm25_corpus, embedder=None)

        # Fuse via RRF
        fused = self._rrf_fuse(
            dense_list=dense_chunks,
            bm25_list=bm25_chunks,
            alpha=self.alpha,
            k=self.rrf_k,
            top_k=query.top_k,
            query_id=str(query.query_id),
        )
        return fused

    @staticmethod
    def _unpack_store(vector_store: Any) -> tuple[Any, Any]:
        """
        Extract (qdrant_client, bm25_corpus) from the vector_store argument.

        Accepts:
            - HybridStore object (has .qdrant and .bm25 attributes)
            - tuple (qdrant_client, bm25_corpus)
        """
        if isinstance(vector_store, tuple) and len(vector_store) == 2:
            return vector_store[0], vector_store[1]
        if hasattr(vector_store, "qdrant") and hasattr(vector_store, "bm25"):
            return vector_store.qdrant, vector_store.bm25
        raise RetrieverError(
            "HybridRetriever requires vector_store to be a tuple "
            "(qdrant_client, bm25_corpus) or a HybridStore object."
        )

    @staticmethod
    def _rrf_fuse(
        dense_list: list[ChunkModel],
        bm25_list: list[ChunkModel],
        alpha: float,
        k: int,
        top_k: int,
        query_id: str,
    ) -> list[ChunkModel]:
        """
        Reciprocal Rank Fusion of two ranked lists.

        RRF score = alpha * 1/(k + dense_rank) + (1-alpha) * 1/(k + bm25_rank)
        Deduplication by chunk_id — the higher RRF score wins.
        """
        rrf_scores: dict[str, float] = {}
        chunk_by_id: dict[str, ChunkModel] = {}

        # Dense contributions
        for rank, chunk in enumerate(dense_list, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + alpha * (1.0 / (k + rank))
            chunk_by_id[chunk.chunk_id] = chunk

        # BM25 contributions
        for rank, chunk in enumerate(bm25_list, start=1):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 - alpha) * (1.0 / (k + rank))
            if chunk.chunk_id not in chunk_by_id:
                chunk_by_id[chunk.chunk_id] = chunk

        # Sort by RRF score descending, take top_k
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        result = []
        for chunk_id in sorted_ids:
            original = chunk_by_id[chunk_id]
            result.append(ChunkModel(
                chunk_id=original.chunk_id,
                doc_id=original.doc_id,
                text=original.text,
                chunk_index=original.chunk_index,
                token_count=original.token_count,
                metadata={
                    **original.metadata,
                    "retriever": "hybrid",
                    "rrf_score": round(rrf_scores[chunk_id], 6),
                    "query_id": query_id,
                },
            ))
        return result

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return {
            "alpha": {
                "type": "float", "default": cls._DEFAULT_ALPHA,
                "min": 0.0, "max": 1.0,
                "description": "Dense weight in RRF fusion. 1.0=pure dense, 0.0=pure BM25.",
            },
            "rrf_k": {
                "type": "int", "default": cls._DEFAULT_RRF_K,
                "min": 1, "max": 200,
                "description": "RRF k constant. Higher = flatter score distribution.",
            },
            "dense_top_k": {
                "type": "int", "default": cls._DEFAULT_DENSE_TOP_K,
                "min": 1, "max": 100,
                "description": "Candidate count from dense leg before fusion.",
            },
            "bm25_top_k": {
                "type": "int", "default": cls._DEFAULT_BM25_TOP_K,
                "min": 1, "max": 100,
                "description": "Candidate count from BM25 leg before fusion.",
            },
            "score_threshold": {
                "type": "float", "default": 0.0, "min": 0.0, "max": 1.0,
                "description": "Minimum score for dense leg (passed to DenseRetriever).",
            },
            "ef": {
                "type": "int", "default": 128, "min": 1, "max": 512,
                "description": "HNSW ef parameter for dense leg.",
            },
        }
