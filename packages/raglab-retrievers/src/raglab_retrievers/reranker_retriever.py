"""
ReRankerRetriever — cross-encoder re-ranking of dense retrieval candidates.

Two-stage retrieval:
    Stage 1: Fetch fetch_k candidates via DenseRetriever (bi-encoder, fast).
    Stage 2: Re-rank all candidates with a cross-encoder model (accurate, slower).
             Cross-encoders score (query, document) pairs jointly — they see
             both simultaneously, enabling fine-grained relevance judgements
             that bi-encoders miss.

Cross-encoder models (via sentence-transformers):
    Default: "cross-encoder/ms-marco-MiniLM-L-6-v2" — fast, strong
    Alternative: "cross-encoder/ms-marco-electra-base" — higher quality, slower

Parameters:
    model_name      : str   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    fetch_k         : int   = 20    — candidates from dense stage (≥ top_k)
    batch_size      : int   = 16    — cross-encoder batch size
    score_threshold : float = None  — drop chunks below this cross-encoder score
    ef              : int   = 128   — HNSW ef for dense stage
"""

from __future__ import annotations

from typing import Any

from raglab_common.exceptions import RetrieverError
from raglab_common.models import ChunkModel, QueryModel

from raglab_retrievers.base import BaseRetriever


class ReRankerRetriever(BaseRetriever):
    """
    Two-stage retrieval: dense candidates → cross-encoder re-ranking. Active in R3.

    Requires sentence-transformers. Requires an embedder for the dense stage.
    The cross-encoder model is loaded lazily on first use.
    """

    retriever_type: str = "reranker"

    _DEFAULT_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _DEFAULT_FETCH_K: int = 20
    _DEFAULT_BATCH_SIZE: int = 16

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = config or {}
        self.model_name: str = cfg.get("model_name", self._DEFAULT_MODEL)
        self.fetch_k: int = int(cfg.get("fetch_k", self._DEFAULT_FETCH_K))
        self.batch_size: int = int(cfg.get("batch_size", self._DEFAULT_BATCH_SIZE))
        self.score_threshold: float | None = cfg.get("score_threshold", None)
        self.ef: int = int(cfg.get("ef", 128))
        self._cross_encoder = None  # lazy init

        if self.fetch_k < 1:
            raise ValueError(f"fetch_k must be >= 1, got {self.fetch_k}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

    def _get_cross_encoder(self) -> Any:
        """Lazy-load cross-encoder model."""
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(self.model_name)
                self._log.info("reranker.model_loaded", model=self.model_name)
            except ImportError as exc:
                raise RetrieverError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
            except Exception as exc:
                raise RetrieverError(
                    f"Failed to load cross-encoder model {self.model_name!r}: {exc}"
                ) from exc
        return self._cross_encoder

    def _retrieve(
        self,
        query: QueryModel,
        vector_store: Any,
        embedder: Any | None,
    ) -> list[ChunkModel]:
        """
        Stage 1: dense fetch → Stage 2: cross-encoder re-rank → top_k.
        """
        if embedder is None:
            raise RetrieverError("ReRankerRetriever requires an embedder for Stage 1.")

        from raglab_retrievers.dense_retriever import DenseRetriever

        # Stage 1: dense candidates
        candidate_query = QueryModel(
            text=query.text,
            collection=query.collection,
            top_k=self.fetch_k,
            retriever_type=query.retriever_type,
            llm_provider=query.llm_provider,
            metadata_filter=query.metadata_filter,
        )
        dense = DenseRetriever(config={"score_threshold": 0.0, "ef": self.ef})
        candidates = dense.retrieve(candidate_query, vector_store, embedder=embedder)

        if not candidates:
            return []

        # Stage 2: cross-encoder re-ranking
        cross_encoder = self._get_cross_encoder()
        pairs = [[query.text, chunk.text] for chunk in candidates]

        try:
            scores = cross_encoder.predict(pairs, batch_size=self.batch_size)
        except Exception as exc:
            raise RetrieverError(f"Cross-encoder prediction failed: {exc}") from exc

        # Sort by cross-encoder score descending
        ranked = sorted(
            zip(candidates, scores), key=lambda x: float(x[1]), reverse=True
        )

        # Apply score threshold if configured
        if self.score_threshold is not None:
            ranked = [(c, s) for c, s in ranked if float(s) >= self.score_threshold]

        # Take top_k
        ranked = ranked[: query.top_k]

        return [
            ChunkModel(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                metadata={
                    **{k: v for k, v in chunk.metadata.items() if k != "vector"},
                    "retriever": "reranker",
                    "reranker_model": self.model_name,
                    "reranker_score": round(float(score), 6),
                    "reranker_rank": rank,
                    "query_id": str(query.query_id),
                },
            )
            for rank, (chunk, score) in enumerate(ranked)
        ]

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return {
            "model_name": {
                "type": "str", "default": cls._DEFAULT_MODEL,
                "options": [
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "cross-encoder/ms-marco-electra-base",
                    "cross-encoder/nli-deberta-v3-small",
                ],
                "description": "Sentence-transformers cross-encoder model for Stage 2 re-ranking.",
            },
            "fetch_k": {
                "type": "int", "default": cls._DEFAULT_FETCH_K, "min": 1, "max": 100,
                "description": "Dense candidates fetched in Stage 1 before re-ranking.",
            },
            "batch_size": {
                "type": "int", "default": cls._DEFAULT_BATCH_SIZE, "min": 1, "max": 128,
                "description": "Cross-encoder inference batch size.",
            },
            "score_threshold": {
                "type": "float", "default": None,
                "description": "Drop chunks below this cross-encoder score after re-ranking.",
            },
            "ef": {
                "type": "int", "default": 128, "min": 1, "max": 512,
                "description": "HNSW ef parameter for Stage 1 dense retrieval.",
            },
        }
