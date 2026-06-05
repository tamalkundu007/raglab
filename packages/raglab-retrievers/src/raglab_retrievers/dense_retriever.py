"""
DenseRetriever — cosine similarity vector search via Qdrant.

The only active retriever in R1. Embeds the query using the provided
embedder callable, then calls the vector store for nearest-neighbour
search with optional metadata filtering.

Parameters (configurable, shown in UI Control Panel):
    score_threshold : float = 0.0   — minimum similarity score (0.0 = no filter)
    ef              : int   = 128   — HNSW ef parameter (search accuracy vs speed)
    with_payload    : bool  = True  — return payload (metadata) alongside vectors
    with_vectors    : bool  = False — return raw vectors in results

Integration contract:
    vector_store must expose:
        .search(collection_name, query_vector, limit, score_threshold,
                query_filter, with_payload, with_vectors)
        returning a list of objects with .payload and .score attributes.

    embedder must be callable:
        embedder(text: str) -> list[float]
"""

from __future__ import annotations

from typing import Any

from raglab_common.exceptions import RetrieverError
from raglab_common.models import ChunkModel, QueryModel

from raglab_retrievers.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """
    Cosine similarity retriever backed by Qdrant vector search.

    Active in R1. Requires an embedder callable to convert query text
    to a vector before search.
    """

    retriever_type: str = "dense"

    _DEFAULT_SCORE_THRESHOLD: float = 0.0
    _DEFAULT_EF: int = 128
    _DEFAULT_WITH_PAYLOAD: bool = True
    _DEFAULT_WITH_VECTORS: bool = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = config or {}
        self.score_threshold: float = float(cfg.get("score_threshold", self._DEFAULT_SCORE_THRESHOLD))
        self.ef: int = int(cfg.get("ef", self._DEFAULT_EF))
        self.with_payload: bool = bool(cfg.get("with_payload", self._DEFAULT_WITH_PAYLOAD))
        self.with_vectors: bool = bool(cfg.get("with_vectors", self._DEFAULT_WITH_VECTORS))

        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError(
                f"score_threshold must be in [0.0, 1.0], got {self.score_threshold}"
            )
        if self.ef < 1:
            raise ValueError(f"ef must be >= 1, got {self.ef}")

    def _retrieve(
        self,
        query: QueryModel,
        vector_store: Any,
        embedder: Any | None,
    ) -> list[ChunkModel]:
        """
        Embed query text and run vector similarity search.

        Args:
            query:        QueryModel with text, collection, top_k, metadata_filter.
            vector_store: Qdrant client (or compatible interface).
            embedder:     Callable(text) -> list[float]. Required for dense retrieval.

        Returns:
            List of ChunkModel ordered by descending similarity score.

        Raises:
            RetrieverError: If embedder is missing or vector store call fails.
        """
        if embedder is None:
            raise RetrieverError(
                "DenseRetriever requires an embedder callable. "
                "Pass embedder=<callable> to retrieve()."
            )

        # Embed query
        try:
            query_vector: list[float] = embedder(query.text)
        except Exception as exc:
            raise RetrieverError(f"Embedding failed: {exc}") from exc

        if not query_vector:
            raise RetrieverError("Embedder returned an empty vector.")

        # Build optional Qdrant filter from query.metadata_filter
        query_filter = self._build_filter(query.metadata_filter) if query.metadata_filter else None

        # Search
        try:
            hits = vector_store.search(
                collection_name=query.collection,
                query_vector=query_vector,
                limit=query.top_k,
                score_threshold=self.score_threshold if self.score_threshold > 0.0 else None,
                query_filter=query_filter,
                with_payload=self.with_payload,
                with_vectors=self.with_vectors,
            )
        except Exception as exc:
            raise RetrieverError(f"Vector store search failed: {exc}") from exc

        return self._hits_to_chunks(hits, query)

    @staticmethod
    def _build_filter(metadata_filter: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a flat metadata_filter dict to a Qdrant filter structure.

        Simple equality match for each key-value pair (AND semantics).
        R3+ will add range, nested, and OR filters.

        Args:
            metadata_filter: e.g. {"doc_id": "abc", "source": "pdf"}

        Returns:
            Qdrant filter dict with "must" conditions.
        """
        must_conditions = [
            {"key": k, "match": {"value": v}}
            for k, v in metadata_filter.items()
        ]
        return {"must": must_conditions}

    @staticmethod
    def _hits_to_chunks(hits: list[Any], query: QueryModel) -> list[ChunkModel]:
        """
        Convert raw vector store hits to ChunkModel instances.

        Handles both Qdrant ScoredPoint objects (with .payload attribute)
        and plain dicts (for testing with mock stores).

        Args:
            hits:  List of scored results from vector store.
            query: Original query (used for metadata enrichment).

        Returns:
            List of ChunkModel instances.
        """
        chunks: list[ChunkModel] = []
        for hit in hits:
            # Support both object-style (Qdrant) and dict-style (mocks/tests)
            if isinstance(hit, dict):
                payload = hit.get("payload", {})
                score = hit.get("score", 0.0)
            else:
                payload = getattr(hit, "payload", {}) or {}
                score = getattr(hit, "score", 0.0)

            chunk = ChunkModel(
                chunk_id=str(payload.get("chunk_id", "")),
                doc_id=str(payload.get("doc_id", "")),
                text=str(payload.get("text", "")),
                chunk_index=int(payload.get("chunk_index", 0)),
                token_count=int(payload.get("token_count", 0)),
                metadata={
                    **{k: v for k, v in payload.items()
                       if k not in ("chunk_id", "doc_id", "text", "chunk_index", "token_count")},
                    "retriever": "dense",
                    "score": score,
                    "query_id": str(query.query_id),
                },
            )
            chunks.append(chunk)
        return chunks

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        """UI-renderable parameter schema for the Control Panel."""
        return {
            "score_threshold": {
                "type": "float",
                "default": cls._DEFAULT_SCORE_THRESHOLD,
                "min": 0.0,
                "max": 1.0,
                "description": (
                    "Minimum cosine similarity score. Results below this are excluded. "
                    "0.0 disables threshold filtering."
                ),
            },
            "ef": {
                "type": "int",
                "default": cls._DEFAULT_EF,
                "min": 1,
                "max": 512,
                "description": (
                    "HNSW ef parameter. Higher values improve recall at the cost of "
                    "latency. 128 is a good default for most collections."
                ),
            },
            "with_payload": {
                "type": "bool",
                "default": cls._DEFAULT_WITH_PAYLOAD,
                "description": "Return chunk metadata alongside search results.",
            },
            "with_vectors": {
                "type": "bool",
                "default": cls._DEFAULT_WITH_VECTORS,
                "description": "Return raw embedding vectors in results (rarely needed).",
            },
        }
