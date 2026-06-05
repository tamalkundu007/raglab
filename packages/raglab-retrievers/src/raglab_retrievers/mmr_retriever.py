"""
MMRRetriever — Maximum Marginal Relevance retrieval.

MMR selects documents that are both relevant to the query AND diverse
from each other. It prevents the classic RAG failure mode where all
top-k chunks are near-duplicates from the same section of a document.

Algorithm (Carbonell & Goldstein, 1998):
    At each step, select the candidate c* that maximises:
        MMR(c*) = λ * sim(query, c) - (1-λ) * max_{d ∈ selected} sim(c, d)

    λ (lambda_mult):
        1.0 → pure relevance (equivalent to DenseRetriever)
        0.0 → pure diversity (pick least similar to already-selected)
        0.5 → balanced (default)

Parameters:
    lambda_mult     : float = 0.5   — relevance vs diversity balance
    fetch_k         : int   = 20    — candidates fetched before MMR (≥ top_k)
    score_threshold : float = 0.0   — minimum relevance score for candidates
    ef              : int   = 128   — HNSW ef for candidate fetch
"""

from __future__ import annotations

import math
from typing import Any

from raglab_common.exceptions import RetrieverError
from raglab_common.models import ChunkModel, QueryModel

from raglab_retrievers.base import BaseRetriever


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class MMRRetriever(BaseRetriever):
    """
    Diversity-aware retrieval via Maximum Marginal Relevance. Active in R3.

    Requires an embedder callable. Uses DenseRetriever internally to
    fetch `fetch_k` candidates, then applies MMR selection to pick `top_k`.
    """

    retriever_type: str = "mmr"

    _DEFAULT_LAMBDA_MULT: float = 0.5
    _DEFAULT_FETCH_K: int = 20

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = config or {}
        self.lambda_mult: float = float(cfg.get("lambda_mult", self._DEFAULT_LAMBDA_MULT))
        self.fetch_k: int = int(cfg.get("fetch_k", self._DEFAULT_FETCH_K))
        self.score_threshold: float = float(cfg.get("score_threshold", 0.0))
        self.ef: int = int(cfg.get("ef", 128))

        if not 0.0 <= self.lambda_mult <= 1.0:
            raise ValueError(f"lambda_mult must be in [0.0, 1.0], got {self.lambda_mult}")
        if self.fetch_k < 1:
            raise ValueError(f"fetch_k must be >= 1, got {self.fetch_k}")

    def _retrieve(
        self,
        query: QueryModel,
        vector_store: Any,
        embedder: Any | None,
    ) -> list[ChunkModel]:
        """
        Fetch fetch_k dense candidates, then select top_k via MMR.
        """
        if embedder is None:
            raise RetrieverError("MMRRetriever requires an embedder callable.")

        from raglab_retrievers.dense_retriever import DenseRetriever

        # Step 1: embed query
        try:
            query_vector: list[float] = embedder(query.text)
        except Exception as exc:
            raise RetrieverError(f"Embedding failed: {exc}") from exc

        # Step 2: fetch fetch_k candidates via DenseRetriever
        candidate_query = QueryModel(
            text=query.text,
            collection=query.collection,
            top_k=self.fetch_k,
            retriever_type=query.retriever_type,
            llm_provider=query.llm_provider,
            metadata_filter=query.metadata_filter,
        )
        dense = DenseRetriever(config={
            "score_threshold": self.score_threshold, "ef": self.ef, "with_vectors": True,
        })
        candidates = dense.retrieve(candidate_query, vector_store, embedder=embedder)

        if not candidates:
            return []

        # Step 3: extract candidate vectors from metadata (with_vectors=True stores them)
        # Fall back to re-embedding chunk texts if vectors not in metadata
        candidate_vectors = self._get_vectors(candidates, embedder)

        # Step 4: MMR selection
        selected_indices = self._mmr_select(
            query_vector=query_vector,
            candidate_vectors=candidate_vectors,
            top_k=min(query.top_k, len(candidates)),
            lambda_mult=self.lambda_mult,
        )

        return [
            ChunkModel(
                chunk_id=candidates[i].chunk_id,
                doc_id=candidates[i].doc_id,
                text=candidates[i].text,
                chunk_index=candidates[i].chunk_index,
                token_count=candidates[i].token_count,
                metadata={
                    **{k: v for k, v in candidates[i].metadata.items() if k != "vector"},
                    "retriever": "mmr",
                    "lambda_mult": self.lambda_mult,
                    "mmr_rank": rank,
                    "query_id": str(query.query_id),
                },
            )
            for rank, i in enumerate(selected_indices)
        ]

    @staticmethod
    def _get_vectors(chunks: list[ChunkModel], embedder: Any) -> list[list[float]]:
        """Extract or compute vectors for each candidate chunk."""
        vectors = []
        for chunk in chunks:
            # DenseRetriever stores vector in metadata when with_vectors=True
            vec = chunk.metadata.get("vector")
            if vec and isinstance(vec, list):
                vectors.append(vec)
            else:
                # Re-embed the chunk text (slightly expensive but correct)
                try:
                    vectors.append(embedder(chunk.text))
                except Exception:
                    vectors.append([0.0])  # fallback — will score 0 in MMR
        return vectors

    @staticmethod
    def _mmr_select(
        query_vector: list[float],
        candidate_vectors: list[list[float]],
        top_k: int,
        lambda_mult: float,
    ) -> list[int]:
        """
        Greedy MMR selection over candidate_vectors.

        Returns indices into candidate_vectors in MMR selection order.
        """
        selected: list[int] = []
        remaining = list(range(len(candidate_vectors)))

        for _ in range(top_k):
            if not remaining:
                break

            best_idx = None
            best_score = float("-inf")

            for i in remaining:
                relevance = _cosine_sim(query_vector, candidate_vectors[i])

                # Diversity: distance from already-selected
                if selected:
                    max_sim_to_selected = max(
                        _cosine_sim(candidate_vectors[i], candidate_vectors[j])
                        for j in selected
                    )
                else:
                    max_sim_to_selected = 0.0

                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim_to_selected

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return selected

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return {
            "lambda_mult": {
                "type": "float", "default": cls._DEFAULT_LAMBDA_MULT,
                "min": 0.0, "max": 1.0,
                "description": "Relevance vs diversity. 1.0=pure relevance, 0.0=pure diversity.",
            },
            "fetch_k": {
                "type": "int", "default": cls._DEFAULT_FETCH_K,
                "min": 1, "max": 100,
                "description": "Dense candidates to fetch before MMR selection (should be > top_k).",
            },
            "score_threshold": {
                "type": "float", "default": 0.0, "min": 0.0, "max": 1.0,
                "description": "Minimum relevance score for initial dense fetch.",
            },
            "ef": {
                "type": "int", "default": 128, "min": 1, "max": 512,
                "description": "HNSW ef parameter for dense candidate fetch.",
            },
        }
