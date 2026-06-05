"""
BM25Retriever — sparse keyword retrieval using rank-bm25.

BM25 (Best Match 25) is the standard sparse retrieval algorithm used in
traditional search engines. It scores documents by term frequency weighted
against inverse document frequency, with length normalisation.

Unlike DenseRetriever, BM25 requires NO embedder — it operates purely on
token overlap between the query and the indexed corpus.

Architecture in RAGLab:
    BM25 does not use Qdrant (vector store). It maintains its own in-memory
    inverted index built from the corpus. This index is passed as `vector_store`
    (duck-typed — BM25Retriever inspects the type and uses BM25Corpus if present).

    In production, the BM25Corpus is built once at startup from all indexed
    chunk texts and stored in app.state. In R3, the retrieval-service manages
    corpus lifecycle.

Parameters:
    k1          : float = 1.5   — term frequency saturation (BM25 k1)
    b           : float = 0.75  — document length normalisation (BM25 b)
    top_n_factor: int   = 3     — internal candidate multiplier before top_k cut
    tokenizer   : str   = "whitespace" — "whitespace" | "word" (same result in practice)

Reuse rule: scoring via rank-bm25; top_k filtering via standard Python slicing.
"""

from __future__ import annotations

import re
from typing import Any

from raglab_common.exceptions import RetrieverError
from raglab_common.models import ChunkModel, QueryModel

from raglab_retrievers.base import BaseRetriever


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Corpus:
    """
    In-memory BM25 index over a list of ChunkModel instances.

    Build once from the full corpus; reuse for many queries.
    Thread-safe for reads (rank_bm25 is stateless after fit).

    Args:
        chunks: List of ChunkModel instances to index.
        k1:     BM25 k1 parameter.
        b:      BM25 b parameter.
    """

    def __init__(self, chunks: list[ChunkModel], k1: float = 1.5, b: float = 0.75) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RetrieverError(
                "rank-bm25 not installed. Run: pip install rank-bm25"
            ) from exc

        self._chunks = chunks
        tokenised_corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenised_corpus, k1=k1, b=b)

    def search(self, query: str, top_k: int) -> list[tuple[ChunkModel, float]]:
        """
        Score all chunks against `query` and return top_k by BM25 score.

        Args:
            query:  Query string.
            top_k:  Number of results to return.

        Returns:
            List of (ChunkModel, score) tuples, ordered by descending score.
        """
        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            zip(self._chunks, scores), key=lambda x: x[1], reverse=True
        )
        return ranked[:top_k]

    @property
    def size(self) -> int:
        return len(self._chunks)


class BM25Retriever(BaseRetriever):
    """
    Sparse keyword retriever using BM25 (rank-bm25). Active in R3.

    Requires a BM25Corpus as `vector_store` (duck-typed).
    No embedder needed — pass embedder=None.
    """

    retriever_type: str = "bm25"

    _DEFAULT_K1: float = 1.5
    _DEFAULT_B: float = 0.75
    _DEFAULT_TOP_N_FACTOR: int = 3

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = config or {}
        self.k1: float = float(cfg.get("k1", self._DEFAULT_K1))
        self.b: float = float(cfg.get("b", self._DEFAULT_B))
        self.top_n_factor: int = int(cfg.get("top_n_factor", self._DEFAULT_TOP_N_FACTOR))

        if self.k1 < 0:
            raise ValueError(f"k1 must be >= 0, got {self.k1}")
        if not 0.0 <= self.b <= 1.0:
            raise ValueError(f"b must be in [0.0, 1.0], got {self.b}")
        if self.top_n_factor < 1:
            raise ValueError(f"top_n_factor must be >= 1, got {self.top_n_factor}")

    def _retrieve(
        self,
        query: QueryModel,
        vector_store: Any,
        embedder: Any | None,
    ) -> list[ChunkModel]:
        """
        BM25 sparse retrieval from a BM25Corpus.

        Args:
            query:        QueryModel (text, collection, top_k, metadata_filter).
            vector_store: Must be a BM25Corpus instance.
            embedder:     Ignored — BM25 needs no embeddings.
        """
        if not isinstance(vector_store, BM25Corpus):
            raise RetrieverError(
                "BM25Retriever requires a BM25Corpus as vector_store. "
                "Build one with: corpus = BM25Corpus(chunks)"
            )

        hits = vector_store.search(query.text, top_k=query.top_k)

        # Apply metadata_filter post-hoc if specified
        if query.metadata_filter:
            hits = [
                (chunk, score) for chunk, score in hits
                if self._matches_filter(chunk, query.metadata_filter)
            ][:query.top_k]

        return [
            ChunkModel(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                metadata={
                    **chunk.metadata,
                    "retriever": "bm25",
                    "score": round(float(score), 6),
                    "query_id": str(query.query_id),
                },
            )
            for chunk, score in hits
        ]

    @staticmethod
    def _matches_filter(chunk: ChunkModel, metadata_filter: dict[str, Any]) -> bool:
        """Simple equality filter on chunk metadata."""
        return all(chunk.metadata.get(k) == v for k, v in metadata_filter.items())

    @classmethod
    def config_schema(cls) -> dict[str, Any]:
        return {
            "k1": {
                "type": "float", "default": cls._DEFAULT_K1,
                "min": 0.0, "max": 3.0,
                "description": "BM25 k1 — term frequency saturation. Higher = more TF weight.",
            },
            "b": {
                "type": "float", "default": cls._DEFAULT_B,
                "min": 0.0, "max": 1.0,
                "description": "BM25 b — document length normalisation. 0=no norm, 1=full norm.",
            },
            "top_n_factor": {
                "type": "int", "default": cls._DEFAULT_TOP_N_FACTOR,
                "min": 1, "max": 10,
                "description": "Internal candidate multiplier before top_k cut.",
            },
        }
