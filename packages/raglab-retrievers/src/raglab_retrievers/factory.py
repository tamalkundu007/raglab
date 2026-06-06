"""
RetrieverFactory — registry-based factory for all RAGLab retrievers.

R1 active: DenseRetriever
R3 active: BM25Retriever, HybridRetriever, MMRRetriever,
           ReRankerRetriever, CompressionRetriever
"""

from __future__ import annotations

from typing import Any

from raglab_common.exceptions import NotImplementedFeatureError
from raglab_common.logging import get_logger
from raglab_common.models import RetrieverType

from raglab_retrievers.base import BaseRetriever
from raglab_retrievers.dense_retriever import DenseRetriever
from raglab_retrievers.bm25_retriever import BM25Retriever
from raglab_retrievers.hybrid_retriever import HybridRetriever
from raglab_retrievers.mmr_retriever import MMRRetriever
from raglab_retrievers.reranker_retriever import ReRankerRetriever
from raglab_retrievers.compression_retriever import CompressionRetriever
from raglab_retrievers.graph_retriever import GraphRetriever

log = get_logger(__name__)

_REGISTRY: dict[str, type[BaseRetriever]] = {
    RetrieverType.DENSE:       DenseRetriever,
    RetrieverType.BM25:        BM25Retriever,
    RetrieverType.HYBRID:      HybridRetriever,
    RetrieverType.MMR:         MMRRetriever,
    RetrieverType.RERANKER:    ReRankerRetriever,
    RetrieverType.COMPRESSION: CompressionRetriever,
    RetrieverType.GRAPH:       GraphRetriever,
}

_ACTIVE_TYPES = set(_REGISTRY.keys())  # all active in R3


class RetrieverFactory:
    """Registry-based factory for RAGLab retrievers."""

    @classmethod
    def create(
        cls,
        retriever_type: str | RetrieverType,
        config: dict[str, Any] | None = None,
    ) -> BaseRetriever:
        key = (
            retriever_type.value
            if isinstance(retriever_type, RetrieverType)
            else str(retriever_type)
        )
        cls_ref = _REGISTRY.get(key)
        if cls_ref is None:
            raise ValueError(
                f"Unknown retriever type {key!r}. Available: {list(_REGISTRY.keys())}"
            )
        log.info("factory.create_retriever", retriever_type=key)
        return cls_ref(config=config)

    @classmethod
    def available(cls) -> list[dict[str, Any]]:
        return [{"type": key, "active": True} for key in _REGISTRY]

    @classmethod
    def schema(cls, retriever_type: str | RetrieverType) -> dict[str, Any]:
        key = (
            retriever_type.value
            if isinstance(retriever_type, RetrieverType)
            else str(retriever_type)
        )
        cls_ref = _REGISTRY.get(key)
        if cls_ref is None:
            raise ValueError(f"Unknown retriever type {key!r}")
        return cls_ref.config_schema()
