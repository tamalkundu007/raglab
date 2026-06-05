"""
RetrieverFactory — registry-based factory for all RAGLab retrievers.

Usage:
    from raglab_retrievers.factory import RetrieverFactory

    retriever = RetrieverFactory.create("dense", config={"score_threshold": 0.7})
    chunks    = retriever.retrieve(query, vector_store, embedder=embed_fn)

R1 active:   DenseRetriever
R3 stubs:    BM25Retriever, HybridRetriever, MMRRetriever,
             ReRankerRetriever, CompressionRetriever

Design mirrors ChunkerFactory exactly — consistent factory pattern
across the codebase.
"""

from __future__ import annotations

from typing import Any

from raglab_common.exceptions import NotImplementedFeatureError
from raglab_common.logging import get_logger
from raglab_common.models import RetrieverType

from raglab_retrievers.base import BaseRetriever
from raglab_retrievers.dense_retriever import DenseRetriever

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Stub generator — mirrors ChunkerFactory pattern exactly
# ---------------------------------------------------------------------------


def _make_stub(name: str, available_in: str) -> type[BaseRetriever]:
    """Create a stub retriever class that raises NotImplementedFeatureError."""

    class _StubRetriever(BaseRetriever):
        retriever_type = name

        def __init__(self, config: dict[str, Any] | None = None) -> None:
            raise NotImplementedFeatureError(
                feature=f"{name.upper()}Retriever",
                available_in=available_in,
            )

        def _retrieve(self, query, vector_store, embedder):
            raise NotImplementedFeatureError(feature=name, available_in=available_in)

        @classmethod
        def config_schema(cls) -> dict[str, Any]:
            return {
                "_stub": {
                    "type": "str",
                    "default": available_in,
                    "description": f"Available in {available_in}.",
                }
            }

    _StubRetriever.__name__ = f"{name.title().replace('_', '')}RetrieverStub"
    return _StubRetriever


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseRetriever]] = {
    # R1 — active
    RetrieverType.DENSE: DenseRetriever,

    # R3 — stubs
    RetrieverType.BM25:        _make_stub("bm25",        "R3"),
    RetrieverType.HYBRID:      _make_stub("hybrid",      "R3"),
    RetrieverType.MMR:         _make_stub("mmr",         "R3"),
    RetrieverType.RERANKER:    _make_stub("reranker",    "R3"),
    RetrieverType.COMPRESSION: _make_stub("compression", "R3"),
}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class RetrieverFactory:
    """Registry-based factory for RAGLab retrievers."""

    @classmethod
    def create(
        cls,
        retriever_type: str | RetrieverType,
        config: dict[str, Any] | None = None,
    ) -> BaseRetriever:
        """
        Instantiate and return a retriever for the given type.

        Args:
            retriever_type: String or RetrieverType enum value.
            config:         Optional parameter dict forwarded to the retriever.

        Returns:
            A BaseRetriever instance.

        Raises:
            ValueError:                  If retriever_type not in registry.
            NotImplementedFeatureError:  If the retriever is a R3+ stub.
        """
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
        """
        Return metadata for all registered retrievers.

        Used by the UI to populate the retriever dropdown with active/stub status.

        Returns:
            List of dicts: {type, active, available_in (if stub)}
        """
        active_types = {RetrieverType.DENSE}
        result = []
        for key, cls_ref in _REGISTRY.items():
            is_active = RetrieverType(key) in active_types
            entry: dict[str, Any] = {"type": key, "active": is_active}
            if not is_active:
                try:
                    schema = cls_ref.config_schema()
                    entry["available_in"] = schema.get("_stub", {}).get("default", "future")
                except Exception:  # noqa: BLE001
                    entry["available_in"] = "future"
            result.append(entry)
        return result

    @classmethod
    def schema(cls, retriever_type: str | RetrieverType) -> dict[str, Any]:
        """
        Return the config schema for a given retriever type.

        Args:
            retriever_type: String or RetrieverType enum value.

        Returns:
            Config schema dict.

        Raises:
            ValueError: If retriever_type is unknown.
        """
        key = (
            retriever_type.value
            if isinstance(retriever_type, RetrieverType)
            else str(retriever_type)
        )
        cls_ref = _REGISTRY.get(key)
        if cls_ref is None:
            raise ValueError(f"Unknown retriever type {key!r}")
        return cls_ref.config_schema()
