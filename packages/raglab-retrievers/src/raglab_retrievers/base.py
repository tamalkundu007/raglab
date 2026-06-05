"""
BaseRetriever — abstract interface for all RAGLab retrievers.

Design rules mirror BaseChunker:
- retrieve() is the public entry point — wraps _retrieve() with logging
  and error handling; never raises, returns [] on failure.
- _retrieve() is the abstract method each concrete retriever implements.
- config_schema() returns a UI-renderable parameter dict.
- Retrievers are stateless with respect to index data; they hold only
  configuration. The vector store client is passed per call so retriever
  instances can be reused across collections.
"""

from __future__ import annotations

import abc
from typing import Any

from raglab_common.logging import get_logger
from raglab_common.models import ChunkModel, QueryModel


class BaseRetriever(abc.ABC):
    """
    Abstract base class for all RAGLab retrievers.

    Subclasses implement `_retrieve()` with the actual search logic.
    The public `retrieve()` wraps `_retrieve()` with logging and
    error handling.
    """

    #: Unique string key used to register this retriever in RetrieverFactory.
    retriever_type: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Args:
            config: Optional dict of retriever-specific parameters.
                    Unrecognised keys are silently ignored.
        """
        self.config: dict[str, Any] = config or {}
        self._log = get_logger(self.__class__.__name__)

    def retrieve(
        self,
        query: QueryModel,
        vector_store: Any,
        embedder: Any | None = None,
    ) -> list[ChunkModel]:
        """
        Public entry point. Delegates to `_retrieve()` with logging.

        Args:
            query:        QueryModel containing text, collection, top_k, filters.
            vector_store: Initialised vector store client (Qdrant, FAISS, etc.).
            embedder:     Optional embedding callable — some retrievers need it
                          (dense), others don't (BM25).

        Returns:
            List of ChunkModel instances ordered by relevance. Empty on failure.
        """
        self._log.info(
            "retriever.start",
            retriever=self.retriever_type,
            query_id=str(query.query_id),
            collection=query.collection,
            top_k=query.top_k,
        )
        try:
            results = self._retrieve(query=query, vector_store=vector_store, embedder=embedder)
            self._log.info(
                "retriever.done",
                retriever=self.retriever_type,
                query_id=str(query.query_id),
                result_count=len(results),
            )
            return results
        except Exception as exc:  # noqa: BLE001
            self._log.error(
                "retriever.error",
                retriever=self.retriever_type,
                query_id=str(query.query_id),
                error=str(exc),
            )
            return []

    @abc.abstractmethod
    def _retrieve(
        self,
        query: QueryModel,
        vector_store: Any,
        embedder: Any | None,
    ) -> list[ChunkModel]:
        """
        Core retrieval logic — implemented by each concrete retriever.

        Args:
            query:        QueryModel (already validated by retrieve()).
            vector_store: Vector store client.
            embedder:     Embedding callable, or None.

        Returns:
            List of ChunkModel instances.
        """

    @classmethod
    @abc.abstractmethod
    def config_schema(cls) -> dict[str, Any]:
        """
        Return a JSON-serialisable dict describing this retriever's parameters.

        Schema shape per parameter (same as chunker schema convention):
            {
              "param_name": {
                "type": "int" | "float" | "bool" | "str" | "list",
                "default": <value>,
                "description": "<human-readable>",
                "min": <optional>,
                "max": <optional>,
                "options": [<optional>],
              }
            }
        """
