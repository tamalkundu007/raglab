"""
BaseChunker — abstract interface for all RAGLab chunkers.

Every chunker must implement `chunk()` and expose `config_schema()`.
The factory pattern means callers never import concrete chunker classes
directly — they go through ChunkerFactory.

Design rules:
- chunk() is synchronous (CPU-bound text processing)
- chunk() always returns a list — empty list for empty/unparseable input
- chunk() never raises; it logs and returns [] on failure
- config_schema() returns a dict describing UI-renderable parameters
"""

from __future__ import annotations

import abc
from typing import Any

from raglab_common.logging import get_logger
from raglab_common.models import ChunkModel

log = get_logger(__name__)


class BaseChunker(abc.ABC):
    """
    Abstract base class for all RAGLab chunkers.

    Subclasses implement `_chunk()` with the actual splitting logic.
    The public `chunk()` method wraps `_chunk()` with logging and
    error handling so every chunker gets consistent behaviour for free.
    """

    #: Unique string key used to register this chunker in ChunkerFactory.
    chunker_type: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Args:
            config: Optional dict of chunker-specific parameters.
                    Unrecognised keys are silently ignored.
        """
        self.config: dict[str, Any] = config or {}
        self._log = get_logger(self.__class__.__name__)

    def chunk(self, text: str, doc_id: str, metadata: dict[str, Any] | None = None) -> list[ChunkModel]:
        """
        Public entry point. Delegates to `_chunk()` with logging.

        Args:
            text:     Raw text content to split.
            doc_id:   Document ID propagated into every ChunkModel.
            metadata: Optional metadata dict attached to every chunk.

        Returns:
            List of ChunkModel instances. Empty list on failure or empty input.
        """
        metadata = metadata or {}
        if not text or not text.strip():
            self._log.warning("chunker.empty_input", chunker=self.chunker_type, doc_id=doc_id)
            return []

        self._log.info(
            "chunker.start",
            chunker=self.chunker_type,
            doc_id=doc_id,
            input_length=len(text),
        )
        try:
            chunks = self._chunk(text, doc_id, metadata)
            self._log.info(
                "chunker.done",
                chunker=self.chunker_type,
                doc_id=doc_id,
                chunk_count=len(chunks),
            )
            return chunks
        except Exception as exc:  # noqa: BLE001
            self._log.error(
                "chunker.error",
                chunker=self.chunker_type,
                doc_id=doc_id,
                error=str(exc),
            )
            return []

    @abc.abstractmethod
    def _chunk(self, text: str, doc_id: str, metadata: dict[str, Any]) -> list[ChunkModel]:
        """
        Core splitting logic — implemented by each concrete chunker.

        Args:
            text:     Non-empty, stripped text.
            doc_id:   Document ID.
            metadata: Caller-supplied metadata dict (already defaulted to {}).

        Returns:
            List of ChunkModel instances.
        """

    @classmethod
    @abc.abstractmethod
    def config_schema(cls) -> dict[str, Any]:
        """
        Return a JSON-serialisable dict describing this chunker's parameters.

        Used by the UI Control Panel to render knobs and by the config-service
        to validate incoming configuration.

        Schema shape per parameter:
            {
              "param_name": {
                "type": "int" | "float" | "bool" | "str" | "list",
                "default": <value>,
                "description": "<human-readable>",
                "min": <optional>,
                "max": <optional>,
                "options": [<optional list for enum-style params>],
              }
            }
        """
