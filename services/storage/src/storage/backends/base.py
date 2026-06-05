"""
BaseStorageBackend — abstract interface for all RAGLab storage backends.

All backends implement the same four operations:
  upload(key, data)  → store bytes at key
  download(key)      → retrieve bytes at key
  delete(key)        → remove object at key
  exists(key)        → check if key exists

Design rules:
  - All methods raise StorageError on failure (never raw SDK exceptions).
  - `key` is always a path-like string: "docs/my-file.pdf", never a URL.
  - Credentials come from the environment / pydantic-settings only.
  - Callers never know which backend is active — they go through StorageFactory.

Active in R1: LocalStorageBackend
Active in R2: S3StorageBackend, AzureBlobStorageBackend
Stub in R7:   GCSStorageBackend
"""

from __future__ import annotations

import abc
from typing import Any

from raglab_common.exceptions import StorageError
from raglab_common.logging import get_logger

log = get_logger(__name__)


class BaseStorageBackend(abc.ABC):
    """Abstract base class for all RAGLab storage backends."""

    #: Unique string key used to register this backend in StorageFactory.
    backend_type: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self._log = get_logger(self.__class__.__name__)

    @abc.abstractmethod
    def upload(self, key: str, data: bytes) -> str:
        """
        Store `data` at `key`.

        Args:
            key:  Path-like string — e.g. "documents/report.pdf".
            data: Raw bytes to store.

        Returns:
            The canonical storage URI for the uploaded object
            (e.g. "s3://bucket/documents/report.pdf").

        Raises:
            StorageError: On any upload failure.
        """

    @abc.abstractmethod
    def download(self, key: str) -> bytes:
        """
        Retrieve bytes stored at `key`.

        Args:
            key: Path-like string.

        Returns:
            Raw bytes.

        Raises:
            StorageError: If key does not exist or download fails.
        """

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        """
        Remove object at `key`.

        Args:
            key: Path-like string.

        Raises:
            StorageError: If deletion fails.
        """

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check whether an object exists at `key`.

        Args:
            key: Path-like string.

        Returns:
            True if the object exists, False otherwise.

        Raises:
            StorageError: On unexpected backend errors (not for missing keys).
        """
