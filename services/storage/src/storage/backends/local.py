"""
LocalStorageBackend — local filesystem storage backend.

Active in R1. Default backend when no cloud provider is configured.
Stores files under a configurable root directory.

Config:
    root: str — absolute path to storage root (default: /app/data/local)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from raglab_common.exceptions import StorageError

from storage.backends.base import BaseStorageBackend


class LocalStorageBackend(BaseStorageBackend):
    """Local filesystem storage. Active in R1."""

    backend_type: str = "local"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._root = Path(self.config.get("root", "/app/data/local"))
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise StorageError(f"Cannot create storage root {self._root}: {exc}") from exc

    def _resolve(self, key: str) -> Path:
        """Resolve key to absolute path under root, preventing path traversal."""
        resolved = (self._root / key).resolve()
        if not str(resolved).startswith(str(self._root.resolve())):
            raise StorageError(f"Path traversal detected for key: {key!r}")
        return resolved

    def upload(self, key: str, data: bytes) -> str:
        path = self._resolve(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self._log.info("storage.upload", backend=self.backend_type, key=key, size=len(data))
            return f"local://{path}"
        except Exception as exc:
            raise StorageError(f"Local upload failed for {key!r}: {exc}") from exc

    def download(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise StorageError(f"Key not found in local storage: {key!r}")
        try:
            data = path.read_bytes()
            self._log.info("storage.download", backend=self.backend_type, key=key, size=len(data))
            return data
        except Exception as exc:
            raise StorageError(f"Local download failed for {key!r}: {exc}") from exc

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        if not path.exists():
            return  # idempotent
        try:
            path.unlink()
            self._log.info("storage.delete", backend=self.backend_type, key=key)
        except Exception as exc:
            raise StorageError(f"Local delete failed for {key!r}: {exc}") from exc

    def exists(self, key: str) -> bool:
        try:
            return self._resolve(key).exists()
        except StorageError:
            return False
