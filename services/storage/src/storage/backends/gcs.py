"""GCSStorageBackend — Google Cloud Storage stub. Activates in R7."""

from __future__ import annotations

from typing import Any

from raglab_common.exceptions import NotImplementedFeatureError

from storage.backends.base import BaseStorageBackend


class GCSStorageBackend(BaseStorageBackend):
    """Google Cloud Storage — stub until R7."""

    backend_type: str = "gcs"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        raise NotImplementedFeatureError("GCSStorageBackend", available_in="R7")

    def upload(self, key: str, data: bytes) -> str:
        raise NotImplementedFeatureError("GCSStorageBackend", available_in="R7")

    def download(self, key: str) -> bytes:
        raise NotImplementedFeatureError("GCSStorageBackend", available_in="R7")

    def delete(self, key: str) -> None:
        raise NotImplementedFeatureError("GCSStorageBackend", available_in="R7")

    def exists(self, key: str) -> bool:
        raise NotImplementedFeatureError("GCSStorageBackend", available_in="R7")
