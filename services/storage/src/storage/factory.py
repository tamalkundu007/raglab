"""
StorageFactory — registry-based factory for RAGLab storage backends.

Usage:
    from storage.factory import StorageFactory

    backend = StorageFactory.create("s3", config={
        "bucket": "raglab-docs",
        "region": "us-east-1",
        "prefix": "prod/",
    })
    uri = backend.upload("documents/report.pdf", pdf_bytes)

Active:  local (R1), s3 (R2), azure_blob (R2)
Stub:    gcs (R7)
"""

from __future__ import annotations

from typing import Any

from raglab_common.exceptions import NotImplementedFeatureError, StorageError
from raglab_common.logging import get_logger
from raglab_common.models import StorageBackend

from storage.backends.base import BaseStorageBackend
from storage.backends.local import LocalStorageBackend
from storage.backends.s3 import S3StorageBackend
from storage.backends.azure_blob import AzureBlobStorageBackend
from storage.backends.gcs import GCSStorageBackend

log = get_logger(__name__)

_REGISTRY: dict[str, type[BaseStorageBackend]] = {
    StorageBackend.LOCAL.value:      LocalStorageBackend,
    StorageBackend.S3.value:         S3StorageBackend,
    StorageBackend.AZURE_BLOB.value: AzureBlobStorageBackend,
    StorageBackend.GCS.value:        GCSStorageBackend,
}

_ACTIVE_BACKENDS = {
    StorageBackend.LOCAL.value,
    StorageBackend.S3.value,
    StorageBackend.AZURE_BLOB.value,
}


class StorageFactory:
    """Registry-based factory for RAGLab storage backends."""

    @classmethod
    def create(
        cls,
        backend_type: str | StorageBackend,
        config: dict[str, Any] | None = None,
    ) -> BaseStorageBackend:
        """
        Instantiate and return a storage backend.

        Args:
            backend_type: StorageBackend enum value or string key.
            config:       Optional configuration dict forwarded to the backend.

        Returns:
            A BaseStorageBackend instance.

        Raises:
            ValueError:                 If backend_type is unknown.
            StorageError:               If backend initialisation fails.
            NotImplementedFeatureError: For stub backends (GCS → R7).
        """
        key = backend_type.value if isinstance(backend_type, StorageBackend) else str(backend_type)
        cls_ref = _REGISTRY.get(key)
        if cls_ref is None:
            available = list(_REGISTRY.keys())
            raise ValueError(
                f"Unknown storage backend {key!r}. Available: {available}"
            )
        log.info("factory.create_storage_backend", backend=key)
        return cls_ref(config=config)

    @classmethod
    def available(cls) -> list[dict[str, Any]]:
        """
        Return metadata for all registered backends.

        Used by the UI and config-service to show active/stub status.
        """
        result = []
        for key in _REGISTRY:
            is_active = key in _ACTIVE_BACKENDS
            entry: dict[str, Any] = {"backend": key, "active": is_active}
            if not is_active:
                entry["available_in"] = "R7" if key == StorageBackend.GCS.value else "future"
            result.append(entry)
        return result
