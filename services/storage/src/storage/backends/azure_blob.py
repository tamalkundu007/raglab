"""
AzureBlobStorageBackend — Azure Blob Storage backend. Active in R2.

Credentials loaded from environment only:
    RAGLAB_AZURE_STORAGE_CONNECTION_STRING  (preferred)
    or RAGLAB_AZURE_STORAGE_ACCOUNT_NAME + RAGLAB_AZURE_STORAGE_ACCOUNT_KEY

azure-storage-blob imported at module level for test patchability.

Config (all optional — fall back to env vars):
    container         : str — blob container name
    connection_string : str — full connection string
    account_name      : str — storage account name (used with account_key)
    account_key       : str — storage account key
    prefix            : str — blob name prefix (default: "")
"""

from __future__ import annotations

from typing import Any

from azure.core.exceptions import ResourceNotFoundError, AzureError
from azure.storage.blob import BlobServiceClient

from raglab_common.exceptions import StorageError

from storage.backends.base import BaseStorageBackend


class AzureBlobStorageBackend(BaseStorageBackend):
    """Azure Blob Storage backend. Activates in R2."""

    backend_type: str = "azure_blob"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._container: str = self.config.get("container", "")
        self._prefix: str = self.config.get("prefix", "")

        if not self._container:
            raise StorageError(
                "AzureBlobStorageBackend requires 'container' in config "
                "or RAGLAB_AZURE_BLOB_CONTAINER env var."
            )

        connection_string: str = self.config.get("connection_string", "")
        account_name: str = self.config.get("account_name", "")
        account_key: str = self.config.get("account_key", "")

        try:
            if connection_string:
                self._service = BlobServiceClient.from_connection_string(connection_string)
            elif account_name and account_key:
                url = f"https://{account_name}.blob.core.windows.net"
                from azure.storage.blob import StorageSharedKeyCredential
                cred = StorageSharedKeyCredential(account_name, account_key)
                self._service = BlobServiceClient(account_url=url, credential=cred)
            else:
                raise StorageError(
                    "AzureBlobStorageBackend requires either 'connection_string' or "
                    "('account_name' + 'account_key') in config."
                )
            self._container_client = self._service.get_container_client(self._container)
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"Failed to initialise Azure Blob client: {exc}") from exc

        self._log.info(
            "storage.azure_blob_init",
            container=self._container,
            prefix=self._prefix or "(none)",
        )

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}{key}" if self._prefix else key

    def upload(self, key: str, data: bytes) -> str:
        blob_name = self._full_key(key)
        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            blob_client.upload_blob(data, overwrite=True)
            account_name = self._service.account_name or "unknown"
            uri = f"https://{account_name}.blob.core.windows.net/{self._container}/{blob_name}"
            self._log.info("storage.upload", backend=self.backend_type, key=blob_name, size=len(data))
            return uri
        except AzureError as exc:
            raise StorageError(f"Azure Blob upload failed for {blob_name!r}: {exc}") from exc
        except Exception as exc:
            raise StorageError(f"Azure Blob upload unexpected error for {blob_name!r}: {exc}") from exc

    def download(self, key: str) -> bytes:
        blob_name = self._full_key(key)
        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            data: bytes = blob_client.download_blob().readall()
            self._log.info("storage.download", backend=self.backend_type, key=blob_name, size=len(data))
            return data
        except ResourceNotFoundError as exc:
            raise StorageError(f"Key not found in Azure Blob: {blob_name!r}") from exc
        except AzureError as exc:
            raise StorageError(f"Azure Blob download failed for {blob_name!r}: {exc}") from exc
        except Exception as exc:
            raise StorageError(f"Azure Blob download unexpected error for {blob_name!r}: {exc}") from exc

    def delete(self, key: str) -> None:
        blob_name = self._full_key(key)
        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
            self._log.info("storage.delete", backend=self.backend_type, key=blob_name)
        except ResourceNotFoundError:
            pass  # idempotent — already gone
        except AzureError as exc:
            raise StorageError(f"Azure Blob delete failed for {blob_name!r}: {exc}") from exc
        except Exception as exc:
            raise StorageError(f"Azure Blob delete unexpected error for {blob_name!r}: {exc}") from exc

    def exists(self, key: str) -> bool:
        blob_name = self._full_key(key)
        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            blob_client.get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False
        except AzureError as exc:
            raise StorageError(f"Azure Blob exists check failed for {blob_name!r}: {exc}") from exc
        except Exception as exc:
            raise StorageError(f"Azure Blob exists unexpected error for {blob_name!r}: {exc}") from exc
