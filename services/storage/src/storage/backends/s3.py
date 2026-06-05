"""
S3StorageBackend — AWS S3 storage backend. Active in R2.

Credentials are loaded exclusively from environment variables via
pydantic-settings (RAGLAB_AWS_ACCESS_KEY_ID, RAGLAB_AWS_SECRET_ACCESS_KEY,
RAGLAB_AWS_REGION, RAGLAB_S3_BUCKET). Never hardcoded.

boto3 is imported at module level for test patchability.

Config (all optional — fall back to env vars):
    bucket  : str — S3 bucket name
    region  : str — AWS region (default: us-east-1)
    prefix  : str — key prefix prepended to all keys (default: "")
              e.g. prefix="raglab/" → key "doc.pdf" → "raglab/doc.pdf"
"""

from __future__ import annotations

from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from raglab_common.exceptions import StorageError

from storage.backends.base import BaseStorageBackend


class S3StorageBackend(BaseStorageBackend):
    """AWS S3 storage backend. Activates in R2."""

    backend_type: str = "s3"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._bucket: str = self.config.get("bucket", "")
        self._region: str = self.config.get("region", "us-east-1")
        self._prefix: str = self.config.get("prefix", "")

        if not self._bucket:
            raise StorageError(
                "S3StorageBackend requires 'bucket' in config or RAGLAB_S3_BUCKET env var."
            )

        try:
            self._client = boto3.client("s3", region_name=self._region)
        except Exception as exc:
            raise StorageError(f"Failed to initialise S3 client: {exc}") from exc

        self._log.info(
            "storage.s3_init",
            bucket=self._bucket,
            region=self._region,
            prefix=self._prefix or "(none)",
        )

    def _full_key(self, key: str) -> str:
        """Prepend configured prefix to the key."""
        return f"{self._prefix}{key}" if self._prefix else key

    def upload(self, key: str, data: bytes) -> str:
        full_key = self._full_key(key)
        try:
            self._client.put_object(Bucket=self._bucket, Key=full_key, Body=data)
            uri = f"s3://{self._bucket}/{full_key}"
            self._log.info("storage.upload", backend=self.backend_type, key=full_key, size=len(data))
            return uri
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"S3 upload failed for {full_key!r}: {exc}") from exc
        except Exception as exc:
            raise StorageError(f"S3 upload unexpected error for {full_key!r}: {exc}") from exc

    def download(self, key: str) -> bytes:
        full_key = self._full_key(key)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=full_key)
            data: bytes = response["Body"].read()
            self._log.info("storage.download", backend=self.backend_type, key=full_key, size=len(data))
            return data
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in ("NoSuchKey", "404"):
                raise StorageError(f"Key not found in S3: {full_key!r}") from exc
            raise StorageError(f"S3 download failed for {full_key!r}: {exc}") from exc
        except Exception as exc:
            raise StorageError(f"S3 download unexpected error for {full_key!r}: {exc}") from exc

    def delete(self, key: str) -> None:
        full_key = self._full_key(key)
        try:
            self._client.delete_object(Bucket=self._bucket, Key=full_key)
            self._log.info("storage.delete", backend=self.backend_type, key=full_key)
        except (BotoCoreError, ClientError) as exc:
            raise StorageError(f"S3 delete failed for {full_key!r}: {exc}") from exc
        except Exception as exc:
            raise StorageError(f"S3 delete unexpected error for {full_key!r}: {exc}") from exc

    def exists(self, key: str) -> bool:
        full_key = self._full_key(key)
        try:
            self._client.head_object(Bucket=self._bucket, Key=full_key)
            return True
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in ("404", "NoSuchKey"):
                return False
            raise StorageError(f"S3 exists check failed for {full_key!r}: {exc}") from exc
        except Exception as exc:
            raise StorageError(f"S3 exists unexpected error for {full_key!r}: {exc}") from exc
