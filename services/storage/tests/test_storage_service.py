"""
Tests for the storage-service.

Covers:
- LocalStorageBackend: upload/download/delete/exists, path traversal guard
- S3StorageBackend: upload/download/delete/exists, prefix handling, error mapping
- AzureBlobStorageBackend: upload/download/delete/exists, prefix handling
- GCSStorageBackend: stub raises NotImplementedFeatureError
- StorageFactory: create, available, unknown backend
- HTTP endpoints: upload, download, delete, exists, backends list
"""

from __future__ import annotations

import base64
import os
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from fastapi.testclient import TestClient

from raglab_common.exceptions import NotImplementedFeatureError, StorageError


# ═══════════════════════════════════════════════════════════════════════════════
# LocalStorageBackend
# ═══════════════════════════════════════════════════════════════════════════════


class TestLocalStorageBackend:
    @pytest.fixture
    def local(self, tmp_path):
        from storage.backends.local import LocalStorageBackend
        return LocalStorageBackend(config={"root": str(tmp_path)})

    def test_upload_creates_file(self, local, tmp_path):
        uri = local.upload("docs/test.txt", b"hello world")
        assert (tmp_path / "docs" / "test.txt").exists()
        assert "local://" in uri

    def test_upload_returns_uri(self, local):
        uri = local.upload("report.pdf", b"PDF bytes")
        assert "report.pdf" in uri

    def test_download_returns_bytes(self, local):
        local.upload("data.bin", b"\x00\x01\x02")
        data = local.download("data.bin")
        assert data == b"\x00\x01\x02"

    def test_download_missing_key_raises(self, local):
        with pytest.raises(StorageError, match="not found"):
            local.download("nonexistent.txt")

    def test_delete_removes_file(self, local, tmp_path):
        local.upload("temp.txt", b"delete me")
        local.delete("temp.txt")
        assert not (tmp_path / "temp.txt").exists()

    def test_delete_missing_key_is_idempotent(self, local):
        local.delete("never_existed.txt")  # should not raise

    def test_exists_true_after_upload(self, local):
        local.upload("present.txt", b"here")
        assert local.exists("present.txt") is True

    def test_exists_false_before_upload(self, local):
        assert local.exists("absent.txt") is False

    def test_upload_creates_nested_dirs(self, local, tmp_path):
        local.upload("a/b/c/deep.txt", b"deep content")
        assert (tmp_path / "a" / "b" / "c" / "deep.txt").exists()

    def test_path_traversal_blocked(self, local):
        with pytest.raises(StorageError, match="traversal"):
            local.upload("../../etc/passwd", b"evil")

    def test_overwrite_existing_file(self, local):
        local.upload("overwrite.txt", b"v1")
        local.upload("overwrite.txt", b"v2")
        assert local.download("overwrite.txt") == b"v2"


# ═══════════════════════════════════════════════════════════════════════════════
# S3StorageBackend
# ═══════════════════════════════════════════════════════════════════════════════


def _make_s3_backend(bucket="test-bucket", prefix="", extra_config=None):
    cfg = {"bucket": bucket, "region": "us-east-1", "prefix": prefix}
    if extra_config:
        cfg.update(extra_config)
    with patch("storage.backends.s3.boto3") as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        from storage.backends.s3 import S3StorageBackend
        backend = S3StorageBackend(config=cfg)
        backend._client = mock_client
        return backend, mock_client


class TestS3StorageBackend:
    def test_missing_bucket_raises(self):
        from storage.backends.s3 import S3StorageBackend
        with patch("storage.backends.s3.boto3"):
            with pytest.raises(StorageError, match="bucket"):
                S3StorageBackend(config={})

    def test_upload_calls_put_object(self):
        backend, client = _make_s3_backend()
        backend.upload("docs/file.pdf", b"pdf bytes")
        client.put_object.assert_called_once_with(
            Bucket="test-bucket", Key="docs/file.pdf", Body=b"pdf bytes"
        )

    def test_upload_returns_s3_uri(self):
        backend, client = _make_s3_backend()
        uri = backend.upload("report.pdf", b"data")
        assert uri == "s3://test-bucket/report.pdf"

    def test_upload_with_prefix(self):
        backend, client = _make_s3_backend(prefix="prod/")
        backend.upload("file.txt", b"data")
        call_kwargs = client.put_object.call_args[1]
        assert call_kwargs["Key"] == "prod/file.txt"

    def test_upload_error_raises_storage_error(self):
        backend, client = _make_s3_backend()
        from botocore.exceptions import BotoCoreError
        client.put_object.side_effect = BotoCoreError()
        with pytest.raises(StorageError, match="S3 upload failed"):
            backend.upload("key", b"data")

    def test_download_calls_get_object(self):
        backend, client = _make_s3_backend()
        mock_body = MagicMock()
        mock_body.read.return_value = b"file content"
        client.get_object.return_value = {"Body": mock_body}
        data = backend.download("file.txt")
        assert data == b"file content"

    def test_download_not_found_raises_storage_error(self):
        backend, client = _make_s3_backend()
        from botocore.exceptions import ClientError
        error_response = {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}
        client.get_object.side_effect = ClientError(error_response, "GetObject")
        with pytest.raises(StorageError, match="not found"):
            backend.download("missing.txt")

    def test_delete_calls_delete_object(self):
        backend, client = _make_s3_backend()
        backend.delete("old.txt")
        client.delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="old.txt"
        )

    def test_exists_true_on_head_success(self):
        backend, client = _make_s3_backend()
        client.head_object.return_value = {"ContentLength": 42}
        assert backend.exists("present.txt") is True

    def test_exists_false_on_404(self):
        backend, client = _make_s3_backend()
        from botocore.exceptions import ClientError
        error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
        client.head_object.side_effect = ClientError(error_response, "HeadObject")
        assert backend.exists("absent.txt") is False

    def test_full_key_with_prefix(self):
        backend, _ = _make_s3_backend(prefix="staging/")
        assert backend._full_key("docs/file.pdf") == "staging/docs/file.pdf"

    def test_full_key_without_prefix(self):
        backend, _ = _make_s3_backend(prefix="")
        assert backend._full_key("docs/file.pdf") == "docs/file.pdf"


# ═══════════════════════════════════════════════════════════════════════════════
# AzureBlobStorageBackend
# ═══════════════════════════════════════════════════════════════════════════════


def _make_azure_backend(container="test-container", prefix="", conn_str="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=a==;EndpointSuffix=core.windows.net"):
    cfg = {"container": container, "connection_string": conn_str, "prefix": prefix}
    with patch("storage.backends.azure_blob.BlobServiceClient") as mock_svc_cls:
        mock_service = MagicMock()
        mock_service.account_name = "testaccount"
        mock_svc_cls.from_connection_string.return_value = mock_service
        mock_container = MagicMock()
        mock_service.get_container_client.return_value = mock_container
        from storage.backends.azure_blob import AzureBlobStorageBackend
        backend = AzureBlobStorageBackend(config=cfg)
        backend._service = mock_service
        backend._container_client = mock_container
        return backend, mock_container


class TestAzureBlobStorageBackend:
    def test_missing_container_raises(self):
        from storage.backends.azure_blob import AzureBlobStorageBackend
        with patch("storage.backends.azure_blob.BlobServiceClient"):
            with pytest.raises(StorageError, match="container"):
                AzureBlobStorageBackend(config={"connection_string": "DefaultEndpointsProtocol=https;AccountName=x;AccountKey=a==;EndpointSuffix=core.windows.net"})

    def test_missing_credentials_raises(self):
        from storage.backends.azure_blob import AzureBlobStorageBackend
        with pytest.raises(StorageError, match="connection_string"):
            AzureBlobStorageBackend(config={"container": "mycontainer"})

    def test_upload_calls_upload_blob(self):
        backend, container = _make_azure_backend()
        mock_blob = MagicMock()
        container.get_blob_client.return_value = mock_blob
        backend.upload("docs/file.pdf", b"pdf bytes")
        mock_blob.upload_blob.assert_called_once_with(b"pdf bytes", overwrite=True)

    def test_upload_returns_azure_uri(self):
        backend, container = _make_azure_backend()
        mock_blob = MagicMock()
        container.get_blob_client.return_value = mock_blob
        uri = backend.upload("report.pdf", b"data")
        assert "blob.core.windows.net" in uri
        assert "report.pdf" in uri

    def test_upload_with_prefix(self):
        backend, container = _make_azure_backend(prefix="staging/")
        mock_blob = MagicMock()
        container.get_blob_client.return_value = mock_blob
        backend.upload("file.txt", b"data")
        container.get_blob_client.assert_called_with("staging/file.txt")

    def test_download_returns_bytes(self):
        backend, container = _make_azure_backend()
        mock_blob = MagicMock()
        mock_stream = MagicMock()
        mock_stream.readall.return_value = b"azure content"
        mock_blob.download_blob.return_value = mock_stream
        container.get_blob_client.return_value = mock_blob
        data = backend.download("file.txt")
        assert data == b"azure content"

    def test_download_not_found_raises(self):
        backend, container = _make_azure_backend()
        mock_blob = MagicMock()
        from azure.core.exceptions import ResourceNotFoundError
        mock_blob.download_blob.side_effect = ResourceNotFoundError("not found")
        container.get_blob_client.return_value = mock_blob
        with pytest.raises(StorageError, match="not found"):
            backend.download("missing.txt")

    def test_delete_calls_delete_blob(self):
        backend, container = _make_azure_backend()
        mock_blob = MagicMock()
        container.get_blob_client.return_value = mock_blob
        backend.delete("old.txt")
        mock_blob.delete_blob.assert_called_once()

    def test_delete_not_found_is_idempotent(self):
        backend, container = _make_azure_backend()
        mock_blob = MagicMock()
        from azure.core.exceptions import ResourceNotFoundError
        mock_blob.delete_blob.side_effect = ResourceNotFoundError("gone")
        container.get_blob_client.return_value = mock_blob
        backend.delete("already_gone.txt")  # should not raise

    def test_exists_true_on_properties_success(self):
        backend, container = _make_azure_backend()
        mock_blob = MagicMock()
        mock_blob.get_blob_properties.return_value = {"size": 100}
        container.get_blob_client.return_value = mock_blob
        assert backend.exists("present.txt") is True

    def test_exists_false_on_not_found(self):
        backend, container = _make_azure_backend()
        mock_blob = MagicMock()
        from azure.core.exceptions import ResourceNotFoundError
        mock_blob.get_blob_properties.side_effect = ResourceNotFoundError("gone")
        container.get_blob_client.return_value = mock_blob
        assert backend.exists("absent.txt") is False


# ═══════════════════════════════════════════════════════════════════════════════
# GCSStorageBackend stub
# ═══════════════════════════════════════════════════════════════════════════════


class TestGCSStorageBackendStub:
    def test_instantiation_raises(self):
        from storage.backends.gcs import GCSStorageBackend
        with pytest.raises(NotImplementedFeatureError) as exc_info:
            GCSStorageBackend()
        assert "R7" in str(exc_info.value)


# ═══════════════════════════════════════════════════════════════════════════════
# StorageFactory
# ═══════════════════════════════════════════════════════════════════════════════


class TestStorageFactory:
    def test_create_local(self, tmp_path):
        from storage.factory import StorageFactory
        from storage.backends.local import LocalStorageBackend
        backend = StorageFactory.create("local", config={"root": str(tmp_path)})
        assert isinstance(backend, LocalStorageBackend)

    def test_create_s3(self):
        from storage.factory import StorageFactory
        from storage.backends.s3 import S3StorageBackend
        with patch("storage.backends.s3.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            backend = StorageFactory.create("s3", config={"bucket": "my-bucket"})
        assert isinstance(backend, S3StorageBackend)

    def test_create_azure_blob(self):
        from storage.factory import StorageFactory
        from storage.backends.azure_blob import AzureBlobStorageBackend
        conn = "DefaultEndpointsProtocol=https;AccountName=x;AccountKey=a==;EndpointSuffix=core.windows.net"
        with patch("storage.backends.azure_blob.BlobServiceClient") as mock_cls:
            mock_svc = MagicMock()
            mock_cls.from_connection_string.return_value = mock_svc
            mock_svc.get_container_client.return_value = MagicMock()
            backend = StorageFactory.create("azure_blob", config={
                "container": "raglab", "connection_string": conn,
            })
        assert isinstance(backend, AzureBlobStorageBackend)

    def test_create_gcs_raises_not_implemented(self):
        from storage.factory import StorageFactory
        with pytest.raises(NotImplementedFeatureError, match="R7"):
            StorageFactory.create("gcs")

    def test_create_unknown_raises_value_error(self):
        from storage.factory import StorageFactory
        with pytest.raises(ValueError, match="Unknown storage backend"):
            StorageFactory.create("dropbox")

    def test_available_lists_all_backends(self):
        from storage.factory import StorageFactory
        backends = {b["backend"] for b in StorageFactory.available()}
        assert {"local", "s3", "azure_blob", "gcs"} == backends

    def test_local_s3_azure_are_active(self):
        from storage.factory import StorageFactory
        entries = {b["backend"]: b for b in StorageFactory.available()}
        assert entries["local"]["active"] is True
        assert entries["s3"]["active"] is True
        assert entries["azure_blob"]["active"] is True

    def test_gcs_is_not_active(self):
        from storage.factory import StorageFactory
        entries = {b["backend"]: b for b in StorageFactory.available()}
        assert entries["gcs"]["active"] is False
        assert entries["gcs"]["available_in"] == "R7"


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP endpoints
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def storage_client(tmp_path):
    from storage.main import app
    from storage.backends.local import LocalStorageBackend
    app.state.backend = LocalStorageBackend(config={"root": str(tmp_path)})
    return TestClient(app)


class TestStorageEndpoints:
    def test_health_ok(self, storage_client):
        r = storage_client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["dependencies"]["backend"] == "local"

    def test_root_shows_active_backend(self, storage_client):
        r = storage_client.get("/")
        assert r.status_code == 200
        assert r.json()["active_backend"] == "local"

    def test_upload_stores_and_returns_uri(self, storage_client):
        payload = base64.b64encode(b"test document content").decode()
        r = storage_client.post(
            "/storage/upload/docs/report.txt",
            json={"data_b64": payload},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["key"] == "docs/report.txt"
        assert body["size"] == len(b"test document content")
        assert body["backend"] == "local"

    def test_download_returns_bytes(self, storage_client):
        content = b"download me"
        payload = base64.b64encode(content).decode()
        storage_client.post("/storage/upload/dl.bin", json={"data_b64": payload})
        r = storage_client.get("/storage/download/dl.bin")
        assert r.status_code == 200
        assert r.content == content

    def test_download_missing_key_returns_404(self, storage_client):
        r = storage_client.get("/storage/download/nonexistent.bin")
        assert r.status_code == 404

    def test_delete_returns_204(self, storage_client):
        payload = base64.b64encode(b"delete me").decode()
        storage_client.post("/storage/upload/del.txt", json={"data_b64": payload})
        r = storage_client.delete("/storage/del.txt")
        assert r.status_code == 204

    def test_exists_true_after_upload(self, storage_client):
        payload = base64.b64encode(b"present").decode()
        storage_client.post("/storage/upload/present.txt", json={"data_b64": payload})
        r = storage_client.get("/storage/exists/present.txt")
        assert r.status_code == 200
        assert r.json()["exists"] is True

    def test_exists_false_for_missing_key(self, storage_client):
        r = storage_client.get("/storage/exists/missing.txt")
        assert r.status_code == 200
        assert r.json()["exists"] is False

    def test_backends_endpoint(self, storage_client):
        r = storage_client.get("/storage/backends")
        assert r.status_code == 200
        backends = {b["backend"] for b in r.json()}
        assert {"local", "s3", "azure_blob", "gcs"} == backends

    def test_invalid_base64_returns_422(self, storage_client):
        r = storage_client.post(
            "/storage/upload/bad.txt",
            json={"data_b64": "not-valid-base64!!!"},
        )
        assert r.status_code == 422

    def test_no_backend_returns_503(self):
        from storage.main import app
        app.state.backend = None
        client = TestClient(app)
        payload = base64.b64encode(b"data").decode()
        r = client.post("/storage/upload/k.txt", json={"data_b64": payload})
        assert r.status_code == 503
