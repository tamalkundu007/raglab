"""Settings for the storage-service."""

from raglab_common.settings import BaseServiceSettings


class StorageSettings(BaseServiceSettings):
    service_name: str = "storage"
    port: int = 8008

    # Active backend (local | s3 | azure_blob | gcs)
    storage_provider: str = "local"

    # Local backend
    local_storage_root: str = "/app/data/local"

    # S3 backend
    s3_bucket: str = ""
    s3_region: str = "us-east-1"
    s3_prefix: str = ""

    # AWS credentials (loaded from env — never hardcoded)
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # Azure Blob backend
    azure_blob_container: str = ""
    azure_storage_connection_string: str = ""
    azure_storage_account_name: str = ""
    azure_storage_account_key: str = ""
    azure_blob_prefix: str = ""
