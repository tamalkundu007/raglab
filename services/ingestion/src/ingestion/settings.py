"""Settings for the ingestion-service."""

from raglab_common.settings import BaseServiceSettings


class IngestionSettings(BaseServiceSettings):
    service_name: str = "ingestion"
    port: int = 8001

    # Upload storage
    local_storage_root: str = "/app/data/local"

    # RabbitMQ publisher
    rabbitmq_publish_timeout: float = 5.0       # seconds per publish
    rabbitmq_confirm_delivery: bool = True       # publisher confirms
