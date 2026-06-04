"""
Base settings for all RAGLab services.

Each service extends BaseServiceSettings and adds its own fields.
Secrets are loaded from environment variables / .env file only —
never hardcoded.

Usage:
    from raglab_common.settings import BaseServiceSettings

    class EmbeddingSettings(BaseServiceSettings):
        model_name: str = "text-embedding-3-small"

    settings = EmbeddingSettings()
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseServiceSettings(BaseSettings):
    """
    Base settings shared by all RAGLab microservices.

    Environment variable prefix: RAGLAB_
    """

    model_config = SettingsConfigDict(
        env_prefix="RAGLAB_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Service identity
    service_name: str = "raglab-service"
    service_version: str = "0.1.0"
    release: str = "R1"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Logging
    log_level: str = "INFO"
    json_logs: bool = False

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "raglab"

    # PostgreSQL
    postgres_dsn: str = "postgresql+asyncpg://raglab:raglab@localhost:5432/raglab"

    # RabbitMQ
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"

    # Internal service URLs
    ingestion_url: str = "http://localhost:8001"
    embedding_url: str = "http://localhost:8002"
    indexing_url: str = "http://localhost:8003"
    retrieval_url: str = "http://localhost:8004"
    llm_url: str = "http://localhost:8005"
    pipeline_url: str = "http://localhost:8006"
    config_url: str = "http://localhost:8007"
    storage_url: str = "http://localhost:8008"
    ui_url: str = "http://localhost:8009"

    # LLM API keys — loaded from env only
    azure_openai_api_key: str = Field(default="", repr=False)
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = ""
    openai_api_key: str = Field(default="", repr=False)
    anthropic_api_key: str = Field(default="", repr=False)
    ollama_base_url: str = "http://localhost:11434"
