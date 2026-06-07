"""Settings for the embedding-service."""

from raglab_common.settings import BaseServiceSettings


class EmbeddingSettings(BaseServiceSettings):
    """
    Embedding-service configuration.

    Loaded from environment variables with RAGLAB_ prefix.
    """

    service_name: str = "embedding"
    port: int = 8002

    # Default embedding model per provider
    openai_embedding_model: str = "text-embedding-3-small"
    azure_openai_embedding_deployment: str = ""   # e.g. "text-embedding-3-small"
    ollama_embedding_model: str = "nomic-embed-text"

    # Embedding dimensions (used for Qdrant collection creation)
    embedding_dimensions: int = 1536  # text-embedding-3-small default

    # Batch size for bulk embedding requests
    embedding_batch_size: int = 32

    # ── Embedding cache (Redis) ────────────────────────────────────────────────
    embedding_cache_enabled: bool = True
    embedding_cache_redis_url: str = "redis://redis:6379/0"
    embedding_cache_ttl_seconds: int = 86400  # 24 hours
