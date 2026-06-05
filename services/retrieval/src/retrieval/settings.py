"""Settings for the retrieval-service."""

from raglab_common.settings import BaseServiceSettings


class RetrievalSettings(BaseServiceSettings):
    service_name: str = "retrieval"
    port: int = 8004

    # Default retrieval params
    default_top_k: int = 5
    default_retriever: str = "dense"
    default_provider: str = "azure_openai"

    # Embedding-service URL for query embedding
    embedding_url: str = "http://embedding:8002"
