"""Settings for the graph-service."""

from raglab_common.settings import BaseServiceSettings


class GraphSettings(BaseServiceSettings):
    service_name: str = "graph"
    port: int = 8010

    # LLM service for entity extraction
    llm_service_url: str = "http://llm:8005"
    default_llm_provider: str = "azure_openai"

    # Graph defaults
    default_collection: str = "raglab"
    max_entities_per_chunk: int = 10
    max_relationships_per_chunk: int = 10
    extraction_timeout_seconds: float = 30.0
