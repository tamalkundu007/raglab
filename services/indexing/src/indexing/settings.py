"""Settings for the indexing-service."""

from raglab_common.settings import BaseServiceSettings


class IndexingSettings(BaseServiceSettings):
    """Indexing-service configuration."""

    service_name: str = "indexing"
    port: int = 8003

    # Qdrant collection defaults
    qdrant_default_collection: str = "raglab"
    qdrant_vector_size: int = 1536          # text-embedding-3-small
    qdrant_distance: str = "Cosine"         # Cosine | Dot | Euclid
    qdrant_on_disk_payload: bool = True     # store payload on disk (large collections)

    # HNSW index params
    qdrant_hnsw_m: int = 16
    qdrant_hnsw_ef_construct: int = 100
