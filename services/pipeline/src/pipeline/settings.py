"""Settings for the pipeline-service."""

from raglab_common.settings import BaseServiceSettings


class PipelineSettings(BaseServiceSettings):
    service_name: str = "pipeline"
    port: int = 8006

    # Consumer
    rabbitmq_prefetch_count: int = 1        # process one message at a time
    pipeline_worker_concurrency: int = 2    # parallel consumer coroutines

    # Internal service URLs (for HTTP calls to embedding + indexing)
    embedding_url: str = "http://embedding:8002"
    indexing_url: str = "http://indexing:8003"
    storage_url: str = "http://storage:8008"
