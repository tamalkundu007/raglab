"""
raglab-common — Shared utilities, models, logging, and exceptions for RAGLab.

Version: 0.1.0
"""

from raglab_common.exceptions import (
    RAGLabError,
    ChunkerError,
    RetrieverError,
    EmbeddingError,
    IndexingError,
    LLMError,
    ConfigError,
    StorageError,
    NotImplementedFeatureError,
)
from raglab_common.logging import get_logger, configure_logging
from raglab_common.models import (
    ChunkModel,
    DocumentModel,
    EmbeddingModel,
    QueryModel,
    ResponseModel,
    HealthModel,
)

__all__ = [
    "RAGLabError",
    "ChunkerError",
    "RetrieverError",
    "EmbeddingError",
    "IndexingError",
    "LLMError",
    "ConfigError",
    "StorageError",
    "NotImplementedFeatureError",
    "get_logger",
    "configure_logging",
    "ChunkModel",
    "DocumentModel",
    "EmbeddingModel",
    "QueryModel",
    "ResponseModel",
    "HealthModel",
]
from raglab_common.tracing import (
    configure_tracing,
    get_tracer,
    traced_span,
    record_event,
    trace_headers,
    current_trace_id,
    make_trace_middleware,
)
