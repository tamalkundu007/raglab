"""
RAGLab exception hierarchy.

All RAGLab exceptions inherit from RAGLabError to allow catch-all handling
at service boundaries while preserving specific exception types internally.
"""


class RAGLabError(Exception):
    """Base exception for all RAGLab errors."""

    def __init__(self, message: str, code: str = "RAGLAB_ERROR") -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.code!r}, message={self.message!r})"


class ChunkerError(RAGLabError):
    """Raised when a chunker fails to process a document."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="CHUNKER_ERROR")


class RetrieverError(RAGLabError):
    """Raised when a retriever fails to fetch results."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="RETRIEVER_ERROR")


class EmbeddingError(RAGLabError):
    """Raised when embedding generation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="EMBEDDING_ERROR")


class IndexingError(RAGLabError):
    """Raised when indexing a document or chunk fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="INDEXING_ERROR")


class LLMError(RAGLabError):
    """Raised when an LLM provider call fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="LLM_ERROR")


class ConfigError(RAGLabError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="CONFIG_ERROR")


class StorageError(RAGLabError):
    """Raised when a storage backend operation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="STORAGE_ERROR")


class NotImplementedFeatureError(RAGLabError):
    """
    Raised when a feature is visible in the UI but not yet implemented.

    Used for R2+ features stubbed in R1. Returns HTTP 501 at service boundary.
    """

    def __init__(self, feature: str, available_in: str = "future release") -> None:
        message = f"'{feature}' is not implemented yet. Available in {available_in}."
        super().__init__(message, code="NOT_IMPLEMENTED")
        self.feature = feature
        self.available_in = available_in
