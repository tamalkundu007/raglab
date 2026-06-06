"""
Shared Pydantic models used across RAGLab services.

These models form the data contract between services. All inter-service
communication uses these types serialized as JSON.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ChunkerType(str, Enum):
    TEXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    HTML = "html"
    EXCEL = "excel"
    PDF_IMAGES = "pdf_images"
    TABLE_STITCH = "table_stitch"


class RetrieverType(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    MMR = "mmr"
    RERANKER = "reranker"
    COMPRESSION = "compression"
    GRAPH = "graph"


class VectorStoreType(str, Enum):
    QDRANT = "qdrant"
    FAISS = "faiss"
    CHROMADB = "chromadb"
    PINECONE = "pinecone"


class LLMProvider(str, Enum):
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    VERTEX = "vertex"


class StorageBackend(str, Enum):
    LOCAL = "local"
    S3 = "s3"
    AZURE_BLOB = "azure_blob"
    GCS = "gcs"


class IngestionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class DocumentModel(BaseModel):
    """A source document submitted for ingestion."""

    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    content_type: str
    storage_path: str
    idempotency_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


class ChunkModel(BaseModel):
    """A single text chunk produced by a chunker."""

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    text: str
    chunk_index: int
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


class EmbeddingModel(BaseModel):
    """An embedding vector paired with its source chunk."""

    chunk_id: str
    doc_id: str
    vector: list[float]
    model: str
    dimensions: int
    created_at: datetime = Field(default_factory=_utcnow)


class QueryModel(BaseModel):
    """An incoming RAG query."""

    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    collection: str
    retriever_type: RetrieverType = RetrieverType.DENSE
    llm_provider: LLMProvider = LLMProvider.AZURE_OPENAI
    top_k: int = Field(default=5, ge=1, le=50)
    metadata_filter: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


class ResponseModel(BaseModel):
    """A generated RAG response with source attribution."""

    query_id: str
    answer: str
    sources: list[ChunkModel] = Field(default_factory=list)
    model: str
    latency_ms: float
    created_at: datetime = Field(default_factory=_utcnow)


class HealthModel(BaseModel):
    """Standard health check response for all services."""

    service: str
    status: str = "ok"
    version: str = "0.1.0"
    release: str = "R1"
    timestamp: datetime = Field(default_factory=_utcnow)
    dependencies: dict[str, str] = Field(default_factory=dict)
