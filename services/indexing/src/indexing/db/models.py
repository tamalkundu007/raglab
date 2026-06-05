"""
PostgreSQL metadata models for the indexing-service.

Two tables:
  - documents   : one row per ingested document
  - chunks      : one row per indexed chunk (FK to documents)

These complement Qdrant (which stores vectors + payload). Postgres is the
source of truth for document-level state, ingestion timestamps, and
queries that span collections (e.g. "list all docs in collection X").

Qdrant stores vectors; Postgres stores metadata. Both reference the same
doc_id and chunk_id UUIDs so joins are possible at query time.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from raglab_common.db import Base
from raglab_common.models import IngestionStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DocumentRecord(Base):
    """
    One row per ingested document.

    doc_id is the canonical identifier shared with Qdrant payloads.
    idempotency_key prevents duplicate ingestion of the same file.
    """

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_documents_idempotency_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True,
        default=lambda: str(uuid.uuid4()),
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    collection: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    chunker_type: Mapped[str] = mapped_column(String(64), nullable=False, default="text")
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=IngestionStatus.PENDING.value,
        server_default=IngestionStatus.PENDING.value, index=True
    )
    idempotency_key: Mapped[str | None] = mapped_column(String(256), nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    doc_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # Relationship
    chunks: Mapped[list[ChunkRecord]] = relationship(
        "ChunkRecord", back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<DocumentRecord doc_id={self.doc_id!r} status={self.status!r}>"


class ChunkRecord(Base):
    """
    One row per indexed chunk.

    chunk_id matches the UUID in Qdrant payload so cross-store lookups
    use the same identifier without a separate mapping table.
    """

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    doc_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("documents.doc_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    collection: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    text_preview: Mapped[str] = mapped_column(String(512), nullable=False)
    chunk_metadata: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    indexed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    # Relationship
    document: Mapped[DocumentRecord] = relationship(
        "DocumentRecord", back_populates="chunks"
    )

    def __repr__(self) -> str:
        return f"<ChunkRecord chunk_id={self.chunk_id!r} doc_id={self.doc_id!r}>"
