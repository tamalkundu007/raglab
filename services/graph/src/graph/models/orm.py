"""
Graph ORM models — Postgres tables for the graph-service.

Tables:
    graph_entities    — unique entities extracted from chunks
    graph_relationships — directed relationships between entities
    graph_runs        — extraction job tracking (doc_id + status)

Design notes:
    - Entities are deduplicated by (name_normalised, entity_type, collection).
      name_normalised = lower().strip() — "Apple" and "apple" are the same entity.
    - Relationships are directed: source_id → target_id with a relation_type label.
    - graph_runs tracks extraction progress so re-ingestion is idempotent.
    - All tables carry created_at for audit; relationships carry weight (float)
      for optional traversal cost assignment.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class GraphRunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class GraphRun(Base):
    """Tracks extraction jobs — one row per (doc_id, collection) pair."""

    __tablename__ = "graph_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(String(512), nullable=False, index=True)
    collection = Column(String(256), nullable=False, default="raglab")
    status = Column(String(32), nullable=False, default=GraphRunStatus.PENDING)
    chunk_count = Column(String(16), nullable=True)  # chunks processed
    entity_count = Column(String(16), nullable=True)
    relationship_count = Column(String(16), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint("doc_id", "collection", name="uq_graph_run_doc_collection"),
    )


class GraphEntity(Base):
    """
    A unique entity extracted from the document corpus.

    Deduplication key: (name_normalised, entity_type, collection).
    Multiple chunks may reference the same entity — source_chunk_ids
    stores a pipe-separated list of chunk UUIDs that mentioned it.
    """

    __tablename__ = "graph_entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(512), nullable=False)
    name_normalised = Column(String(512), nullable=False)
    entity_type = Column(String(128), nullable=False)  # PERSON, ORG, CONCEPT, etc.
    collection = Column(String(256), nullable=False, default="raglab")
    description = Column(Text, nullable=True)
    source_chunk_ids = Column(Text, nullable=True)  # pipe-separated UUIDs
    doc_id = Column(String(512), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    outgoing = relationship(
        "GraphRelationship",
        foreign_keys="GraphRelationship.source_id",
        back_populates="source",
        cascade="all, delete-orphan",
    )
    incoming = relationship(
        "GraphRelationship",
        foreign_keys="GraphRelationship.target_id",
        back_populates="target",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "name_normalised", "entity_type", "collection",
            name="uq_entity_name_type_collection",
        ),
        Index("ix_entity_collection_type", "collection", "entity_type"),
        Index("ix_entity_name_norm", "name_normalised"),
    )


class GraphRelationship(Base):
    """
    A directed relationship between two entities.

    source_id → target_id with a relation_type label.
    weight: optional traversal cost (default 1.0, lower = preferred path).
    """

    __tablename__ = "graph_relationships"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_id = Column(
        UUID(as_uuid=True),
        ForeignKey("graph_entities.id", ondelete="CASCADE"),
        nullable=False,
    )
    relation_type = Column(String(256), nullable=False)  # "WORKS_AT", "RELATED_TO", etc.
    collection = Column(String(256), nullable=False, default="raglab")
    description = Column(Text, nullable=True)
    source_chunk_id = Column(String(512), nullable=True)
    weight = Column(Float, nullable=False, default=1.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    source = relationship(
        "GraphEntity", foreign_keys=[source_id], back_populates="outgoing"
    )
    target = relationship(
        "GraphEntity", foreign_keys=[target_id], back_populates="incoming"
    )

    __table_args__ = (
        UniqueConstraint(
            "source_id", "target_id", "relation_type", "collection",
            name="uq_relationship_source_target_type",
        ),
        Index("ix_rel_source", "source_id"),
        Index("ix_rel_target", "target_id"),
        Index("ix_rel_collection", "collection"),
    )
