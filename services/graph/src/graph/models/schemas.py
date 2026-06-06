"""Pydantic v2 schemas for graph-service API."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class EntitySchema(BaseModel):
    """Entity as returned by the API."""
    id: str
    name: str
    entity_type: str
    collection: str
    description: str | None = None
    source_chunk_ids: list[str] = Field(default_factory=list)
    doc_id: str | None = None

    @classmethod
    def from_orm_entity(cls, e: Any) -> "EntitySchema":
        chunk_ids = (
            [c for c in e.source_chunk_ids.split("|") if c]
            if e.source_chunk_ids else []
        )
        return cls(
            id=str(e.id),
            name=e.name,
            entity_type=e.entity_type,
            collection=e.collection,
            description=e.description,
            source_chunk_ids=chunk_ids,
            doc_id=e.doc_id,
        )


class RelationshipSchema(BaseModel):
    """Relationship as returned by the API."""
    id: str
    source_id: str
    target_id: str
    source_name: str | None = None
    target_name: str | None = None
    relation_type: str
    collection: str
    description: str | None = None
    weight: float = 1.0

    @classmethod
    def from_orm_rel(cls, r: Any) -> "RelationshipSchema":
        return cls(
            id=str(r.id),
            source_id=str(r.source_id),
            target_id=str(r.target_id),
            source_name=r.source.name if r.source else None,
            target_name=r.target.name if r.target else None,
            relation_type=r.relation_type,
            collection=r.collection,
            description=r.description,
            weight=r.weight,
        )


class ExtractedEntity(BaseModel):
    """Entity as returned by the LLM extractor (pre-persistence)."""
    name: str
    entity_type: str
    description: str | None = None


class ExtractedRelationship(BaseModel):
    """Relationship as returned by the LLM extractor (pre-persistence)."""
    source: str
    target: str
    relation_type: str
    description: str | None = None
    weight: float = 1.0


class ExtractionResult(BaseModel):
    """Full result from one LLM extraction call over a chunk."""
    chunk_id: str
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


class ExtractRequest(BaseModel):
    """Request body for POST /graph/extract."""
    doc_id: str
    collection: str = "raglab"
    chunk_ids: list[str] = Field(default_factory=list)
    chunk_texts: list[str] = Field(default_factory=list)
    llm_provider: str = "azure_openai"
    relationship_types: list[str] = Field(
        default_factory=lambda: ["RELATED_TO", "PART_OF", "CAUSES", "USED_BY", "WORKS_AT"]
    )
    max_entities_per_chunk: int = Field(default=10, ge=1, le=50)
    max_relationships_per_chunk: int = Field(default=10, ge=1, le=50)


class ExtractResponse(BaseModel):
    """Response from POST /graph/extract."""
    doc_id: str
    collection: str
    chunks_processed: int
    entities_extracted: int
    relationships_extracted: int
    entities_persisted: int
    relationships_persisted: int
    run_id: str


class GraphStatsResponse(BaseModel):
    """Response from GET /graph/stats."""
    collection: str
    entity_count: int
    relationship_count: int
    entity_types: dict[str, int]
    relation_types: dict[str, int]
