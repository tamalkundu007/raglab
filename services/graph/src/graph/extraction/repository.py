"""
Graph persistence — save/query entities and relationships in Postgres.

All writes are upsert-safe:
    - Entities: INSERT ... ON CONFLICT DO NOTHING (dedup by name_normalised + type + collection).
    - Relationships: INSERT ... ON CONFLICT DO NOTHING (dedup by source + target + type + collection).

This module has no knowledge of NetworkX — graph construction is in graph_builder.py.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from raglab_common.logging import get_logger

from graph.models.orm import GraphEntity, GraphRelationship, GraphRun, GraphRunStatus
from graph.models.schemas import (
    EntitySchema,
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelationship,
    RelationshipSchema,
)

log = get_logger(__name__)


class GraphRepository:
    """
    Async Postgres repository for graph entities and relationships.

    All methods accept an AsyncSession — caller manages transaction scope.
    """

    # ── Entity persistence ─────────────────────────────────────────────────────

    async def upsert_entity(
        self,
        session: AsyncSession,
        entity: ExtractedEntity,
        collection: str,
        doc_id: str,
        chunk_id: str,
    ) -> GraphEntity:
        """
        Upsert an entity. Returns the existing or newly created ORM instance.

        Deduplication: (name_normalised, entity_type, collection).
        If entity exists, append chunk_id to source_chunk_ids.
        """
        name_norm = entity.name.lower().strip()

        # Try to fetch existing
        stmt = select(GraphEntity).where(
            GraphEntity.name_normalised == name_norm,
            GraphEntity.entity_type == entity.entity_type.upper(),
            GraphEntity.collection == collection,
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            # Append chunk_id if not already present
            current_ids = set(existing.source_chunk_ids.split("|")) if existing.source_chunk_ids else set()
            current_ids.add(chunk_id)
            existing.source_chunk_ids = "|".join(sorted(current_ids))
            return existing

        new_entity = GraphEntity(
            id=uuid.uuid4(),
            name=entity.name,
            name_normalised=name_norm,
            entity_type=entity.entity_type.upper(),
            collection=collection,
            description=entity.description,
            source_chunk_ids=chunk_id,
            doc_id=doc_id,
        )
        session.add(new_entity)
        await session.flush()  # get DB-assigned ID
        return new_entity

    async def upsert_relationship(
        self,
        session: AsyncSession,
        rel: ExtractedRelationship,
        source_entity: GraphEntity,
        target_entity: GraphEntity,
        collection: str,
        chunk_id: str,
    ) -> GraphRelationship | None:
        """
        Upsert a relationship between two entities.

        Returns None if source or target entity is missing.
        """
        if source_entity is None or target_entity is None:
            return None

        rel_type = rel.relation_type.upper()

        stmt = select(GraphRelationship).where(
            GraphRelationship.source_id == source_entity.id,
            GraphRelationship.target_id == target_entity.id,
            GraphRelationship.relation_type == rel_type,
            GraphRelationship.collection == collection,
        )
        result = await session.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            return existing

        new_rel = GraphRelationship(
            id=uuid.uuid4(),
            source_id=source_entity.id,
            target_id=target_entity.id,
            relation_type=rel_type,
            collection=collection,
            description=rel.description,
            source_chunk_id=chunk_id,
            weight=rel.weight,
        )
        session.add(new_rel)
        await session.flush()
        return new_rel

    # ── Batch persistence ──────────────────────────────────────────────────────

    async def persist_extraction_result(
        self,
        session: AsyncSession,
        result: ExtractionResult,
        collection: str,
        doc_id: str,
    ) -> tuple[int, int]:
        """
        Persist all entities and relationships from an ExtractionResult.

        Returns:
            (entities_persisted, relationships_persisted) counts.
        """
        entity_map: dict[str, GraphEntity] = {}
        entities_saved = 0

        # Upsert entities first
        for extracted_entity in result.entities:
            orm_entity = await self.upsert_entity(
                session, extracted_entity, collection, doc_id, result.chunk_id
            )
            entity_map[extracted_entity.name] = orm_entity
            entities_saved += 1

        # Upsert relationships
        rels_saved = 0
        for rel in result.relationships:
            source = entity_map.get(rel.source)
            target = entity_map.get(rel.target)
            if source is None or target is None:
                log.warning(
                    "graph.relationship_skipped",
                    reason="entity_not_found",
                    source=rel.source,
                    target=rel.target,
                )
                continue

            orm_rel = await self.upsert_relationship(
                session, rel, source, target, collection, result.chunk_id
            )
            if orm_rel is not None:
                rels_saved += 1

        return entities_saved, rels_saved

    # ── Graph run tracking ─────────────────────────────────────────────────────

    async def create_run(
        self, session: AsyncSession, doc_id: str, collection: str
    ) -> GraphRun:
        run = GraphRun(
            id=uuid.uuid4(),
            doc_id=doc_id,
            collection=collection,
            status=GraphRunStatus.RUNNING,
        )
        session.add(run)
        await session.flush()
        return run

    async def complete_run(
        self,
        session: AsyncSession,
        run: GraphRun,
        entity_count: int,
        relationship_count: int,
        chunk_count: int,
    ) -> None:
        run.status = GraphRunStatus.COMPLETE
        run.entity_count = str(entity_count)
        run.relationship_count = str(relationship_count)
        run.chunk_count = str(chunk_count)

    async def fail_run(
        self, session: AsyncSession, run: GraphRun, error: str
    ) -> None:
        run.status = GraphRunStatus.FAILED
        run.error_message = error[:1000]

    # ── Query helpers ──────────────────────────────────────────────────────────

    async def get_entities(
        self,
        session: AsyncSession,
        collection: str,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[GraphEntity]:
        stmt = select(GraphEntity).where(GraphEntity.collection == collection)
        if entity_type:
            stmt = stmt.where(GraphEntity.entity_type == entity_type.upper())
        stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def get_entity_by_name(
        self,
        session: AsyncSession,
        name: str,
        collection: str,
    ) -> GraphEntity | None:
        stmt = select(GraphEntity).where(
            GraphEntity.name_normalised == name.lower().strip(),
            GraphEntity.collection == collection,
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_relationships(
        self,
        session: AsyncSession,
        collection: str,
        entity_id: str | None = None,
        limit: int = 200,
    ) -> list[GraphRelationship]:
        from sqlalchemy.orm import joinedload
        stmt = (
            select(GraphRelationship)
            .options(
                joinedload(GraphRelationship.source),
                joinedload(GraphRelationship.target),
            )
            .where(GraphRelationship.collection == collection)
        )
        if entity_id:
            stmt = stmt.where(
                (GraphRelationship.source_id == entity_id) |
                (GraphRelationship.target_id == entity_id)
            )
        stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        return list(result.unique().scalars().all())

    async def get_stats(
        self, session: AsyncSession, collection: str
    ) -> dict[str, Any]:
        # Entity count
        entity_count_r = await session.execute(
            select(func.count()).where(GraphEntity.collection == collection)
        )
        entity_count = entity_count_r.scalar() or 0

        # Relationship count
        rel_count_r = await session.execute(
            select(func.count()).where(GraphRelationship.collection == collection)
        )
        rel_count = rel_count_r.scalar() or 0

        # Entity type breakdown
        type_r = await session.execute(
            select(GraphEntity.entity_type, func.count())
            .where(GraphEntity.collection == collection)
            .group_by(GraphEntity.entity_type)
        )
        entity_types = {row[0]: row[1] for row in type_r.all()}

        # Relation type breakdown
        rel_type_r = await session.execute(
            select(GraphRelationship.relation_type, func.count())
            .where(GraphRelationship.collection == collection)
            .group_by(GraphRelationship.relation_type)
        )
        relation_types = {row[0]: row[1] for row in rel_type_r.all()}

        return {
            "entity_count": entity_count,
            "relationship_count": rel_count,
            "entity_types": entity_types,
            "relation_types": relation_types,
        }
