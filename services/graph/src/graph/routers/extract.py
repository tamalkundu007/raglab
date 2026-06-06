"""
Graph-service extraction router.

Endpoints:
  POST /graph/extract          — extract entities + relationships from chunks
  GET  /graph/entities         — list entities in a collection
  GET  /graph/relationships    — list relationships in a collection
  GET  /graph/stats            — entity/relationship counts and type breakdown
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from raglab_common.exceptions import LLMError
from raglab_common.logging import get_logger

from graph.extraction.extractor import EntityRelationshipExtractor
from graph.extraction.repository import GraphRepository
from graph.models.schemas import (
    EntitySchema,
    ExtractRequest,
    ExtractResponse,
    GraphStatsResponse,
    RelationshipSchema,
)

log = get_logger(__name__)
router = APIRouter(prefix="/graph", tags=["graph"])
repo = GraphRepository()


async def get_session(request: Request) -> AsyncSession:
    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        raise HTTPException(status_code=503, detail="Database not initialised.")
    async with session_factory() as session:
        yield session


@router.post("/extract", response_model=ExtractResponse)
async def extract(
    body: ExtractRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> ExtractResponse:
    """
    Extract entities and relationships from a list of chunks.

    Processes each chunk sequentially via the LLM extractor and persists
    results to Postgres. Returns counts of extracted + persisted objects.
    """
    if len(body.chunk_ids) != len(body.chunk_texts):
        raise HTTPException(
            status_code=422,
            detail=f"chunk_ids ({len(body.chunk_ids)}) and chunk_texts "
                   f"({len(body.chunk_texts)}) must have the same length.",
        )

    settings = getattr(request.app.state, "settings", None)
    llm_service_url = getattr(settings, "llm_service_url", "http://llm:8005") if settings else "http://llm:8005"

    extractor = EntityRelationshipExtractor(config={
        "llm_service_url": llm_service_url,
        "llm_provider": body.llm_provider,
        "entity_types": ["PERSON", "ORGANIZATION", "CONCEPT", "TECHNOLOGY", "LOCATION", "PRODUCT"],
        "relation_types": body.relationship_types,
        "max_entities_per_chunk": body.max_entities_per_chunk,
        "max_relationships_per_chunk": body.max_relationships_per_chunk,
    })

    run = await repo.create_run(session, body.doc_id, body.collection)
    total_entities = 0
    total_relationships = 0
    total_entities_persisted = 0
    total_rels_persisted = 0

    try:
        for chunk_id, chunk_text in zip(body.chunk_ids, body.chunk_texts):
            result = extractor.extract_from_chunk(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
            )
            total_entities += len(result.entities)
            total_relationships += len(result.relationships)

            e_saved, r_saved = await repo.persist_extraction_result(
                session, result, body.collection, body.doc_id
            )
            total_entities_persisted += e_saved
            total_rels_persisted += r_saved

        await repo.complete_run(
            session, run,
            entity_count=total_entities_persisted,
            relationship_count=total_rels_persisted,
            chunk_count=len(body.chunk_ids),
        )
        await session.commit()

    except Exception as exc:
        await repo.fail_run(session, run, error=str(exc))
        await session.commit()
        log.error("graph.extract_failed", doc_id=body.doc_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")

    return ExtractResponse(
        doc_id=body.doc_id,
        collection=body.collection,
        chunks_processed=len(body.chunk_ids),
        entities_extracted=total_entities,
        relationships_extracted=total_relationships,
        entities_persisted=total_entities_persisted,
        relationships_persisted=total_rels_persisted,
        run_id=str(run.id),
    )


@router.get("/entities", response_model=list[EntitySchema])
async def list_entities(
    collection: str = "raglab",
    entity_type: str | None = None,
    limit: int = 100,
    session: AsyncSession = Depends(get_session),
) -> list[EntitySchema]:
    """List entities in a collection, optionally filtered by entity_type."""
    entities = await repo.get_entities(session, collection, entity_type, limit)
    return [EntitySchema.from_orm_entity(e) for e in entities]


@router.get("/relationships", response_model=list[RelationshipSchema])
async def list_relationships(
    collection: str = "raglab",
    entity_id: str | None = None,
    limit: int = 200,
    session: AsyncSession = Depends(get_session),
) -> list[RelationshipSchema]:
    """List relationships in a collection, optionally filtered by entity_id."""
    rels = await repo.get_relationships(session, collection, entity_id, limit)
    return [RelationshipSchema.from_orm_rel(r) for r in rels]


@router.get("/stats", response_model=GraphStatsResponse)
async def graph_stats(
    collection: str = "raglab",
    session: AsyncSession = Depends(get_session),
) -> GraphStatsResponse:
    """Entity and relationship counts + type breakdown for a collection."""
    stats = await repo.get_stats(session, collection)
    return GraphStatsResponse(collection=collection, **stats)
