"""
Indexing-service HTTP router.

Endpoints:
  POST /index                          — index a batch of pre-embedded chunks
  GET  /collections/{name}             — collection stats
  POST /collections/{name}/ensure      — idempotent collection creation
  DELETE /collections/{name}           — delete collection (dev/test only)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from raglab_common.exceptions import IndexingError
from raglab_common.logging import get_logger
from raglab_common.models import ChunkModel, EmbeddingModel, IngestionStatus

log = get_logger(__name__)

router = APIRouter(tags=["indexing"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    """Batch index request — pre-embedded chunks ready for Qdrant upsert."""
    collection: str = Field(..., min_length=1, description="Target Qdrant collection name.")
    doc_id: str = Field(..., description="Document ID (propagated to Postgres record).")
    filename: str = Field(default="", description="Original filename (for Postgres record).")
    chunks: list[ChunkModel]
    embeddings: list[EmbeddingModel]
    doc_metadata: dict[str, Any] = Field(default_factory=dict)


class IndexResponse(BaseModel):
    doc_id: str
    collection: str
    chunks_indexed: int
    status: str


class CollectionInfoResponse(BaseModel):
    name: str
    vectors_count: int | None
    indexed_vectors_count: int | None
    status: str


class EnsureCollectionResponse(BaseModel):
    collection: str
    created: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/index", response_model=IndexResponse)
async def index_chunks(body: IndexRequest, request: Request) -> IndexResponse:
    """
    Upsert pre-embedded chunks into Qdrant and record metadata in Postgres.

    Validates that chunks and embeddings lists have the same length.
    Ensures the target collection exists before upserting.
    Writes a DocumentRecord + ChunkRecord rows to Postgres.
    """
    qdrant: Any = getattr(request.app.state, "qdrant", None)
    session_factory: Any = getattr(request.app.state, "session_factory", None)

    if qdrant is None:
        raise HTTPException(status_code=503, detail="Qdrant client not available.")

    if len(body.chunks) != len(body.embeddings):
        raise HTTPException(
            status_code=422,
            detail=f"chunks ({len(body.chunks)}) and embeddings ({len(body.embeddings)}) must match.",
        )

    # Ensure collection exists
    try:
        qdrant.ensure_collection(body.collection)
    except IndexingError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # Upsert into Qdrant
    try:
        count = qdrant.upsert_chunks(body.collection, body.chunks, body.embeddings)
    except IndexingError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # Persist metadata to Postgres (best-effort — non-fatal if DB unavailable)
    if session_factory is not None:
        try:
            await _persist_metadata(session_factory, body, count)
        except Exception as exc:
            log.warning("indexing.postgres_write_failed", error=str(exc), doc_id=body.doc_id)

    return IndexResponse(
        doc_id=body.doc_id,
        collection=body.collection,
        chunks_indexed=count,
        status=IngestionStatus.COMPLETED.value,
    )


async def _persist_metadata(session_factory: Any, body: IndexRequest, chunk_count: int) -> None:
    """Write DocumentRecord and ChunkRecord rows to Postgres."""
    from indexing.db.models import ChunkRecord, DocumentRecord

    async with session_factory() as session:
        # Upsert document record
        from sqlalchemy import select
        result = await session.execute(
            select(DocumentRecord).where(DocumentRecord.doc_id == body.doc_id)
        )
        doc_record = result.scalar_one_or_none()

        if doc_record is None:
            doc_record = DocumentRecord(
                doc_id=body.doc_id,
                filename=body.filename,
                content_type=body.doc_metadata.get("content_type", "text/plain"),
                storage_path=body.doc_metadata.get("storage_path", ""),
                collection=body.collection,
                chunker_type=body.doc_metadata.get("chunker", "text"),
                status=IngestionStatus.COMPLETED.value,
                chunk_count=chunk_count,
                doc_metadata=body.doc_metadata,
            )
            session.add(doc_record)
        else:
            doc_record.status = IngestionStatus.COMPLETED.value
            doc_record.chunk_count = chunk_count

        # Insert chunk records
        for chunk in body.chunks:
            chunk_record = ChunkRecord(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                collection=body.collection,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                text_preview=chunk.text[:500],
                chunk_metadata=chunk.metadata,
            )
            session.add(chunk_record)

        await session.commit()
        log.info("indexing.postgres_written", doc_id=body.doc_id, chunks=chunk_count)


@router.get("/collections/{name}", response_model=CollectionInfoResponse)
async def collection_info(name: str, request: Request) -> CollectionInfoResponse:
    """Return vector count and status for a Qdrant collection."""
    qdrant: Any = getattr(request.app.state, "qdrant", None)
    if qdrant is None:
        raise HTTPException(status_code=503, detail="Qdrant client not available.")
    try:
        info = qdrant.collection_info(name)
        return CollectionInfoResponse(**info)
    except IndexingError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/collections/{name}/ensure", response_model=EnsureCollectionResponse)
async def ensure_collection(name: str, request: Request) -> EnsureCollectionResponse:
    """Idempotent collection creation. Returns created=True if new, False if existed."""
    qdrant: Any = getattr(request.app.state, "qdrant", None)
    if qdrant is None:
        raise HTTPException(status_code=503, detail="Qdrant client not available.")
    try:
        created = qdrant.ensure_collection(name)
        return EnsureCollectionResponse(collection=name, created=created)
    except IndexingError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.delete("/collections/{name}", status_code=204)
async def delete_collection(name: str, request: Request) -> None:
    """Delete a Qdrant collection. Irreversible — dev/test only."""
    qdrant: Any = getattr(request.app.state, "qdrant", None)
    if qdrant is None:
        raise HTTPException(status_code=503, detail="Qdrant client not available.")
    try:
        qdrant.delete_collection(name)
    except IndexingError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
