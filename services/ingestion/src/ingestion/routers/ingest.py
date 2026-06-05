"""
Ingestion-service HTTP router.

Endpoints:
  POST /ingest          — submit a document for async ingestion
  GET  /ingest/{doc_id} — check ingestion status
"""

from __future__ import annotations

import hashlib
import os
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from raglab_common.logging import get_logger
from raglab_common.models import IngestionStatus
from raglab_common.queue import IngestionMessage

log = get_logger(__name__)
router = APIRouter(tags=["ingestion"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Document ingestion request."""
    filename: str = Field(..., min_length=1)
    content_type: str = Field(default="text/plain")
    storage_path: str = Field(..., min_length=1,
                              description="Path where document is already stored.")
    collection: str = Field(default="raglab")
    chunker_type: str = Field(default="text")
    chunker_config: dict[str, Any] = Field(default_factory=dict)
    llm_provider: str = Field(default="azure_openai")
    doc_metadata: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = Field(
        default=None,
        description="Caller-supplied idempotency key. Auto-generated from "
                    "filename+collection if not provided.",
    )


class IngestResponse(BaseModel):
    doc_id: str
    idempotency_key: str
    status: str
    message: str
    duplicate: bool = False


class StatusResponse(BaseModel):
    doc_id: str
    status: str
    detail: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(body: IngestRequest, request: Request) -> IngestResponse:
    """
    Submit a document for async ingestion.

    Idempotency:
        If an idempotency_key is provided and already exists in Postgres,
        the request is acked (202) without re-publishing to RabbitMQ.
        Duplicate detection prevents re-embedding the same document on
        accidental retries or network failures.

    Flow:
        1. Resolve / generate idempotency_key.
        2. Check Postgres for existing record (duplicate guard).
        3. Publish IngestionMessage to RabbitMQ.
        4. Write PENDING DocumentRecord to Postgres.
        5. Return doc_id and status.
    """
    publisher = getattr(request.app.state, "publisher", None)
    session_factory = getattr(request.app.state, "session_factory", None)

    # Resolve idempotency key
    idem_key = body.idempotency_key or _generate_idem_key(body.filename, body.collection)

    # Duplicate check (best-effort — skip if Postgres unavailable)
    if session_factory is not None:
        existing = await _find_existing(session_factory, idem_key)
        if existing is not None:
            log.info("ingestion.duplicate_skipped", idempotency_key=idem_key, doc_id=existing["doc_id"])
            return IngestResponse(
                doc_id=existing["doc_id"],
                idempotency_key=idem_key,
                status=existing["status"],
                message="Duplicate request — document already ingested.",
                duplicate=True,
            )

    # Build message
    doc_id = str(uuid.uuid4())
    message = IngestionMessage(
        idempotency_key=idem_key,
        doc_id=doc_id,
        filename=body.filename,
        content_type=body.content_type,
        storage_path=body.storage_path,
        collection=body.collection,
        chunker_type=body.chunker_type,
        chunker_config=body.chunker_config,
        llm_provider=body.llm_provider,
        doc_metadata=body.doc_metadata,
    )

    # Publish to RabbitMQ
    if publisher is None or not publisher.is_connected:
        raise HTTPException(
            status_code=503,
            detail="Message queue unavailable. Cannot accept ingestion requests.",
        )

    try:
        await publisher.publish(message)
    except Exception as exc:
        log.error("ingestion.publish_failed", error=str(exc), doc_id=doc_id)
        raise HTTPException(status_code=502, detail=f"Queue publish failed: {exc}")

    # Persist PENDING record (best-effort)
    if session_factory is not None:
        await _write_pending(session_factory, message)

    return IngestResponse(
        doc_id=doc_id,
        idempotency_key=idem_key,
        status=IngestionStatus.PENDING.value,
        message="Document accepted for ingestion.",
    )


@router.get("/ingest/{doc_id}", response_model=StatusResponse)
async def ingestion_status(doc_id: str, request: Request) -> StatusResponse:
    """Return the current ingestion status for a document ID."""
    session_factory = getattr(request.app.state, "session_factory", None)
    if session_factory is None:
        raise HTTPException(status_code=503, detail="Database unavailable.")

    record = await _get_status(session_factory, doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    return StatusResponse(
        doc_id=doc_id,
        status=record["status"],
        detail=record.get("error_message"),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_idem_key(filename: str, collection: str) -> str:
    """Deterministic idempotency key from filename + collection."""
    raw = f"{filename}::{collection}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


async def _find_existing(session_factory: Any, idem_key: str) -> dict | None:
    """Return doc_id and status if idempotency_key already in Postgres."""
    try:
        from sqlalchemy import select, text
        from indexing.db.models import DocumentRecord
        async with session_factory() as session:
            result = await session.execute(
                select(DocumentRecord).where(DocumentRecord.idempotency_key == idem_key)
            )
            record = result.scalar_one_or_none()
            if record:
                return {"doc_id": record.doc_id, "status": record.status}
    except Exception as exc:
        log.warning("ingestion.duplicate_check_failed", error=str(exc))
    return None


async def _write_pending(session_factory: Any, message: IngestionMessage) -> None:
    """Write a PENDING DocumentRecord to Postgres."""
    try:
        from indexing.db.models import DocumentRecord
        async with session_factory() as session:
            record = DocumentRecord(
                doc_id=message.doc_id,
                filename=message.filename,
                content_type=message.content_type,
                storage_path=message.storage_path,
                collection=message.collection,
                chunker_type=message.chunker_type,
                status=IngestionStatus.PENDING.value,
                idempotency_key=message.idempotency_key,
                doc_metadata=message.doc_metadata,
            )
            session.add(record)
            await session.commit()
    except Exception as exc:
        log.warning("ingestion.pending_write_failed", error=str(exc), doc_id=message.doc_id)


async def _get_status(session_factory: Any, doc_id: str) -> dict | None:
    """Fetch document status from Postgres."""
    try:
        from sqlalchemy import select
        from indexing.db.models import DocumentRecord
        async with session_factory() as session:
            result = await session.execute(
                select(DocumentRecord).where(DocumentRecord.doc_id == doc_id)
            )
            record = result.scalar_one_or_none()
            if record:
                return {"status": record.status, "error_message": record.error_message}
    except Exception as exc:
        log.warning("ingestion.status_check_failed", error=str(exc))
    return None
