"""
Pipeline-service HTTP router.

Endpoints:
  POST /pipeline/run    — synchronous pipeline trigger (dev/test)
  GET  /pipeline/status — consumer health and queue stats
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from raglab_common.logging import get_logger
from raglab_common.queue import IngestionMessage

log = get_logger(__name__)
router = APIRouter(prefix="/pipeline", tags=["pipeline"])


class RunRequest(BaseModel):
    """Direct pipeline invocation — bypasses RabbitMQ (dev/test only)."""
    filename: str
    storage_path: str
    doc_id: str = ""
    collection: str = "raglab"
    chunker_type: str = "text"
    chunker_config: dict[str, Any] = Field(default_factory=dict)
    llm_provider: str = "azure_openai"
    doc_metadata: dict[str, Any] = Field(default_factory=dict)


class RunResponse(BaseModel):
    doc_id: str
    status: str
    detail: str | None = None


class StatusResponse(BaseModel):
    consumer_running: bool
    rabbitmq_connected: bool


@router.post("/run", response_model=RunResponse)
async def run_pipeline(body: RunRequest, request: Request) -> RunResponse:
    """
    Directly invoke the pipeline for a document (bypasses RabbitMQ).

    Useful for development, testing, and debugging without a broker.
    In production, documents flow through the async queue path.
    """
    from pipeline.runner import run_pipeline as _run, PipelineError
    import uuid

    doc_id = body.doc_id or str(uuid.uuid4())
    message = IngestionMessage(
        idempotency_key=f"direct::{doc_id}",
        doc_id=doc_id,
        filename=body.filename,
        storage_path=body.storage_path,
        collection=body.collection,
        chunker_type=body.chunker_type,
        chunker_config=body.chunker_config,
        llm_provider=body.llm_provider,
        doc_metadata=body.doc_metadata,
    )
    try:
        await _run(message, request.app.state)
        return RunResponse(doc_id=doc_id, status="completed")
    except PipelineError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/status", response_model=StatusResponse)
async def pipeline_status(request: Request) -> StatusResponse:
    """Return consumer and broker connection status."""
    consumer = getattr(request.app.state, "consumer", None)
    consumer_running = getattr(request.app.state, "consumer_running", False)
    rabbitmq_ok = consumer is not None and getattr(consumer, "_connection", None) is not None

    return StatusResponse(
        consumer_running=consumer_running,
        rabbitmq_connected=rabbitmq_ok,
    )
