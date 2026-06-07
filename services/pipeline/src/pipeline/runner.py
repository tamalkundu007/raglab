"""
Pipeline runner — the core ingestion orchestration logic.

Called by the consumer for each IngestionMessage. Executes the full
R1 pipeline synchronously within an async context:

  1. Read document text from storage (local filesystem in R1)
  2. Chunk with ChunkerFactory (TextChunker in R1)
  3. Embed chunks via embedding-service HTTP API
  4. Index into Qdrant + Postgres via indexing-service HTTP API

Each step raises on failure — the consumer handles retry / DLQ routing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from raglab_chunkers import ChunkerFactory
from raglab_common.exceptions import RAGLabError
from raglab_common.logging import get_logger
from raglab_common.models import ChunkModel, EmbeddingModel
from raglab_common.queue import IngestionMessage
from raglab_common.tracing import trace_headers, traced_span, get_tracer
from pipeline.quality_gate import apply_quality_gate

log = get_logger(__name__)


class PipelineError(RAGLabError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="PIPELINE_ERROR")


async def run_pipeline(message: IngestionMessage, app_state: Any) -> None:
    """
    Execute the full ingestion pipeline for one document.

    Args:
        message:    IngestionMessage with document location and config.
        app_state:  FastAPI app.state (carries settings and HTTP clients).

    Raises:
        PipelineError: On any step failure. Consumer handles retry.
    """
    settings = getattr(app_state, "settings", None)
    log.info("pipeline.start", doc_id=message.doc_id, filename=message.filename)

    # Step 1: Read text from storage
    text = await _read_document(message.storage_path)
    log.info("pipeline.read", doc_id=message.doc_id, chars=len(text))

    # Step 2: Chunk
    chunker = ChunkerFactory.create(message.chunker_type, config=message.chunker_config)
    chunks: list[ChunkModel] = chunker.chunk(
        text=text,
        doc_id=message.doc_id,
        metadata={"filename": message.filename, "collection": message.collection},
    )
    if not chunks:
        raise PipelineError(f"Chunker produced 0 chunks for doc_id={message.doc_id}.")
    log.info("pipeline.chunked", doc_id=message.doc_id, chunks=len(chunks))

    # Step 2b: Chunk quality gate (R5) — filter/flag low-quality chunks
    quality_config = getattr(settings, "chunk_quality_config", None) if settings else None
    chunks, gate_summary = apply_quality_gate(chunks, quality_config)
    if gate_summary.get("enabled"):
        log.info("pipeline.quality_gate", doc_id=message.doc_id, **{
            k: gate_summary[k] for k in ("total", "accepted", "flagged", "excluded")
        })
    if not chunks:
        raise PipelineError(
            f"All chunks excluded by quality gate for doc_id={message.doc_id}."
        )

    # Step 3: Embed via embedding-service
    embedding_url = getattr(settings, "embedding_url", "http://embedding:8002") if settings else "http://embedding:8002"
    embeddings = await _embed_chunks(chunks, message.llm_provider, embedding_url)
    log.info("pipeline.embedded", doc_id=message.doc_id, embeddings=len(embeddings))

    # Step 4: Index via indexing-service
    indexing_url = getattr(settings, "indexing_url", "http://indexing:8003") if settings else "http://indexing:8003"
    await _index_chunks(message, chunks, embeddings, indexing_url)
    log.info("pipeline.indexed", doc_id=message.doc_id)


async def _read_document(storage_path: str) -> str:
    """
    Read document text from local filesystem.

    R2+ will route cloud URIs (s3://, az://, gs://) to storage-service.
    In R1, only local paths are supported.
    """
    path = Path(storage_path)
    if not path.exists():
        raise PipelineError(f"Storage path not found: {storage_path}")
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        raise PipelineError(f"Failed to read document: {exc}") from exc


async def _embed_chunks(
    chunks: list[ChunkModel],
    llm_provider: str,
    embedding_url: str,
) -> list[EmbeddingModel]:
    """
    Call embedding-service /embed/batch to get vectors for all chunks.

    Falls back to sequential /embed calls if batch endpoint unavailable.
    """
    texts = [c.text for c in chunks]

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                f"{embedding_url}/embed/batch",
                json={"texts": texts, "provider": llm_provider},
                headers=trace_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            vectors: list[list[float]] = data["vectors"]
        except Exception as exc:
            raise PipelineError(f"Embedding-service call failed: {exc}") from exc

    if len(vectors) != len(chunks):
        raise PipelineError(
            f"Embedding count mismatch: {len(vectors)} vectors for {len(chunks)} chunks."
        )

    return [
        EmbeddingModel(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            vector=vector,
            model=llm_provider,
            dimensions=len(vector),
        )
        for chunk, vector in zip(chunks, vectors)
    ]


async def _index_chunks(
    message: IngestionMessage,
    chunks: list[ChunkModel],
    embeddings: list[EmbeddingModel],
    indexing_url: str,
) -> None:
    """Call indexing-service /index to upsert chunks into Qdrant + Postgres."""
    payload = {
        "collection": message.collection,
        "doc_id": message.doc_id,
        "filename": message.filename,
        "chunks": [c.model_dump(mode="json") for c in chunks],
        "embeddings": [e.model_dump(mode="json") for e in embeddings],
        "doc_metadata": {
            **message.doc_metadata,
            "chunker": message.chunker_type,
            "content_type": message.content_type,
            "storage_path": message.storage_path,
        },
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(f"{indexing_url}/index", json=payload, headers=trace_headers())
            resp.raise_for_status()
        except Exception as exc:
            raise PipelineError(f"Indexing-service call failed: {exc}") from exc
