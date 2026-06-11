"""
Shared RabbitMQ message schemas and queue constants for RAGLab.

Both ingestion-service (publisher) and pipeline-service (consumer) import
from here — the message contract lives in one place.

Exchange topology:
  raglab.ingestion       — main topic exchange
    routing key: ingestion.document   → ingestion_queue (main consumer)
    routing key: ingestion.dlq        → ingestion_dlq   (dead letter)

Idempotency:
  Every message carries an idempotency_key. The consumer checks Postgres
  before processing; duplicate keys are acked and skipped.

Message lifecycle:
  PENDING   → ingestion-service publishes with status=pending
  PROCESSING → pipeline-service sets on pickup
  COMPLETED → pipeline-service sets on success
  FAILED    → pipeline-service sets after max_retries; routes to DLQ
  DEAD_LETTER → DLQ consumer sets (manual intervention required)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

# ── Exchange / Queue names ────────────────────────────────────────────────────

EXCHANGE_NAME = "raglab.ingestion"
EXCHANGE_TYPE = "topic"

QUEUE_INGESTION = "ingestion_queue"
QUEUE_DLQ = "ingestion_dlq"

ROUTING_KEY_DOCUMENT = "ingestion.document"
ROUTING_KEY_DLQ = "ingestion.dlq"

# ── Per-message retry limit ───────────────────────────────────────────────────

MAX_RETRIES = 3


# ── Message schemas ───────────────────────────────────────────────────────────


class IngestionMessage(BaseModel):
    """
    Published by ingestion-service; consumed by pipeline-service.

    All fields must be JSON-serialisable — this payload crosses the wire.
    """

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    idempotency_key: str                        # caller-supplied or auto-generated
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    content_type: str = "text/plain"
    storage_path: str                           # local path or cloud URI
    collection: str = "raglab"                  # target Qdrant collection
    chunker_type: str = "text"                  # ChunkerType enum value
    chunker_config: dict[str, Any] = Field(default_factory=dict)
    llm_provider: str = "azure_openai"          # for embedding model selection
    doc_metadata: dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0
    published_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # R7: tenant isolation — mandatory for all new ingestion jobs
    tenant_id: str = Field(default="default", description="Tenant scope (R7)")
    user_id: str = Field(default="", description="User who triggered ingestion (R7)")

    def to_bytes(self) -> bytes:
        """Serialise to UTF-8 JSON bytes for AMQP body."""
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "IngestionMessage":
        """Deserialise from AMQP message body bytes."""
        return cls.model_validate_json(data.decode("utf-8"))


class DLQMessage(BaseModel):
    """
    Wraps a failed IngestionMessage with failure metadata.
    Published to QUEUE_DLQ after MAX_RETRIES exhausted.
    """

    original: IngestionMessage
    failure_reason: str
    failed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    retry_count: int

    def to_bytes(self) -> bytes:
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "DLQMessage":
        return cls.model_validate_json(data.decode("utf-8"))
