"""
Async RabbitMQ consumer for the pipeline-service.

Consumes from QUEUE_INGESTION, runs the full RAG ingestion pipeline
(read → chunk → embed → index), and handles:

  - Idempotency: checks Postgres before processing; skips duplicates.
  - Retry: nacks with requeue=False after each failure; RabbitMQ dead-letters
    the message. The consumer re-publishes with incremented retry_count up to
    MAX_RETRIES, then routes to DLQ.
  - DLQ: after MAX_RETRIES, wraps in DLQMessage and publishes to QUEUE_DLQ.
    Updates Postgres DocumentRecord to status=dead_letter.

Message ack/nack semantics:
  - Success      → ack
  - Duplicate    → ack (already done)
  - Retriable    → nack(requeue=False) + re-publish with retry_count+1
  - Max retries  → ack + publish DLQMessage
"""

from __future__ import annotations

import asyncio
from typing import Any

import aio_pika
from aio_pika import DeliveryMode, ExchangeType, Message

from raglab_common.exceptions import RAGLabError
from raglab_common.logging import get_logger
from raglab_common.models import IngestionStatus
from raglab_common.queue import (
    EXCHANGE_NAME,
    MAX_RETRIES,
    QUEUE_DLQ,
    QUEUE_INGESTION,
    ROUTING_KEY_DOCUMENT,
    ROUTING_KEY_DLQ,
    DLQMessage,
    IngestionMessage,
)

log = get_logger(__name__)


class RabbitMQConsumer:
    """
    Async RabbitMQ consumer that drives the full ingestion pipeline.

    The consumer is started in a background task at service startup.
    It runs until cancelled (service shutdown).

    Args:
        url:              RabbitMQ connection URL.
        pipeline_runner:  Async callable(IngestionMessage, app_state) -> None.
                          Injected at startup so the consumer stays testable.
        prefetch_count:   Max unacked messages in flight per consumer.
    """

    def __init__(
        self,
        url: str,
        pipeline_runner: Any,
        prefetch_count: int = 1,
    ) -> None:
        self._url = url
        self._pipeline_runner = pipeline_runner
        self._prefetch_count = prefetch_count
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None
        self._exchange: aio_pika.abc.AbstractExchange | None = None
        self._app_state: Any = None

    async def start(self, app_state: Any) -> None:
        """
        Connect to RabbitMQ and start consuming.

        This is a long-running coroutine — run it as a background task.
        Raises on initial connection failure; reconnects automatically
        on subsequent failures via aio_pika.connect_robust.
        """
        self._app_state = app_state
        self._connection = await aio_pika.connect_robust(self._url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=self._prefetch_count)

        self._exchange = await self._channel.declare_exchange(
            EXCHANGE_NAME, ExchangeType.TOPIC, durable=True
        )

        main_queue = await self._channel.declare_queue(
            QUEUE_INGESTION,
            durable=True,
            arguments={
                "x-dead-letter-exchange": EXCHANGE_NAME,
                "x-dead-letter-routing-key": ROUTING_KEY_DLQ,
            },
        )
        await main_queue.bind(self._exchange, ROUTING_KEY_DOCUMENT)

        dlq = await self._channel.declare_queue(QUEUE_DLQ, durable=True)
        await dlq.bind(self._exchange, ROUTING_KEY_DLQ)

        log.info("consumer.started", queue=QUEUE_INGESTION)
        async with main_queue.iterator() as queue_iter:
            async for amqp_message in queue_iter:
                await self._handle(amqp_message)

    async def _handle(self, amqp_message: aio_pika.IncomingMessage) -> None:
        """
        Process a single AMQP message through the full pipeline.

        Ack/nack/DLQ routing logic lives here.
        """
        async with amqp_message.process(ignore_processed=True):
            try:
                message = IngestionMessage.from_bytes(amqp_message.body)
            except Exception as exc:
                log.error("consumer.parse_error", error=str(exc))
                await amqp_message.ack()   # malformed — discard
                return

            log.info(
                "consumer.received",
                doc_id=message.doc_id,
                idempotency_key=message.idempotency_key,
                retry_count=message.retry_count,
            )

            # Idempotency check
            if await self._is_duplicate(message):
                log.info("consumer.duplicate_skipped", idempotency_key=message.idempotency_key)
                await amqp_message.ack()
                return

            # Mark PROCESSING in Postgres
            await self._update_status(message.doc_id, IngestionStatus.PROCESSING.value)

            # Run pipeline
            try:
                await self._pipeline_runner(message, self._app_state)
                await self._update_status(message.doc_id, IngestionStatus.COMPLETED.value)
                log.info("consumer.completed", doc_id=message.doc_id)
                await amqp_message.ack()

            except Exception as exc:
                log.error(
                    "consumer.pipeline_error",
                    doc_id=message.doc_id,
                    retry_count=message.retry_count,
                    error=str(exc),
                )
                if message.retry_count >= MAX_RETRIES:
                    await self._send_to_dlq(message, str(exc))
                    await amqp_message.ack()
                else:
                    # Re-publish with incremented retry_count
                    await self._republish(message, str(exc))
                    await amqp_message.ack()   # ack original; fresh copy re-queued

    async def _is_duplicate(self, message: IngestionMessage) -> bool:
        """Return True if this idempotency_key is already COMPLETED in Postgres."""
        session_factory = getattr(self._app_state, "session_factory", None)
        if session_factory is None:
            return False
        try:
            from sqlalchemy import select
            from indexing.db.models import DocumentRecord
            async with session_factory() as session:
                result = await session.execute(
                    select(DocumentRecord).where(
                        DocumentRecord.idempotency_key == message.idempotency_key,
                        DocumentRecord.status == IngestionStatus.COMPLETED.value,
                    )
                )
                return result.scalar_one_or_none() is not None
        except Exception as exc:
            log.warning("consumer.idempotency_check_failed", error=str(exc))
            return False

    async def _update_status(self, doc_id: str, status: str, error: str | None = None) -> None:
        """Update DocumentRecord status in Postgres (best-effort)."""
        session_factory = getattr(self._app_state, "session_factory", None)
        if session_factory is None:
            return
        try:
            from sqlalchemy import select
            from indexing.db.models import DocumentRecord
            async with session_factory() as session:
                result = await session.execute(
                    select(DocumentRecord).where(DocumentRecord.doc_id == doc_id)
                )
                record = result.scalar_one_or_none()
                if record:
                    record.status = status
                    if error:
                        record.error_message = error
                    await session.commit()
        except Exception as exc:
            log.warning("consumer.status_update_failed", error=str(exc), doc_id=doc_id)

    async def _republish(self, message: IngestionMessage, error: str) -> None:
        """Re-publish with incremented retry_count."""
        if self._exchange is None:
            return
        retried = message.model_copy(update={"retry_count": message.retry_count + 1})
        amqp_msg = Message(
            body=retried.to_bytes(),
            delivery_mode=DeliveryMode.PERSISTENT,
            message_id=retried.message_id,
            content_type="application/json",
            headers={"retry_count": str(retried.retry_count), "last_error": error[:256]},
        )
        try:
            await self._exchange.publish(amqp_msg, routing_key=ROUTING_KEY_DOCUMENT)
            log.info(
                "consumer.republished",
                doc_id=message.doc_id,
                retry_count=retried.retry_count,
            )
        except Exception as exc:
            log.error("consumer.republish_failed", error=str(exc))

    async def _send_to_dlq(self, message: IngestionMessage, reason: str) -> None:
        """Wrap in DLQMessage and publish to dead letter queue."""
        if self._exchange is None:
            return
        dlq_msg = DLQMessage(
            original=message,
            failure_reason=reason,
            retry_count=message.retry_count,
        )
        amqp_msg = Message(
            body=dlq_msg.to_bytes(),
            delivery_mode=DeliveryMode.PERSISTENT,
            content_type="application/json",
            headers={"doc_id": message.doc_id, "failure_reason": reason[:256]},
        )
        try:
            await self._exchange.publish(amqp_msg, routing_key=ROUTING_KEY_DLQ)
            await self._update_status(
                message.doc_id,
                IngestionStatus.DEAD_LETTER.value,
                error=f"DLQ after {message.retry_count} retries: {reason}",
            )
            log.warning(
                "consumer.dlq_routed",
                doc_id=message.doc_id,
                reason=reason[:128],
            )
        except Exception as exc:
            log.error("consumer.dlq_publish_failed", error=str(exc))

    async def close(self) -> None:
        if self._channel and not self._channel.is_closed:
            await self._channel.close()
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
        log.info("consumer.closed")
