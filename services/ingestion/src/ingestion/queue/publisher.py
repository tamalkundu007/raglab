"""
Async RabbitMQ publisher for the ingestion-service.

Publishes IngestionMessage to the raglab.ingestion topic exchange.
Uses publisher confirms (confirm_delivery=True) so every publish is
acknowledged by the broker before the HTTP response is returned.

The publisher maintains a single persistent connection and channel,
reconnecting automatically on failure via aio-pika's robust connection.

Usage (called from lifespan + request handlers):
    publisher = RabbitMQPublisher(url)
    await publisher.connect()
    await publisher.publish(message)
    await publisher.close()
"""

from __future__ import annotations

import asyncio

import aio_pika
from aio_pika import DeliveryMode, ExchangeType, Message

from raglab_common.exceptions import RAGLabError
from raglab_common.logging import get_logger
from raglab_common.queue import (
    EXCHANGE_NAME,
    QUEUE_DLQ,
    QUEUE_INGESTION,
    ROUTING_KEY_DLQ,
    ROUTING_KEY_DOCUMENT,
    IngestionMessage,
)

log = get_logger(__name__)


class PublishError(RAGLabError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="PUBLISH_ERROR")


class RabbitMQPublisher:
    """
    Async RabbitMQ publisher with publisher confirms.

    Declares the exchange and queues on connect so the topology is
    always consistent regardless of startup order.
    """

    def __init__(self, url: str, confirm_delivery: bool = True) -> None:
        self._url = url
        self._confirm_delivery = confirm_delivery
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None
        self._channel: aio_pika.abc.AbstractChannel | None = None
        self._exchange: aio_pika.abc.AbstractExchange | None = None

    async def connect(self) -> None:
        """
        Establish connection, declare exchange and queues.

        Uses aio_pika.connect_robust for automatic reconnection.
        """
        try:
            self._connection = await aio_pika.connect_robust(self._url)
            self._channel = await self._connection.channel()

            if self._confirm_delivery:
                await self._channel.set_qos(prefetch_count=1)

            # Declare topic exchange
            self._exchange = await self._channel.declare_exchange(
                EXCHANGE_NAME,
                ExchangeType.TOPIC,
                durable=True,
            )

            # Declare main queue with DLQ as dead-letter exchange target
            main_queue = await self._channel.declare_queue(
                QUEUE_INGESTION,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": EXCHANGE_NAME,
                    "x-dead-letter-routing-key": ROUTING_KEY_DLQ,
                },
            )
            await main_queue.bind(self._exchange, ROUTING_KEY_DOCUMENT)

            # Declare DLQ (no further dead-lettering)
            dlq = await self._channel.declare_queue(QUEUE_DLQ, durable=True)
            await dlq.bind(self._exchange, ROUTING_KEY_DLQ)

            log.info(
                "publisher.connected",
                url=self._url.split("@")[-1],
                exchange=EXCHANGE_NAME,
            )
        except Exception as exc:
            raise PublishError(f"Failed to connect to RabbitMQ: {exc}") from exc

    async def publish(self, message: IngestionMessage) -> None:
        """
        Publish an IngestionMessage to the ingestion exchange.

        Args:
            message: IngestionMessage to publish.

        Raises:
            PublishError: If not connected or broker rejects the message.
        """
        if self._exchange is None:
            raise PublishError("Publisher not connected. Call connect() first.")

        amqp_message = Message(
            body=message.to_bytes(),
            delivery_mode=DeliveryMode.PERSISTENT,
            message_id=message.message_id,
            content_type="application/json",
            headers={
                "idempotency_key": message.idempotency_key,
                "doc_id": message.doc_id,
                "retry_count": str(message.retry_count),
            },
        )

        try:
            await self._exchange.publish(
                amqp_message,
                routing_key=ROUTING_KEY_DOCUMENT,
            )
            log.info(
                "publisher.published",
                message_id=message.message_id,
                doc_id=message.doc_id,
                idempotency_key=message.idempotency_key,
            )
        except Exception as exc:
            raise PublishError(f"Failed to publish message: {exc}") from exc

    async def close(self) -> None:
        """Close channel and connection gracefully."""
        if self._channel and not self._channel.is_closed:
            await self._channel.close()
        if self._connection and not self._connection.is_closed:
            await self._connection.close()
        log.info("publisher.closed")

    @property
    def is_connected(self) -> bool:
        return (
            self._connection is not None
            and not self._connection.is_closed
        )
