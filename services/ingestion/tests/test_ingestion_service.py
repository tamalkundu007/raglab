"""
Tests for the ingestion-service.

Covers:
- IngestionMessage serialisation round-trip
- _generate_idem_key determinism
- Publisher connect/publish/close (mocked aio-pika)
- /ingest endpoint: success, duplicate, queue-unavailable
- /ingest/{doc_id} status endpoint
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from raglab_common.models import IngestionStatus
from raglab_common.queue import IngestionMessage, DLQMessage, MAX_RETRIES, ROUTING_KEY_DOCUMENT


# ---------------------------------------------------------------------------
# IngestionMessage serialisation
# ---------------------------------------------------------------------------


class TestIngestionMessage:
    def _msg(self) -> IngestionMessage:
        return IngestionMessage(
            idempotency_key="key-001",
            filename="test.txt",
            storage_path="/tmp/test.txt",
            collection="raglab",
        )

    def test_round_trip(self):
        msg = self._msg()
        restored = IngestionMessage.from_bytes(msg.to_bytes())
        assert restored.idempotency_key == msg.idempotency_key
        assert restored.doc_id == msg.doc_id
        assert restored.filename == msg.filename

    def test_to_bytes_is_json(self):
        import json
        msg = self._msg()
        data = json.loads(msg.to_bytes())
        assert data["idempotency_key"] == "key-001"

    def test_unique_doc_ids(self):
        m1 = IngestionMessage(idempotency_key="k1", filename="f.txt", storage_path="/f")
        m2 = IngestionMessage(idempotency_key="k2", filename="f.txt", storage_path="/f")
        assert m1.doc_id != m2.doc_id

    def test_retry_count_default_zero(self):
        msg = self._msg()
        assert msg.retry_count == 0

    def test_model_copy_increments_retry(self):
        msg = self._msg()
        retried = msg.model_copy(update={"retry_count": msg.retry_count + 1})
        assert retried.retry_count == 1
        assert msg.retry_count == 0  # original unchanged


class TestDLQMessage:
    def test_round_trip(self):
        original = IngestionMessage(
            idempotency_key="k", filename="f.txt", storage_path="/f"
        )
        dlq = DLQMessage(original=original, failure_reason="test error", retry_count=3)
        restored = DLQMessage.from_bytes(dlq.to_bytes())
        assert restored.failure_reason == "test error"
        assert restored.original.idempotency_key == "k"


# ---------------------------------------------------------------------------
# Idempotency key generation
# ---------------------------------------------------------------------------


class TestIdempotencyKey:
    def test_deterministic(self):
        from ingestion.routers.ingest import _generate_idem_key
        k1 = _generate_idem_key("file.txt", "raglab")
        k2 = _generate_idem_key("file.txt", "raglab")
        assert k1 == k2

    def test_different_files(self):
        from ingestion.routers.ingest import _generate_idem_key
        k1 = _generate_idem_key("file1.txt", "raglab")
        k2 = _generate_idem_key("file2.txt", "raglab")
        assert k1 != k2

    def test_different_collections(self):
        from ingestion.routers.ingest import _generate_idem_key
        k1 = _generate_idem_key("file.txt", "col1")
        k2 = _generate_idem_key("file.txt", "col2")
        assert k1 != k2

    def test_length_32(self):
        from ingestion.routers.ingest import _generate_idem_key
        k = _generate_idem_key("file.txt", "raglab")
        assert len(k) == 32


# ---------------------------------------------------------------------------
# RabbitMQPublisher (mocked aio-pika)
# ---------------------------------------------------------------------------


class TestRabbitMQPublisher:
    @pytest.fixture
    def mock_connection(self):
        conn = AsyncMock()
        conn.is_closed = False
        channel = AsyncMock()
        channel.is_closed = False
        exchange = AsyncMock()
        channel.declare_exchange = AsyncMock(return_value=exchange)
        queue = AsyncMock()
        queue.bind = AsyncMock()
        channel.declare_queue = AsyncMock(return_value=queue)
        conn.channel = AsyncMock(return_value=channel)
        return conn, channel, exchange

    @pytest.mark.asyncio
    async def test_connect_declares_exchange_and_queues(self, mock_connection):
        conn, channel, exchange = mock_connection
        with patch("ingestion.queue.publisher.aio_pika.connect_robust", return_value=conn):
            from ingestion.queue.publisher import RabbitMQPublisher
            pub = RabbitMQPublisher("amqp://guest:guest@localhost/")
            await pub.connect()
            channel.declare_exchange.assert_called_once()
            assert channel.declare_queue.call_count == 2  # main + DLQ

    @pytest.mark.asyncio
    async def test_publish_sends_message(self, mock_connection):
        conn, channel, exchange = mock_connection
        with patch("ingestion.queue.publisher.aio_pika.connect_robust", return_value=conn):
            from ingestion.queue.publisher import RabbitMQPublisher
            pub = RabbitMQPublisher("amqp://guest:guest@localhost/")
            await pub.connect()
            msg = IngestionMessage(idempotency_key="k", filename="f.txt", storage_path="/f")
            await pub.publish(msg)
            exchange.publish.assert_called_once()
            call_kwargs = exchange.publish.call_args
            assert call_kwargs[1]["routing_key"] == ROUTING_KEY_DOCUMENT

    @pytest.mark.asyncio
    async def test_publish_without_connect_raises(self):
        from ingestion.queue.publisher import RabbitMQPublisher, PublishError
        pub = RabbitMQPublisher("amqp://guest:guest@localhost/")
        msg = IngestionMessage(idempotency_key="k", filename="f.txt", storage_path="/f")
        with pytest.raises(PublishError, match="not connected"):
            await pub.publish(msg)

    @pytest.mark.asyncio
    async def test_is_connected_false_before_connect(self):
        from ingestion.queue.publisher import RabbitMQPublisher
        pub = RabbitMQPublisher("amqp://x")
        assert pub.is_connected is False

    @pytest.mark.asyncio
    async def test_close_graceful(self, mock_connection):
        conn, channel, exchange = mock_connection
        with patch("ingestion.queue.publisher.aio_pika.connect_robust", return_value=conn):
            from ingestion.queue.publisher import RabbitMQPublisher
            pub = RabbitMQPublisher("amqp://x")
            await pub.connect()
            await pub.close()
            channel.close.assert_called_once()
            conn.close.assert_called_once()


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def ingestion_client_with_publisher():
    """TestClient with mock publisher and no Postgres."""
    from ingestion.main import app

    mock_publisher = MagicMock()
    mock_publisher.is_connected = True
    mock_publisher.publish = AsyncMock()

    app.state.publisher = mock_publisher
    app.state.session_factory = None
    return TestClient(app), mock_publisher


class TestIngestionEndpoints:
    def test_health_200(self, ingestion_client_with_publisher):
        client, _ = ingestion_client_with_publisher
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["service"] == "ingestion"

    def test_root_200(self, ingestion_client_with_publisher):
        client, _ = ingestion_client_with_publisher
        r = client.get("/")
        assert r.status_code == 200

    def test_ingest_success(self, ingestion_client_with_publisher):
        client, publisher = ingestion_client_with_publisher
        r = client.post("/ingest", json={
            "filename": "doc.txt",
            "storage_path": "/tmp/doc.txt",
            "collection": "raglab",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == IngestionStatus.PENDING.value
        assert body["duplicate"] is False
        assert "doc_id" in body
        publisher.publish.assert_called_once()

    def test_ingest_uses_provided_idempotency_key(self, ingestion_client_with_publisher):
        client, publisher = ingestion_client_with_publisher
        r = client.post("/ingest", json={
            "filename": "doc.txt",
            "storage_path": "/tmp/doc.txt",
            "idempotency_key": "custom-key-xyz",
        })
        assert r.status_code == 200
        assert r.json()["idempotency_key"] == "custom-key-xyz"

    def test_ingest_queue_unavailable_returns_503(self):
        from ingestion.main import app
        app.state.publisher = None
        app.state.session_factory = None
        client = TestClient(app)
        r = client.post("/ingest", json={
            "filename": "doc.txt",
            "storage_path": "/tmp/doc.txt",
        })
        assert r.status_code == 503

    def test_status_endpoint_no_db_returns_503(self, ingestion_client_with_publisher):
        client, _ = ingestion_client_with_publisher
        r = client.get("/ingest/nonexistent-doc-id")
        assert r.status_code == 503  # no DB wired
