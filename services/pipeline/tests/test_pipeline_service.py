"""
Tests for the pipeline-service.

Covers:
- run_pipeline: read → chunk → embed → index (all steps mocked)
- _read_document: success, missing file
- Consumer: idempotency check, ack/nack/DLQ routing (mocked AMQP)
- HTTP endpoints: /pipeline/run, /pipeline/status, /health
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from raglab_common.models import IngestionStatus
from raglab_common.queue import IngestionMessage, MAX_RETRIES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_message(**kwargs) -> IngestionMessage:
    defaults = dict(
        idempotency_key=str(uuid.uuid4()),
        filename="test.txt",
        storage_path="/tmp/test.txt",
        collection="raglab",
        chunker_type="text",
        llm_provider="azure_openai",
    )
    defaults.update(kwargs)
    return IngestionMessage(**defaults)


# ---------------------------------------------------------------------------
# _read_document
# ---------------------------------------------------------------------------


class TestReadDocument:
    @pytest.mark.asyncio
    async def test_reads_existing_file(self, tmp_path):
        from pipeline.runner import _read_document
        f = tmp_path / "doc.txt"
        f.write_text("Hello RAG world.")
        text = await _read_document(str(f))
        assert text == "Hello RAG world."

    @pytest.mark.asyncio
    async def test_missing_file_raises_pipeline_error(self):
        from pipeline.runner import _read_document, PipelineError
        with pytest.raises(PipelineError, match="not found"):
            await _read_document("/nonexistent/path/doc.txt")


# ---------------------------------------------------------------------------
# run_pipeline — integration (all HTTP calls mocked)
# ---------------------------------------------------------------------------


class TestRunPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_success(self, tmp_path):
        from pipeline.runner import run_pipeline

        doc = tmp_path / "doc.txt"
        doc.write_text(" ".join([f"word{i}" for i in range(200)]))

        msg = make_message(
            storage_path=str(doc),
            chunker_config={"tokenizer": "word_count", "chunk_size": 50, "chunk_overlap": 5},
        )
        app_state = MagicMock()
        app_state.settings = MagicMock()
        app_state.settings.embedding_url = "http://mock-embed"
        app_state.settings.indexing_url = "http://mock-index"

        mock_embed_resp = MagicMock()
        mock_embed_resp.status_code = 200
        mock_embed_resp.json.return_value = {"vectors": [[0.1] * 1536] * 20}  # enough for any chunk count
        mock_embed_resp.raise_for_status = MagicMock()

        mock_index_resp = MagicMock()
        mock_index_resp.status_code = 200
        mock_index_resp.raise_for_status = MagicMock()

        with patch("pipeline.runner.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            async def dynamic_post(url, **kwargs):
                if "embed" in url:
                    # Match vector count to texts sent
                    texts = kwargs.get("json", {}).get("texts", [])
                    resp = MagicMock()
                    resp.status_code = 200
                    resp.json.return_value = {"vectors": [[0.1] * 1536] * len(texts)}
                    resp.raise_for_status = MagicMock()
                    return resp
                return mock_index_resp

            mock_client.post = dynamic_post
            await run_pipeline(msg, app_state)

    @pytest.mark.asyncio
    async def test_empty_chunks_raises_pipeline_error(self, tmp_path):
        from pipeline.runner import run_pipeline, PipelineError

        # Empty file → chunker returns []
        doc = tmp_path / "empty.txt"
        doc.write_text("")

        msg = make_message(storage_path=str(doc))
        app_state = MagicMock()
        with pytest.raises(PipelineError, match="0 chunks"):
            await run_pipeline(msg, app_state)

    @pytest.mark.asyncio
    async def test_embedding_failure_raises_pipeline_error(self, tmp_path):
        from pipeline.runner import run_pipeline, PipelineError

        doc = tmp_path / "doc.txt"
        doc.write_text(" ".join([f"word{i}" for i in range(100)]))

        msg = make_message(
            storage_path=str(doc),
            chunker_config={"tokenizer": "word_count", "chunk_size": 50, "chunk_overlap": 5},
        )
        app_state = MagicMock()
        app_state.settings = MagicMock()
        app_state.settings.embedding_url = "http://mock-embed"
        app_state.settings.indexing_url = "http://mock-index"

        with patch("pipeline.runner.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=Exception("embedding service down"))
            with pytest.raises(PipelineError, match="Embedding-service"):
                await run_pipeline(msg, app_state)


# ---------------------------------------------------------------------------
# Consumer — ack/nack/DLQ routing
# ---------------------------------------------------------------------------


class TestConsumerRouting:
    def _make_consumer(self, pipeline_runner=None):
        from pipeline.queue.consumer import RabbitMQConsumer
        if pipeline_runner is None:
            pipeline_runner = AsyncMock()
        return RabbitMQConsumer("amqp://x", pipeline_runner, prefetch_count=1)

    def _make_amqp_message(self, msg: IngestionMessage):
        amqp_msg = AsyncMock()
        amqp_msg.body = msg.to_bytes()
        amqp_msg.ack = AsyncMock()
        amqp_msg.nack = AsyncMock()
        # Simulate process() context manager
        amqp_msg.process = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=False),
        ))
        return amqp_msg

    @pytest.mark.asyncio
    async def test_successful_pipeline_acks_message(self):
        runner = AsyncMock()
        consumer = self._make_consumer(pipeline_runner=runner)
        consumer._app_state = MagicMock()
        consumer._app_state.session_factory = None
        consumer._exchange = AsyncMock()

        msg = make_message()
        amqp_msg = self._make_amqp_message(msg)

        await consumer._handle(amqp_msg)
        amqp_msg.ack.assert_called_once()
        runner.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_failure_republishes_with_incremented_retry(self):
        runner = AsyncMock(side_effect=Exception("embedding down"))
        consumer = self._make_consumer(pipeline_runner=runner)
        consumer._app_state = MagicMock()
        consumer._app_state.session_factory = None
        exchange = AsyncMock()
        consumer._exchange = exchange

        msg = make_message(retry_count=0)
        amqp_msg = self._make_amqp_message(msg)

        await consumer._handle(amqp_msg)

        # Should ack original and republish
        amqp_msg.ack.assert_called_once()
        exchange.publish.assert_called_once()
        # Republished message has retry_count=1
        published_body = exchange.publish.call_args[0][0].body
        republished = IngestionMessage.from_bytes(published_body)
        assert republished.retry_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_routes_to_dlq(self):
        from raglab_common.queue import DLQMessage, ROUTING_KEY_DLQ
        runner = AsyncMock(side_effect=Exception("persistent failure"))
        consumer = self._make_consumer(pipeline_runner=runner)
        consumer._app_state = MagicMock()
        consumer._app_state.session_factory = None
        exchange = AsyncMock()
        consumer._exchange = exchange

        msg = make_message(retry_count=MAX_RETRIES)
        amqp_msg = self._make_amqp_message(msg)

        await consumer._handle(amqp_msg)

        amqp_msg.ack.assert_called_once()
        exchange.publish.assert_called_once()
        call_kwargs = exchange.publish.call_args[1]
        assert call_kwargs["routing_key"] == ROUTING_KEY_DLQ

    @pytest.mark.asyncio
    async def test_malformed_message_acked_and_discarded(self):
        consumer = self._make_consumer()
        consumer._app_state = MagicMock()
        consumer._exchange = AsyncMock()

        amqp_msg = AsyncMock()
        amqp_msg.body = b"not valid json {"
        amqp_msg.ack = AsyncMock()
        amqp_msg.process = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=None),
            __aexit__=AsyncMock(return_value=False),
        ))

        await consumer._handle(amqp_msg)
        amqp_msg.ack.assert_called_once()


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_client():
    from pipeline.main import app
    app.state.consumer = None
    app.state.consumer_running = False
    app.state.session_factory = None
    app.state.settings = MagicMock()
    app.state.settings.embedding_url = "http://mock-embed"
    app.state.settings.indexing_url = "http://mock-index"
    return TestClient(app)


class TestPipelineEndpoints:
    def test_health_200(self, pipeline_client):
        r = pipeline_client.get("/health")
        assert r.status_code == 200
        assert r.json()["service"] == "pipeline"

    def test_root_200(self, pipeline_client):
        r = pipeline_client.get("/")
        assert r.status_code == 200

    def test_pipeline_status(self, pipeline_client):
        r = pipeline_client.get("/pipeline/status")
        assert r.status_code == 200
        body = r.json()
        assert "consumer_running" in body
        assert "rabbitmq_connected" in body

    def test_pipeline_run_missing_file_returns_502(self, pipeline_client):
        r = pipeline_client.post("/pipeline/run", json={
            "filename": "ghost.txt",
            "storage_path": "/nonexistent/ghost.txt",
        })
        assert r.status_code == 502

    def test_pipeline_run_success(self, tmp_path, pipeline_client):
        doc = tmp_path / "doc.txt"
        doc.write_text(" ".join([f"word{i}" for i in range(100)]))

        with patch("pipeline.runner.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            async def dynamic_post(url, **kwargs):
                if "embed" in url:
                    texts = kwargs.get("json", {}).get("texts", [])
                    resp = MagicMock()
                    resp.status_code = 200
                    resp.json.return_value = {"vectors": [[0.1] * 1536] * len(texts)}
                    resp.raise_for_status = MagicMock()
                    return resp
                resp = MagicMock()
                resp.status_code = 200
                resp.raise_for_status = MagicMock()
                return resp

            mock_client.post = dynamic_post
            r = pipeline_client.post("/pipeline/run", json={
                "filename": "doc.txt",
                "storage_path": str(doc),
                "collection": "raglab",
                "chunker_config": {"tokenizer": "word_count", "chunk_size": 50, "chunk_overlap": 5},
            })

        assert r.status_code == 200
        assert r.json()["status"] == "completed"
