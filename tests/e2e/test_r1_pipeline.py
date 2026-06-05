"""
R1 End-to-End Wiring Smoke Test.

Verifies the complete R1 pipeline:
  TextChunker → EmbeddingModel → QdrantIndexer → DenseRetriever → LLMProvider

All external I/O (Qdrant, embedding APIs, LLM APIs) is mocked so this
test runs in CI with zero infrastructure dependencies.

Test contracts verified:
1. TextChunker produces valid ChunkModel list from plain text.
2. EmbeddingModel objects can be built from chunk data.
3. QdrantIndexer.upsert_chunks receives correctly shaped PointStruct payloads.
4. DenseRetriever receives the query vector and calls vector store with
   correct collection name and top_k.
5. Hits are correctly converted to ChunkModel with score in metadata.
6. BaseLLMProvider assembles a numbered context block and returns ResponseModel.
7. The full pipeline produces a ResponseModel with non-empty answer.

Secondary contracts (service boundaries):
8. IngestionMessage round-trips through serialisation unchanged.
9. RabbitMQPublisher.publish sends AMQP message with correct routing key.
10. api-gateway proxy_request strips hop-by-hop headers.
11. HealthRegistry aggregate_status degrades when a core service is down.
12. UISettings injects gateway_url into Jinja2 template.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── shared models ──────────────────────────────────────────────────────────────
from raglab_common.models import (
    ChunkModel,
    EmbeddingModel,
    HealthModel,
    IngestionStatus,
    LLMProvider,
    QueryModel,
    ResponseModel,
    RetrieverType,
)
from raglab_common.queue import IngestionMessage, ROUTING_KEY_DOCUMENT

# ── chunkers ───────────────────────────────────────────────────────────────────
from raglab_chunkers import ChunkerFactory, TextChunker
from raglab_chunkers._boundary import split_into_windows

# ── retrievers ─────────────────────────────────────────────────────────────────
from raglab_retrievers import DenseRetriever, RetrieverFactory

# ── llm ───────────────────────────────────────────────────────────────────────
from llm.providers.base import BaseLLMProvider

# ── indexing ──────────────────────────────────────────────────────────────────
from indexing.qdrant_client import QdrantIndexer

# ── health registry ────────────────────────────────────────────────────────────
from api_gateway.health_registry import HealthRegistry


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) is a framework that enhances large language
models by providing them with relevant external knowledge at inference time.
Rather than relying solely on parametric knowledge learned during training,
a RAG system retrieves documents from an external corpus and uses them as context.

The retrieval step typically uses dense vector search. A query is embedded into a
high-dimensional vector, and the nearest neighbours in the vector space are
retrieved as context chunks. These chunks are then passed to the language model
alongside the original query.

This approach significantly reduces hallucinations because the model can ground
its answers in retrieved facts. It also allows the knowledge base to be updated
without retraining the underlying model — a critical advantage in production.
""".strip()

DOC_ID = "e2e-doc-001"
COLLECTION = "e2e-test"
VECTOR_DIM = 128  # Small dimension for test speed


def make_embedding(chunk: ChunkModel) -> EmbeddingModel:
    """Create a deterministic embedding from chunk index for reproducibility."""
    vector = [float(chunk.chunk_index % 10) / 10.0 + 0.01] * VECTOR_DIM
    return EmbeddingModel(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        vector=vector,
        model="test-embed",
        dimensions=VECTOR_DIM,
    )


def make_qdrant_hit(chunk: ChunkModel, score: float = 0.92) -> dict:
    """Build a dict-style Qdrant hit from a ChunkModel."""
    return {
        "payload": {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            **chunk.metadata,
        },
        "score": score,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Contract 1 — TextChunker produces valid ChunkModel list
# ═══════════════════════════════════════════════════════════════════════════════


class TestContract1_Chunker:
    def test_chunker_produces_chunks(self):
        chunker = TextChunker(config={
            "tokenizer": "word_count", "chunk_size": 50, "chunk_overlap": 10
        })
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id=DOC_ID)

        assert len(chunks) >= 1
        assert all(isinstance(c, ChunkModel) for c in chunks)

    def test_chunks_have_sequential_indices(self):
        chunker = TextChunker(config={"tokenizer": "word_count", "chunk_size": 50, "chunk_overlap": 5})
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id=DOC_ID)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_chunks_have_unique_ids(self):
        chunker = TextChunker(config={"tokenizer": "word_count", "chunk_size": 50, "chunk_overlap": 5})
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id=DOC_ID)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunks_propagate_doc_id(self):
        chunker = TextChunker(config={"tokenizer": "word_count", "chunk_size": 50, "chunk_overlap": 5})
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id=DOC_ID)
        assert all(c.doc_id == DOC_ID for c in chunks)

    def test_chunks_have_positive_token_count(self):
        chunker = TextChunker(config={"tokenizer": "word_count", "chunk_size": 50, "chunk_overlap": 5})
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id=DOC_ID)
        assert all(c.token_count > 0 for c in chunks)

    def test_chunker_factory_creates_text_chunker(self):
        chunker = ChunkerFactory.create("text", config={"tokenizer": "word_count"})
        assert isinstance(chunker, TextChunker)

    def test_empty_text_returns_empty_list(self):
        chunker = TextChunker(config={"tokenizer": "word_count"})
        assert chunker.chunk("", doc_id=DOC_ID) == []


# ═══════════════════════════════════════════════════════════════════════════════
# Contract 2 — EmbeddingModel objects built from chunks
# ═══════════════════════════════════════════════════════════════════════════════


class TestContract2_Embeddings:
    def _chunks(self) -> list[ChunkModel]:
        chunker = TextChunker(config={"tokenizer": "word_count", "chunk_size": 50, "chunk_overlap": 5})
        return chunker.chunk(SAMPLE_TEXT, doc_id=DOC_ID)

    def test_embedding_per_chunk(self):
        chunks = self._chunks()
        embeddings = [make_embedding(c) for c in chunks]
        assert len(embeddings) == len(chunks)

    def test_embedding_chunk_ids_match(self):
        chunks = self._chunks()
        embeddings = [make_embedding(c) for c in chunks]
        for c, e in zip(chunks, embeddings):
            assert e.chunk_id == c.chunk_id
            assert e.doc_id == c.doc_id

    def test_embedding_vector_dimensions(self):
        chunks = self._chunks()
        embeddings = [make_embedding(c) for c in chunks]
        assert all(e.dimensions == VECTOR_DIM for e in embeddings)
        assert all(len(e.vector) == VECTOR_DIM for e in embeddings)


# ═══════════════════════════════════════════════════════════════════════════════
# Contract 3 — QdrantIndexer upsert shapes payload correctly
# ═══════════════════════════════════════════════════════════════════════════════


class TestContract3_Indexer:
    def _setup(self):
        chunker = TextChunker(config={"tokenizer": "word_count", "chunk_size": 50, "chunk_overlap": 5})
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id=DOC_ID)
        embeddings = [make_embedding(c) for c in chunks]
        return chunks, embeddings

    def test_upsert_calls_qdrant_with_points(self):
        chunks, embeddings = self._setup()
        with patch("indexing.qdrant_client.QdrantClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            indexer = QdrantIndexer(vector_size=VECTOR_DIM)
            count = indexer.upsert_chunks(COLLECTION, chunks, embeddings)

        assert count == len(chunks)
        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == COLLECTION
        assert len(call_kwargs["points"]) == len(chunks)

    def test_payload_contains_required_fields(self):
        chunks, embeddings = self._setup()
        with patch("indexing.qdrant_client.QdrantClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            indexer = QdrantIndexer(vector_size=VECTOR_DIM)
            indexer.upsert_chunks(COLLECTION, chunks, embeddings)

        points = mock_client.upsert.call_args[1]["points"]
        for point in points:
            payload = point.payload
            assert "chunk_id" in payload
            assert "doc_id" in payload
            assert "text" in payload
            assert "chunk_index" in payload
            assert "token_count" in payload
            assert payload["doc_id"] == DOC_ID

    def test_point_ids_match_chunk_ids(self):
        chunks, embeddings = self._setup()
        with patch("indexing.qdrant_client.QdrantClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            indexer = QdrantIndexer(vector_size=VECTOR_DIM)
            indexer.upsert_chunks(COLLECTION, chunks, embeddings)

        points = mock_client.upsert.call_args[1]["points"]
        chunk_ids = {c.chunk_id for c in chunks}
        point_ids = {p.id for p in points}
        assert chunk_ids == point_ids

    def test_mismatch_raises(self):
        from raglab_common.exceptions import IndexingError
        chunks, _ = self._setup()
        with patch("indexing.qdrant_client.QdrantClient"):
            indexer = QdrantIndexer(vector_size=VECTOR_DIM)
            with pytest.raises(IndexingError, match="mismatch"):
                indexer.upsert_chunks(COLLECTION, chunks, [])  # empty embeddings


# ═══════════════════════════════════════════════════════════════════════════════
# Contract 4+5 — DenseRetriever vector store call + hit conversion
# ═══════════════════════════════════════════════════════════════════════════════


class TestContract4_5_Retriever:
    def _chunks(self) -> list[ChunkModel]:
        chunker = TextChunker(config={"tokenizer": "word_count", "chunk_size": 50, "chunk_overlap": 5})
        return chunker.chunk(SAMPLE_TEXT, doc_id=DOC_ID)

    def _query_model(self, top_k: int = 3) -> QueryModel:
        return QueryModel(
            text="What is RAG and how does it reduce hallucinations?",
            collection=COLLECTION,
            top_k=top_k,
            retriever_type=RetrieverType.DENSE,
            llm_provider=LLMProvider.AZURE_OPENAI,
        )

    def test_retriever_calls_vector_store(self):
        chunks = self._chunks()
        hits = [make_qdrant_hit(c) for c in chunks[:3]]
        mock_vs = MagicMock()
        mock_vs.search.return_value = hits

        retriever = DenseRetriever(config={"score_threshold": 0.0, "ef": 64})
        query_vector = [0.42] * VECTOR_DIM
        embedder = lambda text: query_vector  # noqa: E731

        results = retriever.retrieve(self._query_model(top_k=3), mock_vs, embedder=embedder)

        mock_vs.search.assert_called_once()
        call_kwargs = mock_vs.search.call_args[1]
        assert call_kwargs["collection_name"] == COLLECTION
        assert call_kwargs["limit"] == 3
        assert call_kwargs["query_vector"] == query_vector

    def test_results_are_chunk_models(self):
        chunks = self._chunks()
        hits = [make_qdrant_hit(c, score=0.9 - i * 0.05) for i, c in enumerate(chunks[:3])]
        mock_vs = MagicMock()
        mock_vs.search.return_value = hits

        retriever = DenseRetriever()
        results = retriever.retrieve(
            self._query_model(), mock_vs, embedder=lambda t: [0.1] * VECTOR_DIM
        )

        assert all(isinstance(r, ChunkModel) for r in results)
        assert len(results) == 3

    def test_score_in_result_metadata(self):
        chunks = self._chunks()
        hits = [make_qdrant_hit(chunks[0], score=0.93)]
        mock_vs = MagicMock()
        mock_vs.search.return_value = hits

        retriever = DenseRetriever()
        results = retriever.retrieve(
            self._query_model(top_k=1), mock_vs, embedder=lambda t: [0.5] * VECTOR_DIM
        )

        assert results[0].metadata["score"] == pytest.approx(0.93)
        assert results[0].metadata["retriever"] == "dense"

    def test_no_embedder_returns_empty(self):
        mock_vs = MagicMock()
        retriever = DenseRetriever()
        results = retriever.retrieve(self._query_model(), mock_vs, embedder=None)
        assert results == []

    def test_vector_store_error_returns_empty(self):
        mock_vs = MagicMock()
        mock_vs.search.side_effect = Exception("Qdrant unreachable")
        retriever = DenseRetriever()
        results = retriever.retrieve(
            self._query_model(), mock_vs, embedder=lambda t: [0.1] * VECTOR_DIM
        )
        assert results == []

    def test_retriever_factory_creates_dense(self):
        retriever = RetrieverFactory.create("dense")
        assert isinstance(retriever, DenseRetriever)


# ═══════════════════════════════════════════════════════════════════════════════
# Contract 6 — LLM provider assembles context + returns ResponseModel
# ═══════════════════════════════════════════════════════════════════════════════


class _StubLLM(BaseLLMProvider):
    provider = "stub"
    def __init__(self):
        super().__init__()
        self.last_prompt: str = ""
        self.last_system: str = ""

    def _call_api(self, system_prompt, prompt, max_tokens, temperature):
        self.last_prompt = prompt
        self.last_system = system_prompt
        return "RAG reduces hallucinations by grounding answers in retrieved context."

    def _model_name(self):
        return "stub/llm"


class TestContract6_LLM:
    def _chunks(self) -> list[ChunkModel]:
        chunker = TextChunker(config={"tokenizer": "word_count", "chunk_size": 50, "chunk_overlap": 5})
        return chunker.chunk(SAMPLE_TEXT, doc_id=DOC_ID)[:3]

    def test_llm_generates_response_model(self):
        llm = _StubLLM()
        chunks = self._chunks()
        resp = llm.generate("How does RAG work?", chunks)

        assert isinstance(resp, ResponseModel)
        assert resp.answer == "RAG reduces hallucinations by grounding answers in retrieved context."
        assert resp.model == "stub/llm"
        assert resp.latency_ms >= 0

    def test_context_numbered_in_prompt(self):
        llm = _StubLLM()
        chunks = self._chunks()
        llm.generate("q", chunks)
        assert "[1]" in llm.last_prompt
        assert "[2]" in llm.last_prompt

    def test_sources_propagated(self):
        llm = _StubLLM()
        chunks = self._chunks()
        resp = llm.generate("q", chunks)
        assert len(resp.sources) == len(chunks)

    def test_system_prompt_forwarded(self):
        llm = _StubLLM()
        llm.generate("q", self._chunks(), system_prompt="Be concise.")
        assert "Be concise." in llm.last_system

    def test_empty_chunks_still_responds(self):
        llm = _StubLLM()
        resp = llm.generate("What is RAG?", [])
        assert resp.answer != ""


# ═══════════════════════════════════════════════════════════════════════════════
# Contract 7 — Full pipeline wired end-to-end
# ═══════════════════════════════════════════════════════════════════════════════


class TestContract7_FullPipeline:
    """
    Full R1 pipeline: TextChunker → embed → index → retrieve → generate.
    No infrastructure — Qdrant and LLM calls fully mocked.
    """

    def test_full_pipeline_produces_response(self):
        # Step 1: chunk
        chunker = TextChunker(config={"tokenizer": "word_count", "chunk_size": 40, "chunk_overlap": 5})
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id=DOC_ID)
        assert chunks, "Chunker must produce at least one chunk"

        # Step 2: embed (mock)
        embeddings = [make_embedding(c) for c in chunks]

        # Step 3: index (mock Qdrant)
        with patch("indexing.qdrant_client.QdrantClient") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            indexer = QdrantIndexer(vector_size=VECTOR_DIM)
            upserted = indexer.upsert_chunks(COLLECTION, chunks, embeddings)

        assert upserted == len(chunks)

        # Step 4: retrieve (mock search returns top-3 chunks)
        top_chunks = chunks[:3]
        hits = [make_qdrant_hit(c, score=0.95 - i * 0.03) for i, c in enumerate(top_chunks)]
        mock_vs = MagicMock()
        mock_vs.search.return_value = hits

        retriever = DenseRetriever()
        query_model = QueryModel(
            text="What is RAG and how does it reduce hallucinations?",
            collection=COLLECTION,
            top_k=3,
            retriever_type=RetrieverType.DENSE,
            llm_provider=LLMProvider.AZURE_OPENAI,
        )
        retrieved = retriever.retrieve(query_model, mock_vs, embedder=lambda t: [0.5] * VECTOR_DIM)
        assert retrieved, "Retriever must return at least one chunk"

        # Step 5: generate (stub LLM)
        llm = _StubLLM()
        response = llm.generate(
            query=query_model.text,
            chunks=retrieved,
            system_prompt=(
                "Answer using only the provided context. "
                "If unsure, say so."
            ),
        )

        # Verify end-to-end contract
        assert isinstance(response, ResponseModel)
        assert len(response.answer) > 10
        assert len(response.sources) == len(retrieved)
        assert response.model == "stub/llm"
        assert response.latency_ms >= 0
        # Context must include text from retrieved chunks
        assert any(c.text[:20] in llm.last_prompt for c in retrieved)


# ═══════════════════════════════════════════════════════════════════════════════
# Contract 8 — IngestionMessage serialisation
# ═══════════════════════════════════════════════════════════════════════════════


class TestContract8_MessageSerialisation:
    def test_round_trip_preserves_all_fields(self):
        msg = IngestionMessage(
            idempotency_key="e2e-key-001",
            doc_id=DOC_ID,
            filename="sample.txt",
            storage_path="/data/sample.txt",
            collection=COLLECTION,
            chunker_type="text",
            chunker_config={"chunk_size": 500, "tokenizer": "word_count"},
            llm_provider="azure_openai",
            retry_count=0,
        )
        restored = IngestionMessage.from_bytes(msg.to_bytes())
        assert restored.idempotency_key == msg.idempotency_key
        assert restored.doc_id == msg.doc_id
        assert restored.chunker_config == msg.chunker_config
        assert restored.retry_count == 0

    def test_retry_count_increments_cleanly(self):
        msg = IngestionMessage(
            idempotency_key="k", filename="f.txt",
            storage_path="/f", retry_count=2,
        )
        retried = msg.model_copy(update={"retry_count": msg.retry_count + 1})
        assert retried.retry_count == 3
        assert msg.retry_count == 2  # original immutable


# ═══════════════════════════════════════════════════════════════════════════════
# Contract 9 — RabbitMQ publisher routing key
# ═══════════════════════════════════════════════════════════════════════════════


class TestContract9_PublisherRouting:
    @pytest.mark.asyncio
    async def test_publish_uses_correct_routing_key(self):
        from ingestion.queue.publisher import RabbitMQPublisher

        conn = AsyncMock()
        conn.is_closed = False
        channel = AsyncMock()
        channel.is_closed = False
        exchange = AsyncMock()
        queue = AsyncMock()
        queue.bind = AsyncMock()
        channel.declare_exchange = AsyncMock(return_value=exchange)
        channel.declare_queue = AsyncMock(return_value=queue)
        conn.channel = AsyncMock(return_value=channel)

        with patch("ingestion.queue.publisher.aio_pika.connect_robust", return_value=conn):
            pub = RabbitMQPublisher("amqp://guest:guest@localhost/")
            await pub.connect()
            msg = IngestionMessage(
                idempotency_key="k", filename="f.txt", storage_path="/f"
            )
            await pub.publish(msg)

        exchange.publish.assert_called_once()
        routing_key = exchange.publish.call_args[1]["routing_key"]
        assert routing_key == ROUTING_KEY_DOCUMENT


# ═══════════════════════════════════════════════════════════════════════════════
# Contract 10 — api-gateway strips hop-by-hop headers
# ═══════════════════════════════════════════════════════════════════════════════


class TestContract10_ProxyHeaders:
    @pytest.mark.asyncio
    async def test_hop_by_hop_stripped(self):
        from api_gateway.proxy import proxy_request

        captured = {}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"{}"
        mock_resp.headers = {"content-type": "application/json"}

        with patch("api_gateway.proxy.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            async def cap(**kwargs):
                captured.update(kwargs.get("headers", {}))
                return mock_resp

            mock_client.request = cap

            from fastapi import Request
            scope = {
                "type": "http", "method": "GET",
                "path": "/test", "query_string": b"",
                "headers": [
                    (b"host", b"gateway:8000"),
                    (b"connection", b"keep-alive"),
                    (b"x-custom-header", b"value123"),
                    (b"content-type", b"application/json"),
                ],
            }
            req = Request(scope, receive=AsyncMock(return_value={"type": "http.request", "body": b""}))
            await proxy_request(req, "http://downstream:8001/test")

        assert "host" not in captured
        assert "connection" not in captured
        assert "x-custom-header" in captured
        assert "content-type" in captured


# ═══════════════════════════════════════════════════════════════════════════════
# Contract 11 — HealthRegistry aggregate degrades when core service is down
# ═══════════════════════════════════════════════════════════════════════════════


class TestContract11_HealthRegistry:
    def test_all_ok_aggregates_ok(self):
        r = HealthRegistry()
        core = ["ingestion", "embedding", "indexing", "retrieval", "llm", "pipeline"]
        r.configure_urls({s: f"http://{s}:8000" for s in core})
        for svc in r._services.values():
            svc.status = "ok"
        assert r.aggregate_status() == "ok"

    def test_one_core_down_aggregates_degraded(self):
        r = HealthRegistry()
        core = ["ingestion", "embedding", "indexing", "retrieval", "llm", "pipeline"]
        r.configure_urls({s: f"http://{s}:8000" for s in core})
        for svc in r._services.values():
            svc.status = "ok"
        r._services["llm"].status = "unavailable"
        assert r.aggregate_status() == "degraded"

    def test_non_core_down_does_not_degrade(self):
        """Stub services (graph/auth/observability) don't affect aggregate."""
        r = HealthRegistry()
        core = ["ingestion", "embedding", "indexing", "retrieval", "llm", "pipeline"]
        extra = ["graph", "observability", "auth"]
        r.configure_urls({s: f"http://{s}:8000" for s in core + extra})
        for svc in r._services.values():
            svc.status = "ok"
        r._services["graph"].status = "unavailable"
        r._services["auth"].status = "unavailable"
        assert r.aggregate_status() == "ok"  # stubs don't count


# ═══════════════════════════════════════════════════════════════════════════════
# Contract 12 — UI template injects gateway_url
# ═══════════════════════════════════════════════════════════════════════════════


class TestContract12_UITemplate:
    def test_gateway_url_injected(self):
        from fastapi.testclient import TestClient
        from ui.main import app
        from ui.settings import UISettings

        settings = UISettings()
        object.__setattr__(settings, "gateway_url", "http://gateway-test:8000")
        app.state.settings = settings

        client = TestClient(app)
        r = client.get("/")
        assert b"gateway-test" in r.content or b"/api/v1" in r.content

    def test_all_r1_chunker_knobs_present(self):
        from fastapi.testclient import TestClient
        from ui.main import app
        from ui.settings import UISettings

        app.state.settings = UISettings()
        client = TestClient(app)
        r = client.get("/")

        for marker in [b"chunk-size", b"chunk-overlap", b"min-chunk-size",
                       b"tokenizer", b"boundary"]:
            assert marker in r.content, f"Missing knob: {marker}"

    def test_r2_stubs_present_and_disabled(self):
        from fastapi.testclient import TestClient
        from ui.main import app
        from ui.settings import UISettings

        app.state.settings = UISettings()
        client = TestClient(app)
        r = client.get("/")

        # R2 chunkers visible as disabled options
        for marker in [b"PDFChunker", b"DOCXChunker", b"MarkdownChunker"]:
            assert marker in r.content, f"Missing R2 stub: {marker}"
        assert b"disabled" in r.content
