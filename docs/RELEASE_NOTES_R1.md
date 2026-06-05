# RAGLab R1 — Release Notes

**Version:** 0.1.0 · **Theme:** Full Shell + Core Pipeline · **Date:** June 2026

---

## Summary

Release 1 ships a fully wired, production-structured RAG platform as a 13-service uv monorepo. The complete pipeline is live: document ingestion → chunking → embedding → Qdrant indexing → dense retrieval → multi-LLM generation. 369 tests pass with zero infrastructure dependencies.

---

## Stats

| Metric | Value |
|--------|-------|
| Microservices | 13 |
| Internal packages | 3 |
| Tests passing | 369 |
| Tests skipped | 3 (tiktoken BPE blocked in sandboxed CI) |
| Infra required to run tests | Zero |
| E2E contracts verified | 12 |
| Release roadmap | 7 releases |

---

## R1 Active Pipeline

```
Document → Ingest (RabbitMQ + idempotency)
         → TextChunker (fixed-token + boundary backtracking)
         → Embed (Azure OpenAI / OpenAI / Anthropic / Ollama)
         → Index (Qdrant vectors + Postgres metadata)
         → DenseRetriever (cosine similarity)
         → LLM Generate (multi-provider)
         → ResponseModel (answer + sources + latency)
```

---

## What Shipped

### Internal Packages

**raglab-common v0.1.0**
- Pydantic v2 models: `ChunkModel`, `DocumentModel`, `EmbeddingModel`, `QueryModel`, `ResponseModel`, `HealthModel`
- Async SQLAlchemy engine + `get_session()` context manager
- structlog-based `configure_logging()` / `get_logger()`
- `BaseServiceSettings` with `RAGLAB_` env prefix
- RabbitMQ message schemas: `IngestionMessage`, `DLQMessage`, exchange/queue constants

**raglab-chunkers v0.2.0**
- `BaseChunker` with `chunk()` wrapper (logging + silent error recovery)
- `ChunkerFactory` registry: `create()`, `available()`, `schema()`
- `TextChunker`: fixed-token + sentence boundary backtracking per FRS spec
- `_boundary.split_into_windows()`: shared algorithm — all R2+ chunkers reuse this
- R2+ stubs: PDFChunker, DOCXChunker, MarkdownChunker, HTMLChunker, ExcelChunker, TableStitch
- 73 tests (70 pass, 3 skip on tiktoken BPE)

**raglab-retrievers v0.2.0**
- `BaseRetriever` with `retrieve()` wrapper
- `RetrieverFactory` registry: `create()`, `available()`, `schema()`
- `DenseRetriever`: Qdrant cosine similarity, `_build_filter()`, `_hits_to_chunks()`
- R3+ stubs: BM25, Hybrid, MMR, ReRanker, Compression
- 50 tests (mock vector store + embedder)

### Services

| Service | Port | Status | Key Features |
|---------|------|--------|-------------|
| api-gateway | 8000 | Active | HealthRegistry, proxy, 13 routes |
| ingestion | 8001 | Active | RabbitMQ publish, idempotency, DLQ topology |
| embedding | 8002 | Active | 4 providers, batch API |
| indexing | 8003 | Active | Qdrant upsert, Postgres ORM |
| retrieval | 8004 | Active | Dense retrieval, embed+search |
| llm | 8005 | Active | 4 providers, RAG prompt assembly |
| pipeline | 8006 | Active | Consumer, retry logic, DLQ routing |
| config | 8007 | Active | Stub (R2 scope) |
| storage | 8008 | Active | Local filesystem (R1) |
| ui | 8009 | Active | Control Panel, all 7-release knobs |
| graph | 8010 | Stub | Activates R4 |
| observability | 8011 | Stub | Activates R6 |
| auth | 8012 | Stub | Activates R7 |

### Key Design Decisions

**Health-aware routing** — The api-gateway polls all 12 downstream services every 10 seconds. A service marked unavailable gets a 503 before the proxy is even attempted. Core services (ingestion, embedding, indexing, retrieval, llm, pipeline) determine aggregate gateway status; stubs (graph/auth/observability) don't drag the gateway into degraded.

**Idempotency** — Every ingestion request carries an idempotency key (SHA256(filename+collection) or caller-supplied). The consumer checks Postgres before processing; completed documents are acked and skipped. This prevents re-embedding on network retries.

**DLQ pattern** — After `MAX_RETRIES=3` failures, the pipeline wraps the original message in a `DLQMessage` and routes it to `ingestion_dlq` via the topic exchange. Postgres `DocumentRecord.status` is updated to `dead_letter`. Manual intervention required.

**Factory pattern consistency** — `ChunkerFactory` and `RetrieverFactory` follow the same interface (`create()`, `available()`, `schema()`). Stub classes raise `NotImplementedFeatureError` on instantiation. The UI calls `available()` to drive dropdown state.

**Shared boundary algorithm** — `_boundary.split_into_windows()` lives in `raglab-chunkers._boundary`. No R2+ chunker reimplements it — they call it within their structural units (pages, headings, tags, sheets).

---

## E2E Contracts Verified (Phase 10)

12 pipeline contracts tested in-process (zero infrastructure):

1. TextChunker produces valid `ChunkModel` list with sequential indices and unique IDs
2. `EmbeddingModel` objects built correctly from chunk data (chunk_id/doc_id match)
3. `QdrantIndexer.upsert_chunks` sends correctly shaped `PointStruct` payloads
4. `DenseRetriever` calls vector store with correct collection name, top_k, and query vector
5. Qdrant hits converted to `ChunkModel` with score injected into metadata
6. `BaseLLMProvider` assembles numbered context block and returns `ResponseModel`
7. **Full pipeline**: TextChunker → embed → index → retrieve → generate produces a `ResponseModel`
8. `IngestionMessage` round-trips through serialisation unchanged (all fields preserved)
9. `RabbitMQPublisher.publish` sends AMQP message with correct routing key
10. api-gateway `proxy_request` strips hop-by-hop headers (host, connection, etc.)
11. `HealthRegistry.aggregate_status` degrades when a core service is down; stubs don't affect it
12. UI template injects `gateway_url` and all R1 chunker knobs are present

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/tamalkundu007/raglab.git && cd raglab
cp .env.example .env
# Edit .env: set RAGLAB_AZURE_OPENAI_API_KEY, RAGLAB_AZURE_OPENAI_ENDPOINT,
#            RAGLAB_AZURE_OPENAI_CHAT_DEPLOYMENT, RAGLAB_AZURE_OPENAI_EMBEDDING_DEPLOYMENT

# 2. Start infrastructure
docker-compose up -d qdrant postgres rabbitmq

# 3. Start all services
docker-compose up

# 4. Open Control Panel
open http://localhost:8009

# 5. API Gateway (all routes)
open http://localhost:8000/docs

# Run tests (zero infra needed)
uv run pytest packages/ services/ tests/ -q
```

---

## 7-Release Roadmap

| Release | Theme | Status |
|---------|-------|--------|
| **R1** | Full Shell + Core Pipeline | ✅ Done |
| R2 | Advanced Chunking (PDF/DOCX/MD/HTML/Excel) + Cloud Storage (S3/Azure Blob) | 🔜 Next |
| R3 | Advanced Retrievers (BM25/Hybrid/MMR/Re-ranker) + CI/CD + Cloud Deploy | 🔜 Planned |
| R4 | GraphRAG (NetworkX + leidenalg + graph retrieval) | 🔜 Planned |
| R5 | Caching + Performance (Redis semantic cache) | 🔜 Planned |
| R6 | Observability / LLMOps (evaluation metrics + tracing) | 🔜 Planned |
| R7 | Auth + Multi-tenancy + GCS | 🔜 Planned |

---

*Built by [Tamal Kundu](https://tamalkundu.com) · Kundu Corp · June 2026*
