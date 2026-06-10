# RAGLab

> A fully configurable RAG Configuration Generator — microservices monorepo.

[![Release](https://img.shields.io/badge/release-R6-5e9cf5)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://python.org)
[![uv](https://img.shields.io/badge/package%20manager-uv-black)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/linter-ruff-orange)](https://docs.astral.sh/ruff/)

RAGLab lets you configure, run, and compare RAG pipelines through a single Control Panel UI. Swap chunkers, retrievers, vector stores, and LLM providers without changing code — everything is configuration.

---

## Architecture

```
raglab/
├── packages/
│   ├── raglab-common/      # Shared models, logging, exceptions, settings
│   ├── raglab-chunkers/    # Chunker implementations (TextChunker active in R1)
│   └── raglab-retrievers/  # Retriever implementations (DenseRetriever active in R1)
│
└── services/
    ├── api-gateway/        # Single entry point, health-aware routing
    ├── ingestion/          # Document intake, async queue, idempotency
    ├── embedding/          # Vector generation (multi-model) + Redis cache [R5]
    ├── indexing/           # Qdrant indexing + Postgres metadata
    ├── retrieval/          # Retriever execution (7 strategies)
    ├── llm/                # LLM provider abstraction
    ├── pipeline/           # End-to-end orchestration + self-healing gates [R5]
    ├── config/             # Pipeline configuration management
    ├── storage/            # File storage backend (local + S3 + Azure Blob)
    ├── ui/                 # Control Panel + Graph Explorer + Healing Trace
    ├── graph/              # GraphRAG — entity extraction, NetworkX, Leiden [R4]
    ├── observability/      # OTel tracing + 6 views (trace/chunk/retrieval/cost/health) [R6]
    └── auth/               # [R7] Authentication
```

## Release Status

| Release | Theme | Status |
|---------|-------|--------|
| **R1** | Full Shell + Core Pipeline | ✅ Done |
| **R2** | Advanced Chunking + Cloud Storage | ✅ Done |
| **R3** | Retrieval Power + CI/CD | ✅ Done |
| **R4** | Graph RAG + Advanced Document Types | ✅ Done |
| **R5** | Self-Healing RAG + Cost Efficiency | ✅ Done |
| **R6** | Observability + Full Testing | ✅ Done |
| R7 | Auth + Multi-tenancy + GCS + GCP | 🔜 Next |

## R5 Active Components

| Layer | Active | Stubbed (Coming Soon) |
|-------|--------|-----------------------|
| Chunker | TextChunker, PDFChunker, DOCXChunker, MarkdownChunker, HTMLChunker, ExcelChunker, HybridChunker, PDFImageChunker, TableStitchChunker | — (all 9 active) |
| Retriever | DenseRetriever, BM25Retriever, HybridRetriever (RRF), MMRRetriever, ReRankerRetriever, CompressionRetriever, GraphRetriever | — (all 7 active) |
| Vector Store | Qdrant | FAISS, ChromaDB, Pinecone |
| LLM Provider | Azure OpenAI, OpenAI, Anthropic, Ollama | Vertex |
| Storage | Local filesystem, S3, Azure Blob | GCS (R7) |
| Graph | NetworkX + Leiden community detection, graph-service v0.2.0 | Neo4j (optional R6+) |
| Eval / Self-Healing | ChunkQualityScorer, RetrievalHealer, GroundednessChecker (raglab-eval v0.1.0) | — |
| Cache | Redis embedding cache (hit_rate_pct ROI metric) | Semantic cache (R7) |
| Observability | OTel tracing, observability-service v0.2.0, 6 native D3.js views | — |
| Testing | 1,811 tests: unit + integration (55) + E2E (68) | — |
| Infrastructure | Azure AKS + AWS EKS (Terraform), HPA, PDB, Redis HA, IRSA | GCP GKE (R7) |

## Release Branches

Each release is a permanent snapshot branch for step-by-step demos and presentations.

| Branch | Tests | What it showcases |
|--------|-------|-------------------|
| `release/r1` | 369 | Core pipeline — 13 services, TextChunker, DenseRetriever, Control Panel |
| `release/r2` | 564 | + 6 chunkers (PDF/DOCX/MD/HTML/Excel/Hybrid), S3 + Azure Blob |
| `release/r3` | 796 | + BM25, Hybrid RRF, MMR, ReRanker, Compression, Comparison UI, CI/CD |
| `release/r4` | 1,287 | + Graph RAG, PDFImageChunker, TableStitchChunker, D3.js Explorer, Terraform |
| `release/r5` | 1,558 | + Self-healing gates, raglab-eval, embedding cache, Healing Trace UI |

```bash
git checkout release/r1   # demo R1
git checkout release/r5   # current production state
```

## Quick Start

```bash
# 1. Install uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install
git clone https://github.com/tamalkundu007/raglab.git
cd raglab
uv sync

# 3. Configure
cp .env.example .env
# Edit .env with your API keys and infra URLs

# 4. Start infrastructure
docker-compose up -d qdrant postgres rabbitmq

# 5. Start all services
docker-compose up
```

## Development

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=packages --cov=services --cov-report=term-missing

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

## Built With

- **Python 3.12+** · **FastAPI** · **uv monorepo**
- **Qdrant** (vectors) · **PostgreSQL** (metadata) · **RabbitMQ** (async ingestion)
- **Jinja2 + Alpine.js** (UI) · **Docker + docker-compose** (dev)
- **Pydantic v2** · **structlog** · **tiktoken**

---

*Built by [Tamal Kundu](https://tamalkundu.com) · Kundu Corp · 2026*
