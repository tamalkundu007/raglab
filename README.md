# RAGLab

> A fully configurable RAG Configuration Generator — microservices monorepo.

[![Release](https://img.shields.io/badge/release-R1-6B48C8)](CHANGELOG.md)
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
    ├── embedding/          # Vector generation (multi-model)
    ├── indexing/           # Qdrant indexing + Postgres metadata
    ├── retrieval/          # Retriever execution
    ├── llm/                # LLM provider abstraction
    ├── pipeline/           # End-to-end orchestration
    ├── config/             # Pipeline configuration management
    ├── storage/            # File storage backend
    ├── ui/                 # Control Panel (Jinja2 + Alpine.js)
    ├── graph/              # [R4] GraphRAG
    ├── observability/      # [R6] LLMOps monitoring
    └── auth/               # [R7] Authentication
```

## Release Status

| Release | Theme | Status |
|---------|-------|--------|
| **R1** | Full Shell + Core Pipeline | ✅ Done |
| **R2** | **Advanced Chunking + Cloud Storage** | ✅ Done |
| **R3** | **Retrieval Power + CI/CD** | ✅ Done |
| **R4** | GraphRAG | 🔜 Planned |
| R5 | Caching + Performance | 🔜 Planned |
| R6 | Observability / LLMOps | 🔜 Planned |
| R7 | Auth + Multi-tenancy | 🔜 Planned |

## R1 Active Components

| Layer | Active | Stubbed (Coming Soon) |
|-------|--------|-----------------------|
| Chunker | TextChunker | PDF, DOCX, MD, HTML, Excel, TableStitch |
| Retriever | DenseRetriever | BM25, Hybrid, MMR, Re-ranker, Compression |
| Vector Store | Qdrant | FAISS, ChromaDB, Pinecone |
| LLM Provider | Azure OpenAI, OpenAI, Anthropic, Ollama | Vertex |
| Storage | Local filesystem | S3, Azure Blob, GCS |

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
