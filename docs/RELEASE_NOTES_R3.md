# RAGLab R3 — Release Notes

**Version:** 0.3.0 · **Theme:** Retrieval Power + CI/CD · **Date:** June 2026
**Builds on:** R1 (0.1.0) · R2 (0.2.0)

---

## Summary

Release 3 is where retrieval stops being a single mode and becomes the configurable, multi-strategy engine that justifies the "RAG Configuration Generator" positioning. Five retrievers activated, a side-by-side comparison UI, and production-grade CI/CD for Azure and AWS.

---

## Stats

| Metric | Value |
|--------|-------|
| New tests | 427 |
| Total tests passing | 796 |
| Tests skipped | 3 (tiktoken BPE, sandboxed CI) |
| raglab-retrievers version | 0.3.0 |
| Retrievers activated | 5 (BM25, Hybrid, MMR, ReRanker, Compression) |
| CI/CD pipelines | 3 (Azure + AWS active, GCP stub) |
| New UI pages | 1 (Retrieval Comparison) |
| Infra required for tests | Zero |

---

## What Shipped

### raglab-retrievers v0.3.0

All retrievers implement `BaseRetriever`, register in `RetrieverFactory`, and are wired into the UI comparison knobs.

**BM25Retriever** (`retriever_type: "bm25"`)
- `BM25Corpus`: in-memory BM25Okapi index built from a `ChunkModel` list; reusable across queries
- `_tokenize()`: regex word tokenisation (`\b\w+\b`, lowercased)
- No embedder required — pure token overlap scoring
- Post-hoc `metadata_filter` support via equality matching on `ChunkModel.metadata`
- `k1` (TF saturation), `b` (document length normalisation), `top_n_factor` params
- `score` injected into every result's metadata

**HybridRetriever** (`retriever_type: "hybrid"`)
- Reciprocal Rank Fusion of Dense + BM25 results
- RRF formula: `alpha × 1/(k + dense_rank) + (1−alpha) × 1/(k + bm25_rank)`
- Rank-based fusion — sidesteps the dense-vs-sparse score-scale mismatch entirely
- `_rrf_fuse()`: deduplicates by `chunk_id`; both lists contribute independently
- `_unpack_store()`: accepts `(qdrant_client, bm25_corpus)` tuple or `HybridStore` object
- `alpha=1.0` → pure dense; `alpha=0.0` → pure BM25; `alpha=0.5` → balanced
- `rrf_score` injected into every result's metadata
- **Naming distinction (documented and tested):** `HybridRetriever` ≠ `HybridChunker`

**MMRRetriever** (`retriever_type: "mmr"`)
- Maximum Marginal Relevance (Carbonell & Goldstein, 1998)
- `_cosine_sim()`: pure Python dot product / (‖a‖ × ‖b‖), handles zero vectors
- `_mmr_select()`: greedy selection maximising `λ × relevance − (1−λ) × max_sim_to_selected`
- `fetch_k` dense candidates → MMR selection of `top_k`
- `lambda_mult=1.0` → pure relevance (= DenseRetriever); `lambda_mult=0.0` → pure diversity
- `mmr_rank` injected into result metadata

**ReRankerRetriever** (`retriever_type: "reranker"`)
- Two-stage: `DenseRetriever` (fetch_k) → `CrossEncoder` re-ranking
- `sentence-transformers` `CrossEncoder` lazy-loaded on first use (no startup cost)
- Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, strong)
- `batch_size` for inference; `score_threshold` drops below-threshold results
- `reranker_score` + `reranker_rank` + `reranker_model` in metadata
- Composable wrapper: any base retriever can be used for Stage 1

**CompressionRetriever** (`retriever_type: "compression"`)
- Two-stage: `DenseRetriever` (fetch_k) → compression filter
- `strategy="keyword"`: keep chunks with ≥ `min_keyword_overlap` tokens shared with query (no LLM call)
- `strategy="llm"`: LLM extraction (wired for R4 full implementation)
- `_query_tokens()`: regex word tokenisation for overlap check
- `compression_strategy` + `compression_rank` in metadata
- Composable wrapper: any base retriever can be used for Stage 1

### RetrieverFactory updates

All 6 retriever types now active — zero stubs remain in R3. `available()` returns `active=True` for all.

### Retrieval Comparison UI (`GET /compare`)

New page served by the `ui-service`. Shared `_ctx()` context builder eliminates duplicate settings access.

- 6 collapsible strategy toggles (Dense, BM25, Hybrid, MMR, Re-Ranker, Compression)
- Per-strategy config knobs exposed (14 total across strategies + 3 common):
  - Dense: `score_threshold`, `ef`
  - BM25: `k1`, `b`
  - Hybrid: `alpha`, `rrf_k`, `dense_top_k`
  - MMR: `lambda_mult`, `fetch_k`
  - Re-Ranker: `fetch_k`, `model_name`
  - Compression: `strategy`, `min_keyword_overlap`, `fetch_k`
- `runComparison()`: all selected strategies fire concurrently via `Promise.all`
- `computeOverlaps()`: finds `chunk_id`s appearing in ≥ 2 strategy results → gold border + "overlap" badge
- Two view modes: Side by Side / Highlight Overlaps (dims non-overlapping chunks to 30% opacity)
- Summary bar: fastest strategy, most results, count of strategies run
- Score display adapts: `score` / `rrf_score` / `reranker_score` depending on strategy
- Ctrl+Enter / Cmd+Enter keyboard shortcut
- Colour-coded columns via CSS `--s-color` custom property per strategy

### CI/CD

**`ci.yml`** — Runs on push + PR to `main` and `develop`.
- Lint (ruff check + format) → test (pytest, JUnit XML, coverage XML) → Docker build matrix
- All 9 active services in build matrix
- Python 3.12; concurrency: cancel-in-progress on CI (safe)

**`cd-azure.yml`** — Push to `main` → production deploy.
- OIDC federated identity (no long-lived credentials stored)
- Detect changed services via `git diff` → build changed services only → deploy all
- `azure/login@v2` → ACR push → Bicep `deployment group create` → `az containerapp update`
- Secrets via Key Vault `secretRef` — never plain env vars
- Liveness + readiness probes on every Container App
- HTTP scaling (concurrentRequests: 100), min/max replicas per tier
- `cancel-in-progress: false` — deploys never cancelled mid-flight

**`cd-aws.yml`** — Push to `main` → production deploy.
- OIDC via AWS IAM Identity Provider (no `AWS_ACCESS_KEY_ID` stored)
- `aws-actions/configure-aws-credentials@v4` → ECR push → jq-based task def update
- `ecs update-service --force-new-deployment` → `ecs wait services-stable` → ALB health check
- Secrets from AWS Secrets Manager in task definitions — never plain env vars

**`cd-gcp.yml`** — Stub. No `on:` triggers. `if: false` job. Activates in R7.

**Infrastructure:**
- `deploy/azure/main.bicep` — Container Apps for all 9 services, Key Vault refs, probes, scaling
- `deploy/aws/main.tf` — ECS Cluster (Fargate), IAM OIDC role, task definitions, CloudWatch logs
- `deploy/gcp/README.md` — R7 target documented (Cloud Run, GAR, Workload Identity)
- `docs/CI_CD_SETUP.md` — Full setup guide: Azure OIDC commands, AWS OIDC commands, security principles

---

## Design Decisions

**RRF over score interpolation.** Hybrid retrieval uses Reciprocal Rank Fusion, not weighted score averaging. The reason: dense scores and BM25 scores live on completely different scales — normalising them is fragile and dataset-dependent. RRF fuses on ranks (integers), which are always comparable. The `rrf_k=60` default flattens score distribution to reduce sensitivity to rank-1 outliers.

**BM25 in-process vs Qdrant sparse vectors.** `rank-bm25` maintains an in-memory `BM25Corpus` built from indexed `ChunkModel` list. This keeps zero external deps for retrieval, makes testing trivial (no Qdrant extension needed), and keeps the cloud-agnostic story clean. Qdrant native sparse vectors are a valid R4 upgrade path.

**ReRanker and Compression as composable wrappers.** Both wrap any base retriever for Stage 1 — not just Dense. Stage 1 could be BM25 or Hybrid; Stage 2 is always the cross-encoder or compression filter. This is the strategy + decorator pattern applied to retrieval.

**YAML `on:` quoting.** PyYAML's `SafeLoader` parses bare `on` as boolean `True` — a longstanding GOTCHA that silently breaks workflow trigger parsing. All workflow files quote `"on":`.

**No inline Python in YAML.** The original `cd-aws.yml` embedded `python3 -c "..."` inside a bash script inside a YAML block scalar. Single quotes inside the Python broke YAML's simple-key scanner at line 156. Replaced with `jq` (always available on GitHub runners) — cleaner, no quote escaping, faster.

**OIDC everywhere, zero static credentials.** Azure uses Workload Identity Federation (`azure/login@v2`). AWS uses OIDC identity provider + IAM role (`configure-aws-credentials@v4`). No `AWS_ACCESS_KEY_ID`, no `AZURE_CLIENT_SECRET` stored in GitHub secrets.

---

## Interview Angles This Release Unlocks

**"How does hybrid retrieval combine sparse and dense?"**
RRF on ranks, not raw scores — avoids normalising incompatible score scales. `alpha × 1/(k + dense_rank) + (1−alpha) × 1/(k + bm25_rank)`. Most candidates hand-wave score-weighting and get the scale problem wrong.

**"When does BM25 beat dense?"**
Exact keywords, rare terms, codes, product names, acronyms — where embeddings blur lexical specificity. Hybrid exists because neither wins alone across all query types.

**"What's MMR for?"**
Kills near-duplicate chunks in the top-k. Trades a little relevance for diversity via `lambda_mult`. Critical when your corpus has repetitive passages (e.g. policy documents with repeated boilerplate).

**"Re-ranking cost/benefit?"**
Cross-encoders are accurate but expensive. Over-fetch cheaply with a bi-encoder (dense, fast), then re-rank a small candidate set with the cross-encoder. The `fetch_k → top_k` funnel is the pattern.

**"Composable retrievers?"**
Re-ranker and compression wrap any base retriever — Dense, BM25, or Hybrid for Stage 1. Strategy + decorator pattern applied to retrieval. Swapping the base retriever is a config change, not a code change.

---

## Quick Start — R3 Retrievers

```python
from raglab_retrievers import RetrieverFactory, BM25Corpus

# BM25 — build corpus from indexed chunks
corpus = BM25Corpus(chunks, k1=1.5, b=0.75)
retriever = RetrieverFactory.create("bm25")
results = retriever.retrieve(query, corpus)

# Hybrid — RRF fusion of Dense + BM25
retriever = RetrieverFactory.create("hybrid", config={
    "alpha": 0.6,      # 60% dense weight
    "rrf_k": 60,
    "dense_top_k": 20,
})
results = retriever.retrieve(query, (qdrant_client, corpus), embedder=embedder)

# MMR — diversity-aware retrieval
retriever = RetrieverFactory.create("mmr", config={
    "lambda_mult": 0.5,  # balanced relevance/diversity
    "fetch_k": 20,
})
results = retriever.retrieve(query, qdrant_client, embedder=embedder)

# Re-ranker — dense fetch → cross-encoder rerank
retriever = RetrieverFactory.create("reranker", config={
    "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "fetch_k": 20,
    "score_threshold": 0.0,
})
results = retriever.retrieve(query, qdrant_client, embedder=embedder)

# Compression — dense fetch → keyword filter
retriever = RetrieverFactory.create("compression", config={
    "strategy": "keyword",
    "min_keyword_overlap": 1,
    "fetch_k": 20,
})
results = retriever.retrieve(query, qdrant_client, embedder=embedder)
```

---

## 7-Release Roadmap

| Release | Theme | Status |
|---------|-------|--------|
| R1 | Full Shell + Core Pipeline | ✅ Done |
| R2 | Advanced Chunking + Cloud Storage | ✅ Done |
| **R3** | **Retrieval Power + CI/CD** | ✅ **Done** |
| R4 | GraphRAG (NetworkX + leidenalg + graph retrieval) | 🔜 Next |
| R5 | Caching + Performance (Redis semantic cache) | 🔜 Planned |
| R6 | Observability / LLMOps (evaluation metrics + tracing) | 🔜 Planned |
| R7 | Auth + Multi-tenancy + GCS | 🔜 Planned |

---

*Built by [Tamal Kundu](https://tamalkundu.com) · Kundu Corp · June 2026*
