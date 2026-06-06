# RAGLab R4 — Release Notes

**Version:** 0.4.0 · **Theme:** Graph RAG + Advanced Document Types · **Date:** June 2026
**Builds on:** R1 (0.1.0) · R2 (0.2.0) · R3 (0.3.0)

---

## Summary

Release 4 is the hardest release in the RAGLab roadmap. It adds the two document types classical RAG fails on (scanned PDFs, tables spanning page breaks), activates the graph-service that has been scaffolded since R1, and delivers production infrastructure via Terraform. Graph RAG is additive — it runs alongside classical retrieval, never replaces it.

---

## Stats

| Metric | Value |
|--------|-------|
| New tests | 308 |
| Total tests passing | 1,287 |
| raglab-chunkers version | 0.4.0 |
| raglab-retrievers version | 0.4.0 |
| graph-service version | 0.2.0 (was stub 0.1.0) |
| New chunkers activated | 2 (PDFImageChunker, TableStitchChunker) |
| Stubs remaining in raglab-chunkers | 0 — all 8 ChunkerType values active |
| New retrievers | 1 (GraphRetriever) |
| Stubs remaining in raglab-retrievers | 0 — all 7 RetrieverType values active |
| Cloud targets with Terraform | 2 active (Azure, AWS) + 1 stub (GCP, R7) |
| Infra required for tests | Zero |

---

## What Shipped

### PDFImageChunker — raglab-chunkers v0.4.0

Handles the cases `PDFChunker` can't — scanned documents, image-only PDFs, PDFs where the text layer is absent or garbled.

**OCR path:**
- Rasterises each page at configurable DPI (72–600, default 150) via PyMuPDF
- Runs `pytesseract.image_to_string()` with configurable language code
- OCR failure on a single page: warning log, continues to next page — never raises
- Extracted text → `split_into_windows()` (reuse rule — same boundary-backtracking algorithm as every other chunker)
- `chunk_type='text'`, `ocr_engine` in metadata

**Image extraction path:**
- `get_images()` + `extract_image()` per region
- `min_image_area` filter (width × height px²) drops decorative/tiny images
- Image stored as base64 PNG in `metadata["image_bytes"]`
- `chunk_type='image'`, `captioned=False`

**Four modes:** `extract`, `skip`, `both`, `caption`.
**Two OCR engines:** `tesseract`, `none`.

### CaptionService

HTTP client decoupled from `PDFImageChunker`. Called when `image_handling='caption'`.

- POSTs base64 image to llm-service `POST /caption`
- `on_failure='placeholder'`: HTTP error → descriptive fallback text
- `on_failure='raise'`: HTTP error → `ChunkerError`
- Module-level `_requests` alias — fully patchable in tests

### /caption endpoint (llm-service)

- `BaseLLMProvider.caption_image()`: graceful fallback for providers without vision
- `AzureOpenAIProvider.caption_image()`: GPT-4V / GPT-4o via data URL (`data:image/png;base64,...`)
- `AnthropicProvider.caption_image()`: claude-3-* via Anthropic `source` block format
- `POST /caption`: 200 on success, 503 missing provider, 502 LLM error

---

### TableStitchChunker — raglab-chunkers v0.4.0

The insight: PDF tables spanning page breaks produce orphaned rows with no column context. A page-by-page chunker sees continuation rows (`Carol | 91 | Marketing`) with no idea what the columns mean. The downstream LLM answers wrong.

**Detection:**
- pdfplumber extracts tables per page
- Continuation detected when: next page, column count within `column_alignment_tolerance`, total page span ≤ `stitch_threshold`
- `header_repeat_detection=True`: strips repeated header rows on continuation pages

**Emit formats:**
- `markdown` — GFM table, LLM-friendly, padded columns. Default.
- `json` — list of row dicts keyed by header
- `csv` — with header row

**Metadata:** `chunk_type='table'`, `stitched=True/False`, `page_start`, `page_end`, `pages_stitched`, `row_count`, `col_count`.

**Reuse rule:** free text between tables chunked via `split_into_windows()` — no reimplementation.
**All 8 ChunkerType values now active. Zero stubs remain.**

---

### graph-service v0.2.0 (was stub v0.1.0)

The centerpiece of R4. Builds and queries a knowledge graph from ingested chunks.

**ORM (Postgres):**
- `GraphEntity` — dedup by `(name_normalised, entity_type, collection)`. `name_normalised = name.lower().strip()`. `source_chunk_ids`: pipe-separated, appended on re-extraction.
- `GraphRelationship` — directed edge with `relation_type`, `weight`, dedup by `(source_id, target_id, relation_type, collection)`.
- `GraphRun` — job tracking with status lifecycle (PENDING → RUNNING → COMPLETE/FAILED).

**Extractor (`EntityRelationshipExtractor`):**
- Stateless — no DB connection. `llm_caller` injectable for tests.
- `_parse_llm_response()`: strips code fences, filters blank names, uppercases types, skips malformed JSON — never raises.
- Production: HTTP POST to llm-service `/generate`.

**Repository (`GraphRepository`):**
- Async upsert with append-not-replace semantics.
- Missing relationship entity → skip with warning, not crash.

**Graph builder (`GraphBuilder`):**
- Builds `nx.DiGraph` from Postgres: nodes = entity UUIDs, edges = relationships.
- TTL-based in-memory cache per collection. `invalidate()` call after new extraction.
- Phantom entity references silently skipped.
- Community detection: Leiden algorithm (`leidenalg.RBConfigurationVertexPartition`) with configurable `resolution_parameter` and `n_iterations`.
- Fallback: `nx.weakly_connected_components()` when leidenalg unavailable — no crash.
- `_annotate_communities()`: writes `community_id` back onto node attributes.

**Endpoints:**
- `POST /graph/extract` — LLM extraction over chunks → Postgres
- `POST /graph/build` — build NetworkX graph + community detection → cache on `app.state`
- `GET /graph/entities`, `/graph/relationships`, `/graph/stats`
- `GET /graph/communities`, `GET /graph/node/{id}`

---

### GraphRetriever — raglab-retrievers v0.4.0

Three modes — Graph RAG is additive, never a silent replacement.

**`classical`** — delegates to `DenseRetriever`. Tags `graph_mode='classical'`. Zero graph involvement.

**`graph`** — `_find_entry_nodes()`: word-boundary regex match (`\b` + `re.escape`) case-insensitive against graph node names → `_traverse_graph()` BFS up to `traversal_depth` hops, collecting `chunk_ids` from edge attributes. Cycle-safe via `visited_nodes` set.

**`hybrid`** — Dense retrieval (`classical_k` proportional to `1-graph_weight`) + entity matching from query + classical chunk texts + graph traversal (`graph_k` proportional to `graph_weight`) + `_merge_results()` dedup. `graph_weight=0.0` → pure classical. `graph_weight=1.0` → graph only.

**`RetrieverType.GRAPH`** added to raglab-common enum.
**All 7 RetrieverType values now active. Zero stubs remain.**

**Naming distinction (documented and tested):**
- `GraphRetriever` = graph entity matching + traversal (R4)
- `HybridRetriever` = dense + sparse RRF fusion (R3)
- `GraphBuilder` = NetworkX graph construction (graph-service)

---

### Graph Explorer UI

New page at `GET /graph`. D3.js v7 force-directed visualization.

- **Left panel:** collection + graph service URL, build options (community detection toggle, Leiden resolution slider), traversal query (entity name + depth slider), display controls (node size by degree/uniform, colour by community/type, relation labels, link strength)
- **Centre canvas:** D3.js force simulation (forceLink, forceManyBody, forceCenter, forceCollide). Zoom/pan (0.1×–6×). Drag. Arrow markers on directed edges.
- **Right inspector:** entity name, type chip (coloured), description, community badge, outgoing/incoming edge list with relation types.
- `traverseFromEntity()`: client-side BFS highlighting reachable subgraph at configurable depth.
- Community legend: auto-built, max 8 entries.
- Colour palettes: `COMMUNITY_COLOURS` (12), `TYPE_COLOURS` (semantic per entity type).
- `graph_service_url` injected via Jinja2 from `UISettings`.

**Control Panel GraphRAG section activated:**
- Graph Mode select: hybrid / classical / graph (was greyed since R1)
- Community Detection: Leiden / disabled (was greyed)
- Traversal Depth range: 1–5 (was greyed)
- Graph Explorer nav link with R4 purple badge

---

### Terraform — Azure + AWS + GCP stub

**Azure (`infra/terraform/azure/`):**
- `main.tf`: Resource Group, Log Analytics, VNet (3 subnets), ACR (`admin_enabled=false`), AKS (OIDC + Workload Identity + autoscaling), PostgreSQL Flexible Server v16 (private VNet), Key Vault (purge protection enabled, secrets for postgres-dsn), `random_password` — never in tfvars.
- `workloads.tf`: Qdrant (Helm, managed-premium PVC), RabbitMQ (Helm, 3 replicas HA), all 12 services as `kubernetes_deployment` + `kubernetes_service`, liveness/readiness probes, Key Vault CSI secrets.
- Auth: `use_oidc = true` — no `client_secret`.

**AWS (`infra/terraform/aws/`):**
- VPC (3 AZs, public + private + database subnets, NAT Gateways per AZ).
- ECR: 12 repositories, `scan_on_push=true`, lifecycle policy (keep 10 images).
- EKS v1.29: OIDC identity provider from cluster issuer → IRSA (no static pod credentials).
- RDS PostgreSQL v16: `multi_az=true`, `storage_encrypted=true`, `deletion_protection=true`.
- Secrets Manager: `postgres-dsn` (full asyncpg DSN), placeholder secrets for LLM keys.
- CloudWatch log groups per service, 14-day retention.
- No `access_key` / `secret_key` in provider block.

**GCP (`infra/terraform/gcp/`):** stub. Provider block commented. Zero resources created. R7 target documented: GKE Autopilot, Cloud SQL, GAR, Secret Manager, Workload Identity Federation.

**`shared/variables.tf`:** 12 shared variables used across all clouds.

---

## Design Decisions

**Graph RAG as additive, not replacement.** The `GraphRetriever` has three modes and the caller chooses. `classical` mode is identical to calling `DenseRetriever` directly. `hybrid` blends classical and graph results by `graph_weight`. No query is silently rerouted through the graph path.

**Postgres + NetworkX, no Neo4j.** Graph stored relationally (Postgres entities + relationships tables), built in-memory as NetworkX DiGraph for traversal. TTL-cached per collection. Revisit Neo4j only if traversal depth > 3 proves a bottleneck in production.

**BM25 in-process for chunker, Leiden for communities.** Both are optional deps at module level — patchable in tests, graceful fallback if unavailable. The system works (degraded) without either.

**Table stitching is the "you had to be there" insight.** Nobody writes a spec that says "the table on pages 3–4 is one logical table." The chunker has to detect it via header repeat + column count alignment. The insight is that this is an embedding problem masquerading as a parsing problem — without stitching, the retriever will never return a complete answer about that table regardless of query quality.

**OIDC everywhere in Terraform.** No static cloud credentials anywhere. Azure: `use_oidc=true` + federated credential. AWS: OIDC identity provider from EKS cluster → IAM role assumption for GitHub Actions. GCP (R7): Workload Identity Federation. Secret values set by operators post-apply — never via `terraform apply -var`.

---

## Interview Angles This Release Unlocks

**"What does Graph RAG solve that vector RAG can't?"**
Multi-hop questions and connections spread across documents. Vector retrieval finds chunks *similar* to the query. Graph retrieval finds chunks *connected* to the answer even when they don't resemble the query. The "blast radius" framing — surfacing connections nobody wrote down together.

**"How do you build a knowledge graph from documents?"**
LLM entity/relationship extraction over chunks → `GraphEntity` + `GraphRelationship` in Postgres → NetworkX DiGraph built in-memory → Leiden community detection for cluster-level retrieval. Entity dedup by `name_normalised` (lowercase + strip). Defensive JSON parser — malformed LLM output never crashes the pipeline.

**"OCR + multimodal for image PDFs?"**
Text isn't always text. Scanned PDFs need rasterisation + OCR; diagrams need vision-model captioning. `PDFImageChunker` handles both. The `CaptionService` is decoupled — the chunker extracts images, a separate HTTP call sends them to the LLM service for captioning. Failure is a warning, not a crash.

**"Table stitching?"**
Tables spanning page breaks produce orphaned rows with no column context. `TableStitchChunker` detects continuations via header repeat detection + column alignment tolerance, reconstructs the logical table before chunking. The key insight: this is an embedding problem masquerading as a parsing problem.

**"Graph RAG as additive, not replacement?"**
`GraphRetriever` has three modes: `classical` (pure dense, zero graph), `graph` (traversal only), `hybrid` (dense entry + graph expansion). The caller chooses per query. Same discipline as the R3 retrieval layer — match strategy to need, never force one path.

**"Terraform without static credentials?"**
Azure: `use_oidc = true` in provider block + federated credential in Azure AD. AWS: OIDC identity provider from EKS cluster issuer + IAM role assumption. No `client_secret`, no `AWS_ACCESS_KEY_ID` in any file. Secrets in Key Vault / Secrets Manager — operators set values post-apply.

---

## 7-Release Roadmap

| Release | Theme | Status |
|---------|-------|--------|
| R1 | Full Shell + Core Pipeline | ✅ Done |
| R2 | Advanced Chunking + Cloud Storage | ✅ Done |
| R3 | Retrieval Power + CI/CD | ✅ Done |
| **R4** | **Graph RAG + Advanced Document Types** | ✅ **Done** |
| R5 | Caching + Performance (Redis semantic cache) | 🔜 Next |
| R6 | Observability / LLMOps (evaluation metrics + tracing) | 🔜 Planned |
| R7 | Auth + Multi-tenancy + GCS + GCP activation | 🔜 Planned |

---

*Built by [Tamal Kundu](https://tamalkundu.com) · Kundu Corp · June 2026*
