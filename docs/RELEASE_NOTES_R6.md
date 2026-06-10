# RAGLab R6 — Release Notes

**Version:** 0.6.0 · **Theme:** Observability + Full Testing · **Date:** June 2026  
**Builds on:** R1–R5

---

## Summary

Release 6 turns structured logging into a coherent insight layer. Everything R1–R5 logged is now visible: distributed traces, chunk quality scores, retrieval strategy comparisons, token costs, cache hit rates, heal gate firings. Six native D3.js views, no external tool dependencies. Plus the full integration and E2E testing suites that verify every service boundary and user journey.

---

## Stats

| Metric | Value |
|--------|-------|
| New tests (R6) | 253 |
| Total tests passing | 1,811 |
| raglab-common version | 0.2.0 |
| raglab-eval version | 0.1.0 |
| observability-service version | 0.2.0 |
| New observability views | 6 |
| Integration tests | 55 |
| E2E test journeys | 12 |
| Infra required for tests | Zero |

---

## What Shipped

### Phase 1 — OpenTelemetry Instrumentation

`raglab-common/tracing.py` — the OTel backbone for the platform.

- `PostgresSpanExporter`: writes spans to `raglab_events` Postgres table. Buffer on Postgres failure, never raises, never blocks.
- `configure_tracing(service_name, postgres_dsn, otlp_endpoint, enabled)`: call once in lifespan. Registers TracerProvider with BatchSpanProcessor.
- `make_trace_middleware(service_name)`: FastAPI middleware — extracts or generates trace_id per request, injects into `request.state`, structlog contextvars, and `X-Trace-Id` response header.
- `trace_headers()`: returns `{"X-Trace-Id": tid}` for outbound httpx calls — trace ID propagates across service boundaries.
- `_NoopTracer` / `_NoopSpan`: safe no-ops when OTel SDK unavailable.
- `BaseServiceSettings`: three new fields — `tracing_enabled`, `tracing_postgres_dsn`, `tracing_otlp_endpoint`.

Wired into 10 services: api-gateway, ingestion, embedding, indexing, retrieval, llm, pipeline, storage, ui, graph. Pipeline runner injects `trace_headers()` into embed and index outbound calls.

---

### Phase 2 — observability-service v0.2.0

Activated from stub. Read-only throughout — queries `raglab_events`, never writes.

**DB queries (`db/queries.py`):**
- `list_recent_traces()`: GROUP BY trace_id, span_count, services[], has_error, total_duration_ms.
- `get_trace()`: all spans for a trace_id ordered by start_time_ms.
- `get_trace_timeline()`: BFS depth assignment, start_offset_ms relative to trace start. D3-ready.
- `get_service_stats()`: per-service total_spans, error_count, avg/p100 duration over N hours.

**Trace viewer (`/obs/viewer`):** D3.js v7 waterfall timeline. Left column: service tag + operation name (depth-indented for parent→child hierarchy). Bar width = span duration. 10-service colour map. Error spans red. Expandable span detail panel: attributes, events, span ID, parent span ID.

---

### Phase 3 — Chunk Inspector

**`/obs/inspector`** — answers "exactly how was this document chunked?"

Surfaces per-chunk: text, token count, chunk_index, quality_score, quality_passed, quality_action, quality_reason. Reads from `raglab_chunks` metadata (written by pipeline-service quality gate since R5).

Summary bar: total/accepted/flagged/excluded chunk counts + avg quality score per document. Filter by action (accepted/flagged/excluded). Sort by index, score ASC, score DESC. Expandable chunk body with full text and quality reason.

---

### Phase 4 — Retrieval Scorer

**`/obs/retrieval/scorer`** — answers "what did retrieval actually return and at what quality?"

- Recent queries table: strategy pill, result count, top score (colour-coded), duration, healed badge.
- Score distribution: D3 bar chart (5 buckets, red→green: 0.0–0.2 to 0.8–1.0).
- Healing stats bar: total queries, healed count, heal rate %, avg top score.

---

### Phase 5 — Token + Cost Dashboard

**`/obs/cost/dashboard`** — the ROI view.

- Token stats: total tokens, estimated cost (USD), prompt/completion split, request count.
- Daily trend: D3 bar chart (tokens) + line overlay (cost USD), 7 days.
- Cost by provider table: token count, cost, requests, avg latency per provider.
- Cache hit rate panel: visual progress bar + ROI message — "X% hit rate = X% fewer embedding API calls on re-ingestion."

---

### Phase 6 — Pipeline Health + Self-Healing Trace

**`/obs/health/dashboard`** — operational visibility.

- Jobs bar: total/successful/failed count, success rate %, avg duration.
- Self-Healing Gate Firings: gate-card per gate (pass-rate bar, fired count, avg score, action).
- Failed Jobs table: doc_id, filename, duration, error preview.
- Auto-refresh every 15 seconds.

---

### Phase 7 — Integration Testing Suite (55 tests)

`tests/integration/` — zero infra, all mocked.

**`test_gateway_pipeline.py`:** Gateway health, trace ID propagation (X-Trace-Id preserved and generated), IngestionMessage round-trip serialisation, pipeline runner calls embed with correct signature, idempotency key contract, DLQ (PipelineError typed, catchable).

**`test_cross_service.py`:** Ingest→embed→index data shapes, DenseRetriever ChunkModel contract, EmbeddingCache hit/miss with provider+model scoped keys, quality gate metadata injection, GroundednessChecker grounded/ungrounded, trace_id cross-boundary propagation.

---

### Phase 8 — E2E Testing Suite + CI (32 new + CI update)

`tests/e2e/test_r6_journeys.py` — 12 complete user journeys, zero infra.

| Journey | Tests |
|---------|-------|
| 1. Full ingest→embed→index→answer | 3 |
| 2–4. All 7 retrieval strategies create + have .retrieve() | 8 |
| 5. Quality gate: flag_only keeps pipeline running, junk scores low | 3 |
| 6. Retrieval escalation: weak→hybrid, stops on first strong result | 2 |
| 7. Groundedness: grounded passes, empty fails, LLM failure falls back | 3 |
| 8. Graph RAG: graph retriever + PDF/TableStitch chunkers exist | 4 |
| 9. Re-ingestion: same text → same chunk count (deterministic) | 2 |
| 10. Trace ID: unique per request, propagated from client | 3 |
| 11–12. Error paths: FileNotFoundError, all-excluded, unknown type | 4 |

**CI updates (`ci.yml`):** `raglab-eval` added to install matrix. OTel packages added to runtime deps. Test run split into unit / integration / E2E with separate JUnit XML artifacts. Docker build matrix extended: graph + observability added.

---

## Design Decisions

**Observability is read-only.** The observability-service queries `raglab_events`. It never writes to other service tables, never intercepts requests, never changes pipeline behaviour. This is a hard constraint enforced at the architecture level.

**Native views over external tools.** D3.js inside the platform, consistent dark-gold theme with Graph Explorer (R4) and Healing Trace (R5). Demo value: "here is our observability — in the platform, not bouncing to Grafana." The Trace Viewer, Chunk Inspector, Retrieval Scorer, Cost Dashboard, and Pipeline Health pages all live at `/obs/*` served from observability-service.

**OTel formalises what R1 started.** Trace IDs were propagated in structured logs from day one. OTel adds the formal span layer on top — without changing the existing log format. Structured logs and OTel spans coexist; the logs are the human-readable record, the spans are the machine-queryable record.

**Zero infra for all 1,811 tests.** Every test runs without a database, Redis, Qdrant, RabbitMQ, or any LLM API. PostgresSpanExporter buffers on failure. EmbeddingCache degrades gracefully. Pipeline runner accepts injected mocks. This keeps CI fast and removes infrastructure dependencies from the test signal.

---

## Interview Angles Unlocked

**"How do you debug a wrong answer?"** Trace the request across services with the Trace Viewer — every hop timed. Open Chunk Inspector to see the exact chunks retrieved and their quality scores. Open Retrieval Scorer to see the strategy, candidates, and scores. See which heal gates fired. "The answer was bad" becomes "retrieval scored 0.22, healed to hybrid at 0.71, but the source chunk was flagged (quality score 0.38, low information density)."

**"Observability vs monitoring?"** Monitoring tells you something broke. Observability lets you ask why without shipping new code. OTel traces + structured events + the ability to inspect any past request gives you that.

**"Why OpenTelemetry?"** Vendor-neutral. Same cloud-agnostic philosophy as the storage/vector/LLM abstractions throughout the platform. Instrument once, swap backends (Jaeger → Grafana Tempo → cloud-native) without touching service code.

**"Cost observability?"** Per-request token attribution and embedding cache hit rate on `/obs/cost/dashboard`. These are the numbers behind every ROI claim. "Our 73% cache hit rate saves us 73% of embedding API calls on re-ingestion" — traceable to a specific endpoint.

**"Integration vs E2E vs unit testing?"** Unit: logic in isolation (since R1). Integration: service contracts and the async/idempotency/DLQ behaviour between services. E2E: complete user journeys including self-healing paths. Each layer catches a different class of bug: unit catches logic errors, integration catches interface mismatches, E2E catches journey-level regressions.

---

## 7-Release Roadmap

| Release | Theme | Status |
|---------|-------|--------|
| R1 | Full Shell + Core Pipeline | ✅ Done |
| R2 | Advanced Chunking + Cloud Storage | ✅ Done |
| R3 | Retrieval Power + CI/CD | ✅ Done |
| R4 | Graph RAG + Advanced Document Types | ✅ Done |
| R5 | Self-Healing RAG + Cost Efficiency | ✅ Done |
| **R6** | **Observability + Full Testing** | ✅ **Done** |
| R7 | Auth + Multi-tenancy + GCS + GCP | 🔜 Next |

---

*Built by [Tamal Kundu](https://tamalkundu.com) · June 2026*
