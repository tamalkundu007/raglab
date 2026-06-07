# RAGLab R5 — Release Notes

**Version:** 0.5.0 · **Theme:** Self-Healing RAG + Cost Efficiency · **Date:** June 2026
**Builds on:** R1–R4

---

## Summary

Release 5 makes RAGLab self-aware. Three feedback loops — chunk quality, retrieval feedback, groundedness — run detect→score→remediate gates at every stage of the pipeline. Each gate is logged, toggleable, and explainable. Combined with an embedding cache that cuts re-ingestion cost, R5 answers the question every enterprise AI team eventually asks: "how do we stop this from confidently hallucinating?"

---

## Stats

| Metric | Value |
|--------|-------|
| New tests | 271 |
| Total tests passing | 1,558 |
| raglab-eval version | 0.1.0 (new package) |
| raglab-chunkers version | 0.5.0 |
| raglab-retrievers version | 0.5.0 |
| New eval gates | 3 (chunk quality, retrieval feedback, groundedness) |
| Infra required for tests | Zero |

---

## What Shipped

### Phase 1 — Embedding Cache (Redis)

`EmbeddingCache` in embedding-service. Key = SHA-256(`provider:model:text`) → `raglab:embed:{hex}`.

- Cache-aware `POST /embed` and `POST /embed/batch` — batch checks each text individually; only uncached texts hit the provider.
- `GET /embed/cache/stats` — exposes `hit_rate_pct`. This is the ROI metric: "80% cache hit rate = 80% fewer API calls on re-ingestion."
- Graceful degradation: Redis down → `enabled=False`, embedding still works.
- `DELETE /embed/cache/flush` for dev/test.
- Three new `EmbeddingSettings` fields: `embedding_cache_enabled`, `embedding_cache_redis_url`, `embedding_cache_ttl_seconds`.

---

### Phase 2 — raglab-eval v0.1.0

New internal package. Same pattern as raglab-chunkers and raglab-retrievers — independently testable, reusable in R6 observability.

**`ChunkQualityScorer`:**
- Four heuristic subscores: size, boundary integrity, information density, encoding health.
- Boilerplate patterns (page numbers, copyright lines, whitespace-only).
- Bigram repetition ratio for low-information detection.
- U+FFFD check for OCR/encoding garbage.
- `JudgeMode.HEURISTIC_FIRST`: LLM judge called only when heuristic score is in `[llm_trigger_low, llm_trigger_high]` — avoids paying for easy cases.
- `llm_caller` injectable — stateless, zero infra in tests.

**Shared models:** `EvalResult`, `ChunkQualityConfig`, `RetrievalHealConfig`, `GroundednessConfig`. `QuarantineStrategy`, `JudgeMode`, `GroundednessAction` enums.

---

### Phase 3 — Chunk Quality Remediation

`apply_quality_gate()` in pipeline-service. Sits between Step 2 (chunking) and Step 3 (embedding).

- **Accept:** `quality_score`, `quality_passed=True`, `quality_action='accepted'` injected into chunk metadata.
- **Flag (FLAG_ONLY):** chunk kept in pipeline with `quality_passed=False` and `quality_reason`. Indexer sees it but it's marked.
- **Exclude (EXCLUDE):** chunk dropped before embedding. Saves embedding cost.
- **All excluded → `PipelineError`:** explicit failure beats silent empty index.
- `chunk_quality_config: dict | None = None` in `PipelineSettings` — disabled by default.

---

### Phase 4 — Retrieval Feedback Loop

`RetrievalHealer` in raglab-eval. Detects weak retrievals and escalates.

- **Weak signals:** `result_count < min_results` OR `top_score < score_floor`.
- **Escalation:** iterates `escalation_order` (default: `["dense", "hybrid", "bm25"]`), skipping the initial strategy. Bounded by `max_healing_retries`.
- **Best tracking:** keeps the highest-scoring result across all attempts even if nothing fully heals.
- **Healed chunks tagged:** `healed=True`, `original_strategy`, `final_strategy` in metadata.
- Retriever errors on one strategy → continue to next (no crash).
- `retriever_fn` injectable — stateless, no HTTP client.

---

### Phase 5 — Answer Groundedness Check

`GroundednessChecker` in raglab-eval. Verifies the answer is supported by retrieved context.

- `_heuristic_groundedness()`: sentence-level word overlap. Per sentence: `sig_words(sentence) ∩ sig_words(context) / sig_words(sentence) >= 0.5` → grounded.
- Significant words: len > 4, stop-word list removed.
- `JudgeMode.HEURISTIC_FIRST`: LLM judge only in inconclusive band.
- `on_fail`: `re_prompt` / `re_retrieve` / `flag`.
- `GroundednessResult`: `grounded_claims`, `ungrounded_claims`, `groundedness_action`, `answer_preview`.

---

### Phase 6 — UI: Self-Healing Toggles + Healing Trace

**Control Panel — Self-Healing RAG section (R5 Active):**
- Chunk Quality Gate: toggle + min_quality_score slider + quarantine_strategy select.
- Retrieval Feedback Loop: toggle + score_floor slider + escalation_order select.
- Answer Groundedness Check: toggle + groundedness_threshold slider + on_fail action select.
- `updateHealConfig()` JS shows/hides sub-knobs reactively.
- "View Healing Trace" link → `/healing-trace`.

**Healing Trace page (`GET /healing-trace`):**
- Runs a query and displays detect→score→remediate gate decisions.
- Three gate entries: CHUNK QUALITY, RETRIEVAL FEEDBACK, GROUNDEDNESS.
- Coloured icons: pass (green), fail (red), heal (orange), skip (grey).
- Expandable detail panels: reason + structured key/value breakdown.
- Gate summary bar: query time, chunks returned, gates fired, heals applied.
- Chunk list in quality gate detail (first 5 with score badges).
- "Not a black box" — the subtitle says it explicitly.

---

### Phase 7 — Terraform Refinements

**Azure (`azure/scaling.tf`):** HPA (8 services, CPU+memory triggers, 5min scale-down stabilisation), PodDisruptionBudgets (50% min available), Redis HA (Sentinel, managed-premium PVC), dedicated embedding node pool (taint `workload=embedding:NoSchedule`), NetworkPolicy (default-deny + raglab-internal allow + 443 egress), namespace ResourceQuota.

**AWS (`aws/scaling.tf`):** ElastiCache Redis 7.1 (Multi-AZ, encrypted at rest + in transit, connection string in Secrets Manager), IRSA for embedding-service + pipeline-service (OIDC StringEquals scoped to serviceaccount), S3 docs bucket (versioning, AES256 SSE, all public access blocked), CloudWatch Container Insights.

---

## Design Decisions

**Every heal is logged, every gate is toggleable.** The cardinal rule. `enabled=False` is a complete bypass — no partial behaviour. Every `ChunkQualityResult`, `RetrievalHealResult`, `GroundednessResult` carries `score`, `passed`, `reason`, `action_taken`. Observable, not magic.

**HEURISTIC_FIRST over LLM_ALWAYS.** A chunk that scores 0.05 on heuristics doesn't need a gpt-4o-mini call to confirm it's garbage. A chunk scoring 0.85 doesn't need one either. LLM judge cost is concentrated on the genuinely ambiguous middle band [0.35, 0.65].

**raglab-eval as a separate package.** Consistent with chunkers/retrievers. Eval logic has no business living in pipeline-service source. R6 observability can import raglab-eval types directly for dashboards without circular deps.

**Embedding cache key scoped by provider + model + text.** Azure OpenAI and OpenAI share the same model names but different inference endpoints — same text produces different vectors. Provider-scoping prevents cross-provider cache collisions.

---

## Interview Angles Unlocked

**"How do you stop confident hallucinations?"** Groundedness gate post-generation. Per-sentence word overlap against retrieved context. On fail: re-prompt with stricter instruction, re-retrieve more context, or return a low-confidence flag — never emit a confident wrong answer. Failing loudly beats failing silently.

**"How do you detect bad chunks automatically?"** Four heuristic subscores at ingestion: size sanity, boundary integrity, information density (boilerplate + repetition), encoding health (U+FFFD, alpha ratio). LLM judge only for the ambiguous middle. Quarantine strategies: exclude, flag, re-chunk.

**"Self-healing without a black box?"** Every gate is an explicit detect→score→remediate decision with a typed result object, a logged reason, and a config toggle. The UI shows the trace — score, action, reason — per gate per query. Interviewers asking about self-healing are wary of hand-waving; this is the concrete answer.

**"Embedding cost control?"** SHA-256 keyed cache; hit_rate_pct reported on `/embed/cache/stats`. Concrete ROI metric: re-ingesting a corpus where 80% of chunks are unchanged saves 80% of embedding API calls. Graceful degradation when Redis is down.

**"LLM-as-judge tradeoffs?"** Cheap dedicated model (gpt-4o-mini), separate from generation model. HEURISTIC_FIRST mode avoids LLM calls for clear pass/fail cases. Known limits: LLM-judge reliability is ~0.7–0.85 correlation with human judges on groundedness tasks; heuristics are faster and cheaper but miss semantic nuance.

---

## 7-Release Roadmap

| Release | Theme | Status |
|---------|-------|--------|
| R1 | Full Shell + Core Pipeline | ✅ Done |
| R2 | Advanced Chunking + Cloud Storage | ✅ Done |
| R3 | Retrieval Power + CI/CD | ✅ Done |
| R4 | Graph RAG + Advanced Document Types | ✅ Done |
| **R5** | **Self-Healing RAG + Cost Efficiency** | ✅ **Done** |
| R6 | Observability / LLMOps | 🔜 Next |
| R7 | Auth + Multi-tenancy + GCS + GCP | 🔜 Planned |

---

*Built by [Tamal Kundu](https://tamalkundu.com) · Kundu Corp · June 2026*
