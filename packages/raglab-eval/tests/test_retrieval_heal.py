"""
Unit tests for RetrievalHealer (R5 Phase 4).

Zero infrastructure — retriever_fn is injected as a pure Python callable.

Covers:
- RetrievalHealConfig defaults + custom
- _top_score: score/rrf_score/reranker_score, empty list, None metadata
- _is_weak: too few results, top score below floor, adequate results
- heal(): disabled → pass-through, action='skipped'
- heal(): strong initial → no escalation, action='none'
- heal(): weak initial, escalates to next strategy, healed=True
- heal(): healed chunk has healed=True + strategy metadata
- heal(): escalation stops when results become strong
- heal(): all escalations fail → returns best available
- heal(): retriever_fn error → continues to next strategy
- heal(): max_healing_retries respected
- heal(): initial strategy skipped in escalation order
- RetrievalHealResult fields populated correctly
- heal(): result_count in result
- heal(): top_score in result
"""

from __future__ import annotations

import uuid

import pytest

from raglab_common.models import ChunkModel
from raglab_eval.models import RetrievalHealConfig
from raglab_eval.retrieval_heal import RetrievalHealer


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk(score: float = 0.9, chunk_id: str | None = None) -> ChunkModel:
    return ChunkModel(
        chunk_id=chunk_id or str(uuid.uuid4()),
        doc_id="doc-001",
        text="Relevant content about RAG systems and retrieval.",
        chunk_index=0,
        token_count=8,
        metadata={"score": score},
    )


def make_retriever(results_by_strategy: dict[str, list[ChunkModel]]):
    """Returns a retriever_fn that returns preset results per strategy."""
    def retriever_fn(query: str, strategy: str, top_k: int) -> list[ChunkModel]:
        return results_by_strategy.get(strategy, [])
    return retriever_fn


GOOD_RESULTS = [make_chunk(0.85), make_chunk(0.75), make_chunk(0.65)]
WEAK_RESULTS = [make_chunk(0.15)]  # below score_floor=0.3
EMPTY_RESULTS: list[ChunkModel] = []


# ═══════════════════════════════════════════════════════════════════════════════
# RetrievalHealConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestRetrievalHealConfig:
    def test_defaults(self):
        cfg = RetrievalHealConfig()
        assert cfg.enabled is True
        assert cfg.score_floor == 0.3
        assert cfg.min_results == 1
        assert cfg.max_healing_retries == 2
        assert cfg.escalation_order == ["dense", "hybrid", "bm25"]

    def test_custom_config(self):
        cfg = RetrievalHealConfig(
            score_floor=0.5,
            max_healing_retries=3,
            escalation_order=["dense", "mmr"],
        )
        assert cfg.score_floor == 0.5
        assert cfg.max_healing_retries == 3


# ═══════════════════════════════════════════════════════════════════════════════
# _top_score
# ═══════════════════════════════════════════════════════════════════════════════

class TestTopScore:
    def test_score_field(self):
        chunks = [make_chunk(0.8), make_chunk(0.6)]
        assert RetrievalHealer._top_score(chunks) == pytest.approx(0.8)

    def test_rrf_score_field(self):
        chunk = ChunkModel(
            chunk_id="c1", doc_id="d", text="t", chunk_index=0, token_count=1,
            metadata={"rrf_score": 0.72},
        )
        assert RetrievalHealer._top_score([chunk]) == pytest.approx(0.72)

    def test_reranker_score_field(self):
        chunk = ChunkModel(
            chunk_id="c1", doc_id="d", text="t", chunk_index=0, token_count=1,
            metadata={"reranker_score": 0.55},
        )
        assert RetrievalHealer._top_score([chunk]) == pytest.approx(0.55)

    def test_empty_list_returns_none(self):
        assert RetrievalHealer._top_score([]) is None

    def test_no_score_metadata_returns_none(self):
        chunk = ChunkModel(
            chunk_id="c1", doc_id="d", text="t", chunk_index=0, token_count=1,
        )
        assert RetrievalHealer._top_score([chunk]) is None

    def test_returns_highest_score(self):
        chunks = [make_chunk(0.3), make_chunk(0.9), make_chunk(0.6)]
        assert RetrievalHealer._top_score(chunks) == pytest.approx(0.9)


# ═══════════════════════════════════════════════════════════════════════════════
# _is_weak
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsWeak:
    def setup_method(self):
        self.healer = RetrievalHealer(RetrievalHealConfig(
            score_floor=0.3, min_results=2
        ))

    def test_too_few_results_is_weak(self):
        weak, score, reason = self.healer._is_weak([make_chunk(0.9)])
        assert weak is True
        assert "Too few" in reason

    def test_below_score_floor_is_weak(self):
        results = [make_chunk(0.1), make_chunk(0.15)]
        weak, score, reason = self.healer._is_weak(results)
        assert weak is True
        assert "below floor" in reason

    def test_adequate_results_not_weak(self):
        results = [make_chunk(0.85), make_chunk(0.75)]
        weak, score, reason = self.healer._is_weak(results)
        assert weak is False
        assert score == pytest.approx(1.0)

    def test_empty_results_is_weak(self):
        weak, score, reason = self.healer._is_weak([])
        assert weak is True

    def test_weak_score_returns_fractional(self):
        results = [make_chunk(0.1), make_chunk(0.1)]  # both below 0.3
        weak, score, reason = self.healer._is_weak(results)
        assert weak is True
        assert 0.0 < score < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# RetrievalHealer.heal()
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealDisabled:
    def test_disabled_returns_initial_results(self):
        healer = RetrievalHealer(RetrievalHealConfig(enabled=False))
        retriever = make_retriever({})
        results, heal_result = healer.heal("query", GOOD_RESULTS, "dense", retriever)
        assert results == GOOD_RESULTS

    def test_disabled_action_skipped(self):
        healer = RetrievalHealer(RetrievalHealConfig(enabled=False))
        _, heal_result = healer.heal("q", GOOD_RESULTS, "dense", make_retriever({}))
        assert heal_result.action_taken == "skipped"
        assert heal_result.passed is True


class TestHealStrong:
    def test_strong_initial_no_escalation(self):
        healer = RetrievalHealer(RetrievalHealConfig(score_floor=0.3, min_results=1))
        called = []
        def retriever(q, s, k): called.append(s); return GOOD_RESULTS
        _, heal_result = healer.heal("query", GOOD_RESULTS, "dense", retriever)
        assert len(called) == 0  # no escalation

    def test_strong_initial_action_none(self):
        healer = RetrievalHealer()
        _, result = healer.heal("q", GOOD_RESULTS, "dense", make_retriever({}))
        assert result.action_taken == "none"

    def test_strong_initial_passed_true(self):
        healer = RetrievalHealer()
        _, result = healer.heal("q", GOOD_RESULTS, "dense", make_retriever({}))
        assert result.passed is True

    def test_strong_initial_result_count_correct(self):
        healer = RetrievalHealer()
        _, result = healer.heal("q", GOOD_RESULTS, "dense", make_retriever({}))
        assert result.result_count == len(GOOD_RESULTS)


class TestHealWeak:
    def test_weak_initial_triggers_escalation(self):
        healer = RetrievalHealer(RetrievalHealConfig(
            score_floor=0.3, min_results=1,
            escalation_order=["dense", "hybrid", "bm25"]
        ))
        retriever = make_retriever({"hybrid": GOOD_RESULTS})
        results, heal_result = healer.heal("q", WEAK_RESULTS, "dense", retriever)
        assert len(results) == len(GOOD_RESULTS)

    def test_healed_action_is_healed(self):
        healer = RetrievalHealer(RetrievalHealConfig(
            score_floor=0.3, escalation_order=["dense", "hybrid"]
        ))
        retriever = make_retriever({"hybrid": GOOD_RESULTS})
        _, result = healer.heal("q", WEAK_RESULTS, "dense", retriever)
        assert result.action_taken == "healed"

    def test_healed_chunks_tagged(self):
        healer = RetrievalHealer(RetrievalHealConfig(
            score_floor=0.3, escalation_order=["dense", "hybrid"]
        ))
        retriever = make_retriever({"hybrid": GOOD_RESULTS})
        results, _ = healer.heal("q", WEAK_RESULTS, "dense", retriever)
        assert all(c.metadata.get("healed") is True for c in results)

    def test_healed_chunks_have_strategy_metadata(self):
        healer = RetrievalHealer(RetrievalHealConfig(
            score_floor=0.3, escalation_order=["dense", "hybrid"]
        ))
        retriever = make_retriever({"hybrid": GOOD_RESULTS})
        results, _ = healer.heal("q", WEAK_RESULTS, "dense", retriever)
        assert all(c.metadata.get("original_strategy") == "dense" for c in results)
        assert all(c.metadata.get("final_strategy") == "hybrid" for c in results)

    def test_initial_strategy_skipped_in_escalation(self):
        """Dense is initial → should not be tried again in escalation."""
        tried = []
        def retriever(q, s, k): tried.append(s); return GOOD_RESULTS
        healer = RetrievalHealer(RetrievalHealConfig(
            score_floor=0.99,  # force weak even on good results
            escalation_order=["dense", "hybrid", "bm25"],
            max_healing_retries=2,
        ))
        healer.heal("q", WEAK_RESULTS, "dense", retriever)
        assert "dense" not in tried

    def test_max_retries_respected(self):
        tried = []
        def retriever(q, s, k): tried.append(s); return WEAK_RESULTS
        healer = RetrievalHealer(RetrievalHealConfig(
            score_floor=0.99,
            max_healing_retries=1,
            escalation_order=["dense", "hybrid", "bm25"],
        ))
        healer.heal("q", WEAK_RESULTS, "dense", retriever)
        assert len(tried) <= 1

    def test_all_escalations_fail_returns_best_available(self):
        healer = RetrievalHealer(RetrievalHealConfig(
            score_floor=0.99,  # nothing will pass
            escalation_order=["dense", "hybrid"],
        ))
        retriever = make_retriever({"hybrid": WEAK_RESULTS})
        results, result = healer.heal("q", WEAK_RESULTS, "dense", retriever)
        # Returns something even if nothing healed
        assert isinstance(results, list)

    def test_retriever_error_continues_to_next(self):
        tried = []
        def retriever(q, s, k):
            tried.append(s)
            if s == "hybrid":
                raise Exception("HTTP 503")
            return GOOD_RESULTS
        healer = RetrievalHealer(RetrievalHealConfig(
            score_floor=0.3,
            escalation_order=["dense", "hybrid", "bm25"],
        ))
        results, _ = healer.heal("q", WEAK_RESULTS, "dense", retriever)
        assert "hybrid" in tried

    def test_retries_field_incremented(self):
        healer = RetrievalHealer(RetrievalHealConfig(
            score_floor=0.3, escalation_order=["dense", "hybrid"]
        ))
        retriever = make_retriever({"hybrid": GOOD_RESULTS})
        _, result = healer.heal("q", WEAK_RESULTS, "dense", retriever)
        assert result.retries >= 1

    def test_original_strategy_preserved_in_result(self):
        healer = RetrievalHealer(RetrievalHealConfig(
            score_floor=0.3, escalation_order=["dense", "hybrid"]
        ))
        retriever = make_retriever({"hybrid": GOOD_RESULTS})
        _, result = healer.heal("q", WEAK_RESULTS, "dense", retriever)
        assert result.original_strategy == "dense"

    def test_final_strategy_updated_in_result(self):
        healer = RetrievalHealer(RetrievalHealConfig(
            score_floor=0.3, escalation_order=["dense", "hybrid"]
        ))
        retriever = make_retriever({"hybrid": GOOD_RESULTS})
        _, result = healer.heal("q", WEAK_RESULTS, "dense", retriever)
        assert result.final_strategy == "hybrid"

    def test_query_text_in_result(self):
        healer = RetrievalHealer()
        _, result = healer.heal("my question", GOOD_RESULTS, "dense", make_retriever({}))
        assert result.query_text == "my question"

    def test_top_score_in_result(self):
        healer = RetrievalHealer()
        _, result = healer.heal("q", GOOD_RESULTS, "dense", make_retriever({}))
        assert result.top_score is not None
        assert result.top_score == pytest.approx(0.85)
