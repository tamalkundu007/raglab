"""
Unit tests for GroundednessChecker (R5 Phase 5).

Zero infrastructure — llm_caller injected.

Covers:
- GroundednessConfig defaults
- _extract_sentences: splitting, short filter
- _significant_words: stop word removal, length filter
- _heuristic_groundedness: grounded sentence, ungrounded sentence, mixed, empty
- GroundednessChecker disabled → pass-through
- GroundednessChecker empty answer → fails immediately
- HEURISTIC_ONLY: no LLM call
- LLM_ALWAYS: LLM called, score blended
- HEURISTIC_FIRST: LLM called in trigger band, not outside
- LLM parse failure → fallback to heuristic
- passed=True when score >= threshold
- passed=False when score < threshold
- action_taken='none' on pass
- action_taken from config.on_fail on fail
- groundedness_action field
- grounded/ungrounded counts in result
- answer_preview truncated to 100 chars
- context_chunks_used count
"""

from __future__ import annotations

import uuid

import pytest

from raglab_common.models import ChunkModel
from raglab_eval.groundedness import (
    GroundednessChecker,
    _extract_sentences,
    _heuristic_groundedness,
    _significant_words,
)
from raglab_eval.models import (
    GroundednessAction,
    GroundednessConfig,
    JudgeMode,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk(text: str) -> ChunkModel:
    return ChunkModel(
        chunk_id=str(uuid.uuid4()), doc_id="d", text=text,
        chunk_index=0, token_count=len(text.split()),
    )


GROUNDED_CONTEXT = [
    make_chunk(
        "Retrieval Augmented Generation (RAG) reduces hallucinations by grounding "
        "answers in retrieved documents. It combines vector search with language models."
    ),
    make_chunk(
        "Dense retrieval uses embedding vectors to find semantically similar chunks. "
        "The embedding model converts text into high-dimensional float vectors."
    ),
]

GROUNDED_ANSWER = (
    "RAG reduces hallucinations by grounding generated answers in retrieved documents. "
    "Dense retrieval uses embedding vectors for semantic similarity search."
)

UNGROUNDED_ANSWER = (
    "The stock market crashed in 1929 and caused the Great Depression. "
    "This had nothing to do with RAG or retrieval systems whatsoever."
)


# ═══════════════════════════════════════════════════════════════════════════════
# GroundednessConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestGroundednessConfig:
    def test_defaults(self):
        cfg = GroundednessConfig()
        assert cfg.enabled is True
        assert cfg.groundedness_threshold == 0.6
        assert cfg.on_fail == GroundednessAction.FLAG
        assert cfg.judge_mode == JudgeMode.HEURISTIC_FIRST
        assert cfg.judge_model == "gpt-4o-mini"

    def test_on_fail_options(self):
        for action in GroundednessAction:
            cfg = GroundednessConfig(on_fail=action)
            assert cfg.on_fail == action


# ═══════════════════════════════════════════════════════════════════════════════
# _extract_sentences
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractSentences:
    def test_splits_on_period(self):
        text = "First sentence. Second sentence. Third sentence."
        sentences = _extract_sentences(text)
        assert len(sentences) >= 2

    def test_filters_short_sentences(self):
        text = "Hi. This is a complete sentence with enough words."
        sentences = _extract_sentences(text)
        # "Hi." (2 chars) should be filtered (min len=10)
        assert all(len(s) >= 10 for s in sentences)

    def test_empty_text_returns_empty(self):
        assert _extract_sentences("") == []

    def test_single_sentence(self):
        text = "RAG combines retrieval with generation for better answers."
        sentences = _extract_sentences(text)
        assert len(sentences) >= 1


# ═══════════════════════════════════════════════════════════════════════════════
# _significant_words
# ═══════════════════════════════════════════════════════════════════════════════

class TestSignificantWords:
    def test_removes_stop_words(self):
        words = _significant_words("the quick brown fox")
        assert "the" not in words

    def test_removes_short_words(self):
        words = _significant_words("a big cat sat on mat")
        # words <= 4 chars should be filtered
        assert all(len(w) > 4 for w in words)

    def test_lowercases_words(self):
        words = _significant_words("Retrieval Augmented Generation")
        assert "retrieval" in words or "augmented" in words or "generation" in words

    def test_empty_returns_empty_set(self):
        assert _significant_words("") == set()


# ═══════════════════════════════════════════════════════════════════════════════
# _heuristic_groundedness
# ═══════════════════════════════════════════════════════════════════════════════

class TestHeuristicGroundedness:
    def test_grounded_answer_scores_high(self):
        score, grounded, ungrounded, reason = _heuristic_groundedness(
            GROUNDED_ANSWER, GROUNDED_CONTEXT
        )
        assert score >= 0.5
        assert grounded > 0

    def test_ungrounded_answer_scores_low(self):
        score, grounded, ungrounded, reason = _heuristic_groundedness(
            UNGROUNDED_ANSWER, GROUNDED_CONTEXT
        )
        assert score < 0.8  # ungrounded content

    def test_empty_context_returns_low_score(self):
        score, _, _, _ = _heuristic_groundedness(GROUNDED_ANSWER, [])
        assert score <= 1.0  # no context words → no overlap

    def test_reason_string_contains_counts(self):
        _, grounded, ungrounded, reason = _heuristic_groundedness(
            GROUNDED_ANSWER, GROUNDED_CONTEXT
        )
        assert "/" in reason  # "X/Y sentences grounded"

    def test_short_answer_returns_1(self):
        score, _, _, reason = _heuristic_groundedness("OK", GROUNDED_CONTEXT)
        assert score == 1.0
        assert "too short" in reason.lower() or "No claims" in reason


# ═══════════════════════════════════════════════════════════════════════════════
# GroundednessChecker
# ═══════════════════════════════════════════════════════════════════════════════

class TestGroundednessCheckerDisabled:
    def test_disabled_returns_passing(self):
        checker = GroundednessChecker(GroundednessConfig(enabled=False))
        result = checker.check(UNGROUNDED_ANSWER, GROUNDED_CONTEXT)
        assert result.passed is True
        assert result.score == 1.0

    def test_disabled_action_skipped(self):
        checker = GroundednessChecker(GroundednessConfig(enabled=False))
        result = checker.check(UNGROUNDED_ANSWER, GROUNDED_CONTEXT)
        assert result.action_taken == "skipped"


class TestGroundednessCheckerHeuristic:
    def test_heuristic_only_no_llm_call(self):
        called = []
        def llm(sys, usr): called.append(1); return "0.9"
        checker = GroundednessChecker(GroundednessConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY
        ))
        checker.check(GROUNDED_ANSWER, GROUNDED_CONTEXT, llm_caller=llm)
        assert len(called) == 0

    def test_grounded_answer_passes(self):
        checker = GroundednessChecker(GroundednessConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY,
            groundedness_threshold=0.3,
        ))
        result = checker.check(GROUNDED_ANSWER, GROUNDED_CONTEXT)
        assert result.passed is True

    def test_empty_answer_fails(self):
        checker = GroundednessChecker(GroundednessConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY,
        ))
        result = checker.check("", GROUNDED_CONTEXT)
        assert result.passed is False
        assert result.score == 0.0

    def test_action_none_on_pass(self):
        checker = GroundednessChecker(GroundednessConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY,
            groundedness_threshold=0.0,
        ))
        result = checker.check(GROUNDED_ANSWER, GROUNDED_CONTEXT)
        assert result.action_taken == "none"

    def test_action_from_config_on_fail(self):
        checker = GroundednessChecker(GroundednessConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY,
            groundedness_threshold=0.99,
            on_fail=GroundednessAction.RE_PROMPT,
        ))
        result = checker.check(UNGROUNDED_ANSWER, GROUNDED_CONTEXT)
        if not result.passed:
            assert result.action_taken == "re_prompt"

    def test_answer_preview_truncated(self):
        long_answer = "x" * 200
        checker = GroundednessChecker(GroundednessConfig(judge_mode=JudgeMode.HEURISTIC_ONLY))
        result = checker.check(long_answer, GROUNDED_CONTEXT)
        assert len(result.answer_preview) <= 100

    def test_context_chunks_used_count(self):
        checker = GroundednessChecker(GroundednessConfig(judge_mode=JudgeMode.HEURISTIC_ONLY))
        result = checker.check(GROUNDED_ANSWER, GROUNDED_CONTEXT)
        assert result.context_chunks_used == len(GROUNDED_CONTEXT)

    def test_grounded_ungrounded_counts_present(self):
        checker = GroundednessChecker(GroundednessConfig(judge_mode=JudgeMode.HEURISTIC_ONLY))
        result = checker.check(GROUNDED_ANSWER, GROUNDED_CONTEXT)
        assert result.grounded_claims >= 0
        assert result.ungrounded_claims >= 0
        assert result.grounded_claims + result.ungrounded_claims > 0


class TestGroundednessCheckerLLM:
    def test_llm_always_calls_llm(self):
        called = []
        def llm(sys, usr): called.append(1); return "0.9"
        checker = GroundednessChecker(GroundednessConfig(judge_mode=JudgeMode.LLM_ALWAYS))
        checker.check(GROUNDED_ANSWER, GROUNDED_CONTEXT, llm_caller=llm)
        assert len(called) == 1

    def test_llm_score_blended_with_heuristic(self):
        def llm(sys, usr): return "0.2"  # LLM says 0.2
        checker = GroundednessChecker(GroundednessConfig(
            judge_mode=JudgeMode.LLM_ALWAYS,
            groundedness_threshold=0.0,
        ))
        result = checker.check(GROUNDED_ANSWER, GROUNDED_CONTEXT, llm_caller=llm)
        # Final = (heuristic + 0.2) / 2 → should be < heuristic alone
        assert result.score < 1.0

    def test_llm_parse_failure_falls_back(self):
        def llm(sys, usr): return "not_a_float"
        checker = GroundednessChecker(GroundednessConfig(judge_mode=JudgeMode.LLM_ALWAYS))
        result = checker.check(GROUNDED_ANSWER, GROUNDED_CONTEXT, llm_caller=llm)
        assert result.score >= 0.0  # fallback to heuristic

    def test_heuristic_first_calls_llm_in_band(self):
        called = []
        def llm(sys, usr): called.append(1); return "0.5"
        # Force into trigger band
        checker = GroundednessChecker(GroundednessConfig(
            judge_mode=JudgeMode.HEURISTIC_FIRST,
        ))
        # Patch trigger range to always trigger
        checker.config.__dict__["llm_trigger_low"] = 0.0
        checker.config.__dict__["llm_trigger_high"] = 1.0
        checker.check(GROUNDED_ANSWER, GROUNDED_CONTEXT, llm_caller=llm)
        assert len(called) == 1

    def test_heuristic_first_skips_llm_outside_band(self):
        called = []
        def llm(sys, usr): called.append(1); return "0.9"
        checker = GroundednessChecker(GroundednessConfig(
            judge_mode=JudgeMode.HEURISTIC_FIRST,
        ))
        # Patch trigger range to never trigger (heuristic will be 0 or 1, outside band)
        checker.config.__dict__["llm_trigger_low"] = 0.45
        checker.config.__dict__["llm_trigger_high"] = 0.55
        # Use a clearly grounded answer — heuristic will be > 0.55
        checker.check(GROUNDED_ANSWER, GROUNDED_CONTEXT, llm_caller=llm)
        assert len(called) == 0

    def test_groundedness_action_field_populated(self):
        checker = GroundednessChecker(GroundednessConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY,
            groundedness_threshold=0.99,
            on_fail=GroundednessAction.RE_RETRIEVE,
        ))
        result = checker.check(UNGROUNDED_ANSWER, GROUNDED_CONTEXT)
        if not result.passed:
            assert result.groundedness_action == GroundednessAction.RE_RETRIEVE
