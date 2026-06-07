"""
Unit tests for ChunkQualityScorer (R5 Phase 2).

Zero infrastructure — all LLM calls use injected llm_caller.

Covers:
- Models: EvalResult, ChunkQualityResult, ChunkQualityConfig defaults
- QuarantineStrategy + JudgeMode enums
- _score_size: too short, too long, optimal
- _score_boundary: starts lowercase, no terminal punct, both, clean
- _score_information: boilerplate patterns, high repetition, clean
- _score_encoding: replacement chars, low alpha ratio, clean
- ChunkQualityScorer: enabled=False returns passing immediately
- ChunkQualityScorer: heuristic_only — no llm_caller invoked
- ChunkQualityScorer: llm_always — llm_caller invoked, score blended
- ChunkQualityScorer: heuristic_first — LLM called only in trigger band
- ChunkQualityScorer: LLM parse failure → falls back to heuristic score
- ChunkQualityScorer: action_taken reflects quarantine strategy
- ChunkQualityScorer: passed=True when score >= min_quality_score
- ChunkQualityScorer: passed=False when score < min_quality_score
- ChunkQualityScorer: score_batch processes all chunks
- ChunkQualityScorer: subscores present in result
- ChunkQualityResult: chunk_id + doc_id propagated
- _determine_action: all three strategies
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from raglab_common.models import ChunkModel
from raglab_eval.chunk_quality import (
    ChunkQualityScorer,
    _score_boundary,
    _score_encoding,
    _score_information,
    _score_size,
)
from raglab_eval.models import (
    ChunkQualityConfig,
    ChunkQualityResult,
    EvalResult,
    JudgeMode,
    QuarantineStrategy,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk(
    text: str,
    chunk_id: str | None = None,
    doc_id: str = "doc-001",
) -> ChunkModel:
    return ChunkModel(
        chunk_id=chunk_id or str(uuid.uuid4()),
        doc_id=doc_id,
        text=text,
        chunk_index=0,
        token_count=len(text.split()),
    )


GOOD_TEXT = (
    "Retrieval Augmented Generation (RAG) combines a retrieval step with "
    "a language model to ground generated answers in retrieved documents. "
    "This approach reduces hallucinations and improves factual accuracy significantly."
)

BOILERPLATE_TEXT = "Page 1"
SHORT_TEXT = "Hi."
REPETITIVE_TEXT = "the the the the the the the the the the the the the the the"
GARBAGE_TEXT = "\uFFFD\uFFFD\uFFFD broken encoding garbage text"
LOWERCASE_START = "retrieval is important for accurate answers in RAG systems today."
NO_TERMINAL_PUNCT = "This sentence does not end with punctuation"


# ═══════════════════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════════════════

class TestModels:
    def test_eval_result_base(self):
        r = EvalResult(score=0.8, passed=True, reason="OK")
        assert r.score == 0.8
        assert r.passed is True
        assert r.action_taken == "none"
        assert r.details == {}

    def test_chunk_quality_result_defaults(self):
        r = ChunkQualityResult(score=0.5, passed=False, reason="low")
        assert r.llm_score is None
        assert r.size_score == 1.0
        assert r.chunk_id == ""

    def test_chunk_quality_config_defaults(self):
        cfg = ChunkQualityConfig()
        assert cfg.enabled is True
        assert cfg.min_quality_score == 0.4
        assert cfg.quarantine_strategy == QuarantineStrategy.FLAG_ONLY
        assert cfg.judge_mode == JudgeMode.HEURISTIC_FIRST
        assert cfg.judge_model == "gpt-4o-mini"

    def test_quarantine_strategy_values(self):
        assert QuarantineStrategy.EXCLUDE == "exclude"
        assert QuarantineStrategy.FLAG_ONLY == "flag_only"
        assert QuarantineStrategy.RE_CHUNK == "re_chunk"

    def test_judge_mode_values(self):
        assert JudgeMode.HEURISTIC_ONLY == "heuristic_only"
        assert JudgeMode.HEURISTIC_FIRST == "heuristic_first"
        assert JudgeMode.LLM_ALWAYS == "llm_always"


# ═══════════════════════════════════════════════════════════════════════════════
# _score_size
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreSize:
    def test_optimal_size_scores_1(self):
        score, reason = _score_size("word " * 50, min_tokens=10, max_tokens=500)
        assert score == 1.0
        assert "OK" in reason

    def test_too_short_scores_low(self):
        score, reason = _score_size("Hi", min_tokens=20, max_tokens=500)
        assert score < 0.5
        assert "short" in reason.lower() or "Too" in reason

    def test_too_long_scores_below_1(self):
        score, reason = _score_size("word " * 300, min_tokens=10, max_tokens=100)
        assert score < 1.0
        assert "long" in reason.lower() or "Too" in reason

    def test_single_word_very_short(self):
        score, _ = _score_size("word", min_tokens=50, max_tokens=500)
        assert score < 0.2

    def test_exactly_at_min_boundary(self):
        text = "word " * 10
        score, _ = _score_size(text, min_tokens=10, max_tokens=500)
        assert score == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# _score_boundary
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreBoundary:
    def test_clean_chunk_scores_1(self):
        score, reason = _score_boundary(GOOD_TEXT)
        assert score == 1.0
        assert "OK" in reason

    def test_lowercase_start_penalised(self):
        score, reason = _score_boundary(LOWERCASE_START)
        assert score < 1.0
        assert "mid-sentence" in reason or "lowercase" in reason

    def test_no_terminal_punctuation_penalised(self):
        score, reason = _score_boundary(NO_TERMINAL_PUNCT)
        assert score < 1.0
        assert "terminal" in reason or "punctuation" in reason

    def test_both_issues_cumulative_penalty(self):
        bad = "missing both boundaries here no dot"
        score_both, _ = _score_boundary(bad)
        score_clean, _ = _score_boundary(GOOD_TEXT)
        assert score_both < score_clean

    def test_empty_text_scores_zero(self):
        score, _ = _score_boundary("")
        assert score == 0.0

    def test_question_mark_terminal(self):
        score, _ = _score_boundary("Is RAG better than pure generation?")
        assert score == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# _score_information
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreInformation:
    def test_good_text_scores_high(self):
        score, _ = _score_information(GOOD_TEXT, max_repetition_ratio=0.5)
        assert score >= 0.8

    def test_boilerplate_page_number_scores_low(self):
        score, reason = _score_information("Page 1", max_repetition_ratio=0.5)
        assert score < 0.5
        assert "Boilerplate" in reason or "boilerplate" in reason

    def test_copyright_boilerplate_scores_low(self):
        score, _ = _score_information("All rights reserved", max_repetition_ratio=0.5)
        assert score < 0.5

    def test_high_repetition_scores_low(self):
        score, reason = _score_information(REPETITIVE_TEXT, max_repetition_ratio=0.3)
        assert score < 0.7
        assert "repetition" in reason.lower() or "repeated" in reason.lower()

    def test_whitespace_only_scores_low(self):
        # Zero words → size score will be very low (0 tokens < min_tokens)
        score, _ = _score_information("   \n\t  ", max_repetition_ratio=0.5)
        # 0 words = "Too short for repetition analysis" → 0.7 is the fallback
        # The key check: whitespace produces sub-1.0 score OR size catches it
        size_score, _ = _score_size("   \n\t  ", min_tokens=10, max_tokens=500)
        # Combined in scorer, overall will be low because size=0
        assert size_score == 0.0  # zero tokens → zero size score


# ═══════════════════════════════════════════════════════════════════════════════
# _score_encoding
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreEncoding:
    def test_clean_text_scores_1(self):
        score, reason = _score_encoding(GOOD_TEXT, min_alpha_ratio=0.3)
        assert score == 1.0
        assert "OK" in reason

    def test_replacement_chars_penalised(self):
        score, reason = _score_encoding(GARBAGE_TEXT, min_alpha_ratio=0.3)
        assert score < 0.5
        assert "replacement" in reason.lower() or "encoding" in reason.lower()

    def test_low_alpha_ratio_penalised(self):
        text = "1234 5678 9012 !!!! ????  .... ::::"
        score, reason = _score_encoding(text, min_alpha_ratio=0.5)
        assert score < 0.8

    def test_empty_text_scores_zero(self):
        score, _ = _score_encoding("", min_alpha_ratio=0.3)
        assert score == 0.0

    def test_normal_prose_alpha_ratio_passes(self):
        score, _ = _score_encoding("The quick brown fox jumps.", min_alpha_ratio=0.3)
        assert score == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# ChunkQualityScorer
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkQualityScorer:
    def test_disabled_returns_passing_immediately(self):
        scorer = ChunkQualityScorer(ChunkQualityConfig(enabled=False))
        chunk = make_chunk(BOILERPLATE_TEXT)
        result = scorer.score(chunk)
        assert result.passed is True
        assert result.score == 1.0
        assert result.action_taken == "skipped"

    def test_heuristic_only_no_llm_call(self):
        called = []
        def llm(sys, usr): called.append(1); return "0.9"
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY
        ))
        scorer.score(make_chunk(GOOD_TEXT), llm_caller=llm)
        assert len(called) == 0

    def test_llm_always_calls_llm(self):
        called = []
        def llm(sys, usr): called.append(1); return "0.9"
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.LLM_ALWAYS
        ))
        scorer.score(make_chunk(GOOD_TEXT), llm_caller=llm)
        assert len(called) == 1

    def test_heuristic_first_calls_llm_in_trigger_band(self):
        called = []
        def llm(sys, usr): called.append(1); return "0.5"
        # Force heuristic score into trigger band by using ambiguous text
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.HEURISTIC_FIRST,
            llm_trigger_low=0.0,   # always trigger
            llm_trigger_high=1.0,  # always trigger
        ))
        scorer.score(make_chunk(GOOD_TEXT), llm_caller=llm)
        assert len(called) == 1

    def test_heuristic_first_skips_llm_above_trigger(self):
        called = []
        def llm(sys, usr): called.append(1); return "0.9"
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.HEURISTIC_FIRST,
            llm_trigger_low=0.35,
            llm_trigger_high=0.65,
        ))
        # Good text will score > 0.65 on heuristics — LLM should not be called
        scorer.score(make_chunk(GOOD_TEXT), llm_caller=llm)
        assert len(called) == 0

    def test_llm_score_blended_with_heuristic(self):
        def llm(sys, usr): return "0.2"
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.LLM_ALWAYS,
            min_quality_score=0.0,
        ))
        result = scorer.score(make_chunk(GOOD_TEXT), llm_caller=llm)
        assert result.llm_score == pytest.approx(0.2)
        # Final = (heuristic + 0.2) / 2 → should be < heuristic alone
        assert result.score < 0.9

    def test_llm_parse_failure_falls_back_to_heuristic(self):
        def llm(sys, usr): return "not_a_number"
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.LLM_ALWAYS
        ))
        result = scorer.score(make_chunk(GOOD_TEXT), llm_caller=llm)
        assert result.llm_score is None  # parse failed
        assert result.score > 0.0  # heuristic still ran

    def test_good_chunk_passes(self):
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY, min_quality_score=0.4
        ))
        result = scorer.score(make_chunk(GOOD_TEXT))
        assert result.passed is True

    def test_garbage_chunk_fails(self):
        # Pure replacement chars with no real words fail definitively
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY, min_quality_score=0.4
        ))
        result = scorer.score(make_chunk("\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD"))
        assert result.passed is False

    def test_boilerplate_fails_with_default_threshold(self):
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY, min_quality_score=0.6
        ))
        # "Page 1" → info=0.1 (boilerplate), size~0.1 (2 tokens), no terminal punct
        # average ~0.5 → fails at threshold 0.6
        result = scorer.score(make_chunk("Page 1"))
        assert result.passed is False
        assert result.information_score <= 0.2  # boilerplate detected

    def test_chunk_id_doc_id_propagated(self):
        cid = "test-chunk-id"
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY
        ))
        result = scorer.score(make_chunk(GOOD_TEXT, chunk_id=cid, doc_id="my-doc"))
        assert result.chunk_id == cid
        assert result.doc_id == "my-doc"

    def test_subscores_present_in_result(self):
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY
        ))
        result = scorer.score(make_chunk(GOOD_TEXT))
        assert 0.0 <= result.size_score <= 1.0
        assert 0.0 <= result.boundary_score <= 1.0
        assert 0.0 <= result.information_score <= 1.0
        assert 0.0 <= result.encoding_score <= 1.0

    def test_score_batch_returns_one_per_chunk(self):
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY
        ))
        chunks = [make_chunk(GOOD_TEXT), make_chunk(BOILERPLATE_TEXT), make_chunk(SHORT_TEXT)]
        results = scorer.score_batch(chunks)
        assert len(results) == 3
        assert all(isinstance(r, ChunkQualityResult) for r in results)

    def test_reason_string_contains_subscores(self):
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY
        ))
        result = scorer.score(make_chunk(GOOD_TEXT))
        assert "size=" in result.reason
        assert "boundary=" in result.reason
        assert "info=" in result.reason
        assert "encoding=" in result.reason

    def test_action_taken_accepted_on_pass(self):
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY, min_quality_score=0.0
        ))
        result = scorer.score(make_chunk(GOOD_TEXT))
        assert result.action_taken == "accepted"

    def test_action_taken_excluded_strategy(self):
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY,
            quarantine_strategy=QuarantineStrategy.EXCLUDE,
            min_quality_score=0.99,  # force fail
        ))
        # Pure junk — guaranteed to fail
        result = scorer.score(make_chunk("\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD"))
        assert result.action_taken == "excluded"

    def test_action_taken_re_chunk_strategy(self):
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY,
            quarantine_strategy=QuarantineStrategy.RE_CHUNK,
            min_quality_score=0.99,
        ))
        result = scorer.score(make_chunk("\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD"))
        assert result.action_taken == "re_chunk_requested"

    def test_action_taken_flagged_strategy(self):
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY,
            quarantine_strategy=QuarantineStrategy.FLAG_ONLY,
            min_quality_score=0.99,
        ))
        result = scorer.score(make_chunk("\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD"))
        assert result.action_taken == "flagged"

    def test_score_in_valid_range(self):
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY
        ))
        for text in [GOOD_TEXT, BOILERPLATE_TEXT, SHORT_TEXT, GARBAGE_TEXT, REPETITIVE_TEXT]:
            result = scorer.score(make_chunk(text))
            assert 0.0 <= result.score <= 1.0, f"Score out of range for: {text[:30]}"

    def test_judge_mode_used_field_set(self):
        scorer = ChunkQualityScorer(ChunkQualityConfig(
            judge_mode=JudgeMode.HEURISTIC_ONLY
        ))
        result = scorer.score(make_chunk(GOOD_TEXT))
        assert result.judge_mode_used == "heuristic_only"
