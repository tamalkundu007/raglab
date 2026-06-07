"""
raglab-eval shared models.

All eval gates produce an EvalResult subtype carrying:
  score         — float [0.0, 1.0]
  passed        — bool (score >= threshold)
  reason        — human-readable explanation of the score
  action_taken  — what the system did in response (logged, quarantined, re-tried, etc.)
  details       — arbitrary dict for structured tracing

Design principle: every heal is an explicit detect→score→remediate gate.
These models are the typed contract for that gate output.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Shared enums ───────────────────────────────────────────────────────────────

class QuarantineStrategy(str, Enum):
    EXCLUDE   = "exclude"    # remove from index entirely
    FLAG_ONLY = "flag_only"  # keep in index, mark as low-quality
    RE_CHUNK  = "re_chunk"   # attempt re-chunking with adjusted params


class JudgeMode(str, Enum):
    HEURISTIC_ONLY   = "heuristic_only"   # fast, free, no LLM call
    HEURISTIC_FIRST  = "heuristic_first"  # LLM only when heuristics inconclusive
    LLM_ALWAYS       = "llm_always"       # always use LLM judge


class GroundednessAction(str, Enum):
    RE_PROMPT    = "re_prompt"   # re-prompt with stricter grounding instruction
    RE_RETRIEVE  = "re_retrieve" # fetch more context and regenerate
    FLAG         = "flag"        # return answer with low-confidence flag


# ── Base eval result ──────────────────────────────────────────────────────────

class EvalResult(BaseModel):
    """
    Base class for all eval gate outputs.

    Every gate produces a score, a pass/fail decision, a reason string,
    and an action_taken field — making every heal decision fully auditable.
    """
    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    reason: str
    action_taken: str = "none"
    details: dict[str, Any] = Field(default_factory=dict)


# ── Chunk quality ─────────────────────────────────────────────────────────────

class ChunkQualityResult(EvalResult):
    """Output of ChunkQualityScorer for a single chunk."""
    chunk_id: str = ""
    doc_id: str = ""
    # Individual heuristic subscores
    size_score: float = 1.0
    boundary_score: float = 1.0
    information_score: float = 1.0
    encoding_score: float = 1.0
    llm_score: float | None = None  # None = LLM judge not called
    judge_mode_used: str = JudgeMode.HEURISTIC_ONLY


class ChunkQualityConfig(BaseModel):
    """Configuration for chunk quality evaluation."""
    enabled: bool = True
    min_quality_score: float = Field(default=0.4, ge=0.0, le=1.0)
    quarantine_strategy: QuarantineStrategy = QuarantineStrategy.FLAG_ONLY
    judge_mode: JudgeMode = JudgeMode.HEURISTIC_FIRST
    judge_model: str = "gpt-4o-mini"
    judge_provider: str = "azure_openai"
    # Heuristic thresholds
    min_tokens: int = 10
    max_tokens: int = 2000
    min_alpha_ratio: float = 0.3   # minimum fraction of alphabetic chars
    max_repetition_ratio: float = 0.5  # max fraction of repeated n-grams
    # LLM inconclusive range — only call LLM when heuristic score is in this band
    llm_trigger_low: float = 0.35
    llm_trigger_high: float = 0.65


# ── Retrieval feedback ────────────────────────────────────────────────────────

class RetrievalHealResult(EvalResult):
    """Output of RetrievalHealer for a single retrieval attempt."""
    query_text: str = ""
    original_strategy: str = ""
    final_strategy: str = ""
    retries: int = 0
    result_count: int = 0
    top_score: float | None = None


class RetrievalHealConfig(BaseModel):
    """Configuration for retrieval feedback loop."""
    enabled: bool = True
    score_floor: float = Field(default=0.3, ge=0.0, le=1.0)
    min_results: int = 1
    max_healing_retries: int = 2
    escalation_order: list[str] = Field(
        default_factory=lambda: ["dense", "hybrid", "bm25"]
    )


# ── Groundedness ──────────────────────────────────────────────────────────────

class GroundednessResult(EvalResult):
    """Output of GroundednessChecker for a generated answer."""
    answer_preview: str = ""
    context_chunks_used: int = 0
    grounded_claims: int = 0
    ungrounded_claims: int = 0
    groundedness_action: GroundednessAction = GroundednessAction.FLAG


class GroundednessConfig(BaseModel):
    """Configuration for answer groundedness checking."""
    enabled: bool = True
    groundedness_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    on_fail: GroundednessAction = GroundednessAction.FLAG
    judge_mode: JudgeMode = JudgeMode.HEURISTIC_FIRST
    judge_model: str = "gpt-4o-mini"
    judge_provider: str = "azure_openai"
    max_re_retrieve_attempts: int = 1
