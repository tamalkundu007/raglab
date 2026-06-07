"""
ChunkQualityScorer — automated chunk quality evaluation at ingestion time.

Scoring pipeline:
    1. Size score      — is the chunk within min/max token bounds?
    2. Boundary score  — does the chunk start/end at a sentence boundary?
    3. Information score — is the chunk information-dense (not boilerplate/empty)?
    4. Encoding score  — is the chunk valid UTF-8 with expected alpha ratio?
    5. LLM judge       — (optional) fast judge model for inconclusive heuristic scores.

Final score = weighted mean of heuristic subscores (equal weights by default).
If JudgeMode.HEURISTIC_FIRST: LLM judge called only when final heuristic score
is in [llm_trigger_low, llm_trigger_high] — avoids paying for easy cases.

Observable: every score + reason is returned in ChunkQualityResult.
Toggleable: enabled=False returns a passing result immediately.

Design:
    Stateless — no DB. llm_caller injectable for tests (same pattern as extractor).
    Production: HTTP POST to llm-service /generate.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Callable

from raglab_common.logging import get_logger
from raglab_common.models import ChunkModel

from raglab_eval.models import (
    ChunkQualityConfig,
    ChunkQualityResult,
    JudgeMode,
    QuarantineStrategy,
)

log = get_logger(__name__)

# Boilerplate patterns — common low-information fragments
_BOILERPLATE_PATTERNS = [
    re.compile(r"^(page|slide|figure|table|exhibit)\s*\d+\s*$", re.IGNORECASE),
    re.compile(r"^(confidential|proprietary|all rights reserved|copyright).*$", re.IGNORECASE),
    re.compile(r"^[\s\W]+$"),  # whitespace/punctuation only
    re.compile(r"^(\w)\1{5,}$"),  # single character repeated (e.g. "aaaaaaa")
]

_LLM_JUDGE_SYSTEM = (
    "You are a document chunk quality evaluator. "
    "Rate the chunk's usefulness for a RAG retrieval system on a scale of 0.0 to 1.0. "
    "A score of 1.0 means the chunk is clear, complete, and information-rich. "
    "A score of 0.0 means it is garbage, empty, or meaningless. "
    "Output ONLY a single float between 0.0 and 1.0. Nothing else."
)


def _count_tokens_approx(text: str) -> int:
    """Approximate token count: split on whitespace."""
    return len(text.split())


def _score_size(text: str, min_tokens: int, max_tokens: int) -> tuple[float, str]:
    """Score chunk size. Returns (score, reason)."""
    n = _count_tokens_approx(text)
    if n < min_tokens:
        ratio = n / max(min_tokens, 1)
        return round(ratio * 0.5, 3), f"Too short: {n} tokens (min {min_tokens})"
    if n > max_tokens:
        ratio = max_tokens / n
        return round(ratio, 3), f"Too long: {n} tokens (max {max_tokens})"
    return 1.0, f"Size OK: {n} tokens"


def _score_boundary(text: str) -> tuple[float, str]:
    """
    Score sentence boundary integrity.

    Heuristic: chunks starting mid-sentence (lowercase first word after stripping
    leading whitespace/punctuation) or ending without terminal punctuation score lower.
    """
    stripped = text.strip()
    if not stripped:
        return 0.0, "Empty text"

    score = 1.0
    reasons = []

    # Penalise starting mid-sentence (starts with lowercase)
    first_char = stripped[0]
    if first_char.islower():
        score -= 0.3
        reasons.append("starts mid-sentence (lowercase)")

    # Penalise ending without terminal punctuation
    last_char = stripped[-1]
    if last_char not in ".!?\"'":
        score -= 0.2
        reasons.append("no terminal punctuation")

    score = max(0.0, round(score, 3))
    reason = "Boundary OK" if not reasons else "Boundary issues: " + "; ".join(reasons)
    return score, reason


def _score_information(text: str, max_repetition_ratio: float) -> tuple[float, str]:
    """
    Score information density.

    Checks:
    - Boilerplate pattern match (immediate low score)
    - N-gram repetition ratio (high repetition = low information)
    """
    stripped = text.strip()

    # Boilerplate patterns
    for pattern in _BOILERPLATE_PATTERNS:
        if pattern.match(stripped):
            return 0.1, f"Boilerplate match: {pattern.pattern[:40]}"

    # Bigram repetition check
    words = stripped.lower().split()
    if len(words) < 4:
        return 0.7, "Too short for repetition analysis"

    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    unique_bigrams = len(set(bigrams))
    total_bigrams = len(bigrams)
    repetition_ratio = 1.0 - (unique_bigrams / total_bigrams)

    if repetition_ratio > max_repetition_ratio:
        return round(1.0 - repetition_ratio, 3), (
            f"High repetition: {repetition_ratio:.1%} bigrams repeated"
        )

    return 1.0, f"Information density OK (repetition={repetition_ratio:.1%})"


def _score_encoding(text: str, min_alpha_ratio: float) -> tuple[float, str]:
    """
    Score encoding health.

    Checks:
    - Minimum alphabetic character ratio (OCR garbage has low alpha ratio)
    - Presence of replacement characters (U+FFFD — encoding errors)
    - Mojibake detection (common garbled UTF-8 sequences)
    """
    if not text.strip():
        return 0.0, "Empty"

    # Replacement character check
    if "\uFFFD" in text:
        return 0.2, "Contains UTF-8 replacement characters (encoding error)"

    # Alphabetic ratio
    alpha_count = sum(1 for c in text if c.isalpha())
    total = len(text)
    alpha_ratio = alpha_count / total if total > 0 else 0.0

    if alpha_ratio < min_alpha_ratio:
        return round(alpha_ratio / min_alpha_ratio * 0.6, 3), (
            f"Low alpha ratio: {alpha_ratio:.1%} (min {min_alpha_ratio:.1%})"
        )

    return 1.0, f"Encoding OK (alpha ratio={alpha_ratio:.1%})"


class ChunkQualityScorer:
    """
    Scores chunk quality using heuristics and optional LLM judge.

    Observable: every decision logged with score + reason.
    Toggleable: config.enabled=False bypasses all scoring.
    Stateless: llm_caller injectable, no DB dependency.
    """

    def __init__(self, config: ChunkQualityConfig | None = None) -> None:
        self.config = config or ChunkQualityConfig()

    def score(
        self,
        chunk: ChunkModel,
        llm_caller: Callable[[str, str], str] | None = None,
    ) -> ChunkQualityResult:
        """
        Score a single ChunkModel.

        Args:
            chunk:      The ChunkModel to evaluate.
            llm_caller: Optional callable(system, user) → str for LLM judge.
                        If None and judge_mode requires LLM, HTTP call to llm-service.

        Returns:
            ChunkQualityResult with score, passed, reason, and subscores.
        """
        cfg = self.config

        # Master toggle
        if not cfg.enabled:
            return ChunkQualityResult(
                score=1.0, passed=True,
                reason="Quality scoring disabled",
                action_taken="skipped",
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
            )

        text = chunk.text or ""

        # ── Heuristic subscores ────────────────────────────────────────────────
        size_score, size_reason = _score_size(text, cfg.min_tokens, cfg.max_tokens)
        boundary_score, boundary_reason = _score_boundary(text)
        info_score, info_reason = _score_information(text, cfg.max_repetition_ratio)
        encoding_score, enc_reason = _score_encoding(text, cfg.min_alpha_ratio)

        # Equal-weight mean
        heuristic_score = round(
            (size_score + boundary_score + info_score + encoding_score) / 4.0, 4
        )

        # ── LLM judge ─────────────────────────────────────────────────────────
        llm_score: float | None = None
        judge_mode_used = cfg.judge_mode.value

        if cfg.judge_mode == JudgeMode.HEURISTIC_ONLY:
            final_score = heuristic_score

        elif cfg.judge_mode == JudgeMode.LLM_ALWAYS:
            llm_score = self._call_llm_judge(text, llm_caller)
            final_score = round((heuristic_score + llm_score) / 2.0, 4) if llm_score is not None else heuristic_score

        else:  # HEURISTIC_FIRST: LLM only when inconclusive
            if cfg.llm_trigger_low <= heuristic_score <= cfg.llm_trigger_high:
                llm_score = self._call_llm_judge(text, llm_caller)
                if llm_score is not None:
                    final_score = round((heuristic_score + llm_score) / 2.0, 4)
                else:
                    final_score = heuristic_score
            else:
                final_score = heuristic_score

        passed = final_score >= cfg.min_quality_score
        action = self._determine_action(passed, cfg)

        # Build composite reason
        reasons = [
            f"size={size_score:.2f}({size_reason})",
            f"boundary={boundary_score:.2f}({boundary_reason})",
            f"info={info_score:.2f}({info_reason})",
            f"encoding={encoding_score:.2f}({enc_reason})",
        ]
        if llm_score is not None:
            reasons.append(f"llm_judge={llm_score:.2f}")
        reason = f"final={final_score:.3f} | " + " | ".join(reasons)

        log.info(
            "chunk_quality.scored",
            chunk_id=chunk.chunk_id,
            score=final_score,
            passed=passed,
            action=action,
        )

        return ChunkQualityResult(
            score=final_score,
            passed=passed,
            reason=reason,
            action_taken=action,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            size_score=size_score,
            boundary_score=boundary_score,
            information_score=info_score,
            encoding_score=encoding_score,
            llm_score=llm_score,
            judge_mode_used=judge_mode_used,
        )

    def score_batch(
        self,
        chunks: list[ChunkModel],
        llm_caller: Callable[[str, str], str] | None = None,
    ) -> list[ChunkQualityResult]:
        """Score a list of chunks. Returns one result per chunk."""
        return [self.score(chunk, llm_caller=llm_caller) for chunk in chunks]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call_llm_judge(
        self,
        text: str,
        llm_caller: Callable[[str, str], str] | None,
    ) -> float | None:
        """
        Call LLM judge for a quality score.

        Returns float [0.0, 1.0] or None on failure.
        """
        user_prompt = f"Rate the quality of this chunk for RAG retrieval:\n\n{text[:1000]}"

        raw: str = ""
        try:
            if llm_caller is not None:
                raw = llm_caller(_LLM_JUDGE_SYSTEM, user_prompt)
            else:
                raw = self._http_llm_judge(user_prompt)

            score = float(raw.strip())
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            log.warning("chunk_quality.llm_judge_parse_failed", raw=str(raw)[:50])
            return None
        except Exception as exc:
            log.warning("chunk_quality.llm_judge_failed", error=str(exc))
            return None

    def _http_llm_judge(self, user_prompt: str) -> str:
        """HTTP call to llm-service /generate for LLM judge."""
        try:
            from raglab_chunkers.caption_service import _requests
            if _requests is None:
                return ""
            resp = _requests.post(
                "http://llm:8005/generate",
                json={
                    "query": user_prompt,
                    "chunks": [],
                    "provider": self.config.judge_provider,
                    "system_prompt": _LLM_JUDGE_SYSTEM,
                    "max_tokens": 8,
                    "temperature": 0.0,
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json().get("answer", "")
        except Exception:
            return ""

    @staticmethod
    def _determine_action(passed: bool, cfg: ChunkQualityConfig) -> str:
        """Determine what action to take based on pass/fail and quarantine strategy."""
        if passed:
            return "accepted"
        strategy = cfg.quarantine_strategy
        if strategy == QuarantineStrategy.EXCLUDE:
            return "excluded"
        if strategy == QuarantineStrategy.FLAG_ONLY:
            return "flagged"
        if strategy == QuarantineStrategy.RE_CHUNK:
            return "re_chunk_requested"
        return "flagged"
