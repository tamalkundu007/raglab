"""
GroundednessChecker — verify generated answers are grounded in retrieved context.

Detect→Score→Remediate pattern:
    Detect:  claims in answer not supported by retrieved chunks
    Score:   groundedness_ratio = grounded_claims / total_claims
    Remediate: re-prompt / re-retrieve / flag low-confidence

Two scoring strategies (JudgeMode):
    HEURISTIC_ONLY — fast n-gram overlap + keyword presence check.
                     No LLM call. Suitable as a fast pre-filter.
    HEURISTIC_FIRST — heuristic first; LLM judge only when score is in
                      inconclusive band [llm_trigger_low, llm_trigger_high].
    LLM_ALWAYS      — always use LLM-as-judge for groundedness.

Heuristic approach:
    Splits answer into sentences (crude: split on '. ').
    For each sentence: check if any significant word (len>4) appears in the
    concatenated context string. "Grounded" = overlap ratio >= 0.5.

LLM judge approach:
    Sends (answer, context_chunks) to llm-service.
    Prompts the judge model to output a float [0.0, 1.0] groundedness score.
    Cheap model (gpt-4o-mini) — cost isolated from generation model.

Observable: every check logged with score + reason + action.
Toggleable: config.enabled=False is a pass-through.
Stateless: llm_caller injectable.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from raglab_common.logging import get_logger
from raglab_common.models import ChunkModel

from raglab_eval.models import (
    GroundednessAction,
    GroundednessConfig,
    GroundednessResult,
    JudgeMode,
)

log = get_logger(__name__)

_GROUNDEDNESS_SYSTEM = (
    "You are a groundedness evaluator for a RAG system. "
    "Given an answer and retrieved context passages, rate how well the answer "
    "is supported by the context on a scale of 0.0 to 1.0. "
    "1.0 = every claim fully supported by context. "
    "0.0 = answer contains claims not present in context (hallucination risk). "
    "Output ONLY a single float between 0.0 and 1.0. Nothing else."
)

# Words to ignore in overlap check (stop words)
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "this", "that", "these", "those", "with",
    "from", "into", "for", "and", "or", "but", "not", "by", "at", "on",
    "in", "of", "to", "it", "its",
})


def _extract_sentences(text: str) -> list[str]:
    """Split text into sentence-level claims for groundedness checking."""
    # Split on '. ', '! ', '? ' and strip
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _significant_words(text: str) -> set[str]:
    """Extract significant words (len>4, not stop words) for overlap check."""
    words = re.findall(r"\b\w+\b", text.lower())
    return {w for w in words if len(w) > 4 and w not in _STOP_WORDS}


def _heuristic_groundedness(
    answer: str,
    context_chunks: list[ChunkModel],
) -> tuple[float, int, int, str]:
    """
    Fast heuristic groundedness check using word overlap.

    Returns:
        (score, grounded_count, total_count, reason)
    """
    context_text = " ".join(c.text for c in context_chunks).lower()
    context_words = _significant_words(context_text)

    sentences = _extract_sentences(answer)
    if not sentences:
        return 1.0, 0, 0, "No claims to verify (answer too short)"

    grounded = 0
    for sentence in sentences:
        sent_words = _significant_words(sentence)
        if not sent_words:
            grounded += 1  # no significant words = no ungrounded claim
            continue
        overlap = sent_words & context_words
        overlap_ratio = len(overlap) / len(sent_words)
        if overlap_ratio >= 0.5:
            grounded += 1

    total = len(sentences)
    score = round(grounded / total, 4) if total > 0 else 1.0
    reason = (
        f"Heuristic: {grounded}/{total} sentences grounded "
        f"(overlap ratio >= 0.5)"
    )
    return score, grounded, total - grounded, reason


class GroundednessChecker:
    """
    Checks whether a generated answer is grounded in retrieved context.

    Observable: every check logged with score + reason + action.
    Toggleable: config.enabled=False returns a passing result immediately.
    Stateless: llm_caller injectable for tests.
    """

    def __init__(self, config: GroundednessConfig | None = None) -> None:
        self.config = config or GroundednessConfig()

    def check(
        self,
        answer: str,
        context_chunks: list[ChunkModel],
        llm_caller: Callable[[str, str], str] | None = None,
    ) -> GroundednessResult:
        """
        Check whether the answer is grounded in context_chunks.

        Args:
            answer:          Generated answer text.
            context_chunks:  Retrieved chunks used for generation.
            llm_caller:      Optional callable(system, user) → str for LLM judge.

        Returns:
            GroundednessResult with score, passed, reason, and remediation action.
        """
        cfg = self.config

        if not cfg.enabled:
            return GroundednessResult(
                score=1.0, passed=True,
                reason="Groundedness check disabled",
                action_taken="skipped",
                answer_preview=answer[:100],
                context_chunks_used=len(context_chunks),
            )

        if not answer.strip():
            return GroundednessResult(
                score=0.0, passed=False,
                reason="Empty answer",
                action_taken=cfg.on_fail.value,
                answer_preview="",
                context_chunks_used=len(context_chunks),
            )

        # ── Heuristic scoring ──────────────────────────────────────────────────
        h_score, grounded, ungrounded, h_reason = _heuristic_groundedness(
            answer, context_chunks
        )

        # ── LLM judge ─────────────────────────────────────────────────────────
        llm_score: float | None = None

        if cfg.judge_mode == JudgeMode.LLM_ALWAYS:
            llm_score = self._call_llm_judge(answer, context_chunks, llm_caller)
            final_score = round((h_score + llm_score) / 2.0, 4) if llm_score is not None else h_score

        elif cfg.judge_mode == JudgeMode.HEURISTIC_FIRST:
            trigger_low = getattr(cfg, "llm_trigger_low", 0.35)
            trigger_high = getattr(cfg, "llm_trigger_high", 0.75)
            if trigger_low <= h_score <= trigger_high:
                llm_score = self._call_llm_judge(answer, context_chunks, llm_caller)
                final_score = round((h_score + llm_score) / 2.0, 4) if llm_score is not None else h_score
            else:
                final_score = h_score
        else:
            final_score = h_score

        passed = final_score >= cfg.groundedness_threshold
        action = cfg.on_fail if not passed else GroundednessAction.FLAG
        action_str = action.value if not passed else "none"

        reason = h_reason
        if llm_score is not None:
            reason += f" | LLM judge={llm_score:.3f} | final={final_score:.3f}"

        log.info(
            "groundedness.checked",
            score=final_score,
            passed=passed,
            action=action_str,
            grounded=grounded,
            ungrounded=ungrounded,
        )

        return GroundednessResult(
            score=final_score,
            passed=passed,
            reason=reason,
            action_taken=action_str,
            answer_preview=answer[:100],
            context_chunks_used=len(context_chunks),
            grounded_claims=grounded,
            ungrounded_claims=ungrounded,
            groundedness_action=action if not passed else GroundednessAction.FLAG,
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call_llm_judge(
        self,
        answer: str,
        context_chunks: list[ChunkModel],
        llm_caller: Callable[[str, str], str] | None,
    ) -> float | None:
        """Call LLM judge for groundedness score. Returns float or None on failure."""
        context_text = "\n---\n".join(c.text[:300] for c in context_chunks[:5])
        user_prompt = (
            f"Answer:\n{answer[:800]}\n\n"
            f"Context passages:\n{context_text}"
        )

        raw = ""
        try:
            if llm_caller is not None:
                raw = llm_caller(_GROUNDEDNESS_SYSTEM, user_prompt)
            else:
                raw = self._http_llm_judge(user_prompt)
            score = float(raw.strip())
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            log.warning("groundedness.llm_parse_failed", raw=str(raw)[:50])
            return None
        except Exception as exc:
            log.warning("groundedness.llm_failed", error=str(exc))
            return None

    def _http_llm_judge(self, user_prompt: str) -> str:
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
                    "system_prompt": _GROUNDEDNESS_SYSTEM,
                    "max_tokens": 8,
                    "temperature": 0.0,
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json().get("answer", "")
        except Exception:
            return ""
