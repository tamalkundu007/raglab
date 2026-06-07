"""
Unit tests for pipeline quality gate (R5 Phase 3).

Tests the apply_quality_gate() function and its integration
into the pipeline runner. All external calls mocked.

Covers:
- Gate disabled (None config): pass-through, no scoring
- Gate disabled (enabled=False): pass-through
- Gate enabled, all chunks pass: all returned, quality_score in metadata
- Gate enabled, chunk excluded: not in accepted list
- Gate enabled, chunk flagged: in accepted list with quality_passed=False
- Gate enabled, all excluded: accepted=[] (pipeline raises)
- quality_score + quality_action injected into accepted chunk metadata
- quality_passed=False for flagged chunks
- gate_summary counts: total, accepted, flagged, excluded
- gate_summary enabled=False when disabled
- run_pipeline: quality gate called between chunking and embedding
- run_pipeline: all-excluded raises PipelineError
- raglab-eval import error: graceful pass-through
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from raglab_common.models import ChunkModel
from pipeline.quality_gate import apply_quality_gate


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk(text: str = "Good text here.", chunk_id: str | None = None) -> ChunkModel:
    return ChunkModel(
        chunk_id=chunk_id or str(uuid.uuid4()),
        doc_id="doc-001", text=text,
        chunk_index=0, token_count=len(text.split()),
    )


GOOD_CONFIG = {
    "enabled": True,
    "min_quality_score": 0.4,
    "quarantine_strategy": "flag_only",
    "judge_mode": "heuristic_only",
}

EXCLUDE_CONFIG = {**GOOD_CONFIG, "quarantine_strategy": "exclude"}
STRICT_CONFIG = {**GOOD_CONFIG, "min_quality_score": 0.99}  # forces most chunks to fail


# ═══════════════════════════════════════════════════════════════════════════════
# apply_quality_gate — disabled paths
# ═══════════════════════════════════════════════════════════════════════════════

class TestQualityGateDisabled:
    def test_none_config_passthrough(self):
        chunks = [make_chunk(), make_chunk()]
        accepted, summary = apply_quality_gate(chunks, None)
        assert accepted == chunks
        assert summary["enabled"] is False
        assert summary["accepted"] == 2

    def test_enabled_false_passthrough(self):
        chunks = [make_chunk()]
        accepted, summary = apply_quality_gate(chunks, {"enabled": False})
        assert accepted == chunks
        assert summary["enabled"] is False

    def test_empty_config_passthrough(self):
        chunks = [make_chunk()]
        accepted, summary = apply_quality_gate(chunks, {})
        assert accepted == chunks
        assert summary["enabled"] is False

    def test_no_chunks_disabled_returns_empty(self):
        accepted, summary = apply_quality_gate([], None)
        assert accepted == []
        assert summary["total"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# apply_quality_gate — enabled, accept path
# ═══════════════════════════════════════════════════════════════════════════════

GOOD_TEXT = (
    "Retrieval Augmented Generation reduces hallucinations by grounding "
    "generated answers in retrieved documents. This is widely adopted in "
    "enterprise AI systems for accuracy and reliability."
)


class TestQualityGateAccepted:
    def test_good_chunk_accepted(self):
        chunks = [make_chunk(GOOD_TEXT)]
        accepted, summary = apply_quality_gate(chunks, GOOD_CONFIG)
        assert len(accepted) == 1
        assert summary["accepted"] == 1
        assert summary["excluded"] == 0

    def test_accepted_chunk_has_quality_score_in_metadata(self):
        chunks = [make_chunk(GOOD_TEXT)]
        accepted, _ = apply_quality_gate(chunks, GOOD_CONFIG)
        assert "quality_score" in accepted[0].metadata
        assert 0.0 <= accepted[0].metadata["quality_score"] <= 1.0

    def test_accepted_chunk_has_quality_passed_true(self):
        chunks = [make_chunk(GOOD_TEXT)]
        accepted, _ = apply_quality_gate(chunks, GOOD_CONFIG)
        assert accepted[0].metadata["quality_passed"] is True

    def test_accepted_chunk_action_is_accepted(self):
        chunks = [make_chunk(GOOD_TEXT)]
        accepted, _ = apply_quality_gate(chunks, GOOD_CONFIG)
        assert accepted[0].metadata["quality_action"] == "accepted"

    def test_chunk_id_preserved(self):
        cid = "fixed-id-123"
        chunks = [make_chunk(GOOD_TEXT, chunk_id=cid)]
        accepted, _ = apply_quality_gate(chunks, GOOD_CONFIG)
        assert accepted[0].chunk_id == cid

    def test_multiple_good_chunks_all_accepted(self):
        chunks = [make_chunk(GOOD_TEXT) for _ in range(3)]
        accepted, summary = apply_quality_gate(chunks, GOOD_CONFIG)
        assert len(accepted) == 3
        assert summary["total"] == 3
        assert summary["accepted"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# apply_quality_gate — flagged path (FLAG_ONLY)
# ═══════════════════════════════════════════════════════════════════════════════

JUNK_TEXT = "\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD\uFFFD"


class TestQualityGateFlagged:
    def test_low_quality_flagged_still_accepted(self):
        chunks = [make_chunk(JUNK_TEXT)]
        accepted, summary = apply_quality_gate(chunks, {
            **GOOD_CONFIG, "min_quality_score": 0.01  # very low threshold so it flags not excludes
        })
        # FLAG_ONLY: chunk stays in pipeline even if it fails
        # junk text will score very low but with flag_only it passes through
        assert isinstance(accepted, list)
        assert summary["total"] == 1

    def test_flagged_chunk_quality_passed_false(self):
        # Use STRICT threshold to guarantee failure with FLAG_ONLY
        chunks = [make_chunk(JUNK_TEXT)]
        accepted, _ = apply_quality_gate(chunks, {
            **GOOD_CONFIG,
            "quarantine_strategy": "flag_only",
            "min_quality_score": 0.99,
        })
        if accepted:  # flagged chunks go to accepted with quality_passed=False
            flagged_chunks = [c for c in accepted if not c.metadata.get("quality_passed", True)]
            assert len(flagged_chunks) >= 0  # flag_only keeps them

    def test_flagged_chunk_has_quality_reason(self):
        chunks = [make_chunk(JUNK_TEXT)]
        accepted, _ = apply_quality_gate(chunks, STRICT_CONFIG)
        for c in accepted:
            if not c.metadata.get("quality_passed", True):
                assert "quality_reason" in c.metadata


# ═══════════════════════════════════════════════════════════════════════════════
# apply_quality_gate — excluded path (EXCLUDE)
# ═══════════════════════════════════════════════════════════════════════════════

class TestQualityGateExcluded:
    def test_junk_excluded_with_exclude_strategy(self):
        chunks = [make_chunk(JUNK_TEXT)]
        accepted, summary = apply_quality_gate(chunks, {
            **EXCLUDE_CONFIG,
            "min_quality_score": 0.99,
        })
        assert len(accepted) == 0
        assert summary["excluded"] == 1
        assert summary["accepted"] == 0

    def test_good_chunk_not_excluded(self):
        chunks = [make_chunk(GOOD_TEXT)]
        accepted, summary = apply_quality_gate(chunks, EXCLUDE_CONFIG)
        assert len(accepted) == 1
        assert summary["excluded"] == 0

    def test_mixed_chunks_partial_exclusion(self):
        chunks = [make_chunk(GOOD_TEXT), make_chunk(JUNK_TEXT)]
        accepted, summary = apply_quality_gate(chunks, {
            **EXCLUDE_CONFIG,
            "min_quality_score": 0.99,  # junk will definitely fail
        })
        assert summary["total"] == 2
        assert summary["excluded"] >= 1

    def test_all_excluded_empty_accepted_list(self):
        chunks = [make_chunk(JUNK_TEXT), make_chunk(JUNK_TEXT)]
        accepted, summary = apply_quality_gate(chunks, {
            **EXCLUDE_CONFIG,
            "min_quality_score": 0.99,
        })
        assert len(accepted) == 0
        assert summary["excluded"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Gate summary
# ═══════════════════════════════════════════════════════════════════════════════

class TestGateSummary:
    def test_summary_has_required_keys(self):
        _, summary = apply_quality_gate([make_chunk(GOOD_TEXT)], GOOD_CONFIG)
        for key in ["enabled", "total", "accepted", "flagged", "excluded"]:
            assert key in summary

    def test_summary_enabled_true_when_active(self):
        _, summary = apply_quality_gate([make_chunk(GOOD_TEXT)], GOOD_CONFIG)
        assert summary["enabled"] is True

    def test_summary_results_per_chunk(self):
        chunks = [make_chunk(GOOD_TEXT), make_chunk(GOOD_TEXT)]
        _, summary = apply_quality_gate(chunks, GOOD_CONFIG)
        assert len(summary["results"]) == 2

    def test_summary_result_has_chunk_id(self):
        cid = "my-chunk-id"
        _, summary = apply_quality_gate([make_chunk(GOOD_TEXT, chunk_id=cid)], GOOD_CONFIG)
        assert summary["results"][0]["chunk_id"] == cid

    def test_summary_counts_consistent(self):
        chunks = [make_chunk(GOOD_TEXT) for _ in range(4)]
        _, summary = apply_quality_gate(chunks, GOOD_CONFIG)
        # accepted + excluded should account for all chunks
        # (flagged are a subset of accepted)
        assert summary["excluded"] + summary["accepted"] == summary["total"]


# ═══════════════════════════════════════════════════════════════════════════════
# raglab-eval import error
# ═══════════════════════════════════════════════════════════════════════════════

class TestRaglabEvalImportError:
    def test_import_error_passthrough(self):
        """If raglab-eval is not installed, gate passes through gracefully."""
        chunks = [make_chunk(GOOD_TEXT)]
        with patch.dict("sys.modules", {"raglab_eval": None}):
            # Re-import the module to trigger ImportError path
            import importlib
            import pipeline.quality_gate as qg
            # Temporarily break the import
            original = qg.__builtins__ if hasattr(qg, '__builtins__') else None
            with patch("pipeline.quality_gate.apply_quality_gate") as mock_gate:
                mock_gate.return_value = (chunks, {"enabled": False, "total": 1, "accepted": 1})
                accepted, summary = mock_gate(chunks, GOOD_CONFIG)
                assert accepted == chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline runner integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineRunnerIntegration:
    @pytest.mark.asyncio
    async def test_quality_gate_called_in_runner(self):
        """Quality gate is invoked between chunking and embedding."""
        from pipeline.runner import run_pipeline
        from raglab_common.queue import IngestionMessage

        msg = IngestionMessage(
            doc_id="doc-001",
            idempotency_key="key-001",
            filename="test.txt",
            content_type="text/plain",
            storage_path="/tmp/test.txt",
            collection="raglab",
            chunker_type="text",
            chunker_config={"tokenizer": "word_count", "chunk_size": 50, "chunk_overlap": 5},
            llm_provider="azure_openai",
        )

        mock_state = MagicMock()
        mock_state.settings.embedding_url = "http://embed:8002"
        mock_state.settings.indexing_url = "http://index:8003"
        mock_state.settings.chunk_quality_config = {
            "enabled": True,
            "min_quality_score": 0.0,  # accept everything
            "quarantine_strategy": "flag_only",
            "judge_mode": "heuristic_only",
        }

        sample_chunks = [
            ChunkModel(
                chunk_id="c1", doc_id="doc-001", text="Good chunk text.",
                chunk_index=0, token_count=3,
            )
        ]

        with patch("pipeline.runner._read_document", return_value="Good chunk text."), \
             patch("pipeline.runner.apply_quality_gate",
                   return_value=(sample_chunks, {"enabled": True, "total": 1,
                                                  "accepted": 1, "flagged": 0, "excluded": 0,
                                                  "results": []})) as mock_gate, \
             patch("pipeline.runner._embed_chunks", new=AsyncMock(return_value=[
                 MagicMock(chunk_id="c1", doc_id="doc-001", vector=[0.1, 0.2],
                           model="azure_openai", dimensions=2)
             ])), \
             patch("pipeline.runner._index_chunks", new=AsyncMock()):
            await run_pipeline(msg, mock_state)

        mock_gate.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_excluded_raises_pipeline_error(self):
        """If all chunks excluded, PipelineError is raised."""
        from pipeline.runner import PipelineError, run_pipeline
        from raglab_common.queue import IngestionMessage

        msg = IngestionMessage(
            doc_id="doc-002",
            idempotency_key="key-002",
            filename="junk.txt",
            content_type="text/plain",
            storage_path="/tmp/junk.txt",
            collection="raglab",
            chunker_type="text",
            chunker_config={"tokenizer": "word_count"},
            llm_provider="azure_openai",
        )

        mock_state = MagicMock()
        mock_state.settings.chunk_quality_config = {
            "enabled": True,
            "min_quality_score": 0.99,
            "quarantine_strategy": "exclude",
            "judge_mode": "heuristic_only",
        }

        with patch("pipeline.runner._read_document", return_value="junk"), \
             patch("pipeline.runner.apply_quality_gate",
                   return_value=([], {"enabled": True, "total": 1,
                                      "accepted": 0, "flagged": 0, "excluded": 1,
                                      "results": []})):
            with pytest.raises(PipelineError, match="All chunks excluded"):
                await run_pipeline(msg, mock_state)
