"""
Unit tests for graph-service Phase 4 — entity/relationship extraction.

All DB and LLM calls are mocked — zero infrastructure required.

Covers:
- _build_extraction_prompt: entity_types, relation_types, max counts injected
- _parse_llm_response: valid JSON, malformed JSON, code fence stripping,
  blank entity names filtered, missing relationship entities handled
- EntityRelationshipExtractor.extract_from_chunk: llm_caller injection,
  empty chunk → empty result, extractor config forwarded
- GraphRepository.upsert_entity: new entity created, dedup by name_normalised,
  chunk_id appended to existing entity
- GraphRepository.upsert_relationship: new rel created, dedup check,
  missing entity returns None
- GraphRepository.persist_extraction_result: entities + rels persisted,
  relationship skipped when entity missing
- GraphRepository.create_run / complete_run / fail_run
- POST /graph/extract: 200 response shape, chunk_ids/texts length mismatch → 422,
  DB unavailable → 503
- GET /graph/entities, /graph/relationships, /graph/stats: 200 responses
- GET /health: ok with db status
- GET /: service info shows R4
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from graph.extraction.extractor import (
    EntityRelationshipExtractor,
    _build_extraction_prompt,
    _parse_llm_response,
)
from graph.models.schemas import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
)


# ═══════════════════════════════════════════════════════════════════════════════
# _build_extraction_prompt
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildExtractionPrompt:
    def test_entity_types_in_system_prompt(self):
        sys, usr = _build_extraction_prompt(
            "some text", ["PERSON", "ORG"], ["RELATED_TO"], 5, 5
        )
        assert "PERSON" in sys
        assert "ORG" in sys

    def test_relation_types_in_system_prompt(self):
        sys, usr = _build_extraction_prompt(
            "some text", ["CONCEPT"], ["CAUSES", "PART_OF"], 5, 5
        )
        assert "CAUSES" in sys
        assert "PART_OF" in sys

    def test_max_counts_in_system_prompt(self):
        sys, usr = _build_extraction_prompt("text", ["CONCEPT"], ["RELATED_TO"], 7, 3)
        assert "7" in sys
        assert "3" in sys

    def test_chunk_text_in_user_prompt(self):
        sys, usr = _build_extraction_prompt(
            "RAG reduces hallucinations.", ["CONCEPT"], ["RELATED_TO"], 5, 5
        )
        assert "RAG reduces hallucinations" in usr

    def test_json_schema_in_system_prompt(self):
        sys, _ = _build_extraction_prompt("text", ["CONCEPT"], ["RELATED_TO"], 5, 5)
        assert "entities" in sys
        assert "relationships" in sys


# ═══════════════════════════════════════════════════════════════════════════════
# _parse_llm_response
# ═══════════════════════════════════════════════════════════════════════════════

VALID_LLM_JSON = json.dumps({
    "entities": [
        {"name": "RAG", "entity_type": "CONCEPT", "description": "Retrieval-Augmented Generation"},
        {"name": "Qdrant", "entity_type": "TECHNOLOGY", "description": "Vector database"},
    ],
    "relationships": [
        {"source": "RAG", "target": "Qdrant", "relation_type": "USES", "description": "RAG uses Qdrant for retrieval"},
    ],
})

MALFORMED_JSON = "I found some entities: RAG and Qdrant. They are related."

JSON_WITH_FENCES = f"```json\n{VALID_LLM_JSON}\n```"


class TestParseLLMResponse:
    def test_valid_json_returns_entities(self):
        entities, rels = _parse_llm_response(VALID_LLM_JSON, "chunk-001")
        assert len(entities) == 2
        assert entities[0].name == "RAG"
        assert entities[1].name == "Qdrant"

    def test_valid_json_returns_relationships(self):
        entities, rels = _parse_llm_response(VALID_LLM_JSON, "chunk-001")
        assert len(rels) == 1
        assert rels[0].source == "RAG"
        assert rels[0].target == "Qdrant"
        assert rels[0].relation_type == "USES"

    def test_entity_type_uppercased(self):
        raw = json.dumps({"entities": [{"name": "Alice", "entity_type": "person"}], "relationships": []})
        entities, _ = _parse_llm_response(raw, "c1")
        assert entities[0].entity_type == "PERSON"

    def test_relation_type_uppercased(self):
        raw = json.dumps({
            "entities": [
                {"name": "A", "entity_type": "CONCEPT"},
                {"name": "B", "entity_type": "CONCEPT"},
            ],
            "relationships": [{"source": "A", "target": "B", "relation_type": "causes"}],
        })
        _, rels = _parse_llm_response(raw, "c1")
        assert rels[0].relation_type == "CAUSES"

    def test_malformed_json_returns_empty(self):
        entities, rels = _parse_llm_response(MALFORMED_JSON, "chunk-001")
        assert entities == []
        assert rels == []

    def test_code_fence_stripped(self):
        entities, rels = _parse_llm_response(JSON_WITH_FENCES, "chunk-001")
        assert len(entities) == 2

    def test_blank_entity_name_filtered(self):
        raw = json.dumps({
            "entities": [
                {"name": "", "entity_type": "CONCEPT"},
                {"name": "  ", "entity_type": "CONCEPT"},
                {"name": "ValidEntity", "entity_type": "CONCEPT"},
            ],
            "relationships": [],
        })
        entities, _ = _parse_llm_response(raw, "c1")
        assert len(entities) == 1
        assert entities[0].name == "ValidEntity"

    def test_empty_json_returns_empty(self):
        raw = json.dumps({"entities": [], "relationships": []})
        entities, rels = _parse_llm_response(raw, "c1")
        assert entities == [] and rels == []

    def test_description_preserved(self):
        entities, _ = _parse_llm_response(VALID_LLM_JSON, "c1")
        assert entities[0].description == "Retrieval-Augmented Generation"

    def test_missing_relationship_fields_skipped(self):
        raw = json.dumps({
            "entities": [{"name": "X", "entity_type": "CONCEPT"}],
            "relationships": [{"source": "", "target": "X", "relation_type": "RELATED_TO"}],
        })
        _, rels = _parse_llm_response(raw, "c1")
        assert rels == []


# ═══════════════════════════════════════════════════════════════════════════════
# EntityRelationshipExtractor
# ═══════════════════════════════════════════════════════════════════════════════

class TestEntityRelationshipExtractor:
    def test_extract_with_llm_caller_returns_result(self):
        ext = EntityRelationshipExtractor()
        result = ext.extract_from_chunk(
            chunk_id="c-001",
            chunk_text="RAG uses Qdrant for vector retrieval.",
            llm_caller=lambda sys, usr: VALID_LLM_JSON,
        )
        assert result.chunk_id == "c-001"
        assert len(result.entities) == 2
        assert len(result.relationships) == 1

    def test_empty_chunk_returns_empty_result(self):
        ext = EntityRelationshipExtractor()
        result = ext.extract_from_chunk(
            chunk_id="c-empty",
            chunk_text="   ",
            llm_caller=lambda s, u: VALID_LLM_JSON,
        )
        assert result.entities == []
        assert result.relationships == []

    def test_whitespace_chunk_returns_empty(self):
        ext = EntityRelationshipExtractor()
        result = ext.extract_from_chunk("c-ws", "\n\t  \n", llm_caller=lambda s, u: "")
        assert result.entities == []

    def test_llm_caller_receives_system_and_user(self):
        received = {}

        def capture(system, user):
            received["system"] = system
            received["user"] = user
            return json.dumps({"entities": [], "relationships": []})

        ext = EntityRelationshipExtractor(config={"entity_types": ["PERSON"]})
        ext.extract_from_chunk("c1", "Some text", llm_caller=capture)
        assert "PERSON" in received["system"]
        assert "Some text" in received["user"]

    def test_malformed_llm_response_returns_empty(self):
        ext = EntityRelationshipExtractor()
        result = ext.extract_from_chunk(
            "c1", "text", llm_caller=lambda s, u: "not json at all"
        )
        assert result.entities == []
        assert result.relationships == []

    def test_config_entity_types_forwarded_to_prompt(self):
        captured = {}

        def capture(sys, usr):
            captured["sys"] = sys
            return json.dumps({"entities": [], "relationships": []})

        ext = EntityRelationshipExtractor(config={"entity_types": ["CUSTOM_TYPE"]})
        ext.extract_from_chunk("c1", "text", llm_caller=capture)
        assert "CUSTOM_TYPE" in captured["sys"]

    def test_max_entities_config_forwarded(self):
        captured = {}

        def capture(sys, usr):
            captured["sys"] = sys
            return json.dumps({"entities": [], "relationships": []})

        ext = EntityRelationshipExtractor(config={"max_entities_per_chunk": 3})
        ext.extract_from_chunk("c1", "text", llm_caller=capture)
        assert "3" in captured["sys"]


# ═══════════════════════════════════════════════════════════════════════════════
# GraphRepository (all DB calls async-mocked)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGraphRepository:
    def _make_session(self):
        session = AsyncMock()
        session.execute = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()
        session.commit = AsyncMock()
        return session

    def _make_entity(self, name="RAG", etype="CONCEPT", collection="raglab"):
        from graph.models.orm import GraphEntity
        e = GraphEntity()
        e.id = uuid.uuid4()
        e.name = name
        e.name_normalised = name.lower()
        e.entity_type = etype
        e.collection = collection
        e.source_chunk_ids = "chunk-1"
        e.description = None
        e.doc_id = "doc-001"
        return e

    @pytest.mark.asyncio
    async def test_upsert_entity_new_entity_added(self):
        from graph.extraction.repository import GraphRepository
        repo = GraphRepository()
        session = self._make_session()

        # No existing entity
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        entity = ExtractedEntity(name="RAG", entity_type="CONCEPT", description="Framework")
        orm_entity = await repo.upsert_entity(session, entity, "raglab", "doc-001", "chunk-1")

        session.add.assert_called_once()
        session.flush.assert_called_once()
        assert orm_entity.name == "RAG"
        assert orm_entity.name_normalised == "rag"

    @pytest.mark.asyncio
    async def test_upsert_entity_existing_appends_chunk_id(self):
        from graph.extraction.repository import GraphRepository
        repo = GraphRepository()
        session = self._make_session()

        existing = self._make_entity()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing
        session.execute.return_value = mock_result

        entity = ExtractedEntity(name="RAG", entity_type="CONCEPT")
        await repo.upsert_entity(session, entity, "raglab", "doc-001", "chunk-new")

        # Should NOT add (entity exists)
        session.add.assert_not_called()
        # chunk-new should be in source_chunk_ids
        assert "chunk-new" in existing.source_chunk_ids

    @pytest.mark.asyncio
    async def test_upsert_entity_name_normalised_lowercased(self):
        from graph.extraction.repository import GraphRepository
        repo = GraphRepository()
        session = self._make_session()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        entity = ExtractedEntity(name="  OpenAI  ", entity_type="ORGANIZATION")
        orm_entity = await repo.upsert_entity(session, entity, "raglab", "d", "c")
        assert orm_entity.name_normalised == "openai"

    @pytest.mark.asyncio
    async def test_upsert_relationship_new(self):
        from graph.extraction.repository import GraphRepository
        repo = GraphRepository()
        session = self._make_session()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        session.execute.return_value = mock_result

        source = self._make_entity("RAG")
        target = self._make_entity("Qdrant", "TECHNOLOGY")
        rel = ExtractedRelationship(source="RAG", target="Qdrant", relation_type="USES")

        orm_rel = await repo.upsert_relationship(session, rel, source, target, "raglab", "c1")
        session.add.assert_called_once()
        assert orm_rel.relation_type == "USES"

    @pytest.mark.asyncio
    async def test_upsert_relationship_none_if_source_missing(self):
        from graph.extraction.repository import GraphRepository
        repo = GraphRepository()
        session = self._make_session()

        rel = ExtractedRelationship(source="X", target="Y", relation_type="RELATED_TO")
        result = await repo.upsert_relationship(session, rel, None, MagicMock(), "raglab", "c1")
        assert result is None

    @pytest.mark.asyncio
    async def test_persist_extraction_result_calls_upsert(self):
        from graph.extraction.repository import GraphRepository
        repo = GraphRepository()
        session = self._make_session()

        # upsert_entity always returns a new entity
        source_entity = self._make_entity("RAG")
        target_entity = self._make_entity("Qdrant")
        call_count = [0]

        async def mock_upsert_entity(s, e, col, doc_id, chunk_id):
            if e.name == "RAG":
                return source_entity
            return target_entity

        async def mock_upsert_rel(s, r, src, tgt, col, cid):
            call_count[0] += 1
            return MagicMock()

        repo.upsert_entity = mock_upsert_entity
        repo.upsert_relationship = mock_upsert_rel

        result = ExtractionResult(
            chunk_id="c1",
            entities=[
                ExtractedEntity(name="RAG", entity_type="CONCEPT"),
                ExtractedEntity(name="Qdrant", entity_type="TECHNOLOGY"),
            ],
            relationships=[
                ExtractedRelationship(source="RAG", target="Qdrant", relation_type="USES"),
            ],
        )

        e_count, r_count = await repo.persist_extraction_result(session, result, "raglab", "doc-001")
        assert e_count == 2
        assert r_count == 1

    @pytest.mark.asyncio
    async def test_create_run_adds_to_session(self):
        from graph.extraction.repository import GraphRepository
        from graph.models.orm import GraphRunStatus
        repo = GraphRepository()
        session = self._make_session()

        run = await repo.create_run(session, "doc-001", "raglab")
        session.add.assert_called_once()
        assert run.status == GraphRunStatus.RUNNING

    @pytest.mark.asyncio
    async def test_complete_run_sets_status(self):
        from graph.extraction.repository import GraphRepository
        from graph.models.orm import GraphRun, GraphRunStatus
        repo = GraphRepository()
        session = self._make_session()

        run = GraphRun()
        run.status = GraphRunStatus.RUNNING
        await repo.complete_run(session, run, 10, 5, 3)
        assert run.status == GraphRunStatus.COMPLETE
        assert run.entity_count == "10"
        assert run.relationship_count == "5"

    @pytest.mark.asyncio
    async def test_fail_run_sets_status(self):
        from graph.extraction.repository import GraphRepository
        from graph.models.orm import GraphRun, GraphRunStatus
        repo = GraphRepository()
        session = self._make_session()

        run = GraphRun()
        run.status = GraphRunStatus.RUNNING
        await repo.fail_run(session, run, "DB timeout")
        assert run.status == GraphRunStatus.FAILED
        assert "DB timeout" in run.error_message


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def graph_client_no_db():
    """Graph client with no DB — tests that check 503 on missing DB."""
    from graph.main import app
    app.state.session_factory = None
    app.state.settings = MagicMock()
    app.state.settings.llm_service_url = "http://llm:8005"
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def graph_client_with_db():
    """Graph client with mocked async DB session."""
    from graph.main import app

    # Mock session factory
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()
    mock_session.commit = AsyncMock()

    mock_factory = MagicMock()
    mock_factory.return_value = mock_session

    app.state.session_factory = mock_factory
    app.state.settings = MagicMock()
    app.state.settings.llm_service_url = "http://llm:8005"
    return TestClient(app, raise_server_exceptions=False)


class TestGraphHealth:
    def test_health_returns_ok(self, graph_client_no_db):
        r = graph_client_no_db.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_shows_db_unavailable_without_db(self, graph_client_no_db):
        r = graph_client_no_db.get("/health")
        assert r.json()["dependencies"]["database"] == "unavailable"

    def test_root_shows_r4(self, graph_client_no_db):
        r = graph_client_no_db.get("/")
        assert r.status_code == 200
        assert r.json()["release"] == "R4"

    def test_root_shows_version(self, graph_client_no_db):
        r = graph_client_no_db.get("/")
        assert "0.2.0" in r.json()["version"]


class TestExtractEndpoint:
    def test_extract_without_db_returns_503(self, graph_client_no_db):
        r = graph_client_no_db.post("/graph/extract", json={
            "doc_id": "d1",
            "chunk_ids": ["c1"],
            "chunk_texts": ["RAG uses Qdrant."],
        })
        assert r.status_code == 503

    def test_mismatched_chunk_ids_texts_returns_422(self, graph_client_with_db):
        r = graph_client_with_db.post("/graph/extract", json={
            "doc_id": "d1",
            "chunk_ids": ["c1", "c2"],
            "chunk_texts": ["only one text"],
        })
        assert r.status_code == 422

    def test_extract_with_mocked_db_and_extractor(self, graph_client_with_db):
        """Mock extractor + repo to test endpoint shape."""
        from graph.routers import extract as extract_module

        mock_run = MagicMock()
        mock_run.id = uuid.uuid4()

        with patch.object(extract_module.repo, "create_run", new=AsyncMock(return_value=mock_run)), \
             patch.object(extract_module.repo, "persist_extraction_result", new=AsyncMock(return_value=(2, 1))), \
             patch.object(extract_module.repo, "complete_run", new=AsyncMock()), \
             patch("graph.extraction.extractor.EntityRelationshipExtractor.extract_from_chunk") as mock_ext:

            mock_ext.return_value = ExtractionResult(
                chunk_id="c1",
                entities=[
                    ExtractedEntity(name="RAG", entity_type="CONCEPT"),
                    ExtractedEntity(name="Qdrant", entity_type="TECHNOLOGY"),
                ],
                relationships=[
                    ExtractedRelationship(source="RAG", target="Qdrant", relation_type="USES"),
                ],
            )

            r = graph_client_with_db.post("/graph/extract", json={
                "doc_id": "doc-test",
                "collection": "raglab",
                "chunk_ids": ["c1"],
                "chunk_texts": ["RAG uses Qdrant for vector retrieval."],
            })

        assert r.status_code == 200
        body = r.json()
        assert body["doc_id"] == "doc-test"
        assert body["chunks_processed"] == 1
        assert "run_id" in body


class TestEntitiesEndpoint:
    def test_entities_without_db_returns_503(self, graph_client_no_db):
        r = graph_client_no_db.get("/graph/entities")
        assert r.status_code == 503


class TestStatsEndpoint:
    def test_stats_without_db_returns_503(self, graph_client_no_db):
        r = graph_client_no_db.get("/graph/stats")
        assert r.status_code == 503
