"""
EntityRelationshipExtractor — LLM-based entity and relationship extraction.

Calls the llm-service /generate endpoint with a structured JSON extraction prompt.
Parses the LLM response into ExtractedEntity and ExtractedRelationship objects.

Design:
    - Prompt engineering is the core of this module. The prompt must:
        1. Instruct the LLM to output only valid JSON (no prose, no markdown fences).
        2. Define the exact schema: {entities: [...], relationships: [...]}.
        3. Constrain entity types and relationship types to the configured list.
        4. Set count limits to prevent runaway token spend.

    - The parser is defensive: malformed LLM JSON returns an empty result, never raises.
      Extraction failures are logged as warnings, not errors.

    - This is a stateless utility — it holds no DB connection. The graph-service
      router handles persistence after calling extract_from_chunk().
"""

from __future__ import annotations

import json
import re
from typing import Any

from raglab_common.logging import get_logger

from graph.models.schemas import ExtractedEntity, ExtractedRelationship, ExtractionResult

log = get_logger(__name__)

_DEFAULT_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "CONCEPT", "TECHNOLOGY", "LOCATION", "PRODUCT"]
_DEFAULT_RELATION_TYPES = ["RELATED_TO", "PART_OF", "CAUSES", "USED_BY", "WORKS_AT", "INSTANCE_OF"]

_EXTRACTION_SYSTEM_PROMPT = (
    "You are a knowledge graph extraction engine. "
    "Output ONLY valid JSON. No markdown, no prose, no code fences. "
    "Extract entities and relationships from the provided text.\n\n"
    "Required JSON keys: entities (list) and relationships (list).\n"
    "Each entity: name (string), entity_type (string), description (string).\n"
    "Each relationship: source (string), target (string), "
    "relation_type (string), description (string).\n\n"
    "Rules:\n"
    "- entity_type must be one of: {entity_types}\n"
    "- relation_type must be one of: {relation_types}\n"
    "- source and target must exactly match an entity name you listed\n"
    "- description: 1 sentence maximum, factual only\n"
    "- If nothing found: return empty lists for both keys\n"
    "- Max {max_entities} entities, {max_relationships} relationships\n"
    "- Do NOT invent facts not present in the text"
)



def _build_extraction_prompt(
    chunk_text: str,
    entity_types: list[str],
    relation_types: list[str],
    max_entities: int,
    max_relationships: int,
) -> tuple[str, str]:
    """
    Build system + user prompt for entity/relationship extraction.

    Returns:
        (system_prompt, user_prompt)
    """
    system = _EXTRACTION_SYSTEM_PROMPT.format(
        entity_types=", ".join(entity_types),
        relation_types=", ".join(relation_types),
        max_entities=max_entities,
        max_relationships=max_relationships,
    )
    user = f"Extract entities and relationships from this text:\n\n{chunk_text}"
    return system, user


def _parse_llm_response(raw: str, chunk_id: str) -> tuple[list[ExtractedEntity], list[ExtractedRelationship]]:
    """
    Parse LLM JSON response into entity and relationship lists.

    Defensive: malformed JSON → empty result + warning log.
    Strips markdown code fences if present (common LLM tendency).
    """
    # Strip ```json ... ``` fences
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.warning(
            "extractor.json_parse_failed",
            chunk_id=chunk_id,
            error=str(exc),
            raw_preview=raw[:200],
        )
        return [], []

    entities: list[ExtractedEntity] = []
    for e in data.get("entities", []):
        try:
            entities.append(ExtractedEntity(
                name=str(e.get("name", "")).strip(),
                entity_type=str(e.get("entity_type", "CONCEPT")).strip().upper(),
                description=e.get("description"),
            ))
        except Exception:
            continue

    # Filter out blank names
    entities = [e for e in entities if e.name]

    relationships: list[ExtractedRelationship] = []
    entity_names = {e.name for e in entities}
    for r in data.get("relationships", []):
        try:
            src = str(r.get("source", "")).strip()
            tgt = str(r.get("target", "")).strip()
            if not src or not tgt:
                continue
            relationships.append(ExtractedRelationship(
                source=src,
                target=tgt,
                relation_type=str(r.get("relation_type", "RELATED_TO")).strip().upper(),
                description=r.get("description"),
                weight=float(r.get("weight", 1.0)),
            ))
        except Exception:
            continue

    return entities, relationships


class EntityRelationshipExtractor:
    """
    Extracts entities and relationships from chunk text via the llm-service.

    Uses a structured JSON extraction prompt. Parses and validates LLM output.
    Stateless — no DB connection. Caller handles persistence.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.llm_service_url: str = cfg.get("llm_service_url", "http://llm:8005").rstrip("/")
        self.llm_provider: str = cfg.get("llm_provider", "azure_openai")
        self.entity_types: list[str] = cfg.get("entity_types", _DEFAULT_ENTITY_TYPES)
        self.relation_types: list[str] = cfg.get("relation_types", _DEFAULT_RELATION_TYPES)
        self.max_entities: int = int(cfg.get("max_entities_per_chunk", 10))
        self.max_relationships: int = int(cfg.get("max_relationships_per_chunk", 10))
        self.timeout: float = float(cfg.get("timeout_seconds", 30.0))

    def extract_from_chunk(
        self,
        chunk_id: str,
        chunk_text: str,
        llm_caller: Any | None = None,
    ) -> ExtractionResult:
        """
        Extract entities and relationships from a single chunk.

        Args:
            chunk_id:    Chunk identifier (for tracing).
            chunk_text:  The text to extract from.
            llm_caller:  Optional callable(system, user) → str for testing.
                         If None, calls llm-service HTTP endpoint.

        Returns:
            ExtractionResult with entities and relationships lists.
        """
        if not chunk_text.strip():
            return ExtractionResult(chunk_id=chunk_id)

        system_prompt, user_prompt = _build_extraction_prompt(
            chunk_text=chunk_text,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
            max_entities=self.max_entities,
            max_relationships=self.max_relationships,
        )

        raw_response = self._call_llm(system_prompt, user_prompt, llm_caller)
        if not raw_response:
            return ExtractionResult(chunk_id=chunk_id)

        entities, relationships = _parse_llm_response(raw_response, chunk_id)

        log.info(
            "extractor.chunk_complete",
            chunk_id=chunk_id,
            entities=len(entities),
            relationships=len(relationships),
        )

        return ExtractionResult(
            chunk_id=chunk_id,
            entities=entities,
            relationships=relationships,
        )

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        llm_caller: Any | None,
    ) -> str:
        """Call the LLM — via injected caller (tests) or HTTP (production)."""
        if llm_caller is not None:
            return llm_caller(system_prompt, user_prompt)

        # Production: HTTP call to llm-service /generate
        try:
            from raglab_chunkers.caption_service import _requests
            if _requests is None:
                raise ImportError("requests not available")

            resp = _requests.post(
                f"{self.llm_service_url}/generate",
                json={
                    "query": user_prompt,
                    "chunks": [],
                    "provider": self.llm_provider,
                    "system_prompt": system_prompt,
                    "max_tokens": 1024,
                    "temperature": 0.0,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json().get("answer", "")
        except Exception as exc:
            log.warning("extractor.llm_call_failed", error=str(exc))
            return ""
