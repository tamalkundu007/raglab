"""
Graph build router — build NetworkX graph + community detection.

Endpoints:
  POST /graph/build        — build in-memory graph from Postgres for a collection
  GET  /graph/communities  — list detected communities for a collection
  GET  /graph/node/{id}    — fetch a single entity node with its neighbours
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from raglab_common.logging import get_logger

from graph.extraction.graph_builder import GraphBuilder, GraphBuildResult, CommunityInfo
from graph.routers.extract import get_session

log = get_logger(__name__)
router = APIRouter(prefix="/graph", tags=["graph"])

# Singleton builder — shared across requests (manages its own cache)
_builder = GraphBuilder()


class BuildRequest(BaseModel):
    collection: str = "raglab"
    force_rebuild: bool = False
    enable_community_detection: bool = True
    leiden_resolution: float = 1.0


class BuildResponse(BaseModel):
    collection: str
    node_count: int
    edge_count: int
    community_count: int
    communities_detected: bool
    build_time_ms: float
    cached: bool = False
    error: str | None = None


class CommunityResponse(BaseModel):
    community_id: int
    size: int
    entity_names: list[str]
    entity_types: list[str]


class NodeNeighbourResponse(BaseModel):
    entity_id: str
    name: str
    entity_type: str
    description: str
    community_id: int | None
    outgoing: list[dict]
    incoming: list[dict]


@router.post("/build", response_model=BuildResponse)
async def build_graph(
    body: BuildRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> BuildResponse:
    """
    Build the in-memory NetworkX graph from Postgres entities + relationships.

    Optionally runs Leiden community detection. Result is cached on app.state.
    Subsequent calls return the cached graph unless force_rebuild=True.
    """
    global _builder
    _builder = GraphBuilder(config={
        "enable_community_detection": body.enable_community_detection,
        "leiden_resolution": body.leiden_resolution,
    })

    graph, result = await _builder.build(
        session=session,
        collection=body.collection,
        force_rebuild=body.force_rebuild,
    )

    # Store on app.state for Phase 6 retrieval
    request.app.state.graph = graph
    request.app.state.graph_builder = _builder
    request.app.state.graph_collection = body.collection

    if result.error:
        raise HTTPException(status_code=500, detail=result.error)

    return BuildResponse(
        collection=result.collection,
        node_count=result.node_count,
        edge_count=result.edge_count,
        community_count=result.community_count,
        communities_detected=result.communities_detected,
        build_time_ms=result.build_time_ms,
    )


@router.get("/communities", response_model=list[CommunityResponse])
async def list_communities(
    collection: str = "raglab",
    request: Request = None,
) -> list[CommunityResponse]:
    """
    List detected communities for a collection.

    Returns communities from the cached graph. Call /graph/build first.
    """
    builder: GraphBuilder | None = getattr(request.app.state, "graph_builder", None)
    graph = getattr(request.app.state, "graph", None)

    if graph is None or builder is None:
        raise HTTPException(
            status_code=404,
            detail="No graph found. Call POST /graph/build first.",
        )

    communities = builder.get_communities(graph)
    return [
        CommunityResponse(
            community_id=c.community_id,
            size=c.size,
            entity_names=c.entity_names[:20],  # cap for response size
            entity_types=c.entity_types[:20],
        )
        for c in communities
    ]


@router.get("/node/{entity_id}", response_model=NodeNeighbourResponse)
async def get_node(entity_id: str, request: Request) -> NodeNeighbourResponse:
    """
    Fetch a single entity node with its immediate neighbours.

    Returns outgoing and incoming relationships for graph traversal debugging.
    """
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise HTTPException(
            status_code=404,
            detail="No graph found. Call POST /graph/build first.",
        )

    if not graph.has_node(entity_id):
        raise HTTPException(
            status_code=404,
            detail=f"Entity {entity_id!r} not found in graph.",
        )

    attrs = graph.nodes[entity_id]

    outgoing = [
        {
            "target_id": tgt,
            "target_name": graph.nodes[tgt].get("name", ""),
            **graph.edges[entity_id, tgt],
        }
        for tgt in graph.successors(entity_id)
    ]

    incoming = [
        {
            "source_id": src,
            "source_name": graph.nodes[src].get("name", ""),
            **graph.edges[src, entity_id],
        }
        for src in graph.predecessors(entity_id)
    ]

    return NodeNeighbourResponse(
        entity_id=entity_id,
        name=attrs.get("name", ""),
        entity_type=attrs.get("entity_type", ""),
        description=attrs.get("description", ""),
        community_id=attrs.get("community_id"),
        outgoing=outgoing,
        incoming=incoming,
    )
