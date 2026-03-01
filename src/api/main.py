"""
FastAPI Backend — Fraud Detection Dashboard API
=================================================

Serves the graph data, Louvain community detection results, and BFS
fraud-ring explanations as a REST API for the frontend dashboard.

Usage
-----
    # From project root
    uvicorn src.api.main:app --reload --port 8000

All endpoints are prefixed with ``/api``.
"""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.api.graph_service import GraphService

# ═══════════════════════════════════════════════════════════════════════════════
#  Application setup
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Graph Fraud Detection API",
    description=(
        "REST API serving graph data, Louvain community detection, "
        "and BFS fraud-ring explanations for the dashboard frontend."
    ),
    version="0.1.0",
)

# Allow any frontend dev server to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Service singleton ────────────────────────────────────────────────────────
# Loaded once at startup; shared across all requests.

_service: Optional[GraphService] = None


def get_service() -> GraphService:
    """Return the loaded GraphService, initialising on first call."""
    global _service
    if _service is None:
        _service = GraphService().load()
    return _service


@app.on_event("startup")
async def _startup() -> None:
    """Pre-load graph data so the first request is fast."""
    get_service()


# ═══════════════════════════════════════════════════════════════════════════════
#  Health
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/api/health", tags=["health"])
def health():
    """Simple liveness check."""
    svc = get_service()
    return {
        "status": "ok",
        "nodes_loaded": svc.graph.number_of_nodes() if svc.graph else 0,
        "edges_loaded": svc.graph.number_of_edges() if svc.graph else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Graph
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/api/graph/summary", tags=["graph"])
def graph_summary():
    """High-level graph statistics (node/edge counts, fraud rate, etc.)."""
    return get_service().get_graph_summary()


@app.get("/api/graph/nodes", tags=["graph"])
def graph_nodes(
    skip: int = Query(0, ge=0, description="Offset for pagination"),
    limit: int = Query(100, ge=1, le=1000, description="Page size"),
):
    """Paginated list of nodes with features, label, and risk score."""
    return get_service().get_nodes(skip=skip, limit=limit)


@app.get("/api/graph/nodes/{node_id}", tags=["graph"])
def graph_node(node_id: int):
    """Single node detail including features, risk, and neighbors."""
    result = get_service().get_node(node_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    return result


@app.get("/api/graph/edges", tags=["graph"])
def graph_edges(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    """Paginated edge list with source, target, amount, is_fraud."""
    return get_service().get_edges(skip=skip, limit=limit)


# ═══════════════════════════════════════════════════════════════════════════════
#  Communities
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/api/communities", tags=["communities"])
def communities():
    """All Louvain communities with metrics and suspicious flag."""
    return get_service().get_communities()


@app.get("/api/communities/{cluster_id}", tags=["communities"])
def community_detail(cluster_id: int):
    """Single community detail including member node IDs."""
    result = get_service().get_community(cluster_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Community {cluster_id} not found"
        )
    return result


@app.get("/api/communities/{cluster_id}/subgraph", tags=["communities"])
def community_subgraph(cluster_id: int):
    """Subgraph (nodes + edges) of a community for frontend visualisation."""
    result = get_service().get_community_subgraph_data(cluster_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Community {cluster_id} not found"
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Explanations
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/api/explanations", tags=["explanations"])
def explanations():
    """BFS explanations for all suspicious communities."""
    return get_service().get_all_explanations()


@app.get("/api/explanations/{cluster_id}", tags=["explanations"])
def explanation_detail(cluster_id: int):
    """BFS explanation for a specific suspicious community."""
    result = get_service().get_explanation(cluster_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No explanation for community {cluster_id}",
        )
    return result
