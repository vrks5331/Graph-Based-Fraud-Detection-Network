"""
BFS Explainer — Breadth-First Search Traversal for Fraud Ring Explanation
=========================================================================

This module sits at the **end** of the fraud-detection pipeline:

    GNN (risk scores)  →  Louvain (community detection)  →  **BFS Explainer**

Purpose
-------
Once Louvain flags a suspicious community and extracts its subgraph, the BFS
explainer **walks** that subgraph starting from the highest-risk node.  The
traversal order produces a human-readable narrative of how fraud flows through
the ring — which accounts are involved, how they're connected, and what the
transaction amounts look like.

Key entry points
----------------
- ``explain_cluster``    – explain a single flagged community subgraph.
- ``explain_all_clusters`` – explain every flagged community from
  ``run_louvain_detection``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx


# ── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class TraversalStep:
    """One step in the BFS traversal of a fraud-ring subgraph.

    Each step records the edge that was followed and the node that was
    discovered, along with contextual information for the explanation.

    Attributes
    ----------
    depth : int
        BFS depth (0 = the seed / root node).
    node : object
        The node discovered at this step.
    node_risk : float
        GNN fraud-risk score for this node (0.0–1.0).
    parent : object | None
        The node from which this node was reached (``None`` for the root).
    edge_weight : float | None
        Transaction amount on the edge from ``parent`` → ``node``
        (``None`` for the root).
    edge_data : dict
        Full attribute dict of the edge (timestamps, labels, etc.).
    """

    depth: int
    node: object
    node_risk: float
    parent: object = None
    edge_weight: float = None
    edge_data: dict = field(default_factory=dict)


@dataclass
class ClusterExplanation:
    """Complete BFS explanation for one suspicious community.

    Attributes
    ----------
    cluster_id : int
        The Louvain community ID.
    seed_node : object
        Highest-risk node where the BFS started.
    num_nodes : int
        Total nodes in the community subgraph.
    num_edges : int
        Total edges in the community subgraph.
    traversal : List[TraversalStep]
        Ordered BFS traversal steps (root first).
    summary : str
        Auto-generated plain-English summary of the fraud ring.
    """

    cluster_id: int
    seed_node: object
    num_nodes: int
    num_edges: int
    traversal: List[TraversalStep]
    summary: str


# ── Seed Selection ───────────────────────────────────────────────────────────


def _select_seed_node(
    subgraph: nx.Graph,
    node_risk: Dict[object, float],
) -> object:
    """Pick the best starting node for BFS traversal.

    Strategy: choose the node with the **highest GNN risk score**.  If there
    are ties, break by highest degree (most connections = most central in the
    ring).  This ensures the explanation starts at the "ringleader" and fans
    out to accomplices.

    Parameters
    ----------
    subgraph : nx.Graph
        The community subgraph to traverse.
    node_risk : Dict[node, float]
        Per-node fraud risk from the GNN.

    Returns
    -------
    object
        The chosen seed node.
    """
    # Build (risk, degree, node) tuples — sort descending by risk then degree.
    candidates = [
        (node_risk.get(node, 0.0), subgraph.degree(node), node)
        for node in subgraph.nodes()
    ]
    # max picks the candidate with the highest (risk, degree) tuple.
    _, _, best = max(candidates, key=lambda t: (t[0], t[1]))
    return best


# ── BFS Traversal ────────────────────────────────────────────────────────────


def _bfs_traverse(
    subgraph: nx.Graph,
    seed: object,
    node_risk: Dict[object, float],
    weight_attr: str = "weight",
) -> List[TraversalStep]:
    """Perform breadth-first search on the community subgraph.

    Starting from ``seed``, we visit every reachable node layer by layer.
    Each visit is recorded as a ``TraversalStep`` with full edge context so
    downstream consumers (dashboards, reports) can render the explanation.

    Parameters
    ----------
    subgraph : nx.Graph
        The community subgraph (from ``get_community_subgraph``).
    seed : object
        The node to start BFS from (typically the highest-risk node).
    node_risk : Dict[node, float]
        GNN fraud scores.
    weight_attr : str
        Edge attribute that stores the transaction weight (default ``"weight"``).
        Also checks ``"amount"`` as a fallback, since the original graph may
        use ``"amount"`` while Louvain's projected graph uses ``"weight"``.

    Returns
    -------
    List[TraversalStep]
        Ordered list of traversal steps (root at index 0).
    """
    visited: Set[object] = set()
    # Queue entries: (current_node, parent_node, edge_data_dict, depth)
    queue: deque[Tuple[object, Optional[object], dict, int]] = deque()

    # Enqueue the seed with no parent.
    queue.append((seed, None, {}, 0))
    steps: List[TraversalStep] = []

    while queue:
        current, parent, edge_data, depth = queue.popleft()

        if current in visited:
            continue
        visited.add(current)

        # Extract the transaction weight from the edge data dict.
        # Try the explicit weight_attr first, then common fallbacks.
        edge_weight = None
        if parent is not None:
            edge_weight = float(
                edge_data.get(weight_attr,
                              edge_data.get("amount",
                                            edge_data.get("weight", 0.0)))
            )

        steps.append(
            TraversalStep(
                depth=depth,
                node=current,
                node_risk=node_risk.get(current, 0.0),
                parent=parent,
                edge_weight=edge_weight,
                edge_data=dict(edge_data),  # defensive copy
            )
        )

        # Enqueue unvisited neighbours.  Sorting by risk (descending) makes
        # the traversal deterministic and prioritises high-risk paths.
        neighbours = []
        for neighbour in subgraph.neighbors(current):
            if neighbour not in visited:
                edata = subgraph.edges[current, neighbour]
                risk = node_risk.get(neighbour, 0.0)
                neighbours.append((risk, neighbour, edata))

        # Sort descending by risk so higher-risk branches come first.
        neighbours.sort(key=lambda t: t[0], reverse=True)

        for _, neighbour, edata in neighbours:
            queue.append((neighbour, current, edata, depth + 1))

    return steps


# ── Summary Generation ───────────────────────────────────────────────────────


def _build_summary(
    cluster_id: int,
    steps: List[TraversalStep],
    num_nodes: int,
    num_edges: int,
) -> str:
    """Generate a plain-English summary of the fraud ring.

    The summary is meant for analysts and dashboards — it should be readable
    without any graph theory background.

    Parameters
    ----------
    cluster_id : int
        Louvain community ID.
    steps : List[TraversalStep]
        BFS traversal steps.
    num_nodes : int
        Total nodes in the community.
    num_edges : int
        Total edges in the community.

    Returns
    -------
    str
        Multi-line summary string.
    """
    if not steps:
        return f"Cluster {cluster_id}: empty community (no nodes to explain)."

    # Identify the seed (root) and compute aggregate stats.
    seed = steps[0]
    max_depth = max(s.depth for s in steps)
    avg_risk = sum(s.node_risk for s in steps) / len(steps)

    # Collect all transaction amounts (skip the root which has no parent edge).
    amounts = [s.edge_weight for s in steps if s.edge_weight is not None]
    total_amount = sum(amounts) if amounts else 0.0

    lines = [
        f"Fraud Ring (Cluster {cluster_id})",
        f"{'=' * 40}",
        f"Nodes: {num_nodes}  |  Edges: {num_edges}  |  BFS Depth: {max_depth}",
        f"Average Risk Score: {avg_risk:.2f}",
        f"Total Transaction Volume: ${total_amount:,.2f}",
        "",
        f"Root (highest-risk node): {seed.node} (risk={seed.node_risk:.2f})",
        "",
        "Traversal Order:",
    ]

    for step in steps:
        indent = "  " * step.depth
        if step.parent is None:
            lines.append(f"  {indent}[Depth {step.depth}] {step.node} "
                         f"(risk={step.node_risk:.2f}) ← START")
        else:
            amt = f"${step.edge_weight:,.2f}" if step.edge_weight else "N/A"
            lines.append(f"  {indent}[Depth {step.depth}] {step.parent} → "
                         f"{step.node} (risk={step.node_risk:.2f}, "
                         f"amount={amt})")

    return "\n".join(lines)


# ── Public API ───────────────────────────────────────────────────────────────


def explain_cluster(
    subgraph: nx.Graph,
    cluster_id: int,
    node_risk: Optional[Dict[object, float]] = None,
    weight_attr: str = "weight",
) -> ClusterExplanation:
    """Explain a single suspicious community via BFS traversal.

    This is the function you call after Louvain has extracted a community
    subgraph.  It:

    1. Picks the highest-risk node as the BFS seed.
    2. Traverses the subgraph breadth-first.
    3. Builds a human-readable summary.

    Parameters
    ----------
    subgraph : nx.Graph
        The community subgraph (from ``get_community_subgraph`` or
        ``run_louvain_detection``'s ``"suspicious_subgraphs"``).
    cluster_id : int
        The Louvain community ID (for labelling).
    node_risk : Dict[node, float] | None
        Per-node GNN fraud scores.  Defaults to 0.0 for all nodes if omitted.
    weight_attr : str
        Edge attribute storing transaction weight.

    Returns
    -------
    ClusterExplanation
        Full explanation including traversal steps and summary text.
    """
    node_risk = node_risk or {}

    if subgraph.number_of_nodes() == 0:
        return ClusterExplanation(
            cluster_id=cluster_id,
            seed_node=None,
            num_nodes=0,
            num_edges=0,
            traversal=[],
            summary=f"Cluster {cluster_id}: empty community (no nodes to explain).",
        )

    # Step 1: Pick the seed — highest-risk, highest-degree node.
    seed = _select_seed_node(subgraph, node_risk)

    # Step 2: BFS traverse from the seed outward.
    steps = _bfs_traverse(subgraph, seed, node_risk, weight_attr=weight_attr)

    # Step 3: Generate the analyst-friendly summary.
    summary = _build_summary(
        cluster_id=cluster_id,
        steps=steps,
        num_nodes=subgraph.number_of_nodes(),
        num_edges=subgraph.number_of_edges(),
    )

    return ClusterExplanation(
        cluster_id=cluster_id,
        seed_node=seed,
        num_nodes=subgraph.number_of_nodes(),
        num_edges=subgraph.number_of_edges(),
        traversal=steps,
        summary=summary,
    )


def explain_all_clusters(
    suspicious_subgraphs: Dict[int, nx.Graph],
    node_risk: Optional[Dict[object, float]] = None,
    weight_attr: str = "weight",
) -> List[ClusterExplanation]:
    """Explain every flagged community from ``run_louvain_detection``.

    Convenience wrapper that iterates over the ``"suspicious_subgraphs"``
    dict returned by ``run_louvain_detection`` and explains each one.

    Usage
    -----
    ::

        from src.detection.louvain import run_louvain_detection
        from src.explainability.bfs_explainer import explain_all_clusters

        results = run_louvain_detection(graph, node_risk=gnn_scores)
        explanations = explain_all_clusters(
            results["suspicious_subgraphs"],
            node_risk=gnn_scores,
        )
        for expl in explanations:
            print(expl.summary)

    Parameters
    ----------
    suspicious_subgraphs : Dict[int, nx.Graph]
        Mapping of cluster_id → community subgraph.
    node_risk : Dict[node, float] | None
        Per-node GNN fraud scores.
    weight_attr : str
        Edge attribute storing transaction weight.

    Returns
    -------
    List[ClusterExplanation]
        One explanation per flagged cluster, sorted by cluster_id.
    """
    explanations = [
        explain_cluster(
            subgraph=sub,
            cluster_id=cid,
            node_risk=node_risk,
            weight_attr=weight_attr,
        )
        for cid, sub in suspicious_subgraphs.items()
    ]
    # Sort by cluster_id for deterministic output.
    explanations.sort(key=lambda e: e.cluster_id)
    return explanations
