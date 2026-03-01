"""
Louvain Community Detection for Fraud Ring Identification
=========================================================

This module sits **downstream** of the GNN and **upstream** of the BFS explainer
in the fraud-detection pipeline:

    Raw data  →  GNN (per-node risk scores)  →  **Louvain** (community clustering)  →  BFS Explainer

Workflow
--------
1. The GNN produces a risk score for every account node.
2. Louvain groups nodes into densely-connected communities.
3. Each community is evaluated using structural metrics (density, size) and
   the GNN risk scores (average / max risk) to flag suspicious fraud rings.
4. Flagged communities are extracted as NetworkX subgraphs so the BFS
   explainer can traverse them and produce human-readable explanations.

Key entry points
----------------
- ``run_louvain_detection``  – full pipeline; returns partition, metrics,
  flagged clusters, **and** their subgraphs (ready for BFS).
- ``get_community_subgraph`` / ``get_suspicious_subgraphs`` – helpers the
  BFS explainer can call directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

import networkx as nx

try:
    import community as community_louvain  # python-louvain package
except ImportError:  # pragma: no cover
    community_louvain = None


# ── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class ClusterMetrics:
    """Per-community summary statistics used to decide if a cluster is suspicious.

    Attributes
    ----------
    cluster_id : int
        Unique identifier for this community (assigned by Louvain).
    size : int
        Number of nodes in the community.
    density : float
        Edge density of the community subgraph (0.0–1.0).
        A higher density means accounts are more interconnected.
    average_node_risk : float
        Mean of the GNN risk scores for all nodes in the community.
    max_node_risk : float
        Highest GNN risk score among the community's nodes.
    """

    cluster_id: int
    size: int
    density: float
    average_node_risk: float
    max_node_risk: float


# ── Graph Projection ────────────────────────────────────────────────────────


def _to_weighted_undirected(
    graph: nx.Graph, weight_attr: str = "amount"
) -> nx.Graph:
    """Create a simple weighted undirected graph from a (possibly directed /
    multi-edge) transaction graph.

    Louvain requires an undirected graph, so we collapse parallel edges by
    **summing** their weights.  For example, if account A sends account B
    three payments of $10, $20, and $30, the projected edge A↔B gets
    weight = $60.

    Parameters
    ----------
    graph : nx.Graph | nx.DiGraph | nx.MultiGraph | nx.MultiDiGraph
        The original transaction graph.
    weight_attr : str
        Edge attribute that carries the monetary amount (default ``"amount"``).
        Falls back to ``"weight"`` if ``weight_attr`` is missing on an edge.

    Returns
    -------
    nx.Graph
        Undirected, simple graph with a single ``"weight"`` attribute per edge.
    """
    projected = nx.Graph()

    # Determine the correct edge iterator depending on whether the graph
    # stores multiple edges between the same pair of nodes.
    if graph.is_multigraph():
        edge_iter = graph.edges(data=True, keys=True)
        for u, v, _, data in edge_iter:
            weight = float(data.get(weight_attr, data.get("weight", 1.0)))
            if projected.has_edge(u, v):
                projected[u][v]["weight"] += weight
            else:
                projected.add_edge(u, v, weight=weight)
    else:
        edge_iter = graph.edges(data=True)
        for u, v, data in edge_iter:
            weight = float(data.get(weight_attr, data.get("weight", 1.0)))
            if projected.has_edge(u, v):
                projected[u][v]["weight"] += weight
            else:
                projected.add_edge(u, v, weight=weight)

    # Keep isolated nodes so they each receive their own singleton community
    # rather than being silently dropped.
    for node in graph.nodes():
        if node not in projected:
            projected.add_node(node)

    return projected


# ── Community Detection ──────────────────────────────────────────────────────


def detect_louvain_communities(
    graph: nx.Graph,
    resolution: float = 1.0,
    random_state: int = 42,
    weight_attr: str = "amount",
) -> Dict[object, int]:
    """Run Louvain community detection and return a node → cluster_id mapping.

    Parameters
    ----------
    graph : nx.Graph
        Transaction graph (directed / multigraph variants are accepted).
    resolution : float
        Louvain resolution parameter.  Higher values produce smaller, more
        granular communities — useful for isolating tight fraud rings.
    random_state : int
        Seed for reproducibility.
    weight_attr : str
        Edge attribute to use as weight (default ``"amount"``).

    Returns
    -------
    Dict[node, int]
        Mapping of each node to its community (cluster) ID.
    """
    if graph.number_of_nodes() == 0:
        return {}

    # Project to a simple undirected graph (Louvain requirement).
    working_graph = _to_weighted_undirected(graph, weight_attr=weight_attr)

    # Prefer the dedicated python-louvain package; fall back to the built-in
    # NetworkX implementation if python-louvain is not installed.
    if community_louvain is not None:
        return community_louvain.best_partition(
            working_graph,
            weight="weight",
            resolution=resolution,
            random_state=random_state,
        )

    # NetworkX fallback returns a list of sets; convert to {node: cluster_id}.
    communities = nx.community.louvain_communities(
        working_graph,
        weight="weight",
        resolution=resolution,
        seed=random_state,
    )
    partition: Dict[object, int] = {}
    for cluster_id, members in enumerate(communities):
        for node in members:
            partition[node] = cluster_id
    return partition


# ── Partition Helpers ────────────────────────────────────────────────────────


def _invert_partition(partition: Dict[object, int]) -> Dict[int, Set[object]]:
    """Invert a node→cluster_id map into a cluster_id→{nodes} map.

    This is used internally to iterate over communities without scanning the
    entire partition dict for each cluster.
    """
    clusters: Dict[int, Set[object]] = {}
    for node, cluster_id in partition.items():
        clusters.setdefault(cluster_id, set()).add(node)
    return clusters


# ── Cluster Metrics ──────────────────────────────────────────────────────────


def compute_cluster_metrics(
    graph: nx.Graph,
    partition: Dict[object, int],
    node_risk: Optional[Dict[object, float]] = None,
) -> List[ClusterMetrics]:
    """Compute structural and risk metrics for every community.

    Parameters
    ----------
    graph : nx.Graph
        The original transaction graph (used to compute density).
    partition : Dict[node, int]
        Node → cluster_id mapping from ``detect_louvain_communities``.
    node_risk : Dict[node, float] | None
        GNN-produced fraud probability per node (0.0–1.0).
        If ``None``, all risk values default to 0.0.

    Returns
    -------
    List[ClusterMetrics]
        One entry per community, sorted descending by
        (average_node_risk, density, size) so the most suspicious clusters
        appear first.
    """
    if not partition:
        return []

    node_risk = node_risk or {}
    clusters = _invert_partition(partition)
    metrics: List[ClusterMetrics] = []

    for cluster_id, members in clusters.items():
        subgraph = graph.subgraph(members)

        # Gather per-node risk scores; missing nodes default to 0.0.
        risks = [float(node_risk.get(node, 0.0)) for node in members]
        avg_risk = sum(risks) / len(risks) if risks else 0.0
        max_risk = max(risks) if risks else 0.0

        metrics.append(
            ClusterMetrics(
                cluster_id=cluster_id,
                size=len(members),
                density=nx.density(subgraph) if len(members) > 1 else 0.0,
                average_node_risk=avg_risk,
                max_node_risk=max_risk,
            )
        )

    # Most suspicious clusters first.
    metrics.sort(
        key=lambda m: (m.average_node_risk, m.density, m.size), reverse=True
    )
    return metrics


# ── Suspicious Cluster Flagging ──────────────────────────────────────────────


def flag_suspicious_clusters(
    cluster_metrics: Iterable[ClusterMetrics],
    min_size: int = 3,
    min_average_risk: float = 0.7,
    min_density: float = 0.2,
) -> List[ClusterMetrics]:
    """Apply threshold rules to flag communities that look like fraud rings.

    A cluster is flagged when **all three** conditions are met:

    - ``size >= min_size``         → ring must have at least N members
    - ``average_node_risk >= min_average_risk`` → overall GNN suspicion is high
    - ``density >= min_density``   → members are tightly interconnected

    Parameters
    ----------
    cluster_metrics : Iterable[ClusterMetrics]
        Output of ``compute_cluster_metrics``.
    min_size : int
        Minimum number of nodes to qualify as a ring (default 3).
    min_average_risk : float
        Minimum average GNN risk score (default 0.7).
    min_density : float
        Minimum edge density (default 0.2).

    Returns
    -------
    List[ClusterMetrics]
        Only the clusters that pass all three thresholds.
    """
    flagged: List[ClusterMetrics] = []
    for cluster in cluster_metrics:
        if (
            cluster.size >= min_size
            and cluster.average_node_risk >= min_average_risk
            and cluster.density >= min_density
        ):
            flagged.append(cluster)
    return flagged


# ── Subgraph Extraction (for BFS Explainer) ─────────────────────────────────
#
# These functions let the BFS explainer pull out a community as a standalone
# NetworkX subgraph.  BFS then walks that subgraph starting from the
# highest-risk node to build a human-readable explanation of the fraud ring.
#


def get_community_subgraph(
    graph: nx.Graph,
    partition: Dict[object, int],
    cluster_id: int,
) -> nx.Graph:
    """Extract the subgraph for a single community.

    This is the **core building block** for the BFS explainer: given a
    community ID, it returns the induced subgraph containing only the
    nodes assigned to that community and the edges between them.

    Parameters
    ----------
    graph : nx.Graph
        The full transaction graph.
    partition : Dict[node, int]
        Node → cluster_id mapping from ``detect_louvain_communities``.
    cluster_id : int
        The community to extract.

    Returns
    -------
    nx.Graph
        Induced subgraph for the specified community.  Preserves all
        original node / edge attributes (amounts, timestamps, etc.) so
        the BFS explainer has full context.

    Raises
    ------
    ValueError
        If ``cluster_id`` does not exist in the partition.
    """
    members = {
        node for node, cid in partition.items() if cid == cluster_id
    }
    if not members:
        raise ValueError(
            f"cluster_id {cluster_id} not found in partition. "
            f"Valid IDs: {sorted(set(partition.values()))}"
        )
    return graph.subgraph(members).copy()


def get_all_community_subgraphs(
    graph: nx.Graph,
    partition: Dict[object, int],
) -> Dict[int, nx.Graph]:
    """Extract subgraphs for **every** community in the partition.

    Parameters
    ----------
    graph : nx.Graph
        The full transaction graph.
    partition : Dict[node, int]
        Node → cluster_id mapping.

    Returns
    -------
    Dict[int, nx.Graph]
        Mapping of cluster_id → induced subgraph.
    """
    clusters = _invert_partition(partition)
    return {
        cid: graph.subgraph(members).copy()
        for cid, members in clusters.items()
    }


def get_suspicious_subgraphs(
    graph: nx.Graph,
    partition: Dict[object, int],
    flagged: List[ClusterMetrics],
) -> Dict[int, nx.Graph]:
    """Extract subgraphs **only** for the flagged suspicious clusters.

    This is the primary entry point for the BFS explainer:

    1. ``run_louvain_detection`` identifies suspicious clusters.
    2. This function returns their subgraphs.
    3. BFS traverses each subgraph (starting from the highest-risk node)
       to produce a human-readable fraud-ring explanation.

    Parameters
    ----------
    graph : nx.Graph
        The full transaction graph.
    partition : Dict[node, int]
        Node → cluster_id mapping.
    flagged : List[ClusterMetrics]
        Suspicious clusters from ``flag_suspicious_clusters``.

    Returns
    -------
    Dict[int, nx.Graph]
        Mapping of cluster_id → induced subgraph for each flagged cluster.
    """
    return {
        cm.cluster_id: get_community_subgraph(graph, partition, cm.cluster_id)
        for cm in flagged
    }


# ── End-to-End Pipeline ─────────────────────────────────────────────────────


def run_louvain_detection(
    graph: nx.Graph,
    node_risk: Optional[Dict[object, float]] = None,
    resolution: float = 1.0,
    random_state: int = 42,
    weight_attr: str = "amount",
    min_size: int = 3,
    min_average_risk: float = 0.7,
    min_density: float = 0.2,
) -> Dict[str, object]:
    """Run the full Louvain fraud-ring detection pipeline.

    This is the single function you call to go from a transaction graph +
    GNN risk scores to a set of flagged fraud-ring subgraphs ready for
    BFS explanation.

    Parameters
    ----------
    graph : nx.Graph
        Transaction graph (any NetworkX graph type).
    node_risk : Dict[node, float] | None
        Per-node fraud score from the GNN (0.0 = safe, 1.0 = fraud).
    resolution : float
        Louvain resolution (higher → smaller communities).
    random_state : int
        Random seed for reproducibility.
    weight_attr : str
        Edge attribute storing transaction amounts.
    min_size : int
        Minimum cluster size to flag as suspicious.
    min_average_risk : float
        Minimum average risk to flag as suspicious.
    min_density : float
        Minimum density to flag as suspicious.

    Returns
    -------
    dict
        ``"partition"``            – node → cluster_id mapping
        ``"cluster_metrics"``      – list of ``ClusterMetrics`` (all clusters)
        ``"suspicious_clusters"``  – list of ``ClusterMetrics`` (flagged only)
        ``"suspicious_subgraphs"`` – dict of cluster_id → nx.Graph subgraphs
                                     for every flagged cluster, ready for the
                                     BFS explainer
    """
    # Step 1: Partition the graph into communities.
    partition = detect_louvain_communities(
        graph=graph,
        resolution=resolution,
        random_state=random_state,
        weight_attr=weight_attr,
    )

    # Step 2: Compute structural + risk metrics per community.
    metrics = compute_cluster_metrics(
        graph=graph, partition=partition, node_risk=node_risk
    )

    # Step 3: Flag clusters that exceed the suspicion thresholds.
    suspicious = flag_suspicious_clusters(
        cluster_metrics=metrics,
        min_size=min_size,
        min_average_risk=min_average_risk,
        min_density=min_density,
    )

    # Step 4: Extract subgraphs for flagged clusters so the BFS explainer
    #         can traverse them directly.
    suspicious_subs = get_suspicious_subgraphs(
        graph=graph, partition=partition, flagged=suspicious
    )

    return {
        "partition": partition,
        "cluster_metrics": metrics,
        "suspicious_clusters": suspicious,
        "suspicious_subgraphs": suspicious_subs,
    }