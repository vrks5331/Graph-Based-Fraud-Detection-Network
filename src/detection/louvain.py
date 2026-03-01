from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

import networkx as nx

try:
    import community as community_louvain  # python-louvain
except ImportError:  # pragma: no cover
    community_louvain = None


@dataclass #basically a constructer
class ClusterMetrics: 
    cluster_id: int
    size: int
    density: float
    average_node_risk: float
    max_node_risk: float


def _to_weighted_undirected(
    graph: nx.Graph, weight_attr: str = "amount"
) -> nx.Graph:
    """
    Build a weighted undirected projection for Louvain.
    Directed and multi-edge transaction graphs are collapsed by summing
    edge weights between each account pair.
    """
    projected = nx.Graph()

    if graph.is_multigraph(): #multigraph multiple edges connecting same nodes 
        #(ex person A pays person B $1 5 times turns into Person A pays person B $5 1 time)
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

    # Retain isolated nodes so they receive singleton communities.
    for node in graph.nodes():
        if node not in projected:
            projected.add_node(node)

    return projected


def detect_louvain_communities(
    graph: nx.Graph,
    resolution: float = 1.0,
    random_state: int = 42,
    weight_attr: str = "amount",
) -> Dict[object, int]:
    """
    Run Louvain community detection and return node -> cluster_id mapping.
    """
    if graph.number_of_nodes() == 0:
        return {}

    working_graph = _to_weighted_undirected(graph, weight_attr=weight_attr)

    if community_louvain is not None:
        return community_louvain.best_partition(
            working_graph, weight="weight", resolution=resolution, random_state=random_state
        )

    # Fallback for environments where python-louvain is unavailable.
    communities = nx.community.louvain_communities(
        working_graph, weight="weight", resolution=resolution, seed=random_state
    )
    partition: Dict[object, int] = {}
    for cluster_id, members in enumerate(communities):
        for node in members:
            partition[node] = cluster_id
    return partition


def _invert_partition(partition: Dict[object, int]) -> Dict[int, Set[object]]:
    clusters: Dict[int, Set[object]] = {}
    for node, cluster_id in partition.items():
        clusters.setdefault(cluster_id, set()).add(node)
    return clusters


def compute_cluster_metrics(
    graph: nx.Graph, partition: Dict[object, int], node_risk: Optional[Dict[object, float]] = None
) -> List[ClusterMetrics]:
    """
    Compute structural + risk metrics per community.
    """
    if not partition:
        return []

    node_risk = node_risk or {}
    clusters = _invert_partition(partition)
    metrics: List[ClusterMetrics] = []

    for cluster_id, members in clusters.items():
        subgraph = graph.subgraph(members)
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

    metrics.sort(key=lambda x: (x.average_node_risk, x.density, x.size), reverse=True)
    return metrics


def flag_suspicious_clusters(
    cluster_metrics: Iterable[ClusterMetrics],
    min_size: int = 3,
    min_average_risk: float = 0.7,
    min_density: float = 0.2,
) -> List[ClusterMetrics]:
    """
    Apply threshold rules to flag likely fraud rings.
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
    """
    End-to-end Louvain detection pipeline.
    Returns partition map, per-cluster metrics, and flagged suspicious clusters.
    """
    partition = detect_louvain_communities(
        graph=graph,
        resolution=resolution,
        random_state=random_state,
        weight_attr=weight_attr,
    )
    metrics = compute_cluster_metrics(graph=graph, partition=partition, node_risk=node_risk)
    suspicious = flag_suspicious_clusters(
        cluster_metrics=metrics,
        min_size=min_size,
        min_average_risk=min_average_risk,
        min_density=min_density,
    )
    return {
        "partition": partition, # Puts the people into groups
        "cluster_metrics": metrics, #Shows the data of the groups
        "suspicious_clusters": suspicious, #Shows the suspicous people in the group
    }
