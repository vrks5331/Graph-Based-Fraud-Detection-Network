"""Tests for the Louvain community detection module.

Covers: empty graphs, single-community triangles, multi-community separation,
suspicious flagging, and subgraph extraction for BFS integration.
"""

import networkx as nx
import pytest

from src.detection.louvain import (
    ClusterMetrics,
    compute_cluster_metrics,
    detect_louvain_communities,
    flag_suspicious_clusters,
    get_all_community_subgraphs,
    get_community_subgraph,
    run_louvain_detection,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def empty_graph():
    """An empty graph with no nodes or edges."""
    return nx.Graph()


@pytest.fixture
def triangle_graph():
    """A simple triangle: 3 nodes, 3 edges, all weight 1."""
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)])
    return G


@pytest.fixture
def two_cliques_graph():
    """Two disconnected 4-node cliques (should land in separate communities)."""
    G = nx.Graph()
    # Clique A: nodes 0–3
    for i in range(4):
        for j in range(i + 1, 4):
            G.add_edge(i, j, weight=10.0)
    # Clique B: nodes 10–13
    for i in range(10, 14):
        for j in range(i + 1, 14):
            G.add_edge(i, j, weight=10.0)
    return G


@pytest.fixture
def risky_node_scores():
    """High risk scores for clique A, low for clique B."""
    return {
        0: 0.9, 1: 0.85, 2: 0.95, 3: 0.8,
        10: 0.1, 11: 0.05, 12: 0.15, 13: 0.1,
    }


# ── Tests: Empty Graph ──────────────────────────────────────────────────────


class TestEmptyGraph:
    def test_detect_returns_empty(self, empty_graph):
        assert detect_louvain_communities(empty_graph) == {}

    def test_run_pipeline_returns_empty(self, empty_graph):
        result = run_louvain_detection(empty_graph)
        assert result["partition"] == {}
        assert result["cluster_metrics"] == []
        assert result["suspicious_clusters"] == []
        assert result["suspicious_subgraphs"] == {}


# ── Tests: Single Community (Triangle) ───────────────────────────────────────


class TestTriangleGraph:
    def test_single_community(self, triangle_graph):
        partition = detect_louvain_communities(triangle_graph)
        # All 3 nodes should share the same community ID.
        assert len(set(partition.values())) == 1
        assert set(partition.keys()) == {0, 1, 2}

    def test_subgraph_has_all_edges(self, triangle_graph):
        partition = detect_louvain_communities(triangle_graph)
        cid = list(set(partition.values()))[0]
        sub = get_community_subgraph(triangle_graph, partition, cid)
        assert set(sub.nodes()) == {0, 1, 2}
        assert sub.number_of_edges() == 3


# ── Tests: Two Disconnected Cliques ─────────────────────────────────────────


class TestTwoCliques:
    def test_two_communities(self, two_cliques_graph):
        partition = detect_louvain_communities(two_cliques_graph)
        community_ids = set(partition.values())
        assert len(community_ids) == 2

    def test_cliques_in_separate_communities(self, two_cliques_graph):
        partition = detect_louvain_communities(two_cliques_graph)
        clique_a_ids = {partition[n] for n in range(4)}
        clique_b_ids = {partition[n] for n in range(10, 14)}
        # Each clique should map to exactly one community, and they differ.
        assert len(clique_a_ids) == 1
        assert len(clique_b_ids) == 1
        assert clique_a_ids != clique_b_ids

    def test_all_subgraphs(self, two_cliques_graph):
        partition = detect_louvain_communities(two_cliques_graph)
        subgraphs = get_all_community_subgraphs(two_cliques_graph, partition)
        assert len(subgraphs) == 2
        for sub in subgraphs.values():
            assert sub.number_of_nodes() == 4


# ── Tests: Suspicious Flagging ───────────────────────────────────────────────


class TestSuspiciousFlagging:
    def test_high_risk_clique_flagged(self, two_cliques_graph, risky_node_scores):
        result = run_louvain_detection(
            two_cliques_graph,
            node_risk=risky_node_scores,
            min_size=3,
            min_average_risk=0.5,
            min_density=0.5,
        )
        # Only the high-risk clique A should be flagged.
        assert len(result["suspicious_clusters"]) == 1
        flagged = result["suspicious_clusters"][0]
        assert flagged.average_node_risk >= 0.5
        assert flagged.size == 4

    def test_no_false_positives(self, two_cliques_graph, risky_node_scores):
        result = run_louvain_detection(
            two_cliques_graph,
            node_risk=risky_node_scores,
            min_size=3,
            min_average_risk=0.99,  # Very strict threshold.
            min_density=0.5,
        )
        # Even the risky clique averages ~0.875, below 0.99.
        assert len(result["suspicious_clusters"]) == 0
        assert result["suspicious_subgraphs"] == {}


# ── Tests: Subgraph Extraction ───────────────────────────────────────────────


class TestSubgraphExtraction:
    def test_get_community_subgraph_preserves_edges(self, two_cliques_graph):
        partition = detect_louvain_communities(two_cliques_graph)
        cid_a = partition[0]
        sub = get_community_subgraph(two_cliques_graph, partition, cid_a)
        # 4 nodes, C(4,2) = 6 edges in a complete subgraph.
        assert sub.number_of_nodes() == 4
        assert sub.number_of_edges() == 6

    def test_invalid_cluster_id_raises(self, two_cliques_graph):
        partition = detect_louvain_communities(two_cliques_graph)
        with pytest.raises(ValueError, match="cluster_id 999 not found"):
            get_community_subgraph(two_cliques_graph, partition, 999)

    def test_suspicious_subgraphs_match_flagged(
        self, two_cliques_graph, risky_node_scores
    ):
        result = run_louvain_detection(
            two_cliques_graph,
            node_risk=risky_node_scores,
            min_size=3,
            min_average_risk=0.5,
            min_density=0.5,
        )
        # suspicious_subgraphs keys should exactly match flagged cluster IDs.
        flagged_ids = {c.cluster_id for c in result["suspicious_clusters"]}
        assert set(result["suspicious_subgraphs"].keys()) == flagged_ids


# ── Tests: Cluster Metrics ───────────────────────────────────────────────────


class TestClusterMetrics:
    def test_density_complete_graph(self, triangle_graph):
        partition = detect_louvain_communities(triangle_graph)
        metrics = compute_cluster_metrics(triangle_graph, partition)
        # A triangle is a complete graph → density should be 1.0.
        assert len(metrics) == 1
        assert metrics[0].density == pytest.approx(1.0)

    def test_risk_aggregation(self, two_cliques_graph, risky_node_scores):
        partition = detect_louvain_communities(two_cliques_graph)
        metrics = compute_cluster_metrics(
            two_cliques_graph, partition, node_risk=risky_node_scores
        )
        # Sorted descending by risk, so the first entry is the risky clique.
        assert metrics[0].average_node_risk > metrics[1].average_node_risk
        assert metrics[0].max_node_risk == pytest.approx(0.95)

    def test_flag_thresholds(self):
        clusters = [
            ClusterMetrics(0, size=5, density=0.8, average_node_risk=0.9, max_node_risk=0.95),
            ClusterMetrics(1, size=2, density=0.5, average_node_risk=0.9, max_node_risk=0.9),  # too small
            ClusterMetrics(2, size=5, density=0.8, average_node_risk=0.3, max_node_risk=0.5),  # low risk
        ]
        flagged = flag_suspicious_clusters(clusters, min_size=3, min_average_risk=0.7, min_density=0.2)
        assert len(flagged) == 1
        assert flagged[0].cluster_id == 0
