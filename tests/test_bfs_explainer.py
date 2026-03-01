"""Tests for the BFS explainer module.

Covers: empty subgraphs, single-node communities, BFS traversal order,
seed selection, summary generation, and full Louvain → BFS integration.
"""

import networkx as nx
import pytest

from src.detection.louvain import (
    run_louvain_detection,
    get_community_subgraph,
    detect_louvain_communities,
)
from src.explainability.bfs_explainer import (
    ClusterExplanation,
    explain_all_clusters,
    explain_cluster,
    _select_seed_node,
    _bfs_traverse,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def empty_subgraph():
    """An empty graph with no nodes."""
    return nx.Graph()


@pytest.fixture
def single_node_subgraph():
    """A graph with one isolated node."""
    G = nx.Graph()
    G.add_node("A")
    return G


@pytest.fixture
def linear_chain():
    """A simple chain: A — B — C — D with increasing edge weights.

    A (risk=0.9) → B (risk=0.6) → C (risk=0.3) → D (risk=0.1)
    """
    G = nx.Graph()
    G.add_edge("A", "B", weight=100.0)
    G.add_edge("B", "C", weight=200.0)
    G.add_edge("C", "D", weight=300.0)
    return G


@pytest.fixture
def linear_chain_risk():
    """Risk scores for the linear chain fixture."""
    return {"A": 0.9, "B": 0.6, "C": 0.3, "D": 0.1}


@pytest.fixture
def triangle_ring():
    """A triangular fraud ring: X — Y — Z — X with equal weights."""
    G = nx.Graph()
    G.add_edge("X", "Y", weight=500.0)
    G.add_edge("Y", "Z", weight=500.0)
    G.add_edge("Z", "X", weight=500.0)
    return G


@pytest.fixture
def triangle_risk():
    """Risk scores for the triangle ring."""
    return {"X": 0.95, "Y": 0.85, "Z": 0.80}


@pytest.fixture
def two_cliques_graph():
    """Two disconnected 4-node cliques for Louvain integration testing."""
    G = nx.Graph()
    # Clique A: nodes 0–3 (high-risk)
    for i in range(4):
        for j in range(i + 1, 4):
            G.add_edge(i, j, weight=10.0)
    # Clique B: nodes 10–13 (low-risk)
    for i in range(10, 14):
        for j in range(i + 1, 14):
            G.add_edge(i, j, weight=10.0)
    return G


@pytest.fixture
def two_cliques_risk():
    """High risk for clique A, low risk for clique B."""
    return {
        0: 0.9, 1: 0.85, 2: 0.95, 3: 0.8,
        10: 0.1, 11: 0.05, 12: 0.15, 13: 0.1,
    }


# ── Tests: Empty & Single-Node Subgraphs ────────────────────────────────────


class TestEdgeCases:
    def test_empty_subgraph(self, empty_subgraph):
        """BFS on an empty subgraph returns a valid but empty explanation."""
        expl = explain_cluster(empty_subgraph, cluster_id=0)
        assert expl.num_nodes == 0
        assert expl.traversal == []
        assert expl.seed_node is None
        assert "empty" in expl.summary.lower()

    def test_single_node(self, single_node_subgraph):
        """BFS on a single node returns one traversal step with depth 0."""
        risk = {"A": 0.75}
        expl = explain_cluster(single_node_subgraph, cluster_id=1, node_risk=risk)
        assert expl.num_nodes == 1
        assert expl.seed_node == "A"
        assert len(expl.traversal) == 1
        assert expl.traversal[0].depth == 0
        assert expl.traversal[0].parent is None


# ── Tests: Seed Selection ────────────────────────────────────────────────────


class TestSeedSelection:
    def test_highest_risk_selected(self, linear_chain, linear_chain_risk):
        """The node with the highest risk score should be selected as seed."""
        seed = _select_seed_node(linear_chain, linear_chain_risk)
        assert seed == "A"  # A has risk 0.9, the highest

    def test_tie_broken_by_degree(self):
        """When two nodes have equal risk, the one with higher degree wins."""
        G = nx.Graph()
        G.add_edge("hub", "spoke1", weight=1.0)
        G.add_edge("hub", "spoke2", weight=1.0)
        G.add_edge("hub", "spoke3", weight=1.0)
        G.add_edge("leaf", "spoke1", weight=1.0)
        # hub has degree 3, leaf has degree 1 — same risk
        risk = {"hub": 0.9, "spoke1": 0.5, "spoke2": 0.5, "spoke3": 0.5, "leaf": 0.9}
        seed = _select_seed_node(G, risk)
        assert seed == "hub"


# ── Tests: BFS Traversal ────────────────────────────────────────────────────


class TestBFSTraversal:
    def test_visits_all_nodes(self, linear_chain, linear_chain_risk):
        """BFS should visit every node in the subgraph."""
        steps = _bfs_traverse(linear_chain, "A", linear_chain_risk)
        visited_nodes = {s.node for s in steps}
        assert visited_nodes == {"A", "B", "C", "D"}

    def test_traversal_order_is_bfs(self, linear_chain, linear_chain_risk):
        """In a linear chain starting from A, BFS order is A → B → C → D."""
        steps = _bfs_traverse(linear_chain, "A", linear_chain_risk)
        order = [s.node for s in steps]
        assert order == ["A", "B", "C", "D"]

    def test_depths_are_correct(self, linear_chain, linear_chain_risk):
        """Depth should increase by 1 at each step in a chain."""
        steps = _bfs_traverse(linear_chain, "A", linear_chain_risk)
        depths = [s.depth for s in steps]
        assert depths == [0, 1, 2, 3]

    def test_parent_links(self, linear_chain, linear_chain_risk):
        """Each step should reference the correct parent node."""
        steps = _bfs_traverse(linear_chain, "A", linear_chain_risk)
        parents = [s.parent for s in steps]
        assert parents == [None, "A", "B", "C"]

    def test_edge_weights_captured(self, linear_chain, linear_chain_risk):
        """Edge weights should be recorded in each traversal step."""
        steps = _bfs_traverse(linear_chain, "A", linear_chain_risk)
        # Root has no edge weight; others have the chain weights.
        assert steps[0].edge_weight is None
        assert steps[1].edge_weight == 100.0
        assert steps[2].edge_weight == 200.0
        assert steps[3].edge_weight == 300.0

    def test_triangle_all_visited(self, triangle_ring, triangle_risk):
        """BFS on a triangle visits all 3 nodes within depth ≤ 1."""
        steps = _bfs_traverse(triangle_ring, "X", triangle_risk)
        assert len(steps) == 3
        # Seed at depth 0, the other two at depth 1.
        assert steps[0].depth == 0
        assert {s.depth for s in steps[1:]} == {1}


# ── Tests: Explain Cluster ───────────────────────────────────────────────────


class TestExplainCluster:
    def test_explanation_structure(self, triangle_ring, triangle_risk):
        """explain_cluster returns a well-formed ClusterExplanation."""
        expl = explain_cluster(
            triangle_ring, cluster_id=42, node_risk=triangle_risk
        )
        assert isinstance(expl, ClusterExplanation)
        assert expl.cluster_id == 42
        assert expl.num_nodes == 3
        assert expl.num_edges == 3
        assert expl.seed_node == "X"  # highest risk
        assert len(expl.traversal) == 3

    def test_summary_contains_key_info(self, triangle_ring, triangle_risk):
        """The summary should mention cluster ID, node count, and seed node."""
        expl = explain_cluster(
            triangle_ring, cluster_id=7, node_risk=triangle_risk
        )
        assert "Cluster 7" in expl.summary
        assert "3" in expl.summary  # num nodes
        assert "X" in expl.summary  # seed node

    def test_no_risk_defaults_to_zero(self, linear_chain):
        """When node_risk is None, all risks should default to 0.0."""
        expl = explain_cluster(linear_chain, cluster_id=0)
        for step in expl.traversal:
            assert step.node_risk == 0.0


# ── Tests: Full Louvain → BFS Integration ────────────────────────────────────


class TestLouvainBFSIntegration:
    def test_end_to_end_pipeline(self, two_cliques_graph, two_cliques_risk):
        """Run Louvain → extract suspicious subgraphs → BFS explain them."""
        # Run the full Louvain detection pipeline.
        results = run_louvain_detection(
            two_cliques_graph,
            node_risk=two_cliques_risk,
            min_size=3,
            min_average_risk=0.5,
            min_density=0.5,
        )

        # Only the high-risk clique A should be flagged.
        assert len(results["suspicious_clusters"]) == 1

        # Explain all flagged clusters.
        explanations = explain_all_clusters(
            results["suspicious_subgraphs"],
            node_risk=two_cliques_risk,
        )

        assert len(explanations) == 1
        expl = explanations[0]

        # The explanation should cover the 4-node clique.
        assert expl.num_nodes == 4
        assert expl.num_edges == 6  # complete graph K4

        # Seed should be node 2 (risk=0.95, highest in clique A).
        assert expl.seed_node == 2

        # BFS should visit all 4 nodes.
        visited = {s.node for s in expl.traversal}
        assert visited == {0, 1, 2, 3}

    def test_no_suspicious_means_no_explanations(self, two_cliques_graph, two_cliques_risk):
        """If no clusters are flagged, explain_all_clusters returns empty."""
        results = run_louvain_detection(
            two_cliques_graph,
            node_risk=two_cliques_risk,
            min_average_risk=0.99,  # Unreachable threshold.
        )
        explanations = explain_all_clusters(
            results["suspicious_subgraphs"],
            node_risk=two_cliques_risk,
        )
        assert explanations == []

    def test_subgraph_isolation(self, two_cliques_graph, two_cliques_risk):
        """BFS should only traverse within the extracted community subgraph,
        never leaking into other communities."""
        results = run_louvain_detection(
            two_cliques_graph,
            node_risk=two_cliques_risk,
            min_size=3,
            min_average_risk=0.5,
            min_density=0.5,
        )
        explanations = explain_all_clusters(
            results["suspicious_subgraphs"],
            node_risk=two_cliques_risk,
        )
        for expl in explanations:
            traversed_nodes = {s.node for s in expl.traversal}
            # No node from the low-risk clique B (nodes 10–13) should appear.
            assert traversed_nodes.isdisjoint({10, 11, 12, 13})

    def test_individual_community_subgraph(self, two_cliques_graph):
        """Extract a single community subgraph and explain it."""
        partition = detect_louvain_communities(two_cliques_graph)
        cid_a = partition[0]
        sub = get_community_subgraph(two_cliques_graph, partition, cid_a)

        risk = {0: 0.9, 1: 0.85, 2: 0.95, 3: 0.8}
        expl = explain_cluster(sub, cluster_id=cid_a, node_risk=risk)

        assert expl.num_nodes == 4
        assert expl.seed_node == 2
        assert len(expl.traversal) == 4
