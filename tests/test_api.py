"""
Unit tests for the FastAPI backend and GraphService.

These tests are **self-contained** — they synthesise small graphs from scratch
rather than depending on the real data files, so they generalise to any dataset.
"""

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

# ── Fixtures: build a tiny processed-data directory on the fly ───────────────


def _make_processed_dir(tmp_path: Path, num_nodes: int = 20) -> Path:
    """Create a minimal set of CSV/JSON files that GraphService can load.

    Constructs a small graph with:
    - ``num_nodes`` nodes (IDs 0..num_nodes-1)
    - A dense "fraud ring" among nodes 0-4 (clique)
    - Sparse legit connections among remaining nodes
    - Labels: nodes 0-4 = fraud (1), rest = legit (0)
    """
    out = tmp_path / "processed"
    out.mkdir()

    # -- node features --
    rows = []
    for i in range(num_nodes):
        rows.append(
            {
                "node_id": i,
                "init_balance": 100.0 + i,
                "account_type_enc": 0,
                "country_enc": 0,
                "tx_behavior_id": 1,
                "out_tx_count": float(i % 5 + 1),
                "out_tx_sum": float((i % 5 + 1) * 50),
                "out_tx_mean": 50.0,
                "out_tx_std": 0.0,
                "out_tx_max": 50.0,
                "in_tx_count": float(i % 4),
                "in_tx_sum": float((i % 4) * 30),
                "in_tx_mean": 30.0,
                "in_tx_std": 0.0,
                "in_tx_max": 30.0,
                "out_unique_nbrs": 2.0,
                "in_unique_nbrs": 1.0,
                "total_tx_count": float(i % 5 + i % 4 + 1),
                "net_flow": 20.0,
                "flow_ratio": 1.5,
                "tx_time_span": 100.0,
                "unique_counterparts": 3.0,
            }
        )
    pd.DataFrame(rows).to_csv(out / "node_features.csv", index=False)

    # -- edge index & edge features --
    edges_src, edges_tgt = [], []
    amounts, timestamps, is_frauds, log_amounts = [], [], [], []

    # Dense clique among fraud ring (nodes 0-4)
    for i in range(5):
        for j in range(5):
            if i != j:
                edges_src.append(i)
                edges_tgt.append(j)
                amounts.append(500.0)
                timestamps.append(0)
                is_frauds.append(1)
                log_amounts.append(6.2)

    # Sparse legit edges
    for i in range(5, num_nodes - 1):
        edges_src.append(i)
        edges_tgt.append(i + 1)
        amounts.append(50.0)
        timestamps.append(0)
        is_frauds.append(0)
        log_amounts.append(3.9)

    pd.DataFrame({"source_id": edges_src, "target_id": edges_tgt}).to_csv(
        out / "edge_index.csv", index=False
    )
    pd.DataFrame(
        {
            "amount": amounts,
            "timestamp": timestamps,
            "is_fraud": is_frauds,
            "log_amount": log_amounts,
        }
    ).to_csv(out / "edge_features.csv", index=False)

    # -- node labels --
    labels = [{"node_id": i, "label": 1 if i < 5 else 0} for i in range(num_nodes)]
    pd.DataFrame(labels).to_csv(out / "node_labels.csv", index=False)

    # -- graph summary --
    summary = {
        "num_nodes": num_nodes,
        "num_edges": len(edges_src),
        "density": 0.05,
        "num_fraud_nodes": 5,
        "num_legit_nodes": num_nodes - 5,
        "fraud_rate_pct": round(5 / num_nodes * 100, 2),
    }
    with open(out / "graph_summary.json", "w") as f:
        json.dump(summary, f)

    # -- pipeline meta --
    meta = {
        "num_nodes": num_nodes,
        "num_edges": len(edges_src),
        "node_feature_dim": 21,
        "edge_feature_dim": 4,
    }
    with open(out / "pipeline_meta.json", "w") as f:
        json.dump(meta, f)

    return out


@pytest.fixture(scope="module")
def processed_dir(tmp_path_factory):
    """Create a shared processed-data directory for all tests in this module."""
    return _make_processed_dir(tmp_path_factory.mktemp("data"))


# ═══════════════════════════════════════════════════════════════════════════════
#  1. GraphService unit tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGraphService:
    """Tests for GraphService data loading and pipeline execution."""

    @pytest.fixture(autouse=True)
    def _setup(self, processed_dir):
        from src.api.graph_service import GraphService

        self.svc = GraphService(processed_dir=processed_dir).load()

    # -- loading --

    def test_graph_loaded(self):
        """Graph should be a non-empty DiGraph."""
        assert self.svc.graph is not None
        assert isinstance(self.svc.graph, nx.DiGraph)
        assert self.svc.graph.number_of_nodes() == 20

    def test_node_features_loaded(self):
        """Node features DataFrame should have the correct shape."""
        assert self.svc.node_features is not None
        assert len(self.svc.node_features) == 20
        assert "node_id" in self.svc.node_features.columns
        assert "init_balance" in self.svc.node_features.columns

    def test_edge_data_loaded(self):
        """Edge index and features should be aligned."""
        assert len(self.svc.edge_index) == len(self.svc.edge_features)
        assert len(self.svc.edge_index) > 0

    def test_labels_loaded(self):
        """Labels dict should map node IDs to 0 or 1."""
        assert len(self.svc.node_labels) == 20
        assert self.svc.node_labels[0] == 1  # fraud node
        assert self.svc.node_labels[10] == 0  # legit node

    def test_json_meta_loaded(self):
        """graph_summary and pipeline_meta should be loaded."""
        assert self.svc.graph_summary is not None
        assert self.svc.graph_summary["num_nodes"] == 20
        assert self.svc.pipeline_meta is not None

    # -- mock risk scores --

    def test_risk_scores_generated(self):
        """Every node should have a risk score between 0 and 1."""
        assert len(self.svc.node_risk) == 20
        for nid, score in self.svc.node_risk.items():
            assert 0.0 <= score <= 1.0

    def test_fraud_nodes_get_high_risk(self):
        """Fraud-labelled nodes should have risk >= 0.65."""
        for nid in range(5):
            assert self.svc.node_risk[nid] >= 0.65

    def test_legit_nodes_get_low_risk(self):
        """Legit-labelled nodes should have risk <= 0.35."""
        for nid in range(5, 20):
            assert self.svc.node_risk[nid] <= 0.35

    # -- pipeline --

    def test_partition_exists(self):
        """Louvain partition should assign every node a community."""
        assert self.svc.partition is not None
        assert len(self.svc.partition) == 20

    def test_cluster_metrics_computed(self):
        """At least one cluster metric should exist."""
        assert self.svc.cluster_metrics is not None
        assert len(self.svc.cluster_metrics) > 0

    # -- query: graph summary --

    def test_get_graph_summary(self):
        """Summary should include both base stats and community info."""
        s = self.svc.get_graph_summary()
        assert "num_nodes" in s
        assert "num_communities" in s
        assert "num_suspicious" in s

    # -- query: nodes --

    def test_get_nodes_pagination(self):
        """get_nodes should respect skip and limit."""
        all_nodes = self.svc.get_nodes(skip=0, limit=100)
        assert len(all_nodes) == 20

        first_five = self.svc.get_nodes(skip=0, limit=5)
        assert len(first_five) == 5

        second_five = self.svc.get_nodes(skip=5, limit=5)
        assert len(second_five) == 5
        assert first_five[0]["node_id"] != second_five[0]["node_id"]

    def test_get_nodes_includes_risk_and_label(self):
        """Each node dict should have risk_score and label."""
        nodes = self.svc.get_nodes(skip=0, limit=1)
        assert "risk_score" in nodes[0]
        assert "label" in nodes[0]

    def test_get_node_found(self):
        """Valid node_id should return a populated dict."""
        n = self.svc.get_node(0)
        assert n is not None
        assert n["node_id"] == 0
        assert "risk_score" in n
        assert "community_id" in n
        assert "out_neighbors" in n
        assert "in_neighbors" in n

    def test_get_node_not_found(self):
        """Invalid node_id should return None."""
        assert self.svc.get_node(99999) is None

    # -- query: edges --

    def test_get_edges_pagination(self):
        """get_edges should return at most `limit` entries."""
        edges = self.svc.get_edges(skip=0, limit=5)
        assert len(edges) <= 5
        assert "source_id" in edges[0]
        assert "target_id" in edges[0]
        assert "amount" in edges[0]

    # -- query: communities --

    def test_get_communities(self):
        """Should return a list of community dicts with metrics."""
        comms = self.svc.get_communities()
        assert isinstance(comms, list)
        assert len(comms) > 0
        c = comms[0]
        assert "cluster_id" in c
        assert "size" in c
        assert "density" in c
        assert "is_suspicious" in c

    def test_get_community_found(self):
        """Requesting a valid cluster_id should return a detail dict."""
        comms = self.svc.get_communities()
        cid = comms[0]["cluster_id"]
        detail = self.svc.get_community(cid)
        assert detail is not None
        assert "member_nodes" in detail
        assert len(detail["member_nodes"]) == detail["size"]

    def test_get_community_not_found(self):
        assert self.svc.get_community(999999) is None

    # -- query: subgraph --

    def test_get_community_subgraph_data(self):
        """Subgraph data should include nodes and edges lists."""
        comms = self.svc.get_communities()
        cid = comms[0]["cluster_id"]
        sg = self.svc.get_community_subgraph_data(cid)
        assert sg is not None
        assert "nodes" in sg
        assert "edges" in sg
        assert isinstance(sg["nodes"], list)

    def test_get_community_subgraph_not_found(self):
        assert self.svc.get_community_subgraph_data(999999) is None


# ═══════════════════════════════════════════════════════════════════════════════
#  2. FastAPI endpoint tests (using TestClient)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def client(processed_dir):
    """Create a FastAPI TestClient backed by our synthetic data."""
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from src.api.graph_service import GraphService
    from src.api.main import app, get_service

    svc = GraphService(processed_dir=processed_dir).load()

    # Override the service singleton so the app uses our test data
    def _override():
        return svc

    with patch("src.api.main.get_service", _override):
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["nodes_loaded"] == 20


class TestGraphEndpoints:
    def test_summary(self, client):
        r = client.get("/api/graph/summary")
        assert r.status_code == 200
        data = r.json()
        assert "num_nodes" in data
        assert "num_communities" in data

    def test_nodes_default(self, client):
        r = client.get("/api/graph/nodes")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 20  # all 20 fit in default limit=100

    def test_nodes_pagination(self, client):
        r = client.get("/api/graph/nodes?skip=0&limit=3")
        assert r.status_code == 200
        assert len(r.json()) == 3

    def test_node_detail(self, client):
        r = client.get("/api/graph/nodes/0")
        assert r.status_code == 200
        data = r.json()
        assert data["node_id"] == 0
        assert "risk_score" in data

    def test_node_not_found(self, client):
        r = client.get("/api/graph/nodes/99999")
        assert r.status_code == 404

    def test_edges_default(self, client):
        r = client.get("/api/graph/edges")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert "source_id" in data[0]
        assert "amount" in data[0]

    def test_edges_pagination(self, client):
        r = client.get("/api/graph/edges?skip=0&limit=5")
        assert r.status_code == 200
        assert len(r.json()) <= 5


class TestCommunityEndpoints:
    def test_communities_list(self, client):
        r = client.get("/api/communities")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert "cluster_id" in data[0]
        assert "is_suspicious" in data[0]

    def test_community_detail(self, client):
        comms = client.get("/api/communities").json()
        cid = comms[0]["cluster_id"]
        r = client.get(f"/api/communities/{cid}")
        assert r.status_code == 200
        data = r.json()
        assert "member_nodes" in data

    def test_community_not_found(self, client):
        r = client.get("/api/communities/999999")
        assert r.status_code == 404

    def test_community_subgraph(self, client):
        comms = client.get("/api/communities").json()
        cid = comms[0]["cluster_id"]
        r = client.get(f"/api/communities/{cid}/subgraph")
        assert r.status_code == 200
        data = r.json()
        assert "nodes" in data
        assert "edges" in data

    def test_community_subgraph_not_found(self, client):
        r = client.get("/api/communities/999999/subgraph")
        assert r.status_code == 404


class TestExplanationEndpoints:
    def test_explanations_list(self, client):
        r = client.get("/api/explanations")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        # May be empty if no suspicious clusters meet thresholds
        if len(data) > 0:
            assert "cluster_id" in data[0]
            assert "summary" in data[0]
            assert "traversal" in data[0]

    def test_explanation_not_found(self, client):
        r = client.get("/api/explanations/999999")
        assert r.status_code == 404

    def test_explanation_detail_if_exists(self, client):
        """If there are explanations, verify the detail endpoint works."""
        explanations = client.get("/api/explanations").json()
        if len(explanations) > 0:
            cid = explanations[0]["cluster_id"]
            r = client.get(f"/api/explanations/{cid}")
            assert r.status_code == 200
            data = r.json()
            assert data["cluster_id"] == cid
            assert "traversal" in data
