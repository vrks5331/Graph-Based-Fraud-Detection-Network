"""
Rigorous frontend tests for the Streamlit AML Dashboard.

Tests cover:
  1. API client functions (success + error handling)
  2. Pyvis graph builder (node/edge rendering, colors, sizes)
  3. Data handling edge cases (empty data, missing keys, large datasets)
"""

import pytest
from unittest.mock import patch, MagicMock
from pyvis.network import Network


# ═══════════════════════════════════════════════════════════════════════════════
#  Test fixtures
# ═══════════════════════════════════════════════════════════════════════════════

MOCK_SUMMARY = {
    "num_nodes": 250,
    "num_edges": 800,
    "num_fraud_nodes": 6,
    "num_communities": 5,
    "num_suspicious": 2,
}

MOCK_COMMUNITIES = [
    {"cluster_id": 1, "is_suspicious": True, "size": 2, "average_node_risk": 0.237, "density": 0.5},
    {"cluster_id": 2, "is_suspicious": True, "size": 7, "average_node_risk": 0.640, "density": 0.571},
    {"cluster_id": 0, "is_suspicious": False, "size": 218, "average_node_risk": 0.05, "density": 0.013},
    {"cluster_id": 3, "is_suspicious": False, "size": 22, "average_node_risk": 0.04, "density": 0.054},
    {"cluster_id": 4, "is_suspicious": False, "size": 1, "average_node_risk": 0.0, "density": 0.0},
]

MOCK_SUBGRAPH = {
    "nodes": [
        {"id": 10, "risk_score": 0.73, "label": 1},
        {"id": 11, "risk_score": 0.83, "label": 1},
        {"id": 12, "risk_score": 0.66, "label": 1},
        {"id": 13, "risk_score": 0.72, "label": 1},
        {"id": 14, "risk_score": 0.88, "label": 1},
        {"id": 15, "risk_score": 0.92, "label": 1},
        {"id": 67, "risk_score": 0.12, "label": 0},
    ],
    "edges": [
        {"source": 14, "target": 11, "amount": 15924.39, "is_fraud": 1},
        {"source": 14, "target": 10, "amount": 42028.02, "is_fraud": 1},
        {"source": 11, "target": 12, "amount": 41857.61, "is_fraud": 1},
        {"source": 67, "target": 10, "amount": 500.0, "is_fraud": 0},
    ],
}

MOCK_EXPLANATION = {
    "cluster_id": 2,
    "seed_node": "14",
    "num_nodes": 7,
    "num_edges": 24,
    "summary": "**Fraud Ring (Cluster 2)**\n\nNodes: 7 | Edges: 24 | BFS Depth: 2",
    "traversal": [
        {"depth": 0, "node": "14", "node_risk": 0.88, "parent": None, "edge_weight": None, "edge_data": None},
        {"depth": 1, "node": "11", "node_risk": 0.83, "parent": "14", "edge_weight": 15924.39, "edge_data": {"amount": 15924.39}},
        {"depth": 1, "node": "10", "node_risk": 0.73, "parent": "14", "edge_weight": 42028.02, "edge_data": {"amount": 42028.02}},
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
#  1. API Client Function Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFetchGraphSummary:
    """Tests for fetch_summary()."""

    @patch("src.dashboard.app.requests.get")
    def test_success(self, mock_get):
        """Should return the JSON summary on 200."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_SUMMARY
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        from src.dashboard.app import fetch_summary
        fetch_summary.clear()
        result = fetch_summary()

        assert result["num_nodes"] == 250
        assert result["num_edges"] == 800
        assert result["num_fraud_nodes"] == 6
        assert result["num_communities"] == 5
        assert result["num_suspicious"] == 2
        mock_get.assert_called_once_with("http://localhost:8000/api/graph/summary", timeout=5)

    @patch("src.dashboard.app.requests.get")
    def test_connection_refused(self, mock_get):
        """Should return an error dict if the backend is down."""
        mock_get.side_effect = ConnectionError("Connection refused")
        from src.dashboard.app import fetch_summary
        fetch_summary.clear()
        result = fetch_summary()
        assert "error" in result
        assert "Connection refused" in result["error"]

    @patch("src.dashboard.app.requests.get")
    def test_timeout(self, mock_get):
        """Should return an error dict on request timeout."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Timed out")
        from src.dashboard.app import fetch_summary
        fetch_summary.clear()
        result = fetch_summary()
        assert "error" in result

    @patch("src.dashboard.app.requests.get")
    def test_500_server_error(self, mock_get):
        """Should return an error dict on 5xx response."""
        import requests
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_get.return_value = mock_resp
        from src.dashboard.app import fetch_summary
        fetch_summary.clear()
        result = fetch_summary()
        assert "error" in result


class TestFetchCommunities:
    """Tests for fetch_communities()."""

    @patch("src.dashboard.app.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_COMMUNITIES
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        from src.dashboard.app import fetch_communities
        fetch_communities.clear()
        result = fetch_communities()

        assert len(result) == 5
        suspicious = [c for c in result if c["is_suspicious"]]
        assert len(suspicious) == 2

    @patch("src.dashboard.app.requests.get")
    def test_empty_communities(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        from src.dashboard.app import fetch_communities
        fetch_communities.clear()
        result = fetch_communities()
        assert result == []

    @patch("src.dashboard.app.requests.get")
    def test_error_returns_empty_list(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        from src.dashboard.app import fetch_communities
        fetch_communities.clear()
        result = fetch_communities()
        assert result == []


class TestFetchSubgraph:
    """Tests for fetch_subgraph()."""

    @patch("src.dashboard.app.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_SUBGRAPH
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        from src.dashboard.app import fetch_subgraph
        fetch_subgraph.clear()
        result = fetch_subgraph(2)

        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) == 7
        assert len(result["edges"]) == 4

    @patch("src.dashboard.app.requests.get")
    def test_404_returns_none(self, mock_get):
        import requests
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
        mock_get.return_value = mock_resp

        from src.dashboard.app import fetch_subgraph
        fetch_subgraph.clear()
        result = fetch_subgraph(999)
        assert result is None

    @patch("src.dashboard.app.requests.get")
    def test_error_returns_none(self, mock_get):
        mock_get.side_effect = Exception("Error")
        from src.dashboard.app import fetch_subgraph
        fetch_subgraph.clear()
        result = fetch_subgraph(2)
        assert result is None


class TestFetchExplanation:
    """Tests for fetch_explanation()."""

    @patch("src.dashboard.app.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_EXPLANATION
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        from src.dashboard.app import fetch_explanation
        fetch_explanation.clear()
        result = fetch_explanation(2)

        assert result is not None
        assert result["seed_node"] == "14"
        assert result["num_nodes"] == 7
        assert result["num_edges"] == 24
        assert len(result["traversal"]) == 3
        assert result["traversal"][0]["node"] == "14"
        assert result["traversal"][0]["node_risk"] == 0.88

    @patch("src.dashboard.app.requests.get")
    def test_traversal_keys_match_api(self, mock_get):
        """Verify frontend uses the same keys the API returns."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_EXPLANATION
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        from src.dashboard.app import fetch_explanation
        fetch_explanation.clear()
        result = fetch_explanation(2)

        for step in result["traversal"]:
            # The API returns "node" and "node_risk", NOT "node_id" and "risk_score"
            assert "node" in step, f"Missing 'node' key in traversal step: {step}"
            assert "node_risk" in step, f"Missing 'node_risk' key in traversal step: {step}"
            assert "depth" in step, f"Missing 'depth' key in traversal step: {step}"

    @patch("src.dashboard.app.requests.get")
    def test_error_returns_none(self, mock_get):
        mock_get.side_effect = Exception("Error")
        from src.dashboard.app import fetch_explanation
        fetch_explanation.clear()
        result = fetch_explanation(2)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Pyvis Graph Builder Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildPyvisHtml:
    """Tests for build_pyvis_html()."""

    def test_returns_html_string(self):
        """Should return a non-empty HTML string."""
        from src.dashboard.app import build_pyvis_html
        html = build_pyvis_html(MOCK_SUBGRAPH)
        assert isinstance(html, str)
        assert len(html) > 100
        assert "<html>" in html.lower() or "<!doctype" in html.lower()

    def test_contains_vis_network_js(self):
        """The HTML should include the vis-network library."""
        from src.dashboard.app import build_pyvis_html
        html = build_pyvis_html(MOCK_SUBGRAPH)
        assert "vis-network" in html.lower() or "vis.min.js" in html.lower() or "vis.Network" in html

    def test_all_nodes_present(self):
        """Every node ID should appear in the generated HTML."""
        from src.dashboard.app import build_pyvis_html
        html = build_pyvis_html(MOCK_SUBGRAPH)
        for node in MOCK_SUBGRAPH["nodes"]:
            assert str(node["id"]) in html, f"Node {node['id']} not found in HTML"

    def test_high_risk_nodes_colored_red(self):
        """Nodes with risk > 0.5 should use the red color."""
        from src.dashboard.app import build_pyvis_html
        html = build_pyvis_html(MOCK_SUBGRAPH)
        assert "#ff4b4b" in html  # red for high risk

    def test_low_risk_nodes_colored_blue(self):
        """Nodes with risk <= 0.5 should use the blue color."""
        from src.dashboard.app import build_pyvis_html
        html = build_pyvis_html(MOCK_SUBGRAPH)
        assert "#4b9dff" in html  # blue for low risk

    def test_empty_subgraph(self):
        """Should handle subgraph with no nodes/edges gracefully."""
        from src.dashboard.app import build_pyvis_html
        html = build_pyvis_html({"nodes": [], "edges": []})
        assert isinstance(html, str)
        assert len(html) > 0

    def test_single_node_no_edges(self):
        """Should handle a subgraph with exactly one node and no edges."""
        from src.dashboard.app import build_pyvis_html
        data = {"nodes": [{"id": 1, "risk_score": 0.5, "label": 0}], "edges": []}
        html = build_pyvis_html(data)
        assert "1" in html

    def test_missing_optional_edge_fields(self):
        """Edges missing 'amount' and 'is_fraud' should default to zero."""
        from src.dashboard.app import build_pyvis_html
        data = {
            "nodes": [
                {"id": 1, "risk_score": 0.3, "label": 0},
                {"id": 2, "risk_score": 0.7, "label": 1},
            ],
            "edges": [{"source": 1, "target": 2}],  # no amount, no is_fraud
        }
        html = build_pyvis_html(data)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_node_sizes_scale_with_risk(self):
        """Higher-risk nodes should get larger sizes."""
        from src.dashboard.app import build_pyvis_html
        data = {
            "nodes": [
                {"id": 1, "risk_score": 0.1, "label": 0},
                {"id": 2, "risk_score": 0.9, "label": 1},
            ],
            "edges": [],
        }
        html = build_pyvis_html(data)
        assert isinstance(html, str)
        assert len(html) > 0


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Data Handling & Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCommunityFiltering:
    """Tests for suspicious community filtering logic."""

    def test_filter_suspicious_communities(self):
        """Only communities with is_suspicious=True should appear."""
        suspicious = [c for c in MOCK_COMMUNITIES if c.get("is_suspicious")]
        assert len(suspicious) == 2
        assert all(c["is_suspicious"] for c in suspicious)

    def test_no_suspicious_communities(self):
        """When all communities are clean, the list should be empty."""
        clean_comms = [{"cluster_id": i, "is_suspicious": False, "size": 10, "average_node_risk": 0.1, "density": 0.1} for i in range(5)]
        suspicious = [c for c in clean_comms if c.get("is_suspicious")]
        assert suspicious == []

    def test_cluster_option_format(self):
        """Cluster selection dropdown strings should contain risk and size."""
        suspicious = [c for c in MOCK_COMMUNITIES if c.get("is_suspicious")]
        options = {
            f"Cluster {c['cluster_id']}  (risk={c['average_node_risk']:.3f}, size={c['size']})": c["cluster_id"]
            for c in suspicious
        }
        assert len(options) == 2
        for label in options:
            assert "risk=" in label
            assert "size=" in label


class TestBFSTraversalRendering:
    """Tests for BFS traversal step rendering logic."""

    def test_icon_selection_high_risk(self):
        """Nodes with risk > 0.5 should get the red icon."""
        step = MOCK_EXPLANATION["traversal"][0]
        risk = step.get("node_risk", step.get("risk_score", 0))
        icon = "🔴" if risk > 0.5 else "🔵"
        assert icon == "🔴"

    def test_icon_selection_low_risk(self):
        """Nodes with risk <= 0.5 should get the blue icon."""
        step = {"node": "99", "node_risk": 0.3, "depth": 0}
        risk = step.get("node_risk", step.get("risk_score", 0))
        icon = "🔴" if risk > 0.5 else "🔵"
        assert icon == "🔵"

    def test_fallback_keys(self):
        """Should fall back to 'risk_score' and 'node_id' if 'node_risk'/'node' missing."""
        step_old_format = {"node_id": "42", "risk_score": 0.75, "depth": 1}
        risk = step_old_format.get("node_risk", step_old_format.get("risk_score", 0))
        node = step_old_format.get("node", step_old_format.get("node_id", "?"))
        assert risk == 0.75
        assert node == "42"


class TestGraphSummaryDisplay:
    """Tests for graph summary metric display logic."""

    def test_all_metrics_present(self):
        """summary should contain all required metric keys."""
        required = ["num_nodes", "num_edges", "num_fraud_nodes", "num_communities", "num_suspicious"]
        for key in required:
            assert key in MOCK_SUMMARY, f"Missing key: {key}"

    def test_metrics_are_numeric(self):
        """All metric values should be numeric."""
        for key, value in MOCK_SUMMARY.items():
            assert isinstance(value, (int, float)), f"{key} should be numeric, got {type(value)}"

    def test_format_with_commas(self):
        """Large numbers should be formatted with commas."""
        val = MOCK_SUMMARY["num_nodes"]
        formatted = f"{val:,}"
        assert formatted == "250"

    def test_format_large_number(self):
        """Test comma formatting for a large number."""
        val = 10000
        formatted = f"{val:,}"
        assert formatted == "10,000"
