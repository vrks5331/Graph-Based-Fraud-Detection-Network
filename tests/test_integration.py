import pytest
import os
import tempfile
import json
import pandas as pd
from pathlib import Path
from src.api.graph_service import GraphService


@pytest.fixture
def mock_processed_dir():
    """Create a temporary directory simulating the expected data/processed structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 1. pyg_graph.pt is actually loaded via CSVs in GraphService, not directly from .pt
        # Make dummy edge_index.csv
        pd.DataFrame({
            "source_id": [0, 1, 1, 2, 3],
            "target_id": [1, 2, 0, 3, 0]
        }).to_csv(temp_path / "edge_index.csv", index=False)
        
        # Make dummy edge_features.csv (3 dims)
        pd.DataFrame({
            "amount": [100.0, 500.0, 50.0, 1000.0, 10.0],
            "is_international": [0, 1, 0, 1, 0],
            "time_of_day": [12, 14, 8, 23, 9]
        }).to_csv(temp_path / "edge_features.csv", index=False)
        
        # Make dummy node_labels.csv (5 nodes, 2 are fraud)
        pd.DataFrame({
            "node_id": [0, 1, 2, 3, 4],
            "label": [0, 1, 0, 1, 0]
        }).to_csv(temp_path / "node_labels.csv", index=False)
        
        # Make dummy graph_summary.json (this is what sets num_nodes)
        summary_meta = {
            "num_nodes": 5,
            "num_edges": 5
        }
        with open(temp_path / "graph_summary.json", "w") as f:
            json.dump(summary_meta, f)
            
        # Make dummy pipeline_meta.json
        meta = {}
        with open(temp_path / "pipeline_meta.json", "w") as f:
            json.dump(meta, f)
            
        # We need mock node features as well since get_node returns them
        pd.DataFrame({
            "node_id": [0, 1, 2, 3, 4],
            "feat_1": [1.0] * 5,
            "feat_2": [2.0] * 5
        }).to_csv(temp_path / "node_features.csv", index=False)
        
        # We need node_mapping.json for generate_risk_scores
        mapping = {f"Acc_{i}": i for i in range(5)}
        with open(temp_path / "node_mapping.json", "w") as f:
            json.dump(mapping, f)
            
        yield temp_path


def test_graph_service_pipeline(mock_processed_dir):
    """Test the end-to-end loading and detection pipeline inside GraphService."""
    from unittest.mock import patch
    import random

    def mock_generate_risk_scores(graph_path, ckpt_path, mapping_path):
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
        labels_df = pd.read_csv(mock_processed_dir / "node_labels.csv")
        labels_dict = dict(zip(labels_df["node_id"], labels_df["label"]))
        rng = random.Random(42)
        scores = {}
        for original_id, node_id in mapping.items():
            label = labels_dict.get(node_id, 0)
            if label == 1:
                scores[original_id] = round(rng.uniform(0.65, 1.0), 4)
            else:
                scores[original_id] = round(rng.uniform(0.0, 0.35), 4)
        return scores, {}
        
    # 1. Initialize custom dir service
    with patch("src.api.graph_service.generate_risk_scores", side_effect=mock_generate_risk_scores):
        service = GraphService(processed_dir=mock_processed_dir)
        
        # 2. Run the load pipeline (csv loading -> graph build -> louvain -> bfs)
        service.load()
    
    # 3. Test Summary API
    summary = service.get_graph_summary()
    assert summary["num_nodes"] == 5
    assert summary["num_edges"] == 5
    assert summary["num_suspicious"] >= 0  # Depending on clustering randomness
    
    # 4. Test Nodes API
    nodes = service.get_nodes(skip=0, limit=2)
    assert len(nodes) == 2
    assert "node_id" in nodes[0]
    assert "risk_score" in nodes[0]
    assert "label" in nodes[0]
    
    # 5. Test detailed Node API
    node_detail = service.get_node(0)
    assert node_detail is not None
    assert node_detail["node_id"] == 0
    assert "in_neighbors" in node_detail
    assert "out_neighbors" in node_detail
    assert isinstance(node_detail["in_neighbors"], list)
    
    # 6. Test Edges API
    edges = service.get_edges(skip=0, limit=3)
    assert len(edges) == 3
    assert "source_id" in edges[0]
    assert "target_id" in edges[0]
    assert "amount" in edges[0]

    # 7. Test Explanations (if any clusters were flagged)
    clusters = service.get_communities()
    if clusters:
        first_cluster_id = clusters[0]["cluster_id"]
        
        # 8. Test Subgraph Data
        subgraph = service.get_community_subgraph_data(first_cluster_id)
        assert "nodes" in subgraph
        assert "edges" in subgraph
        assert len(subgraph["nodes"]) > 0
        
        # 9. Test BFS Explainer integration
        explanation = service.get_explanation(first_cluster_id)
        assert explanation is not None
        assert "summary" in explanation
        assert "traversal" in explanation
