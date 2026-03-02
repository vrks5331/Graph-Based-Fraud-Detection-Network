"""
Graph Service — Data Loading & Pipeline Orchestration
======================================================

Singleton service that:

1. Loads processed graph artifacts (node features, edge index, labels)
2. Builds a NetworkX DiGraph from those artifacts
3. Generates mock GNN risk scores (swap for real model output later)
4. Runs Louvain community detection → cluster metrics → suspicious flagging
5. Runs BFS explainer on each suspicious community

The service is initialised once at FastAPI startup and shared across all
request handlers via dependency injection.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.models.inference import generate_risk_scores
import networkx as nx
import pandas as pd

# ── Internal pipeline imports ────────────────────────────────────────────────
from src.detection.louvain import (
    ClusterMetrics,
    run_louvain_detection,
)
from src.explainability.bfs_explainer import (
    ClusterExplanation,
    explain_all_clusters,
)

# ── Path defaults ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED = _PROJECT_ROOT / "data" / "testing_processed"


# ═══════════════════════════════════════════════════════════════════════════════
#  GraphService
# ═══════════════════════════════════════════════════════════════════════════════


class GraphService:
    """Loads graph data and exposes query methods for the API layer."""

    # ── Construction ────────────────────────────────────────────────────────

    def __init__(self, processed_dir: Optional[Path] = None) -> None:
        self._dir = Path(processed_dir) if processed_dir else _PROCESSED

        # Populated by load()
        self.graph: Optional[nx.DiGraph] = None
        self.node_features: Optional[pd.DataFrame] = None
        self.node_labels: Optional[Dict[int, int]] = None
        self.node_risk: Optional[Dict[int, float]] = None
        self.edge_index: Optional[pd.DataFrame] = None
        self.edge_features: Optional[pd.DataFrame] = None
        self.graph_summary: Optional[Dict[str, Any]] = None
        self.pipeline_meta: Optional[Dict[str, Any]] = None

        # Pipeline results
        self.partition: Optional[Dict[int, int]] = None
        self.cluster_metrics: Optional[List[ClusterMetrics]] = None
        self.suspicious_ids: Optional[List[int]] = None
        self.community_subgraphs: Optional[Dict[int, nx.Graph]] = None
        self.explanations: Optional[Dict[int, ClusterExplanation]] = None

    # ── 1. Data Loading ─────────────────────────────────────────────────────

    def load(self) -> "GraphService":
        """Load all processed artifacts and run the pipeline.  Returns *self*."""
        self._load_csvs()
        self._build_graph()
        self._load_json_meta()
        self._generate_gnn_risk_scores()
        self._run_pipeline()
        return self

    # -- helpers --

    def _load_csvs(self) -> None:
        self.node_features = pd.read_csv(self._dir / "node_features.csv")
        self.edge_index = pd.read_csv(self._dir / "edge_index.csv")
        self.edge_features = pd.read_csv(self._dir / "edge_features.csv")

        labels_df = pd.read_csv(self._dir / "node_labels.csv")
        self.node_labels = dict(zip(labels_df["node_id"], labels_df["label"]))

    def _build_graph(self) -> None:
        """Build a NetworkX DiGraph from edge_index + edge_features."""
        G = nx.DiGraph()

        # Add nodes with features
        for _, row in self.node_features.iterrows():
            node_id = int(row["node_id"])
            attrs = row.drop("node_id").to_dict()
            G.add_node(node_id, **attrs)

        # Add edges with features
        for i in range(len(self.edge_index)):
            src = int(self.edge_index.iloc[i]["source_id"])
            tgt = int(self.edge_index.iloc[i]["target_id"])
            edge_attrs = self.edge_features.iloc[i].to_dict()
            G.add_edge(src, tgt, **edge_attrs)

        self.graph = G

    def _load_json_meta(self) -> None:
        summary_path = self._dir / "graph_summary.json"
        meta_path = self._dir / "pipeline_meta.json"
        if summary_path.exists():
            with open(summary_path) as f:
                self.graph_summary = json.load(f)
        if meta_path.exists():
            with open(meta_path) as f:
                self.pipeline_meta = json.load(f)

    # ── 2. Mock Risk Scores ─────────────────────────────────────────────────
    #
    # Replace this method with real GNN inference once the model is trained.

    def _generate_gnn_risk_scores(self) -> None:
        """Call the GNN inference script to populate node risk scores."""
        # Paths are dynamically resolved based on the init _dir so it works for testing too
        graph_path = self._dir / "pyg_graph.pt"
        ckpt_path = self._dir / "gnn_checkpoint.pt"
        mapping_path = self._dir / "node_mapping.json"

        # Generate the risk scores
        scores_dict, _ = generate_risk_scores(
            graph_path=graph_path,
            ckpt_path=ckpt_path,
            mapping_path=mapping_path,
        )

        # The inference output keys are strings (from node_mapping), but self.node_risk
        # uses the raw integer node_ids (from node_features.csv "node_id").
        # Fortunately, the inference module reads node_mapping.json which has keys as original IDs
        # and values as integers, and its output is { original_id : probability }
        
        # We need to map the original string IDs back to the node_id in the graph
        # Wait, the frontend and self.node_labels.items() uses integer node_ids.
        # Let's read the mapping
        import json
        with open(mapping_path, "r") as f:
            mapping = json.load(f)

        # mapping: { "Acc_123": 0, "Acc_456": 1 }
        # scores_dict: { "Acc_123": 0.95, "Acc_456": 0.01 }
        
        self.node_risk = {}
        for original_id, prob in scores_dict.items():
            if original_id in mapping:
                node_id = mapping[original_id]
                # Fallback to float just in case
                self.node_risk[node_id] = float(prob)


    # ── 3. Pipeline ─────────────────────────────────────────────────────────

    def _run_pipeline(self) -> None:
        """Run Louvain → metrics → flagging → BFS explanations."""
        result = run_louvain_detection(
            self.graph,
            node_risk=self.node_risk,
            min_size=2,            # Lowered for testing
            min_average_risk=0.1,  # Lowered for testing
            min_density=0.1,
        )

        self.partition = result["partition"]
        self.cluster_metrics = result["cluster_metrics"]
        self.suspicious_ids = [cm.cluster_id for cm in result["suspicious_clusters"]]
        self.community_subgraphs = result.get("suspicious_subgraphs", {})

        # BFS explanations for suspicious clusters
        self.explanations = {}
        if self.community_subgraphs:
            explanation_list = explain_all_clusters(
                self.community_subgraphs,
                node_risk=self.node_risk,
            )
            for exp in explanation_list:
                self.explanations[exp.cluster_id] = exp

    # ═════════════════════════════════════════════════════════════════════════
    #  Query helpers (called by API route handlers)
    # ═════════════════════════════════════════════════════════════════════════

    def get_graph_summary(self) -> Dict[str, Any]:
        """Return high-level graph statistics."""
        # Start with graph_summary.json if present, else build from pipeline_meta
        if self.graph_summary:
            base = dict(self.graph_summary)
        elif self.pipeline_meta:
            base = {
                "num_nodes": self.pipeline_meta.get("num_nodes", self.graph.number_of_nodes() if self.graph else 0),
                "num_edges": self.pipeline_meta.get("num_edges", self.graph.number_of_edges() if self.graph else 0),
                "num_fraud_nodes": self.pipeline_meta.get("num_fraud_nodes", 0),
            }
        else:
            base = {
                "num_nodes": self.graph.number_of_nodes() if self.graph else 0,
                "num_edges": self.graph.number_of_edges() if self.graph else 0,
            }
        base.update(
            {
                "num_communities": len(set(self.partition.values())),
                "num_suspicious": len(self.suspicious_ids),
            }
        )
        return base

    # -- nodes --

    def get_nodes(
        self, skip: int = 0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Paginated list of nodes with features + risk + label."""
        rows = self.node_features.iloc[skip : skip + limit]
        result = []
        for _, row in rows.iterrows():
            nid = int(row["node_id"])
            entry = row.to_dict()
            entry["node_id"] = nid
            entry["label"] = self.node_labels.get(nid, 0)
            entry["risk_score"] = self.node_risk.get(nid, 0.0)
            result.append(entry)
        return result

    def get_node(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Single node with features, label, risk, and neighbor list."""
        if node_id not in self.graph:
            return None

        node_row = self.node_features[
            self.node_features["node_id"] == node_id
        ]
        if node_row.empty:
            return None

        entry = node_row.iloc[0].to_dict()
        entry["node_id"] = node_id
        entry["label"] = self.node_labels.get(node_id, 0)
        entry["risk_score"] = self.node_risk.get(node_id, 0.0)
        entry["community_id"] = self.partition.get(node_id)

        # Neighbors
        successors = list(self.graph.successors(node_id))
        predecessors = list(self.graph.predecessors(node_id))
        entry["out_neighbors"] = successors[:50]  # cap for large hubs
        entry["in_neighbors"] = predecessors[:50]
        return entry

    # -- edges --

    def get_edges(
        self, skip: int = 0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Paginated edge list."""
        rows_ei = self.edge_index.iloc[skip : skip + limit]
        rows_ef = self.edge_features.iloc[skip : skip + limit]
        result = []
        for i in range(len(rows_ei)):
            entry = {
                "source_id": int(rows_ei.iloc[i]["source_id"]),
                "target_id": int(rows_ei.iloc[i]["target_id"]),
            }
            entry.update(rows_ef.iloc[i].to_dict())
            result.append(entry)
        return result

    # -- communities --

    def get_communities(self) -> List[Dict[str, Any]]:
        """All communities with metrics and a suspicious flag."""
        result = []
        for cm in self.cluster_metrics:
            d = asdict(cm)
            d["is_suspicious"] = cm.cluster_id in self.suspicious_ids
            result.append(d)
        return result

    def get_community(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """Single community detail including member node IDs."""
        match = [cm for cm in self.cluster_metrics if cm.cluster_id == cluster_id]
        if not match:
            return None

        cm = match[0]
        d = asdict(cm)
        d["is_suspicious"] = cm.cluster_id in self.suspicious_ids
        # Member nodes
        members = [n for n, c in self.partition.items() if c == cluster_id]
        d["member_nodes"] = members
        return d

    def get_community_subgraph_data(
        self, cluster_id: int
    ) -> Optional[Dict[str, Any]]:
        """Nodes + edges of a community for frontend visualization."""
        members = [n for n, c in self.partition.items() if c == cluster_id]
        if not members:
            return None

        sub = self.graph.subgraph(members)
        nodes = []
        for n in sub.nodes():
            nodes.append(
                {
                    "id": n,
                    "risk_score": self.node_risk.get(n, 0.0),
                    "label": self.node_labels.get(n, 0),
                }
            )

        edges = []
        for u, v, data in sub.edges(data=True):
            edges.append(
                {
                    "source": u,
                    "target": v,
                    "amount": data.get("amount", 0),
                    "is_fraud": data.get("is_fraud", 0),
                }
            )

        return {"cluster_id": cluster_id, "nodes": nodes, "edges": edges}

    def get_full_graph_data(self) -> Dict[str, Any]:
        """Return ALL nodes and edges for a full-network overview visualization."""
        nodes = []
        for n in self.graph.nodes():
            community = self.partition.get(n)
            nodes.append({
                "id": n,
                "risk_score": self.node_risk.get(n, 0.0),
                "label": self.node_labels.get(n, 0),
                "community": community,
                "is_suspicious": community in self.suspicious_ids if community is not None else False,
            })

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "amount": data.get("amount", 0),
                "is_fraud": data.get("is_fraud", 0),
            })

        return {"num_nodes": len(nodes), "num_edges": len(edges), "nodes": nodes, "edges": edges}

    # -- explanations --

    def get_all_explanations(self) -> List[Dict[str, Any]]:
        """BFS explanations for every suspicious community."""
        result = []
        for exp in self.explanations.values():
            result.append(self._explanation_to_dict(exp))
        return result

    def get_explanation(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """BFS explanation for a single cluster."""
        exp = self.explanations.get(cluster_id)
        if exp is None:
            return None
        return self._explanation_to_dict(exp)

    @staticmethod
    def _explanation_to_dict(exp: ClusterExplanation) -> Dict[str, Any]:
        return {
            "cluster_id": exp.cluster_id,
            "seed_node": exp.seed_node,
            "num_nodes": exp.num_nodes,
            "num_edges": exp.num_edges,
            "summary": exp.summary,
            "traversal": [asdict(step) for step in exp.traversal],
        }
