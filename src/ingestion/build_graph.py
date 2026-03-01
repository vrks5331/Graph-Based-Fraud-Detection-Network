"""
build_graph.py — Graph Construction & Visualization
=====================================================

Purpose
-------
Reads the preprocessed artifacts produced by ``preprocess.py`` and
constructs:

1. A **PyTorch Geometric (PyG) ``Data`` object** — the primary training
   input for the GNN.  Contains node features, edge index, edge
   features, and per‑node fraud labels as dense tensors.

2. A **NetworkX ``DiGraph``** — used for classical graph analytics
   (centrality, community detection) and for powering the visualisation.

3. A **visual representation** of a representative subgraph, rendered
   as both a static matplotlib PNG and an interactive Pyvis HTML page.
   Because the full graph has 10 000 nodes and ~1.3 M edges,
   visualising it directly would be unreadable; we therefore sample a
   neighbourhood around fraud nodes so the picture is informative.

Output files  (all written to  data/processed/)
-------------------------------------------------
  ├─ pyg_graph.pt              # serialised PyG Data object
  ├─ graph_summary.json        # high‑level graph statistics
  ├─ graph_visualisation.png   # static matplotlib render
  └─ graph_visualisation.html  # interactive Pyvis render

Usage
-----
    python -m src.ingestion.build_graph       # from project root
    python src/ingestion/build_graph.py       # direct invocation
"""

# ──────────────────────────────────────────────
#  Imports
# ──────────────────────────────────────────────
import json
import logging
import random
from pathlib import Path

import matplotlib
# Use the non‑interactive Agg backend so the script works on headless
# servers / CI environments without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from pyvis.network import Network

# ──────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────
PROCESSED_PATH = Path("data/processed")

NODE_FEATURES_FILE = PROCESSED_PATH / "node_features.csv"
EDGE_INDEX_FILE    = PROCESSED_PATH / "edge_index.csv"
EDGE_FEATURES_FILE = PROCESSED_PATH / "edge_features.csv"
NODE_LABELS_FILE   = PROCESSED_PATH / "node_labels.csv"
NODE_MAPPING_FILE  = PROCESSED_PATH / "node_mapping.json"


# ═══════════════════════════════════════════════
#  1. LOAD PREPROCESSED ARTIFACTS
# ═══════════════════════════════════════════════

def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Load every artifact that ``preprocess.py`` wrote to disk.

    Returns
    -------
    node_features : pd.DataFrame  — shape (N, F+1) including ``node_id``
    edge_index    : pd.DataFrame  — shape (E, 2)   [source_id, target_id]
    edge_features : pd.DataFrame  — shape (E, D)   per‑edge feature vectors
    node_labels   : pd.DataFrame  — shape (N, 2)   [node_id, label]
    node_mapping  : dict          — {raw_account_id → int_node_id}
    """
    log.info("Loading preprocessed artifacts from %s/", PROCESSED_PATH)

    node_features = pd.read_csv(NODE_FEATURES_FILE)
    edge_index    = pd.read_csv(EDGE_INDEX_FILE)
    edge_features = pd.read_csv(EDGE_FEATURES_FILE)
    node_labels   = pd.read_csv(NODE_LABELS_FILE)

    with open(NODE_MAPPING_FILE, "r") as f:
        node_mapping = json.load(f)

    log.info(
        "  nodes=%d  edges=%d  node_feat_dim=%d  edge_feat_dim=%d",
        len(node_features),
        len(edge_index),
        node_features.shape[1] - 1,   # exclude node_id column
        edge_features.shape[1],
    )
    return node_features, edge_index, edge_features, node_labels, node_mapping


# ═══════════════════════════════════════════════
#  2. BUILD PyTorch Geometric DATA OBJECT
# ═══════════════════════════════════════════════

def build_pyg_data(
    node_features: pd.DataFrame,
    edge_index: pd.DataFrame,
    edge_features: pd.DataFrame,
    node_labels: pd.DataFrame,
) -> Data:
    """
    Assemble a PyTorch Geometric ``Data`` object from the raw DataFrames.

    Layout of the resulting ``Data``
    --------------------------------
    data.x          : FloatTensor  [N, F]   — node feature matrix
    data.edge_index : LongTensor   [2, E]   — COO edge index
    data.edge_attr  : FloatTensor  [E, D]   — edge feature matrix
    data.y          : LongTensor   [N]      — binary node labels (0/1)
    data.num_nodes  : int                   — N

    Notes
    -----
    • ``edge_index`` is stored in COO format [2, E] as required by PyG
      message‑passing layers.
    • All tensors are float32 / int64 — the standard precision for GPU
      training.  We intentionally exclude the ``is_fraud`` column from
      the edge features used for training (it would leak the label).
    """

    # ── Node features → tensor ────────────────────────────────
    # Drop the ``node_id`` column — it is an identifier, not a feature.
    feature_cols = [c for c in node_features.columns if c != "node_id"]
    x = torch.tensor(
        node_features[feature_cols].values,
        dtype=torch.float32,
    )

    # ── Edge index → [2, E] tensor ────────────────────────────
    ei = torch.tensor(
        edge_index[["source_id", "target_id"]].values.T,   # transpose → [2, E]
        dtype=torch.long,
    )

    # ── Edge features → tensor ────────────────────────────────
    # Exclude ``is_fraud`` from training features to prevent label leakage.
    # We keep it attached as a separate attribute for analysis.
    train_edge_feat_cols = [c for c in edge_features.columns if c != "is_fraud"]
    edge_attr = torch.tensor(
        edge_features[train_edge_feat_cols].values,
        dtype=torch.float32,
    )

    # Keep the raw edge fraud flags separately for evaluation / analysis.
    edge_fraud = torch.tensor(
        edge_features["is_fraud"].values,
        dtype=torch.long,
    )

    # ── Labels → tensor ───────────────────────────────────────
    y = torch.tensor(
        node_labels["label"].values,
        dtype=torch.long,
    )

    # ── Assemble Data object ──────────────────────────────────
    data = Data(
        x=x,
        edge_index=ei,
        edge_attr=edge_attr,
        y=y,
    )
    # Attach supplementary attributes (not used during forward pass
    # but useful for later analysis and explainability).
    data.edge_fraud = edge_fraud
    data.feature_names = feature_cols
    data.edge_feature_names = train_edge_feat_cols

    log.info(
        "PyG Data object built:  x=%s  edge_index=%s  edge_attr=%s  y=%s",
        list(x.shape), list(ei.shape), list(edge_attr.shape), list(y.shape),
    )
    return data


# ═══════════════════════════════════════════════
#  3. BUILD NetworkX GRAPH
# ═══════════════════════════════════════════════

def build_networkx_graph(
    node_features: pd.DataFrame,
    edge_index: pd.DataFrame,
    edge_features: pd.DataFrame,
    node_labels: pd.DataFrame,
) -> nx.DiGraph:
    """
    Build a NetworkX directed graph with rich node / edge attributes.

    This graph is useful for:
    - Classical graph metrics (degree, centrality, PageRank, etc.)
    - Community detection (Louvain, label propagation)
    - Powering the Pyvis interactive visualisation
    - Serving as a reference during model explainability

    Node attributes
    ---------------
    Every column in ``node_features`` is stored as a node attribute,
    plus the fraud ``label``.

    Edge attributes
    ---------------
    ``amount``, ``timestamp``, ``is_fraud``, ``log_amount``.
    """
    G = nx.DiGraph()

    # ── Add nodes with attributes ──────────────────────────────
    # Merge labels into node features for convenience.
    node_data = node_features.merge(node_labels, on="node_id", how="left")

    for _, row in node_data.iterrows():
        nid = int(row["node_id"])
        attrs = row.drop("node_id").to_dict()
        G.add_node(nid, **attrs)

    # ── Add edges with attributes ──────────────────────────────
    for i in range(len(edge_index)):
        src = int(edge_index.iloc[i]["source_id"])
        tgt = int(edge_index.iloc[i]["target_id"])
        eattrs = edge_features.iloc[i].to_dict()
        G.add_edge(src, tgt, **eattrs)

    log.info(
        "NetworkX DiGraph built:  %d nodes,  %d edges",
        G.number_of_nodes(), G.number_of_edges(),
    )
    return G


# ═══════════════════════════════════════════════
#  4. GRAPH STATISTICS
# ═══════════════════════════════════════════════

def compute_graph_stats(G: nx.DiGraph, data: Data) -> dict:
    """
    Compute and log key graph statistics for sanity‑checking and
    documentation.

    Metrics include degree distribution summaries, connected‑component
    counts, density, and fraud prevalence.

    Returns
    -------
    stats : dict   — written to ``graph_summary.json``
    """
    degrees = [d for _, d in G.degree()]
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]

    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        # Connected components (on undirected projection)
        "num_weakly_connected_components": nx.number_weakly_connected_components(G),
        # Degree statistics
        "degree_mean": float(np.mean(degrees)),
        "degree_median": float(np.median(degrees)),
        "degree_max": int(np.max(degrees)),
        "in_degree_mean": float(np.mean(in_degrees)),
        "out_degree_mean": float(np.mean(out_degrees)),
        # Fraud prevalence
        "num_fraud_nodes": int(data.y.sum().item()),
        "num_legit_nodes": int((data.y == 0).sum().item()),
        "fraud_rate_pct": round(100.0 * data.y.sum().item() / data.y.shape[0], 2),
        # Tensor shapes (for quick reference)
        "tensor_shapes": {
            "x": list(data.x.shape),
            "edge_index": list(data.edge_index.shape),
            "edge_attr": list(data.edge_attr.shape),
            "y": list(data.y.shape),
        },
    }

    log.info("Graph statistics:")
    for k, v in stats.items():
        if k != "tensor_shapes":
            log.info("  %-40s %s", k, v)

    return stats


# ═══════════════════════════════════════════════
#  5. SUBGRAPH SAMPLING  (for visualisation)
# ═══════════════════════════════════════════════

def sample_visualisation_subgraph(
    G: nx.DiGraph,
    n_seed_fraud: int = 8,
    n_seed_legit: int = 4,
    hops: int = 1,
) -> nx.DiGraph:
    """
    Extract a small, *representative* subgraph centred on fraud nodes.

    Strategy
    --------
    1. Pick ``n_seed_fraud`` random fraud nodes as seeds.
    2. Pick ``n_seed_legit`` random legitimate nodes as seeds (to show
       normal transaction patterns for contrast).
    3. Expand each seed by ``hops`` hops in the undirected sense
       (considering both in‑ and out‑neighbours).
    4. Induce the subgraph on the union of all discovered nodes.

    This keeps the visual compact (~50‑150 nodes) while showing a mix
    of fraud and legitimate topology.

    Parameters
    ----------
    G             : full directed graph
    n_seed_fraud  : number of fraud seed nodes to sample
    n_seed_legit  : number of legitimate seed nodes to sample
    hops          : neighbourhood expansion radius

    Returns
    -------
    subG : nx.DiGraph   — induced subgraph with all original attributes
    """
    random.seed(42)  # reproducible sampling

    # Separate fraud and legitimate node IDs
    fraud_nodes = [n for n, d in G.nodes(data=True) if d.get("label", 0) == 1]
    legit_nodes = [n for n, d in G.nodes(data=True) if d.get("label", 0) == 0]

    # Sample seeds (clamp to available count)
    seed_fraud = random.sample(fraud_nodes, min(n_seed_fraud, len(fraud_nodes)))
    seed_legit = random.sample(legit_nodes, min(n_seed_legit, len(legit_nodes)))
    seeds = seed_fraud + seed_legit

    # BFS expansion on undirected view
    undirected = G.to_undirected()
    expanded_nodes = set(seeds)

    for seed in seeds:
        # nx.single_source_shortest_path_length gives all nodes within `hops`
        neighbours = nx.single_source_shortest_path_length(
            undirected, seed, cutoff=hops,
        )
        expanded_nodes.update(neighbours.keys())

    subG = G.subgraph(expanded_nodes).copy()

    log.info(
        "Sampled visualisation subgraph: %d nodes, %d edges  "
        "(from %d fraud seeds + %d legit seeds, %d hops)",
        subG.number_of_nodes(), subG.number_of_edges(),
        len(seed_fraud), len(seed_legit), hops,
    )
    return subG


# ═══════════════════════════════════════════════
#  6. STATIC MATPLOTLIB VISUALISATION
# ═══════════════════════════════════════════════

def visualise_matplotlib(
    subG: nx.DiGraph,
    out_path: Path,
) -> None:
    """
    Render a static PNG of the sampled subgraph using matplotlib.

    Visual encoding
    ---------------
    • **Node colour** — red = fraud (label 1), steel‑blue = legitimate (label 0)
    • **Node size**   — proportional to total transaction count (degree proxy)
    • **Node label**  — shows ``node_id`` and the fraud/legit tag
    • **Edge colour** — orange‑red if the edge is fraudulent, light grey otherwise
    • **Edge width**  — scaled by log(amount) for readability
    • **Edge label**  — shows transaction amount (on a subset to avoid clutter)
    """
    fig, ax = plt.subplots(figsize=(20, 16))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # ── Layout ─────────────────────────────────────────────────
    # Kamada–Kawai gives well‑separated clusters for small graphs.
    pos = nx.kamada_kawai_layout(subG)

    # ── Node properties ────────────────────────────────────────
    node_ids = list(subG.nodes())
    labels_map = nx.get_node_attributes(subG, "label")
    total_tx_map = nx.get_node_attributes(subG, "total_tx_count")

    node_colors = [
        "#ff4b4b" if labels_map.get(n, 0) == 1 else "#4b9dff"
        for n in node_ids
    ]
    # Scale node sizes: min 200, max 1200
    raw_sizes = [total_tx_map.get(n, 1) for n in node_ids]
    max_sz = max(raw_sizes) if raw_sizes else 1
    node_sizes = [200 + 1000 * (s / max_sz) for s in raw_sizes]

    # ── Edge properties ────────────────────────────────────────
    edge_fraud_map = nx.get_edge_attributes(subG, "is_fraud")
    edge_amount_map = nx.get_edge_attributes(subG, "amount")

    edge_colors = [
        "#ff6347" if edge_fraud_map.get(e, 0) == 1 else "#3a3f4b"
        for e in subG.edges()
    ]
    edge_widths = [
        0.5 + 1.5 * np.log1p(edge_amount_map.get(e, 1)) / 10
        for e in subG.edges()
    ]

    # ── Draw edges ─────────────────────────────────────────────
    nx.draw_networkx_edges(
        subG, pos, ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.6,
        arrows=True,
        arrowsize=10,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        min_source_margin=10,
        min_target_margin=10,
    )

    # ── Draw nodes ─────────────────────────────────────────────
    nx.draw_networkx_nodes(
        subG, pos, ax=ax,
        nodelist=node_ids,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#ffffff",
        linewidths=0.8,
        alpha=0.92,
    )

    # ── Node labels ────────────────────────────────────────────
    # Show node ID + fraud/legit tag
    node_labels = {}
    for n in node_ids:
        tag = "F" if labels_map.get(n, 0) == 1 else "L"
        node_labels[n] = f"{n}\n({tag})"

    nx.draw_networkx_labels(
        subG, pos, labels=node_labels, ax=ax,
        font_size=6, font_color="#e0e0e0", font_weight="bold",
    )

    # ── Edge labels (only on fraud edges to avoid clutter) ─────
    fraud_edge_labels = {}
    for e in subG.edges():
        if edge_fraud_map.get(e, 0) == 1:
            amt = edge_amount_map.get(e, 0)
            fraud_edge_labels[e] = f"${amt:,.0f}"

    nx.draw_networkx_edge_labels(
        subG, pos, edge_labels=fraud_edge_labels, ax=ax,
        font_size=5, font_color="#ffaa00",
        bbox=dict(boxstyle="round,pad=0.15", fc="#1a1a2e", ec="none", alpha=0.8),
    )

    # ── Legend ─────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor="#ff4b4b", edgecolor="white", label="Fraud node"),
        mpatches.Patch(facecolor="#4b9dff", edgecolor="white", label="Legitimate node"),
        mpatches.Patch(facecolor="#ff6347", edgecolor="white", label="Fraud edge"),
        mpatches.Patch(facecolor="#3a3f4b", edgecolor="white", label="Normal edge"),
    ]
    legend = ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=10,
        facecolor="#1a1a2e",
        edgecolor="#4b9dff",
        labelcolor="#e0e0e0",
        framealpha=0.9,
    )
    legend.get_frame().set_linewidth(1.5)

    # ── Title & metadata annotation ────────────────────────────
    n_fraud_sub = sum(1 for n in node_ids if labels_map.get(n, 0) == 1)
    n_legit_sub = len(node_ids) - n_fraud_sub

    ax.set_title(
        f"Transaction Graph — Subgraph Sample\n"
        f"{subG.number_of_nodes()} nodes  ·  {subG.number_of_edges()} edges  ·  "
        f"{n_fraud_sub} fraud  ·  {n_legit_sub} legit",
        fontsize=16, fontweight="bold", color="#e0e0e0", pad=20,
    )

    # Annotation box with node‑size explanation
    ax.annotate(
        "Node size ∝ total transaction count\n"
        "Edge width ∝ log(amount)\n"
        "Labels: F = fraud, L = legit",
        xy=(0.99, 0.01), xycoords="axes fraction",
        fontsize=8, color="#aaaaaa",
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a2e", ec="#4b9dff", alpha=0.85),
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    log.info("Static visualisation saved → %s", out_path)


# ═══════════════════════════════════════════════
#  7. INTERACTIVE PYVIS VISUALISATION
# ═══════════════════════════════════════════════

def visualise_pyvis(
    subG: nx.DiGraph,
    out_path: Path,
) -> None:
    """
    Build an interactive HTML visualisation using Pyvis.

    Hover over any node or edge to see its full attribute table.
    Nodes are colour‑coded by label and sized by degree; edges are
    colour‑coded by fraud status.
    """
    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#0e1117",
        font_color="#e0e0e0",
        notebook=False,
    )

    # Physics / layout settings for a readable result
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.05,
        damping=0.9,
    )

    # ── Add nodes ──────────────────────────────────────────────
    labels_map = nx.get_node_attributes(subG, "label")
    total_tx_map = nx.get_node_attributes(subG, "total_tx_count")
    balance_map = nx.get_node_attributes(subG, "init_balance")
    in_tx_map = nx.get_node_attributes(subG, "in_tx_count")
    out_tx_map = nx.get_node_attributes(subG, "out_tx_count")
    net_flow_map = nx.get_node_attributes(subG, "net_flow")

    for n in subG.nodes():
        is_fraud = labels_map.get(n, 0) == 1
        color = "#ff4b4b" if is_fraud else "#4b9dff"
        tag = "FRAUD" if is_fraud else "LEGIT"

        # Tooltip with key metrics
        title = (
            f"<b>Node {n}</b> — {tag}<br>"
            f"─────────────────<br>"
            f"Init balance: ${balance_map.get(n, 0):,.2f}<br>"
            f"In-txns: {int(in_tx_map.get(n, 0))}<br>"
            f"Out-txns: {int(out_tx_map.get(n, 0))}<br>"
            f"Total txns: {int(total_tx_map.get(n, 0))}<br>"
            f"Net flow: ${net_flow_map.get(n, 0):,.2f}<br>"
        )

        size = 10 + 20 * np.log1p(total_tx_map.get(n, 0))

        net.add_node(
            n,
            label=f"{n} ({tag[0]})",
            title=title,
            color=color,
            size=float(size),
            borderWidth=2,
            borderWidthSelected=4,
        )

    # ── Add edges ──────────────────────────────────────────────
    edge_fraud_map = nx.get_edge_attributes(subG, "is_fraud")
    edge_amount_map = nx.get_edge_attributes(subG, "amount")
    edge_ts_map = nx.get_edge_attributes(subG, "timestamp")

    for src, tgt in subG.edges():
        e = (src, tgt)
        is_fraud_edge = edge_fraud_map.get(e, 0) == 1
        amt = edge_amount_map.get(e, 0)
        ts = edge_ts_map.get(e, 0)

        color = "#ff6347" if is_fraud_edge else "#3a3f4b"
        width = 1 + 2 * np.log1p(amt) / 10

        fraud_tag = "YES" if is_fraud_edge else "NO"
        title = (
            f"<b>Edge {src} → {tgt}</b><br>"
            f"─────────────────<br>"
            f"Amount: ${amt:,.2f}<br>"
            f"Timestamp: {int(ts)}<br>"
            f"Fraudulent: {fraud_tag}<br>"
        )

        # Show amount as edge label only on fraud edges for clarity
        edge_label = f"${amt:,.0f}" if is_fraud_edge else ""

        net.add_edge(
            src, tgt,
            title=title,
            label=edge_label,
            color=color,
            width=float(width),
            arrows="to",
            font={"size": 8, "color": "#ffaa00", "strokeWidth": 0},
        )

    # ── Save ───────────────────────────────────────────────────
    net.save_graph(str(out_path))
    log.info("Interactive visualisation saved → %s", out_path)


# ═══════════════════════════════════════════════
#  8. SAVE ARTIFACTS
# ═══════════════════════════════════════════════

def save_artifacts(data: Data, stats: dict) -> None:
    """
    Persist the PyG Data object and graph summary statistics.

    The ``.pt`` file can be loaded directly in a training script::

        data = torch.load("data/processed/pyg_graph.pt")
    """
    # Save PyG Data
    pt_path = PROCESSED_PATH / "pyg_graph.pt"
    torch.save(data, pt_path)
    log.info("PyG graph saved → %s  (%.1f MB)",
             pt_path, pt_path.stat().st_size / 1e6)

    # Save summary statistics
    stats_path = PROCESSED_PATH / "graph_summary.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Graph summary saved → %s", stats_path)


# ═══════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════

def main() -> None:
    """
    End‑to‑end graph construction and visualisation pipeline.

    Steps
    -----
    1. Load preprocessed CSV / JSON artifacts.
    2. Build PyG Data object (primary training input).
    3. Build NetworkX graph (analytics + visualisation).
    4. Compute and persist graph‑level statistics.
    5. Sample a small representative subgraph.
    6. Render static matplotlib visualisation.
    7. Render interactive Pyvis visualisation.
    8. Save all output artifacts.
    """
    log.info("=" * 60)
    log.info("  GRAPH FRAUD DETECTION — Graph Construction")
    log.info("=" * 60)

    # ── Step 1: Load artifacts ──
    log.info("[1/8] Loading preprocessed artifacts …")
    node_features, edge_index, edge_features, node_labels, node_mapping = (
        load_artifacts()
    )

    # ── Step 2: Build PyG Data ──
    log.info("[2/8] Building PyTorch Geometric Data object …")
    data = build_pyg_data(node_features, edge_index, edge_features, node_labels)

    # ── Step 3: Build NetworkX graph ──
    log.info("[3/8] Building NetworkX directed graph …")
    # NOTE: This iterates over 1.3M edges — it's the slowest step.
    G = build_networkx_graph(node_features, edge_index, edge_features, node_labels)

    # ── Step 4: Graph statistics ──
    log.info("[4/8] Computing graph statistics …")
    stats = compute_graph_stats(G, data)

    # ── Step 5: Sample subgraph ──
    log.info("[5/8] Sampling visualisation subgraph …")
    subG = sample_visualisation_subgraph(
        G,
        n_seed_fraud=8,
        n_seed_legit=4,
        hops=1,
    )

    # ── Step 6: Static visualisation ──
    log.info("[6/8] Rendering static matplotlib visualisation …")
    visualise_matplotlib(subG, PROCESSED_PATH / "graph_visualisation.png")

    # ── Step 7: Interactive visualisation ──
    log.info("[7/8] Rendering interactive Pyvis visualisation …")
    visualise_pyvis(subG, PROCESSED_PATH / "graph_visualisation.html")

    # ── Step 8: Save artifacts ──
    log.info("[8/8] Saving graph artifacts …")
    save_artifacts(data, stats)

    log.info("=" * 60)
    log.info("  Graph construction & visualisation complete ✓")
    log.info("=" * 60)


# ──────────────────────────────────────────────
#  Script entry‑point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()