"""
Streamlit Dashboard — AML Fraud Detection
==========================================
Connects to the FastAPI backend and renders:
  1. Global graph summary with metrics
  2. Community table
  3. Interactive Pyvis graph visualization per suspicious cluster
  4. BFS traversal explanations
"""

import streamlit as st
import requests
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import os

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="AML Fraud Graph Dashboard",
    page_icon="🕵️",
)

# ── Backend URL ─────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000/api"


# ═══════════════════════════════════════════════════════════════════════════════
#  API helper functions
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=30)
def fetch_summary():
    """GET /api/graph/summary"""
    try:
        r = requests.get(f"{API_BASE}/graph/summary", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=30)
def fetch_communities():
    """GET /api/communities"""
    try:
        r = requests.get(f"{API_BASE}/communities", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


@st.cache_data(ttl=30)
def fetch_subgraph(cluster_id: int):
    """GET /api/communities/{id}/subgraph"""
    try:
        r = requests.get(f"{API_BASE}/communities/{cluster_id}/subgraph", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=30)
def fetch_explanation(cluster_id: int):
    """GET /api/explanations/{id}"""
    try:
        r = requests.get(f"{API_BASE}/explanations/{cluster_id}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=30)
def fetch_full_graph():
    """GET /api/graph/full — all nodes and edges."""
    try:
        r = requests.get(f"{API_BASE}/graph/full", timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Pyvis graph builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_pyvis_html(subgraph_data: dict) -> str:
    """Build a Pyvis network from subgraph JSON and return HTML string."""
    net = Network(
        height="550px",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="#ffffff",
        cdn_resources="in_line",   # Bundle CSS/JS inline to avoid broken CDN URLs
    )
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=150,
        spring_strength=0.05,
    )
    # Force Pyvis to set proper canvas options
    net.set_options("""
    {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "font": {"color": "#ffffff", "size": 14}
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
        "smooth": {"type": "curvedCW", "roundness": 0.2}
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.05
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 200,
        "zoomView": true,
        "dragView": true
      }
    }
    """)

    # Nodes
    for node in subgraph_data.get("nodes", []):
        nid = node["id"]
        risk = node.get("risk_score", 0.0)
        lbl = "FRAUD" if node.get("label") == 1 else "LEGIT"

        # Color: red gradient for high risk, blue for low
        if risk > 0.5:
            color = "#ff4b4b"
        else:
            color = "#4b9dff"

        size = 20 + risk * 30  # scale node size by risk

        hover = f"Node {nid}\nRisk: {risk:.3f}\nLabel: {lbl}"
        net.add_node(
            nid,
            label=str(nid),
            title=hover,
            color=color,
            size=size,
        )

    # Edges
    for edge in subgraph_data.get("edges", []):
        src = edge["source"]
        tgt = edge["target"]
        amt = edge.get("amount", 0)
        fraud_flag = edge.get("is_fraud", 0)

        edge_color = "#ff6b6b" if fraud_flag else "#555555"
        hover = f"${amt:,.2f}" + (" [FRAUD]" if fraud_flag else "")

        net.add_edge(src, tgt, title=hover, color=edge_color, width=1 + (amt / 10000))

    # Generate HTML directly as a string (avoids Windows cp1252 encoding issues)
    html = net.generate_html()
    return html


def build_full_graph_html(graph_data: dict) -> str:
    """Build a Pyvis network for the FULL dataset overview.

    Color scheme:
      - Red   (#ff4b4b) = fraud node (label == 1) or in a suspicious community
      - Blue  (#4b9dff) = legitimate node in a clean community
      - Gray edges for normal, red edges for flagged fraud
    """
    net = Network(
        height="650px",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="#ffffff",
        cdn_resources="in_line",
    )
    net.set_options("""
    {
      "nodes": {
        "borderWidth": 1,
        "font": {"color": "#ffffff", "size": 10}
      },
      "edges": {
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.3}},
        "smooth": {"type": "continuous"}
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 100}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "zoomView": true
      }
    }
    """)

    for node in graph_data.get("nodes", []):
        nid = node["id"]
        risk = node.get("risk_score", 0.0)
        label_val = node.get("label", 0)
        is_sus = node.get("is_suspicious", False)
        comm = node.get("community", "?")

        # Fraud or in suspicious community → red, else blue
        is_fraud = label_val == 1 or is_sus
        color = "#ff4b4b" if is_fraud else "#4b9dff"
        size = 8 + risk * 15

        tag = "FRAUD" if label_val == 1 else ("SUSPICIOUS" if is_sus else "LEGIT")
        hover = f"Node {nid}\nRisk: {risk:.3f}\nCommunity: {comm}\nStatus: {tag}"

        net.add_node(nid, label=str(nid), title=hover, color=color, size=size)

    for edge in graph_data.get("edges", []):
        fraud_flag = edge.get("is_fraud", 0)
        edge_color = "#ff6b6b" if fraud_flag else "#33335566"
        net.add_edge(edge["source"], edge["target"], color=edge_color, width=0.5)

    return net.generate_html()


# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

st.sidebar.title("🕵️ AML Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔍 Cluster Investigation"],
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Page: Overview
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📊 Overview":
    st.title("📊 Graph-Based Fraud Detection Overview")
    st.caption("Real-time metrics from the GNN + Louvain + BFS pipeline")

    summary = fetch_summary()

    if "error" in summary:
        st.error(f"⚠️ Cannot reach backend: `{summary['error']}`")
        st.info("Start the API with: `uvicorn src.api.main:app --reload --port 8000`")
        st.stop()

    # ── KPI metrics ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏦 Accounts (Nodes)", f"{summary.get('num_nodes', '?'):,}")
    c2.metric("💸 Transactions (Edges)", f"{summary.get('num_edges', '?'):,}")
    c3.metric("🚨 Fraud Nodes", f"{summary.get('num_fraud_nodes', '?'):,}")
    c4.metric("🧩 Communities", f"{summary.get('num_communities', '?'):,}")

    st.divider()

    # ── Full-network graph ─────────────────────────────────────────────────
    st.subheader("Full Transaction Network")
    st.caption("All accounts and transactions. Red = fraud/suspicious, Blue = legitimate.")

    full_graph = fetch_full_graph()
    if full_graph and full_graph.get("nodes"):
        full_html = build_full_graph_html(full_graph)
        components.html(full_html, height=670, scrolling=False)
    else:
        st.warning("Could not load full graph data.")

    st.divider()

    # ── Communities table ───────────────────────────────────────────────────
    st.subheader("All Detected Communities")
    comms = fetch_communities()

    if comms:
        df = pd.DataFrame(comms)
        display_cols = [c for c in ["cluster_id", "is_suspicious", "size", "average_node_risk", "density"] if c in df.columns]
        st.dataframe(
            df[display_cols].style.applymap(
                lambda v: "background-color: #ff4b4b22" if v is True else "",
                subset=["is_suspicious"] if "is_suspicious" in display_cols else [],
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No communities detected yet.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Page: Cluster Investigation
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Cluster Investigation":
    st.title("🔍 Suspicious Cluster Investigation")

    comms = fetch_communities()

    if not comms:
        st.warning("No communities found. Make sure the backend is running.")
        st.stop()

    suspicious = [c for c in comms if c.get("is_suspicious")]

    if not suspicious:
        st.info("No suspicious communities flagged. All clusters look clean! ✅")
        st.stop()

    # ── Cluster selector ────────────────────────────────────────────────────
    options = {
        f"Cluster {c['cluster_id']}  (risk={c['average_node_risk']:.3f}, size={c['size']})": c["cluster_id"]
        for c in suspicious
    }
    choice = st.selectbox("Select a suspicious cluster:", list(options.keys()))
    cid = options[choice]

    st.divider()

    # ── Two-column layout: explanation + graph ──────────────────────────────
    left, right = st.columns([1, 2])

    with left:
        st.subheader("📝 BFS Explanation")
        explanation = fetch_explanation(cid)
        if explanation:
            st.metric("Seed Node", explanation.get("seed_node", "N/A"))
            st.markdown(explanation.get("summary", "_No summary available._"))

            with st.expander("🔎 Full BFS Traversal", expanded=False):
                for step in explanation.get("traversal", []):
                    risk = step.get("node_risk", step.get("risk_score", 0))
                    node = step.get("node", step.get("node_id", "?"))
                    icon = "🔴" if risk > 0.5 else "🔵"
                    st.write(
                        f"{icon} **Node {node}** — "
                        f"depth {step['depth']}, risk {risk:.3f}"
                    )
        else:
            st.info("No BFS explanation available for this cluster.")

    with right:
        st.subheader("🌐 Interactive Graph Visualization")
        subgraph = fetch_subgraph(cid)

        if subgraph and subgraph.get("nodes"):
            html = build_pyvis_html(subgraph)
            components.html(html, height=580, scrolling=False)

            st.caption(
                f"🔴 = High risk (>0.5)  |  🔵 = Low risk  |  "
                f"{len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges"
            )
        else:
            st.error("Could not load subgraph data for this cluster.")
