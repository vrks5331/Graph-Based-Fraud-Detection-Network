# Graph-Based Fraud Detection Network

A graph-based anti-money-laundering system that combines **GNN node classification**, **Louvain community detection**, and **BFS explainability** to identify and explain fraud rings in transaction networks.

## Architecture

```
Raw CSVs ─→ preprocess.py ─→ Processed Artifacts ─→ build_graph.py ─→ PyG Graph
                                      │
                                      ├─→ GNN Model (node-level risk scores)
                                      │
                                      ├─→ Louvain (community/fraud-ring detection)
                                      │         │
                                      │         └─→ BFS Explainer (ring explanations)
                                      │
                                      └─→ FastAPI ─→ Frontend Dashboard
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | AMLSim synthetic CSVs (accounts, transactions, alerts) |
| Preprocessing | pandas, scikit-learn |
| Graph | NetworkX, PyTorch Geometric |
| GNN Model | PyTorch Geometric (in progress) |
| Community Detection | python-louvain (Louvain algorithm) |
| Explainability | BFS traversal |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit + Pyvis (in progress) |


## Project Structure

```
graph-fraud-detection/
├── data/
│   ├── raw/                    # AMLSim CSVs (accounts, transactions, alerts)
│   └── processed/              # Pipeline output (node_features, edge_index, etc.)
├── src/
│   ├── api/
│   │   ├── main.py             # FastAPI app (10 REST endpoints)
│   │   └── graph_service.py    # Data loading & pipeline orchestration
│   ├── detection/
│   │   ├── louvain.py          # Louvain community detection + subgraph extraction
│   │   └── risk_scoring.py     # (Stub) Post-processing for GNN scores
│   ├── explainability/
│   │   └── bfs_explainer.py    # BFS traversal for fraud-ring explanations
│   ├── ingestion/
│   │   ├── preprocess.py       # Raw CSV → processed artifacts pipeline
│   │   └── build_graph.py      # NetworkX/PyG graph construction
│   ├── models/
│   │   ├── gnn_model.py        # (Stub) GNN architecture
│   │   └── inference.py        # (Stub) Model inference
│   └── dashboard/
│       └── app.py              # (Stub) Streamlit dashboard
├── tests/
│   ├── test_louvain.py         # 32 tests for Louvain module
│   ├── test_bfs_explainer.py   # Tests for BFS explainer
│   └── test_api.py             # 37 tests for API + GraphService
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.11+
- pip

### Install Dependencies

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### Run the Preprocessing Pipeline

This step transforms the raw CSVs in `data/raw/` into processed artifacts:

```bash
python -m src.ingestion.preprocess
```

Output files are written to `data/processed/`.

---

## Running the API

```bash
# From project root
uvicorn src.api.main:app --reload --port 8000
```

The API starts at **http://localhost:8000**.  Interactive docs are at **http://localhost:8000/docs** (Swagger UI).

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check (returns node/edge counts) |
| `GET` | `/api/graph/summary` | Graph-level stats (node count, edge count, fraud rate, community count) |
| `GET` | `/api/graph/nodes?skip=0&limit=100` | Paginated node list with features, label, and risk score |
| `GET` | `/api/graph/nodes/{node_id}` | Single node detail (features, risk, community, neighbors) |
| `GET` | `/api/graph/edges?skip=0&limit=100` | Paginated edge list (source, target, amount, is_fraud) |
| `GET` | `/api/communities` | All Louvain communities with metrics and suspicious flag |
| `GET` | `/api/communities/{cluster_id}` | Community detail including member node IDs |
| `GET` | `/api/communities/{cluster_id}/subgraph` | Community subgraph (nodes + edges) for frontend visualization |
| `GET` | `/api/explanations` | BFS explanations for all suspicious communities |
| `GET` | `/api/explanations/{cluster_id}` | BFS explanation for a specific community |

### Example: Fetch Graph Summary

```bash
curl http://localhost:8000/api/graph/summary
```

```json
{
  "num_nodes": 10000,
  "num_edges": 68936,
  "fraud_rate_pct": 16.85,
  "num_communities": 42,
  "num_suspicious": 3
}
```

### Example: Fetch a Community Subgraph (for visualization)

```bash
curl http://localhost:8000/api/communities/5/subgraph
```

Returns nodes and edges in the community — ready for Pyvis or any graph visualization library.

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run only API tests
python -m pytest tests/test_api.py -v

# Run only Louvain tests
python -m pytest tests/test_louvain.py -v
```

All tests are self-contained and use synthetic data (no dependency on real CSV files).

---

## Pipeline Components

### 1. Preprocessing (`src/ingestion/preprocess.py`)
Transforms raw AMLSim CSVs into ML-ready artifacts: node features (21 dimensions), edge index, edge features, node labels, and a PyG graph object.

### 2. Louvain Community Detection (`src/detection/louvain.py`)
Groups accounts into communities using the Louvain modularity algorithm. Computes per-cluster metrics (size, density, average risk) and flags suspicious clusters that exceed configurable thresholds.

### 3. BFS Explainer (`src/explainability/bfs_explainer.py`)
Traverses each suspicious community starting from the highest-risk node, building a step-by-step human-readable explanation of the fraud ring structure.

### 4. FastAPI Backend (`src/api/`)
Serves all pipeline results as a REST API. Uses mock GNN risk scores (derived from ground-truth labels) until the real GNN model is trained. The mock scores are generated in `graph_service.py` — swap `_generate_mock_risk_scores()` for the real model's inference output when ready.

### 5. GNN Model (`src/models/`) — *In Progress*
Will produce per-node fraud probability scores that feed into both the Louvain pipeline (for risk-weighted community flagging) and the BFS explainer (for seed node selection).

---

## Fraud Patterns Detected

- **Fraud rings** — dense clusters of tightly connected accounts with high aggregate risk
- **High-risk accounts** — individual nodes with elevated GNN-predicted fraud probability
- **Suspicious subgraph extraction** — isolated community subgraphs ready for visual inspection and BFS traversal

