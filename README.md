# Graph-Based Fraud Detection Network

A graph-based anti-money-laundering system that combines **GNN node classification**, **Louvain community detection**, and **BFS explainability** to identify and explain fraud rings in transaction networks.

---

## Architecture

```
Raw CSVs ──→ preprocess.py ──→ Processed Artifacts ──→ PyG Graph
                                       │
                                       ├──→ GNN Model (node-level risk scores)
                                       │
                                       ├──→ Louvain (community/fraud-ring detection)
                                       │         │
                                       │         └──→ BFS Explainer (ring explanations)
                                       │
                                       └──→ FastAPI ──→ Streamlit Dashboard + Pyvis
```

## Tech Stack

| Layer | Technology |
|---|---|
| Data | AMLSim synthetic CSVs (accounts, transactions, alerts) |
| Preprocessing | pandas, scikit-learn |
| Graph | NetworkX, PyTorch Geometric |
| GNN Model | PyTorch (GraphSAGE via `torch_geometric`) |
| Community Detection | python-louvain (Louvain algorithm) |
| Explainability | BFS traversal with NLP summary generation |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit + Pyvis (interactive graph visualization) |

---

## Project Structure

```
graph-fraud-detection/
├── data/
│   ├── raw/                        # AMLSim CSVs (accounts, transactions, alerts)
│   ├── processed/                  # Preprocessed artifacts from the real dataset
│   ├── testing/                    # Dummy CSV data for quick testing
│   └── testing_processed/          # Preprocessed artifacts from the dummy data
├── scripts/
│   └── generate_dummy_data.py      # Generates 250-node, 800-edge test dataset
├── src/
│   ├── api/
│   │   ├── main.py                 # FastAPI app (11 REST endpoints)
│   │   └── graph_service.py        # Data loading & pipeline orchestration
│   ├── detection/
│   │   ├── louvain.py              # Louvain community detection + flagging
│   │   └── risk_scoring.py         # Post-processing for GNN scores
│   ├── explainability/
│   │   └── bfs_explainer.py        # BFS traversal for fraud-ring explanations
│   ├── ingestion/
│   │   ├── preprocess.py           # Raw CSV → processed artifacts pipeline
│   │   └── build_graph.py          # NetworkX / PyG graph construction
│   ├── models/
│   │   ├── gnn_model.py            # 2-layer GraphSAGE classifier (FraudGNN)
│   │   └── train.py                # Training pipeline (NeighborLoader, stratified masks)
│   └── dashboard/
│       └── app.py                  # Streamlit dashboard with Pyvis graphs
├── tests/
│   ├── test_model.py               # GNN architecture unit tests
│   ├── test_louvain.py             # Louvain community detection tests
│   ├── test_bfs_explainer.py       # BFS explainer tests
│   ├── test_frontend.py            # Streamlit dashboard tests (32 tests)
│   ├── test_integration.py         # End-to-end GraphService pipeline tests
│   └── test_api.py                 # API endpoint tests
└── requirements.txt
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Clone the repo and enter the directory
git clone <repo-url>
cd graph-fraud-detection

# Create a virtual environment (recommended)
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install all packages
pip install -r requirements.txt
```

### 2. Preprocess the Data

If you have the real AMLSim dataset in `data/raw/`:

```bash
python -m src.ingestion.preprocess
```

If you don't have the real dataset, generate dummy testing data:

```bash
python scripts/generate_dummy_data.py
python -m src.ingestion.preprocess
```

> The dummy data generator creates 250 accounts and 800 transactions with an embedded fraud ring for demonstration purposes.

### 3. Start the Backend API

```bash
uvicorn src.api.main:app --reload --port 8000
```

The API starts at **http://localhost:8000**. Interactive Swagger docs are at **http://localhost:8000/docs**.

### 4. Start the Frontend Dashboard

Open a **second terminal**, activate the virtual environment, and run:

```bash
streamlit run src/dashboard/app.py
```

The dashboard opens automatically at **http://localhost:8501**.

### 5. (Optional) Train the GNN Model

```bash
python -m src.models.train --epochs 30 --lr 0.001
```

The trained model weights are saved to `models/fraud_gnn.pt`.

---

## Running Tests

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_model.py -v        # GNN architecture
python -m pytest tests/test_louvain.py -v       # Louvain detection
python -m pytest tests/test_bfs_explainer.py -v # BFS explainer
python -m pytest tests/test_frontend.py -v      # Streamlit dashboard
python -m pytest tests/test_integration.py -v   # End-to-end pipeline
python -m pytest tests/test_api.py -v           # API endpoints
```

All tests are self-contained using synthetic data — no dependency on real CSV files.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check (node/edge counts) |
| `GET` | `/api/graph/summary` | Graph-level stats (nodes, edges, fraud count, communities) |
| `GET` | `/api/graph/nodes?skip=0&limit=100` | Paginated node list with features, label, and risk score |
| `GET` | `/api/graph/nodes/{node_id}` | Single node detail (features, risk, community, neighbors) |
| `GET` | `/api/graph/edges?skip=0&limit=100` | Paginated edge list (source, target, amount, is_fraud) |
| `GET` | `/api/graph/full` | Full graph (all nodes + edges) for overview visualization |
| `GET` | `/api/communities` | All Louvain communities with metrics and suspicious flag |
| `GET` | `/api/communities/{cluster_id}` | Community detail including member node IDs |
| `GET` | `/api/communities/{cluster_id}/subgraph` | Community subgraph (nodes + edges) for Pyvis |
| `GET` | `/api/explanations` | BFS explanations for all suspicious communities |
| `GET` | `/api/explanations/{cluster_id}` | BFS explanation for a specific community |

---

## Pipeline Components

### 1. Preprocessing (`src/ingestion/preprocess.py`)
Transforms raw AMLSim CSVs into ML-ready artifacts: **21-dimensional node features**, edge index, edge features, node labels, and a PyG `Data` object.

### 2. GNN Model (`src/models/gnn_model.py`)
A 2-layer **GraphSAGE** classifier (`FraudGNN`) with edge-feature injection. Takes `(node_features, edge_index, edge_attr)` and outputs per-node fraud probability logits. Trained with `NeighborLoader`, stratified masking, and class-weighted cross-entropy loss.

### 3. Louvain Community Detection (`src/detection/louvain.py`)
Groups accounts into communities using the Louvain modularity algorithm. Computes per-cluster metrics (size, density, average risk) and flags clusters exceeding configurable thresholds as suspicious.

### 4. BFS Explainer (`src/explainability/bfs_explainer.py`)
Starting from the highest-risk node in each suspicious community, performs a breadth-first traversal to produce a step-by-step human-readable explanation of the fraud ring structure, including transaction amounts and risk propagation.

### 5. FastAPI Backend (`src/api/`)
Serves all pipeline results as a REST API. Currently uses mock risk scores derived from ground-truth labels (the `_generate_mock_risk_scores()` method in `graph_service.py` can be swapped for real GNN inference output).

### 6. Streamlit Dashboard (`src/dashboard/app.py`)
Two-page interactive dashboard:
- **Overview** — KPI metrics, full-network Pyvis graph (250 nodes, 800 edges with fraud color-coding), and community summary table
- **Cluster Investigation** — Select any suspicious cluster to see its BFS explanation and interactive subgraph visualization

---

## Fraud Patterns Detected

- **Fraud rings** — dense clusters of tightly connected accounts with high aggregate risk
- **High-risk accounts** — individual nodes with elevated GNN-predicted fraud probability
- **Suspicious subgraph extraction** — isolated community subgraphs ready for interactive visual inspection and BFS traversal
