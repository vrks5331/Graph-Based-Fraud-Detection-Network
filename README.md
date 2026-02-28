# Graph Based Fraud Detection Network

Architecture:
- Neo4j: Graph Database
- FastAPI: API
- Streamlit: Dashboard
- PyTorch Geometric: GNN Model

Dataset → Graph Builder → PyG → GNN → Risk Scoring
                                ↓
                              Louvain
                                ↓
                               BFS
                                ↓
                             Streamlit




Fraud Patterns Detected: 
- Fraud rings (dense clusters)
- High-risk accounts
- Suspicious subgraph extraction
