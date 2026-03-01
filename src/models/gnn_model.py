"""
gnn_model.py — GraphSAGE Fraud Classifier
==========================================

Architecture-only module.  Defines the GNN that maps
(node_features, edge_index, edge_attr) → per-node logits.

No loss function, optimizer, or training logic lives here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FraudGNN(nn.Module):
    """
    2-layer GraphSAGE classifier for node-level fraud detection.

    Edge features are incorporated by projecting them into per-edge
    scalars and using them to weight the adjacency (via sparse
    multiplication on the source-node embeddings before aggregation).

    Parameters
    ----------
    in_channels : int
        Number of input node features (21 in our graph).
    hidden_channels : int
        Width of each hidden GraphSAGE layer (default 64).
    out_channels : int
        Number of output classes (default 2: legit / fraud).
    edge_dim : int
        Dimensionality of edge feature vectors (3 in our graph).
    dropout : float
        Dropout probability applied after each conv layer.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 2,
        edge_dim: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── Edge feature projection ───────────────────────
        # Projects D-dim edge features → hidden_channels so they
        # can be added to neighbour messages after the first conv.
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.ReLU(),
        )

        # ── GraphSAGE convolutions ────────────────────────
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        # ── Batch normalisation ───────────────────────────
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        # ── Classification head ───────────────────────────
        self.classifier = nn.Linear(hidden_channels, out_channels)

        self.dropout = dropout

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x          : FloatTensor [N, F]   — node feature matrix
        edge_index : LongTensor  [2, E]   — COO edge list
        edge_attr  : FloatTensor [E, D]   — per-edge features

        Returns
        -------
        logits : FloatTensor [N, 2] — raw class scores (apply softmax
                 externally for probabilities).
        """
        # ── Encode edge features ─────────────────────────
        # Map edge features to hidden dim; scatter-add onto target
        # nodes to inject edge information into the node embeddings.
        edge_enc = self.edge_encoder(edge_attr)          # [E, H]
        target_nodes = edge_index[1]                      # [E]
        edge_agg = torch.zeros(x.size(0), edge_enc.size(1),
                               device=x.device)
        edge_agg.scatter_add_(0, target_nodes.unsqueeze(1)
                              .expand_as(edge_enc), edge_enc)

        # ── Layer 1 ──────────────────────────────────────
        h = self.conv1(x, edge_index)
        h = h + edge_agg                                  # fuse edge info
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ── Layer 2 ──────────────────────────────────────
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # ── Classification head ──────────────────────────
        logits = self.classifier(h)
        return logits
