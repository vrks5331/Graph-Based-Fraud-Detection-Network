"""
inference.py — GNN Risk Score Generation
=========================================

Loads a trained GNN checkpoint and produces a ``Dict[str, float]``
mapping each original account ID to a fraud risk score (0.0 → 1.0).

This output is consumed by the downstream pipeline:
  - ``detection/louvain.py``   → community detection with ``node_risk``
  - ``explainability/bfs_explainer.py`` → narrative explanation

Usage
-----
    # As a module
    from src.models.inference import generate_risk_scores
    scores = generate_risk_scores()          # Dict[str, float]

    # CLI — writes risk_scores.json
    python -m src.models.inference
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.gnn_model import FraudGNN

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────
PROCESSED      = Path("data/processed")
GRAPH_PATH     = PROCESSED / "pyg_graph.pt"
CKPT_PATH      = PROCESSED / "gnn_checkpoint.pt"
MAPPING_PATH   = PROCESSED / "node_mapping.json"
SCORES_PATH    = PROCESSED / "risk_scores.json"
SCORES_META    = PROCESSED / "risk_scores_meta.json"


# ═══════════════════════════════════════════════════════════
#  CORE
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def generate_risk_scores(
    graph_path: Path = GRAPH_PATH,
    ckpt_path: Path = CKPT_PATH,
    mapping_path: Path = MAPPING_PATH,
) -> dict[str, float]:
    """
    Load the trained GNN and return per-node fraud risk scores.

    Returns
    -------
    scores : Dict[str, float]
        Keys   = original account IDs (from ``node_mapping.json``)
        Values = fraud probability ∈ [0.0, 1.0]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load graph ────────────────────────────────────────
    data = torch.load(graph_path, weights_only=False)
    log.info("Graph loaded  nodes=%d", data.num_nodes)

    # ── Load model ────────────────────────────────────────
    model = FraudGNN(
        in_channels=data.x.size(1),
        edge_dim=data.edge_attr.size(1),
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    log.info("Checkpoint loaded from %s", ckpt_path)

    # ── Full-graph forward pass ───────────────────────────
    data = data.to(device)
    infer_start = time.time()
    logits = model(data.x, data.edge_index, data.edge_attr)      # [N, 2]
    probs  = F.softmax(logits, dim=1)[:, 1]                      # fraud prob
    probs  = probs.cpu().tolist()
    infer_time = time.time() - infer_start

    # ── Map node indices → original account IDs ───────────
    #    node_mapping.json is  { original_id : int_index }
    #    We need the inverse   { int_index : original_id }
    with open(mapping_path, "r") as f:
        raw_mapping: dict[str, int] = json.load(f)

    inv_mapping = {v: k for k, v in raw_mapping.items()}

    scores: dict[str, float] = {}
    for idx, prob in enumerate(probs):
        account_id = inv_mapping.get(idx, str(idx))
        scores[account_id] = round(prob, 6)

    log.info(
        "Risk scores generated  nodes=%d  mean=%.4f  max=%.4f",
        len(scores),
        sum(scores.values()) / len(scores),
        max(scores.values()),
    )
    # attach inference metadata for callers
    meta = {"inference_seconds": round(infer_time, 3), "nodes": len(scores)}
    return scores, meta


def save_risk_scores(scores: dict[str, float], out_path: Path = SCORES_PATH, meta: dict | None = None):
    """Write risk scores and optional metadata to JSON for downstream consumption."""
    out_path.write_text(json.dumps(scores, indent=2))
    log.info("Risk scores saved to %s", out_path)
    if meta is not None:
        try:
            SCORES_META.write_text(json.dumps(meta, indent=2))
            log.info("Risk score metadata saved to %s", SCORES_META)
        except Exception:
            log.exception("Failed to write risk score metadata to %s", SCORES_META)


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    scores, meta = generate_risk_scores()
    save_risk_scores(scores, meta=meta)
