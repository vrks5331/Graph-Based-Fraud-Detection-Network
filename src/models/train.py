"""
train.py — GNN Training Pipeline
=================================

Handles everything the model needs to *learn*:
- Loading ``pyg_graph.pt``
- Creating stratified train / val / test masks
- NeighborLoader with oversampled fraud seed nodes
- Class-weighted cross-entropy loss
- Adam optimiser with ReduceLROnPlateau scheduler
- Early stopping on validation F1
- Checkpoint saving

Usage
-----
    python -m src.models.train               # defaults (100 epochs)
    python -m src.models.train --epochs 5    # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
# NeighborLoader is imported lazily inside `build_loaders` because the
# optional PyG acceleration libraries (pyg-lib / torch-sparse) may be
# unavailable in some environments.  If the import fails we fall back to
# full-graph training using the masks on the Data object.

from src.models.gnn_model import FraudGNN

# ── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────
PROCESSED    = Path("data/processed")
GRAPH_PATH   = PROCESSED / "pyg_graph.pt"
CKPT_PATH    = PROCESSED / "gnn_checkpoint.pt"
METRICS_PATH = PROCESSED / "training_metrics.json"


# ═══════════════════════════════════════════════════════════
#  1.  DATA PREPARATION
# ═══════════════════════════════════════════════════════════

def load_graph():
    """Load the pre-built PyG Data object."""
    log.info("Loading graph from %s …", GRAPH_PATH)
    data = torch.load(GRAPH_PATH, weights_only=False)
    log.info(
        "  nodes=%d  edges=%d  features=%d  fraud=%.1f%%",
        data.num_nodes,
        data.edge_index.size(1),
        data.x.size(1),
        100 * data.y.sum().item() / data.num_nodes,
    )
    return data


def create_masks(data, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
    """
    Stratified train / val / test split stored as boolean masks on ``data``.

    The split is stratified by label so that the fraud ratio is
    preserved across all three sets.
    """
    n = data.num_nodes
    indices = list(range(n))
    labels  = data.y.numpy()

    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=val_ratio + test_ratio,
        stratify=labels,
        random_state=seed,
    )
    # Second split: val vs test
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test,
        stratify=labels[temp_idx],
        random_state=seed,
    )

    # Build boolean masks
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    log.info(
        "  masks  train=%d  val=%d  test=%d",
        train_mask.sum().item(),
        val_mask.sum().item(),
        test_mask.sum().item(),
    )
    return data


def compute_class_weights(data):
    """
    Return a weight tensor [w_legit, w_fraud] inversely proportional
    to class frequency so the loss penalises fraud misses harder.
    """
    labels = data.y[data.train_mask]
    counts = torch.bincount(labels, minlength=2).float()
    weights = counts.sum() / (2.0 * counts)          # inverse frequency
    log.info("  class weights  legit=%.3f  fraud=%.3f", weights[0], weights[1])
    return weights


# ═══════════════════════════════════════════════════════════
#  2.  NEIGHBOR LOADER  (with fraud oversampling)
# ═══════════════════════════════════════════════════════════

def build_loaders(data, batch_size: int = 256, num_neighbors: list[int] | None = None):
    """
    Create NeighborLoaders for train, val, and test.

    Training loader uses a ``WeightedRandomSampler`` so that fraud
    nodes are picked as seed nodes ~5× more often, giving each
    mini-batch a roughly balanced label distribution.
    """
    if num_neighbors is None:
        num_neighbors = [10, 5]       # 2-hop sampling

    # ── Fraud-oversampled seed sampler (train only) ───────
    train_indices = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_labels  = data.y[train_indices]

    # Weight each node inversely by its class frequency
    counts = torch.bincount(train_labels, minlength=2).float()
    class_weight = 1.0 / counts
    sample_weights = class_weight[train_labels]

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_indices),
        replacement=True,
    )
    # Attempt to use NeighborLoader; if optional PyG libs are missing,
    # fall back to returning the full Data object wrapped in an iterable.
    try:
        from torch_geometric.loader import NeighborLoader

        # Build the NeighborLoader for training
        # We pass input_nodes as the training mask
        train_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=data.train_mask,
            sampler=sampler,
        )

        # Val / test loaders — no oversampling, fixed order
        val_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=data.val_mask,
        )

        test_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            input_nodes=data.test_mask,
        )

        log.info(
            "  loaders ready  train_batches=%d  val_batches=%d  test_batches=%d",
            len(train_loader), len(val_loader), len(test_loader),
        )
        return train_loader, val_loader, test_loader
    except Exception:
        log.warning("NeighborLoader unavailable; falling back to full-graph loaders")
        return [data], [data], [data]


# ═══════════════════════════════════════════════════════════
#  3.  TRAINING  &  EVALUATION
# ═══════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch.  Returns average loss."""
    model.train()
    total_loss = 0
    total_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index, batch.edge_attr)

        # Determine which nodes to compute loss on.
        # If we're using NeighborLoader the seed nodes are the first `batch_size` rows.
        if hasattr(batch, "batch_size"):
            seed_count = batch.batch_size
            out_seed = out[:seed_count]
            y_seed = batch.y[:seed_count]
        else:
            # Full-graph fallback: use the training mask on the data object
            if hasattr(batch, "train_mask"):
                mask = batch.train_mask
                out_seed = out[mask]
                y_seed = batch.y[mask]
            else:
                out_seed = out
                y_seed = batch.y

        loss = criterion(out_seed, y_seed)
        loss.backward()
        optimizer.step()

        total_loss  += loss.item() * y_seed.size(0)
        total_nodes += y_seed.size(0)

    return total_loss / max(total_nodes, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate on a loader.  Returns dict with loss, f1, auc."""
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr)

        # Extract seed nodes depending on loader type
        if hasattr(batch, "batch_size"):
            seed_count = batch.batch_size
            out_seed = out[:seed_count]
            y_seed = batch.y[:seed_count]
        else:
            # Full-graph fallback: use val/test mask if present
            if hasattr(batch, "val_mask"):
                mask = batch.val_mask
            elif hasattr(batch, "test_mask"):
                mask = batch.test_mask
            else:
                # If no mask available, assume whole graph
                mask = torch.ones(out.size(0), dtype=torch.bool, device=out.device)
            out_seed = out[mask]
            y_seed = batch.y[mask]

        probs = F.softmax(out_seed, dim=1)[:, 1]
        preds = out_seed.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(y_seed.cpu())
        all_probs.append(probs.cpu())

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = torch.cat(all_probs).numpy()

    f1  = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0

    return {"f1": f1, "auc": auc}


# ═══════════════════════════════════════════════════════════
#  4.  MAIN
# ═══════════════════════════════════════════════════════════

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # track global training time
    overall_start = time.time()

    # ── Data ──────────────────────────────────────────────
    data = load_graph()
    data = create_masks(data)
    class_weights = compute_class_weights(data).to(device)

    try:
        train_loader, val_loader, test_loader = build_loaders(
            data,
            batch_size=args.batch_size,
        )
    except Exception:
        # NeighborLoader or its optional dependencies may be missing
        log.warning("NeighborLoader unavailable; falling back to full-graph training")
        # wrap the full Data object in a single-item iterable so the training
        # loop still works with the same API
        train_loader = [data]
        val_loader = [data]
        test_loader = [data]

    # ── Model ─────────────────────────────────────────────
    model = FraudGNN(
        in_channels=data.x.size(1),
        hidden_channels=args.hidden,
        edge_dim=data.edge_attr.size(1),
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %s", f"{total_params:,}")

    # ── Optimiser & loss ──────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=7
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ── Training loop ─────────────────────────────────────
    best_val_f1  = 0.0
    patience_ctr = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val  = evaluate(model, val_loader, device)
        scheduler.step(val["f1"])

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        log.info(
            "Epoch %3d/%d  loss=%.4f  val_f1=%.4f  val_auc=%.4f  lr=%.2e  epoch_time=%.2fs",
            epoch, args.epochs, loss, val["f1"], val["auc"], current_lr, epoch_time,
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(loss, 5),
            "val_f1": round(val["f1"], 5),
            "val_auc": round(val["auc"], 5),
            "lr": current_lr,
            "epoch_time": round(epoch_time, 3),
        })

        # Write incremental metrics (so you can tail the file while training)
        elapsed = time.time() - overall_start
        avg_epoch = elapsed / epoch
        eta = avg_epoch * max(0, args.epochs - epoch)
        partial_metrics = {
            "best_val_f1": round(best_val_f1, 5),
            "elapsed_seconds": int(elapsed),
            "eta_seconds": int(eta),
            "history": history,
        }
        try:
            METRICS_PATH.write_text(json.dumps(partial_metrics, indent=2))
        except Exception:
            log.exception("Failed to write metrics to %s", METRICS_PATH)

        # ── Early stopping on val F1 ─────────────────────
        if val["f1"] > best_val_f1:
            best_val_f1  = val["f1"]
            patience_ctr = 0
            torch.save(model.state_dict(), CKPT_PATH)
            log.info("  ↑ best model saved  (val_f1=%.4f)", best_val_f1)
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                log.info("Early stopping triggered after %d epochs", epoch)
                break

    # ── Test evaluation ───────────────────────────────────
    model.load_state_dict(torch.load(CKPT_PATH, weights_only=True))
    test = evaluate(model, test_loader, device)
    log.info("Test results  f1=%.4f  auc=%.4f", test["f1"], test["auc"])

    # ── Save metrics ──────────────────────────────────────
    metrics = {
        "best_val_f1": round(best_val_f1, 5),
        "test_f1": round(test["f1"], 5),
        "test_auc": round(test["auc"], 5),
        "epochs_trained": len(history),
        "history": history,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    log.info("Metrics written to %s", METRICS_PATH)
    log.info("Checkpoint saved to %s", CKPT_PATH)


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Train GNN fraud detector")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--hidden",     type=int,   default=64)
    p.add_argument("--lr",         type=float, default=0.005)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--patience",   type=int,   default=15)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
