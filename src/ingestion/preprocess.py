"""
preprocess.py — Graph-Ready Data Preprocessing Pipeline
========================================================

Purpose
-------
Transforms raw AMLSim CSV data (accounts, transactions, alerts) into
graph-ready artifacts suitable for building a PyTorch Geometric (PyG)
heterogeneous / homogeneous graph and training a Graph Neural Network
for fraud detection.

Pipeline overview
-----------------
1. Load raw CSVs with explicit dtypes (memory-efficient).
2. Clean & validate — drop duplicates, handle missing values, type-cast.
3. Create a deterministic integer node-ID mapping (required by PyG).
4. Engineer per-node features from account attributes + transaction stats.
5. Engineer per-edge features from transaction attributes.
6. Derive node-level fraud labels from accounts and alerts.
7. Persist all artifacts to  data/processed/  as CSVs & JSON.

Output artifacts  (all written to data/processed/)
---------------------------------------------------
  ├─ node_mapping.json          # {original_account_id: int_node_id}
  ├─ node_features.csv          # rows = nodes, columns = feature dims
  ├─ edge_index.csv             # source_id, target_id  (integer pairs)
  ├─ edge_features.csv          # per-edge feature vector  (parallel to edge_index)
  ├─ node_labels.csv            # node_id, label  (0 = legit, 1 = fraud)
  └─ pipeline_meta.json         # run metadata  (counts, timestamp, etc.)

Usage
-----
    python -m src.ingestion.preprocess          # from project root
    python src/ingestion/preprocess.py          # direct invocation
"""

# ──────────────────────────────────────────────
#  Imports
# ──────────────────────────────────────────────
import pandas as pd
import numpy as np
import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

# ──────────────────────────────────────────────
#  Logging Setup
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Path Configuration
# ──────────────────────────────────────────────
# All paths are relative to the project root so the script works whether
# invoked via `python -m src.ingestion.preprocess` or directly.

RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")

ACCOUNTS_FILE   = RAW_PATH / "accounts.csv"
TRANSACTIONS_FILE = RAW_PATH / "transactions.csv"
ALERTS_FILE     = RAW_PATH / "alerts.csv"

# Make sure the output directory exists before writing anything.
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════
#  1. DATA LOADING
# ═══════════════════════════════════════════════

def load_data():
    """
    Load the three raw AMLSim CSVs.

    Notes
    -----
    - We specify *explicit dtypes* to prevent pandas from up-casting integer
      columns to float when NaN values are present.
    - Only necessary columns are loaded from the (very large) transactions
      file to keep memory usage reasonable (~60 MB raw).

    Returns
    -------
    accounts : pd.DataFrame
    transactions : pd.DataFrame
    alerts : pd.DataFrame
    """
    log.info("Loading accounts from %s", ACCOUNTS_FILE)
    accounts = pd.read_csv(
        ACCOUNTS_FILE,
        dtype={
            "ACCOUNT_ID": np.int64,
            "INIT_BALANCE": np.float64,
            "TX_BEHAVIOR_ID": np.int64,
        },
    )

    log.info("Loading transactions from %s  (this may take a moment)", TRANSACTIONS_FILE)
    transactions = pd.read_csv(
        TRANSACTIONS_FILE,
        usecols=[
            "TX_ID",
            "SENDER_ACCOUNT_ID",
            "RECEIVER_ACCOUNT_ID",
            "TX_AMOUNT",
            "TIMESTAMP",
            "IS_FRAUD",
            "ALERT_ID",
        ],
        dtype={
            "TX_ID": np.int64,
            "SENDER_ACCOUNT_ID": np.int64,
            "RECEIVER_ACCOUNT_ID": np.int64,
            "TX_AMOUNT": np.float64,
            "TIMESTAMP": np.int64,
            "ALERT_ID": np.int64,
        },
    )

    log.info("Loading alerts from %s", ALERTS_FILE)
    alerts = pd.read_csv(
        ALERTS_FILE,
        dtype={
            "ALERT_ID": np.int64,
            "TX_ID": np.int64,
            "SENDER_ACCOUNT_ID": np.int64,
            "RECEIVER_ACCOUNT_ID": np.int64,
            "TX_AMOUNT": np.float64,
            "TIMESTAMP": np.int64,
        },
    )

    log.info(
        "Loaded  accounts=%d  transactions=%d  alerts=%d",
        len(accounts), len(transactions), len(alerts),
    )
    return accounts, transactions, alerts


# ═══════════════════════════════════════════════
#  2. CLEANING & VALIDATION
# ═══════════════════════════════════════════════

def clean_accounts(accounts: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the accounts table.

    Steps
    -----
    1. Drop duplicate ACCOUNT_IDs (keep first occurrence).
    2. Fill any missing INIT_BALANCE with the column median — a safer
       imputation than mean because balances can be highly skewed.
    3. Encode categorical columns as integers so every column is numeric
       (GNNs require numeric tensors).
    """
    before = len(accounts)
    accounts = accounts.drop_duplicates(subset=["ACCOUNT_ID"], keep="first")
    after = len(accounts)
    if before != after:
        log.warning("Dropped %d duplicate accounts.", before - after)

    # --- Impute missing balances with the median ---
    if accounts["INIT_BALANCE"].isna().any():
        median_balance = accounts["INIT_BALANCE"].median()
        accounts["INIT_BALANCE"] = accounts["INIT_BALANCE"].fillna(median_balance)
        log.info("Imputed %d missing INIT_BALANCE values with median=%.2f",
                 accounts["INIT_BALANCE"].isna().sum(), median_balance)

    # --- Encode ACCOUNT_TYPE as integer ---
    # The raw data has a single category "I", but we future-proof with a
    # general label-encoding step.
    accounts["ACCOUNT_TYPE_ENC"] = (
        accounts["ACCOUNT_TYPE"].astype("category").cat.codes
    )

    # --- Encode COUNTRY as integer ---
    accounts["COUNTRY_ENC"] = (
        accounts["COUNTRY"].astype("category").cat.codes
    )

    return accounts


def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the transactions table.

    Steps
    -----
    1. Drop duplicate TX_IDs.
    2. Remove self-loops (sender == receiver) — they add no structural
       information and can confuse message-passing layers.
    3. Drop rows where amount <= 0 (invalid / placeholder rows).
    """
    before = len(transactions)
    transactions = transactions.drop_duplicates(subset=["TX_ID"], keep="first")
    dup_removed = before - len(transactions)
    if dup_removed:
        log.warning("Dropped %d duplicate transactions.", dup_removed)

    # Remove self-loops
    self_loop_mask = (
        transactions["SENDER_ACCOUNT_ID"] == transactions["RECEIVER_ACCOUNT_ID"]
    )
    n_self = self_loop_mask.sum()
    if n_self:
        transactions = transactions[~self_loop_mask]
        log.warning("Removed %d self-loop transactions.", n_self)

    # Remove non-positive amounts
    bad_amount_mask = transactions["TX_AMOUNT"] <= 0
    n_bad = bad_amount_mask.sum()
    if n_bad:
        transactions = transactions[~bad_amount_mask]
        log.warning("Removed %d transactions with amount <= 0.", n_bad)

    log.info("Transactions after cleaning: %d rows", len(transactions))
    return transactions


def clean_alerts(alerts: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the alerts table.

    Steps
    -----
    1. Drop full-row duplicates.
    2. Validate that every alert has a real ALERT_ID (> 0).
    """
    before = len(alerts)
    alerts = alerts.drop_duplicates()
    after = len(alerts)
    if before != after:
        log.warning("Dropped %d duplicate alert rows.", before - after)

    # Sanity-check: ALERT_IDs should be positive
    bad_ids = alerts["ALERT_ID"] <= 0
    if bad_ids.any():
        log.warning("Found %d alerts with non-positive ALERT_ID; dropping.", bad_ids.sum())
        alerts = alerts[~bad_ids]

    log.info("Alerts after cleaning: %d rows", len(alerts))
    return alerts


# ═══════════════════════════════════════════════
#  3. NODE-ID MAPPING
# ═══════════════════════════════════════════════

def create_node_mapping(accounts: pd.DataFrame, transactions: pd.DataFrame) -> dict:
    """
    Build a *deterministic* mapping from raw ACCOUNT_ID → contiguous int.

    Why both tables?
    ----------------
    The accounts table is the canonical list of nodes, but we union it with
    accounts that appear in transactions to catch any orphan senders/receivers
    not present in accounts.csv.  The mapping is sorted so it stays stable
    across runs.

    Returns
    -------
    mapping : dict[int, int]
        {raw_account_id: integer_node_id}
    """
    account_ids_from_accounts = set(accounts["ACCOUNT_ID"].unique())
    account_ids_from_txns = set(
        pd.concat([
            transactions["SENDER_ACCOUNT_ID"],
            transactions["RECEIVER_ACCOUNT_ID"],
        ]).unique()
    )

    # Union of both sources, sorted for determinism
    all_ids = sorted(account_ids_from_accounts | account_ids_from_txns)
    mapping = {int(acc_id): idx for idx, acc_id in enumerate(all_ids)}

    # Persist for later use (graph construction, inference, explainability)
    out_path = PROCESSED_PATH / "node_mapping.json"
    with open(out_path, "w") as f:
        json.dump(mapping, f)
    log.info("Node mapping: %d unique nodes → %s", len(mapping), out_path)

    return mapping


# ═══════════════════════════════════════════════
#  4. NODE FEATURE ENGINEERING
# ═══════════════════════════════════════════════

def engineer_node_features(
    accounts: pd.DataFrame,
    transactions: pd.DataFrame,
    mapping: dict,
) -> pd.DataFrame:
    """
    Combine *static account attributes* with *transaction-derived statistics*
    to produce a rich feature vector for every node.

    Feature groups
    --------------
    A) Account-level (from accounts.csv)
        - init_balance       : initial account balance  (continuous)
        - account_type_enc   : encoded account type     (categorical → int)
        - country_enc        : encoded country          (categorical → int)
        - tx_behavior_id     : transaction behaviour ID (categorical int)

    B) Outgoing transaction statistics
        - out_tx_count       : number of outgoing transactions
        - out_tx_sum         : total outgoing amount
        - out_tx_mean        : mean outgoing amount
        - out_tx_std         : std-dev of outgoing amounts  (captures burstiness)
        - out_tx_max         : largest single outgoing transaction

    C) Incoming transaction statistics
        - in_tx_count, in_tx_sum, in_tx_mean, in_tx_std, in_tx_max

    D) Derived / structural
        - total_tx_count     : in + out count  (degree proxy)
        - net_flow           : out_tx_sum − in_tx_sum   (money direction)
        - flow_ratio         : out_tx_sum / (in_tx_sum + 1)  (avoids div-by-zero)
        - tx_time_span       : max(timestamp) − min(timestamp)  per node
        - unique_counterparts: number of distinct accounts interacted with

    Notes
    -----
    - All features are **numeric** and ready for torch tensor conversion.
    - NaN is filled with 0 for nodes that have no incoming or outgoing txns
      (e.g., a node that only receives never sends).
    """

    # ── A) Account-level features ──────────────────────────────
    accounts = accounts.copy()
    accounts["node_id"] = accounts["ACCOUNT_ID"].map(mapping)

    account_feats = accounts[[
        "node_id",
        "INIT_BALANCE",
        "ACCOUNT_TYPE_ENC",
        "COUNTRY_ENC",
        "TX_BEHAVIOR_ID",
    ]].rename(columns={
        "INIT_BALANCE":     "init_balance",
        "ACCOUNT_TYPE_ENC": "account_type_enc",
        "COUNTRY_ENC":      "country_enc",
        "TX_BEHAVIOR_ID":   "tx_behavior_id",
    })

    # ── B) Outgoing statistics ──────────────────────────────────
    txns = transactions.copy()
    txns["source_id"] = txns["SENDER_ACCOUNT_ID"].map(mapping)
    txns["target_id"] = txns["RECEIVER_ACCOUNT_ID"].map(mapping)

    outgoing = txns.groupby("source_id").agg(
        out_tx_count  = ("TX_AMOUNT", "count"),
        out_tx_sum    = ("TX_AMOUNT", "sum"),
        out_tx_mean   = ("TX_AMOUNT", "mean"),
        out_tx_std    = ("TX_AMOUNT", "std"),
        out_tx_max    = ("TX_AMOUNT", "max"),
    )

    # ── C) Incoming statistics ──────────────────────────────────
    incoming = txns.groupby("target_id").agg(
        in_tx_count  = ("TX_AMOUNT", "count"),
        in_tx_sum    = ("TX_AMOUNT", "sum"),
        in_tx_mean   = ("TX_AMOUNT", "mean"),
        in_tx_std    = ("TX_AMOUNT", "std"),
        in_tx_max    = ("TX_AMOUNT", "max"),
    )

    # ── D) Derived structural features ─────────────────────────

    # Time span of activity per node (union of in/out timestamps)
    out_time = txns.groupby("source_id")["TIMESTAMP"].agg(["min", "max"])
    in_time  = txns.groupby("target_id")["TIMESTAMP"].agg(["min", "max"])
    out_time.columns = ["out_tmin", "out_tmax"]
    in_time.columns  = ["in_tmin",  "in_tmax"]

    # Unique counterparts (distinct neighbours)
    out_uniq = txns.groupby("source_id")["target_id"].nunique().rename("out_unique_nbrs")
    in_uniq  = txns.groupby("target_id")["source_id"].nunique().rename("in_unique_nbrs")

    # ── Merge everything on node_id ─────────────────────────────
    # Start with a scaffold of all node IDs (ensures nodes with no txns get a row)
    all_node_ids = pd.DataFrame({"node_id": sorted(mapping.values())})

    features = all_node_ids \
        .merge(account_feats,  on="node_id", how="left") \
        .merge(outgoing,       left_on="node_id", right_index=True, how="left") \
        .merge(incoming,       left_on="node_id", right_index=True, how="left") \
        .merge(out_time,       left_on="node_id", right_index=True, how="left") \
        .merge(in_time,        left_on="node_id", right_index=True, how="left") \
        .merge(out_uniq.to_frame(), left_on="node_id", right_index=True, how="left") \
        .merge(in_uniq.to_frame(),  left_on="node_id", right_index=True, how="left")

    # Fill NaN for nodes with no transactions in a given direction
    features = features.fillna(0)

    # Compute high-level derived features
    features["total_tx_count"] = features["out_tx_count"] + features["in_tx_count"]

    # Net flow: positive means net sender
    features["net_flow"] = features["out_tx_sum"] - features["in_tx_sum"]

    # Flow ratio: how much more a node sends than receives
    features["flow_ratio"] = features["out_tx_sum"] / (features["in_tx_sum"] + 1.0)

    # Activity time span (union of in/out windows)
    features["tx_time_span"] = (
        features[["out_tmax", "in_tmax"]].max(axis=1)
        - features[["out_tmin", "in_tmin"]].min(axis=1)
    )

    # Total unique counterparts
    features["unique_counterparts"] = (
        features["out_unique_nbrs"] + features["in_unique_nbrs"]
    )

    # Drop intermediate time columns (they served their purpose)
    features = features.drop(
        columns=["out_tmin", "out_tmax", "in_tmin", "in_tmax"],
        errors="ignore",
    )

    log.info(
        "Node features: %d nodes × %d features",
        len(features), features.shape[1] - 1,  # -1 for node_id
    )
    return features


# ═══════════════════════════════════════════════
#  5. EDGE INDEX & EDGE FEATURES
# ═══════════════════════════════════════════════

def build_edge_artifacts(
    transactions: pd.DataFrame,
    mapping: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the edge index and per-edge feature matrix.

    Edge index
    ----------
    Two-column DataFrame [source_id, target_id] with contiguous integer IDs.
    This maps directly to PyG's ``edge_index`` tensor  (shape [2, E]).

    Edge features
    -------------
    Per-edge numeric attributes that can become ``edge_attr`` in PyG.
    Currently includes:
        - amount          : transaction amount  (continuous)
        - timestamp       : raw timestamp       (ordinal int)
        - is_fraud        : ground-truth edge fraud flag  (kept for analysis;
                            you may choose to exclude it from training features)
        - log_amount      : log1p(amount) — stabilises large-value skew

    Returns
    -------
    edge_index    : pd.DataFrame  — columns [source_id, target_id]
    edge_features : pd.DataFrame  — columns per feature (rows align 1-to-1
                                     with edge_index)
    """
    txns = transactions.copy()
    txns["source_id"] = txns["SENDER_ACCOUNT_ID"].map(mapping)
    txns["target_id"] = txns["RECEIVER_ACCOUNT_ID"].map(mapping)

    # ── Edge index ──
    edge_index = txns[["source_id", "target_id"]].reset_index(drop=True)

    # ── Edge features ──
    edge_features = pd.DataFrame({
        "amount":    txns["TX_AMOUNT"].values,
        "timestamp": txns["TIMESTAMP"].values,
        "is_fraud":  txns["IS_FRAUD"].astype(int).values,
        "log_amount": np.log1p(txns["TX_AMOUNT"].values),
    })

    log.info("Edge artifacts: %d edges, %d edge features",
             len(edge_index), edge_features.shape[1])

    return edge_index, edge_features


# ═══════════════════════════════════════════════
#  6. NODE LABELS
# ═══════════════════════════════════════════════

def create_labels(
    accounts: pd.DataFrame,
    alerts: pd.DataFrame,
    mapping: dict,
) -> pd.DataFrame:
    """
    Produce per-node binary fraud labels.

    Labelling logic
    ---------------
    A node is labelled **fraud (1)** if EITHER:
      • its IS_FRAUD flag in accounts.csv is True, OR
      • the account appears as a sender or receiver in the alerts table.

    All other nodes are labelled **legitimate (0)**.

    This two-source approach is more robust than relying on a single table
    because:
      – accounts.csv gives us a direct fraud flag per account.
      – alerts.csv catches accounts involved in suspicious *patterns*
        (fan-in, cycles, etc.) that may not be individually flagged.

    Returns
    -------
    labels : pd.DataFrame  — columns [node_id, label]
    """
    # Start with all nodes labelled 0
    all_node_ids = sorted(mapping.values())
    labels = pd.DataFrame({"node_id": all_node_ids, "label": 0})

    # ── Source 1: IS_FRAUD flag from accounts ──
    fraud_accts_raw = accounts.loc[accounts["IS_FRAUD"] == True, "ACCOUNT_ID"]
    fraud_nodes_from_accts = set(
        fraud_accts_raw.map(mapping).dropna().astype(int)
    )

    # ── Source 2: accounts appearing in the alerts table ──
    alerted_senders   = alerts["SENDER_ACCOUNT_ID"].map(mapping).dropna().astype(int)
    alerted_receivers = alerts["RECEIVER_ACCOUNT_ID"].map(mapping).dropna().astype(int)
    fraud_nodes_from_alerts = set(alerted_senders) | set(alerted_receivers)

    # ── Combine & apply ──
    all_fraud_nodes = fraud_nodes_from_accts | fraud_nodes_from_alerts
    labels.loc[labels["node_id"].isin(all_fraud_nodes), "label"] = 1

    n_fraud = labels["label"].sum()
    n_total = len(labels)
    log.info(
        "Labels: %d fraud / %d total  (%.2f%% positive rate)",
        n_fraud, n_total, 100.0 * n_fraud / n_total,
    )

    return labels


# ═══════════════════════════════════════════════
#  7. SAVE ALL OUTPUTS
# ═══════════════════════════════════════════════

def save_outputs(
    node_features: pd.DataFrame,
    edge_index: pd.DataFrame,
    edge_features: pd.DataFrame,
    labels: pd.DataFrame,
    mapping: dict,
) -> None:
    """
    Persist every artifact to ``data/processed/``.

    File inventory
    --------------
    node_features.csv   — one row per node; first column is ``node_id``
    edge_index.csv      — [source_id, target_id] per edge
    edge_features.csv   — per-edge feature vector  (rows aligned with edge_index)
    node_labels.csv     — [node_id, label]
    pipeline_meta.json  — bookkeeping: row counts, feature dims, timestamp
    node_mapping.json   — already saved by create_node_mapping(), listed here
                          for completeness
    """
    node_features.to_csv(PROCESSED_PATH / "node_features.csv", index=False)
    edge_index.to_csv(PROCESSED_PATH / "edge_index.csv", index=False)
    edge_features.to_csv(PROCESSED_PATH / "edge_features.csv", index=False)
    labels.to_csv(PROCESSED_PATH / "node_labels.csv", index=False)

    # ── Pipeline metadata ──
    meta = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "num_nodes": len(node_features),
        "num_edges": len(edge_index),
        "node_feature_dim": node_features.shape[1] - 1,   # exclude node_id col
        "edge_feature_dim": edge_features.shape[1],
        "num_fraud_nodes": int(labels["label"].sum()),
        "num_legit_nodes": int((labels["label"] == 0).sum()),
        "output_files": [
            "node_mapping.json",
            "node_features.csv",
            "edge_index.csv",
            "edge_features.csv",
            "node_labels.csv",
            "pipeline_meta.json",
        ],
    }
    with open(PROCESSED_PATH / "pipeline_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("All artifacts saved to %s/", PROCESSED_PATH)


# ═══════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════

def main() -> None:
    """
    Entry-point: orchestrates the full preprocessing pipeline end-to-end.

    Steps executed in order
    -----------------------
    1. Load raw data
    2. Clean & validate each table
    3. Create node mapping  (account_id → int)
    4. Engineer node features
    5. Build edge index + edge features
    6. Create node labels
    7. Save all outputs
    """
    log.info("=" * 60)
    log.info("  GRAPH FRAUD DETECTION — Preprocessing Pipeline")
    log.info("=" * 60)

    # ── Step 1: Load ──
    log.info("[1/7] Loading raw data …")
    accounts, transactions, alerts = load_data()

    # ── Step 2: Clean ──
    log.info("[2/7] Cleaning & validating …")
    accounts     = clean_accounts(accounts)
    transactions = clean_transactions(transactions)
    alerts       = clean_alerts(alerts)

    # ── Step 3: Node mapping ──
    log.info("[3/7] Creating node mapping …")
    mapping = create_node_mapping(accounts, transactions)

    # ── Step 4: Node features ──
    log.info("[4/7] Engineering node features …")
    node_features = engineer_node_features(accounts, transactions, mapping)

    # ── Step 5: Edge artifacts ──
    log.info("[5/7] Building edge index & edge features …")
    edge_index, edge_features = build_edge_artifacts(transactions, mapping)

    # ── Step 6: Labels ──
    log.info("[6/7] Creating node labels …")
    labels = create_labels(accounts, alerts, mapping)

    # ── Step 7: Save ──
    log.info("[7/7] Saving outputs …")
    save_outputs(node_features, edge_index, edge_features, labels, mapping)

    log.info("=" * 60)
    log.info("  Pipeline complete ✓")
    log.info("=" * 60)


# ──────────────────────────────────────────────
#  Script entry-point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()