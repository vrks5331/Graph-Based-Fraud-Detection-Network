import pandas as pd
import json
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------

RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")

ACCOUNTS_FILE = RAW_PATH / "accounts.csv"
TRANSACTIONS_FILE = RAW_PATH / "transactions.csv"
ALERTS_FILE = RAW_PATH / "alerts.csv"

PROCESSED_PATH.mkdir(parents=True, exist_ok=True)


# -----------------------------
# LOAD DATA
# -----------------------------

def load_data():
    """
    Loads raw AMLSim datasets.
    Only loads necessary columns to reduce memory usage.
    """

    accounts = pd.read_csv(ACCOUNTS_FILE)

    transactions = pd.read_csv(
        TRANSACTIONS_FILE,
        usecols=["origAccount", "destAccount", "amount"]
    )

    alerts = pd.read_csv(ALERTS_FILE)

    return accounts, transactions, alerts


# -----------------------------
# CREATE NODE MAPPING
# -----------------------------

def create_account_mapping(transactions):
    """
    GNNs require integer node indices.
    Map account IDs → integer node IDs.
    """

    unique_accounts = pd.concat([
        transactions["origAccount"],
        transactions["destAccount"]
    ]).unique()

    mapping = {acc: idx for idx, acc in enumerate(unique_accounts)}

    with open(PROCESSED_PATH / "node_mapping.json", "w") as f:
        json.dump(mapping, f)

    return mapping


# -----------------------------
# APPLY MAPPING
# -----------------------------

def apply_mapping(transactions, mapping):

    transactions["source_id"] = transactions["origAccount"].map(mapping)
    transactions["target_id"] = transactions["destAccount"].map(mapping)

    return transactions


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

def engineer_node_features(transactions):
    """
    Create structural features per account.
    These become node features for the GNN.
    """

    # outgoing stats
    outgoing = transactions.groupby("source_id").agg(
        out_tx_count=("amount", "count"),
        out_tx_sum=("amount", "sum"),
        out_tx_mean=("amount", "mean")
    )

    # incoming stats
    incoming = transactions.groupby("target_id").agg(
        in_tx_count=("amount", "count"),
        in_tx_sum=("amount", "sum"),
        in_tx_mean=("amount", "mean")
    )

    # merge features
    features = outgoing.join(incoming, how="outer").fillna(0)

    # total degree proxy
    features["total_tx"] = (
        features["out_tx_count"] + features["in_tx_count"]
    )

    return features.reset_index().rename(columns={"index": "node_id"})


# -----------------------------
# CREATE LABELS FROM ALERTS
# -----------------------------

def create_labels(alerts, mapping):
    """
    Convert alerts → node fraud labels.
    If an account appears in an alert, mark as fraud.
    """

    labels = pd.DataFrame({
        "node_id": list(mapping.values()),
        "label": 0
    })

    alerted_accounts = alerts["accountID"].map(mapping).dropna().astype(int)

    labels.loc[labels["node_id"].isin(alerted_accounts), "label"] = 1

    return labels


# -----------------------------
# SAVE OUTPUTS
# -----------------------------

def save_outputs(transactions, features, labels):

    transactions[["source_id", "target_id", "amount"]].to_csv(
        PROCESSED_PATH / "transactions_processed.csv",
        index=False
    )

    features.to_csv(PROCESSED_PATH / "node_features.csv", index=False)
    labels.to_csv(PROCESSED_PATH / "node_labels.csv", index=False)


# -----------------------------
# MAIN PIPELINE
# -----------------------------

def main():

    print("Loading data...")
    accounts, transactions, alerts = load_data()

    print("Creating node mapping...")
    mapping = create_account_mapping(transactions)

    print("Applying mapping...")
    transactions = apply_mapping(transactions, mapping)

    print("Engineering node features...")
    features = engineer_node_features(transactions)

    print("Creating labels...")
    labels = create_labels(alerts, mapping)

    print("Saving outputs...")
    save_outputs(transactions, features, labels)

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()