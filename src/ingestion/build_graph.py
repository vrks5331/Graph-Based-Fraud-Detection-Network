import pandas as pd
import networkx as nx
import numpy as np

def build_transaction_graph(
    accounts_path="data/processed/accounts_processed.csv",
    transactions_path="data/processed/transactions_processed.csv",
    alerts_path="data/processed/alerts_processed.csv"
):
    """
    Builds a directed financial transaction graph.

    Nodes: accounts
    Edges: transactions

    Node attributes: account info
    Edge attributes: amount, timestamp, fraud, alert
    """

    accounts = pd.read_csv(accounts_path)
    transactions = pd.read_csv(transactions_path)
    alerts = pd.read_csv(alerts_path)
    
    G = nx.DiGraph()

    for _, row in accounts.iterrows():

        G.add_node(
            account_id = row['account_id'],
            balance=row['balance'],
            account_type=row['account_type'],
            customer_id=row['customer_id']
        )