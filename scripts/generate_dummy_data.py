import pandas as pd
import numpy as np
import random
from pathlib import Path

def generate_dummy_data():
    """Generates a small generalized set of dummy AMLSim data for testing."""
    output_dir = Path("data/testing")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating dummy data for testing...")

    # 1. Accounts
    num_accounts = 250
    accounts = []
    for i in range(1, num_accounts + 1):
        accounts.append({
            "ACCOUNT_ID": i,
            "CUSTOMER_ID": f"C_{i}",
            "INIT_BALANCE": round(random.uniform(10.0, 100000.0), 2),
            "COUNTRY": random.choice(["US", "UK", "CA", "DE", "FR"]),
            "ACCOUNT_TYPE": random.choice(["checking", "savings", "business"]),
            "TX_BEHAVIOR_ID": random.randint(1, 10),
            "IS_FRAUD": (10 <= i <= 15)  # nodes 10-15 are the fraud ring
        })
    df_accounts = pd.DataFrame(accounts)
    
    # 2. Transactions & Alerts
    fraud_nodes = list(range(10, 16))
    num_transactions = 800
    transactions = []
    alerts = []
    
    alert_id_counter = 1
    
    for idx in range(1, num_transactions + 1):
        is_fraud_tx = random.random() < 0.05
        
        if is_fraud_tx:
            # Force fraud ring interactions
            src = random.choice(fraud_nodes)
            dst = random.choice(fraud_nodes)
            while dst == src:
                dst = random.choice(fraud_nodes)
            amt = round(random.uniform(5000.0, 50000.0), 2)
            is_fraud_flag = 1
            cur_alert_id = alert_id_counter
            alert_id_counter += 1
            
            alerts.append({
                "ALERT_ID": cur_alert_id,
                "TX_ID": idx,
                "SENDER_ACCOUNT_ID": src,
                "RECEIVER_ACCOUNT_ID": dst,
                "TX_AMOUNT": amt,
                "TIMESTAMP": idx
            })
        else:
            # Normal interactions
            src = random.randint(1, num_accounts)
            dst = random.randint(1, num_accounts)
            while dst == src:
                dst = random.randint(1, num_accounts)
            amt = round(random.uniform(10.0, 1000.0), 2)
            is_fraud_flag = 0
            cur_alert_id = -1
            
        transactions.append({
            "TX_ID": idx,
            "SENDER_ACCOUNT_ID": src,
            "RECEIVER_ACCOUNT_ID": dst,
            "TX_AMOUNT": amt,
            "TIMESTAMP": idx,
            "IS_FRAUD": is_fraud_flag,
            "ALERT_ID": cur_alert_id,
            "TX_TYPE": random.choice(["TRANSFER", "PAYMENT", "WITHDRAWAL"])
        })
        
    df_transactions = pd.DataFrame(transactions)
    df_alerts = pd.DataFrame(alerts)
    
    # Save to data/testing
    df_accounts.to_csv(output_dir / "accounts.csv", index=False)
    df_transactions.to_csv(output_dir / "transactions.csv", index=False)
    df_alerts.to_csv(output_dir / "alerts.csv", index=False)
    
    print(f"Dummy data generated in {output_dir}")
    print(f"- Accounts: {len(df_accounts)}")
    print(f"- Transactions: {len(df_transactions)}")
    print(f"- Alerts: {len(df_alerts)}")

if __name__ == "__main__":
    generate_dummy_data()
