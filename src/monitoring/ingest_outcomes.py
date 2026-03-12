import os
import random
from datetime import datetime, timezone
import psycopg

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "fraud_db")
DB_USER = os.getenv("DB_USER", "fraud")
DB_PASSWORD = os.getenv("DB_PASSWORD", "fraud")

def main():
    # Simulate delayed labels for a few existing tx ids
    tx_ids = [f"tx_{i:06d}" for i in range(1, 51)] + ["tx_live_0001"]
    sample = random.sample(tx_ids, k=min(10, len(tx_ids)))

    with psycopg.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, autocommit=True) as conn:
        with conn.cursor() as cur:
            for txid in sample:
                # Simulate rare fraud labels (~2%)
                label = random.random() < 0.02
                cur.execute(
                    """
                    INSERT INTO outcomes (transaction_id, label, label_time)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (transaction_id) DO NOTHING
                    """,
                    (txid, label, datetime.now(timezone.utc)),
                )
                print("labeled", txid, "=", label)

if __name__ == "__main__":
    main()
