import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import psycopg
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "fraud_db")
DB_USER = os.getenv("DB_USER", "fraud")
DB_PASSWORD = os.getenv("DB_PASSWORD", "fraud")

WINDOW_HOURS = int(os.getenv("WINDOW_HOURS", "24"))

def main():
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=WINDOW_HOURS)

    with psycopg.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD) as conn:
        q = """
        SELECT p.transaction_id, p.score, p.decision, p.model_name, p.model_version, p.event_time,
               o.label, o.label_time
        FROM predictions p
        JOIN outcomes o ON o.transaction_id = p.transaction_id
        WHERE p.event_time >= %s
        ORDER BY p.event_time DESC
        """
        df = pd.read_sql(q, conn, params=(window_start,))

    if df.empty:
        print("No joined prediction/outcome rows in window.")
        return

    y_true = df["label"].astype(int).values
    y_score = df["score"].astype(float).values
    y_pred = df["decision"].astype(int).values

    auprc = average_precision_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("WINDOW_START:", window_start.isoformat())
    print("N_JOINED:", len(df))
    print("AUPRC:", auprc)
    print("PRECISION:", prec)
    print("RECALL:", rec)
    print("F1:", f1)
    print("BY_MODEL_VERSION:")
    print(df.groupby(["model_name","model_version"])["transaction_id"].count().to_string())

if __name__ == "__main__":
    main()
