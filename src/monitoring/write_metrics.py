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

# Guardrails / triggers (tune later)
MIN_AUPRC = float(os.getenv("MIN_AUPRC", "0.20"))
MIN_RECALL = float(os.getenv("MIN_RECALL", "0.80"))

def main():
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=WINDOW_HOURS)

    with psycopg.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD) as conn:
        q = """
        SELECT p.transaction_id, p.score, p.decision, p.model_name, p.model_version, p.event_time,
               o.label
        FROM predictions p
        JOIN outcomes o ON o.transaction_id = p.transaction_id
        WHERE p.event_time >= %s
        ORDER BY p.event_time DESC
        """
        df = pd.read_sql(q, conn, params=(window_start,))

        if df.empty:
            print("No joined rows; nothing to write.")
            return

        y_true = df["label"].astype(int).values
        y_score = df["score"].astype(float).values
        y_pred = df["decision"].astype(int).values

        auprc = average_precision_score(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Choose latest model seen in this window (by event_time)
        latest = df.sort_values("event_time").iloc[-1]
        model_name = str(latest["model_name"])
        model_version = str(latest["model_version"])

        should_retrain = False
        reasons = []
        if auprc == auprc and auprc < MIN_AUPRC:  # auprc==auprc checks not-NaN
            should_retrain = True
            reasons.append(f"AUPRC<{MIN_AUPRC}")
        if rec < MIN_RECALL:
            should_retrain = True
            reasons.append(f"RECALL<{MIN_RECALL}")

        reason = ",".join(reasons) if reasons else None

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO metrics_runs
                (run_time, window_start, window_end, model_name, model_version,
                 auprc, f1, precision, recall, latency_p95_ms, drift_score, should_retrain, reason)
                VALUES
                (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    now, window_start, now, model_name, model_version,
                    float(auprc) if auprc == auprc else None,
                    float(f1), float(prec), float(rec),
                    None, None, should_retrain, reason
                ),
            )

    print("WROTE metrics_runs row")
    print("MODEL:", model_name, model_version)
    print("AUPRC:", auprc, "PREC:", prec, "RECALL:", rec, "F1:", f1)
    print("SHOULD_RETRAIN:", should_retrain, "REASON:", reason)

if __name__ == "__main__":
    main()
