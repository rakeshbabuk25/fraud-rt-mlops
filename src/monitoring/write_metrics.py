import os
from datetime import datetime, timezone, timedelta
import math

import pandas as pd
import psycopg

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "fraud_db")
DB_USER = os.getenv("DB_USER", "fraud")
DB_PASSWORD = os.getenv("DB_PASSWORD", "fraud")

WINDOW_HOURS = int(os.getenv("WINDOW_HOURS", "24"))

# Minimum evidence before we allow retrain decisions
MIN_JOINED_ROWS = int(os.getenv("MIN_JOINED_ROWS", "200"))
MIN_POS_LABELS = int(os.getenv("MIN_POS_LABELS", "5"))
MIN_REF_SAMPLES = int(os.getenv("MIN_REF_SAMPLES", "200"))
MIN_CUR_SAMPLES = int(os.getenv("MIN_CUR_SAMPLES", "200"))

# KPI triggers
MIN_AUPRC = float(os.getenv("MIN_AUPRC", "0.20"))
MIN_RECALL = float(os.getenv("MIN_RECALL", "0.80"))

# Drift (PSI on Amount)
REF_DAYS = int(os.getenv("REF_DAYS", "7"))
CUR_HOURS = int(os.getenv("CUR_HOURS", "6"))
PSI_BINS = int(os.getenv("PSI_BINS", "10"))
PSI_EPS = float(os.getenv("PSI_EPS", "1e-6"))
MAX_PSI = float(os.getenv("MAX_PSI", "0.25"))

def psi(expected, actual, eps=1e-6):
    total = 0.0
    for e, a in zip(expected, actual):
        e = max(float(e), eps)
        a = max(float(a), eps)
        total += (a - e) * math.log(a / e)
    return total

def hist_probs(values, bin_edges):
    counts = pd.cut(values, bins=bin_edges, include_lowest=True).value_counts(sort=False).values
    probs = counts / max(counts.sum(), 1)
    return probs

def compute_amount_psi(conn, now):
    cur_start = now - timedelta(hours=CUR_HOURS)
    ref_end = cur_start
    ref_start = ref_end - timedelta(days=REF_DAYS)

    q = """
    SELECT event_time,
           (features->>'Amount')::double precision AS amount
    FROM predictions
    WHERE event_time >= %s AND event_time < %s
      AND (features ? 'Amount')
    """

    ref = pd.read_sql(q, conn, params=(ref_start, ref_end))
    cur = pd.read_sql(q, conn, params=(cur_start, now))

    if ref.empty or cur.empty:
        return None

    ref_amount = ref["amount"].dropna()
    cur_amount = cur["amount"].dropna()
    if ref_amount.empty or cur_amount.empty:
        return None

    # Guardrail: PSI is unstable on tiny samples
    if len(ref_amount) < MIN_REF_SAMPLES or len(cur_amount) < MIN_CUR_SAMPLES:
        return None

    qs = [i / PSI_BINS for i in range(PSI_BINS + 1)]
    edges = ref_amount.quantile(qs).values.tolist()
    edges[0] = min(edges[0], ref_amount.min())
    edges[-1] = max(edges[-1], ref_amount.max())

    deduped = [edges[0]]
    for x in edges[1:]:
        if x <= deduped[-1]:
            x = deduped[-1] + 1e-9
        deduped.append(x)

    ref_p = hist_probs(ref_amount, deduped)
    cur_p = hist_probs(cur_amount, deduped)
    return float(psi(ref_p, cur_p, eps=PSI_EPS))

def average_precision(y_true, y_score):
    # Manual AUPRC (Average Precision) implementation
    df = pd.DataFrame({"y": y_true, "s": y_score}).sort_values("s", ascending=False)
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    pos = int(df["y"].sum())
    if pos == 0:
        return float("nan")
    for _, r in df.iterrows():
        if int(r["y"]) == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / pos)
    # AP = sum over recall increments * precision
    ap = 0.0
    prev_recall = 0.0
    for p, rec in zip(precisions, recalls):
        ap += p * (rec - prev_recall)
        prev_recall = rec
    return ap

def prf(y_true, y_pred):
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
    return precision, recall, f1

def main():
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=WINDOW_HOURS)

    with psycopg.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD) as conn:
        q = """
        WITH latest AS (
            SELECT model_name, model_version
            FROM predictions
            WHERE event_time >= %s
            ORDER BY event_time DESC
            LIMIT 1
        )
        SELECT p.transaction_id, p.score, p.decision, p.model_name, p.model_version, p.event_time,
               o.label
        FROM predictions p
        JOIN outcomes o ON o.transaction_id = p.transaction_id
        JOIN latest l ON l.model_name = p.model_name AND l.model_version = p.model_version
        WHERE p.event_time >= %s
        ORDER BY p.event_time DESC
        """
        df = pd.read_sql(q, conn, params=(window_start, window_start))

        if df.empty:
            print("No joined rows; nothing to write.")
            return

        # Guardrail: do not make retrain decisions without enough labeled data
        n_joined = int(len(df))
        n_pos = int(df['label'].astype(int).sum())
        if n_joined < MIN_JOINED_ROWS or n_pos < MIN_POS_LABELS:
            latest = df.sort_values('event_time').iloc[-1]
            model_name = str(latest['model_name'])
            model_version = str(latest['model_version'])
            reason = f"INSUFFICIENT_LABELS(joined={n_joined},pos={n_pos})"
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO metrics_runs
                    (run_time, window_start, window_end, model_name, model_version,
                     auprc, f1, precision, recall, latency_p95_ms, drift_score, should_retrain, reason)
                    VALUES
                    (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (now, window_start, now, model_name, model_version,
                     None, None, None, None, None,
                     None, False, reason)
                )
            print("WROTE metrics_runs row")
            print("MODEL:", model_name, model_version)
            print("SHOULD_RETRAIN:", False, "REASON:", reason)
            return

        y_true = df["label"].astype(int).tolist()
        y_score = df["score"].astype(float).tolist()
        y_pred = df["decision"].astype(int).tolist()

        auprc = average_precision(y_true, y_score) if len(set(y_true)) > 1 else float("nan")
        prec, rec, f1 = prf(y_true, y_pred)

        latest = df.sort_values("event_time").iloc[-1]
        model_name = str(latest["model_name"])
        model_version = str(latest["model_version"])

        drift_score = compute_amount_psi(conn, now)

        should_retrain = False
        reasons = []
        if auprc == auprc and auprc < MIN_AUPRC:
            should_retrain = True
            reasons.append(f"AUPRC<{MIN_AUPRC}")
        if rec < MIN_RECALL:
            should_retrain = True
            reasons.append(f"RECALL<{MIN_RECALL}")
        if drift_score is not None and drift_score > MAX_PSI:
            should_retrain = True
            reasons.append(f"PSI>{MAX_PSI}")

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
                    None,
                    drift_score,
                    should_retrain,
                    reason
                ),
            )

    print("WROTE metrics_runs row")
    print("MODEL:", model_name, model_version)
    print("AUPRC:", auprc, "PREC:", prec, "RECALL:", rec, "F1:", f1)
    print("DRIFT_PSI_AMOUNT:", drift_score)
    print("SHOULD_RETRAIN:", should_retrain, "REASON:", reason)

if __name__ == "__main__":
    main()
