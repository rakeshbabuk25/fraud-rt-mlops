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

# Windows
REF_DAYS = int(os.getenv("REF_DAYS", "7"))          # reference window (older)
CUR_HOURS = int(os.getenv("CUR_HOURS", "6"))        # current window (recent)

# PSI settings
N_BINS = int(os.getenv("PSI_BINS", "10"))
EPS = float(os.getenv("PSI_EPS", "1e-6"))

def psi(expected, actual, eps=1e-6):
    """
    Population Stability Index:
      sum( (a - e) * ln(a/e) )
    expected/actual are probability vectors over bins.
    """
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

def main():
    now = datetime.now(timezone.utc)
    cur_start = now - timedelta(hours=CUR_HOURS)
    ref_end = cur_start
    ref_start = ref_end - timedelta(days=REF_DAYS)

    with psycopg.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD) as conn:
        # Pull Amount from JSONB features for both windows
        q = """
        SELECT event_time,
               (features->>'Amount')::double precision AS amount,
               model_name,
               model_version
        FROM predictions
        WHERE event_time >= %s AND event_time < %s
          AND (features ? 'Amount')
        ORDER BY event_time DESC
        """
        ref = pd.read_sql(q, conn, params=(ref_start, ref_end))
        cur = pd.read_sql(q, conn, params=(cur_start, now))

    if ref.empty or cur.empty:
        print("Not enough data for PSI.")
        print("REF_ROWS:", len(ref), "CUR_ROWS:", len(cur))
        return

    # Build bins from reference distribution (quantiles)
    ref_amount = ref["amount"].dropna()
    cur_amount = cur["amount"].dropna()

    # Quantile bin edges; ensure uniqueness
    qs = [i / N_BINS for i in range(N_BINS + 1)]
    edges = ref_amount.quantile(qs).values.tolist()
    edges[0] = min(edges[0], ref_amount.min())
    edges[-1] = max(edges[-1], ref_amount.max())
    # Make edges strictly increasing (dedupe)
    deduped = [edges[0]]
    for x in edges[1:]:
        if x <= deduped[-1]:
            x = deduped[-1] + 1e-9
        deduped.append(x)
    bin_edges = deduped

    ref_p = hist_probs(ref_amount, bin_edges)
    cur_p = hist_probs(cur_amount, bin_edges)
    score = psi(ref_p, cur_p, eps=EPS)

    # Choose latest model in current window for reporting
    latest = cur.sort_values("event_time").iloc[-1]
    model_name = str(latest["model_name"])
    model_version = str(latest["model_version"])

    print("REF_WINDOW:", ref_start.isoformat(), "->", ref_end.isoformat(), "rows=", len(ref))
    print("CUR_WINDOW:", cur_start.isoformat(), "->", now.isoformat(), "rows=", len(cur))
    print("MODEL:", model_name, model_version)
    print("PSI_AMOUNT:", score)

if __name__ == "__main__":
    main()
