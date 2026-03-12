#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# 1) compute + store metrics (includes PSI drift)
python3 src/monitoring/write_metrics.py

# 2) check latest decision
LATEST="$(docker exec -i fraud_postgres psql -U fraud -d fraud_db -t -A -c "SELECT should_retrain FROM metrics_runs ORDER BY id DESC LIMIT 1;")"
LATEST="$(echo "$LATEST" | tr -d '[:space:]')"

if [[ "${LATEST}" == "t" || "${LATEST}" == "true" || "${LATEST}" == "True" ]]; then
  echo "[host] Retrain triggered"
  docker compose run --rm trainer
  docker compose up -d --build fraud_api
else
  echo "[host] No retrain needed"
fi
