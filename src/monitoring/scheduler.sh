#!/usr/bin/env bash
set -euo pipefail

INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
REPO_DIR="${REPO_DIR:-/repo}"

echo "Scheduler started. Interval: ${INTERVAL_SECONDS}s"
echo "Repo: ${REPO_DIR}"
echo "Will run: write_metrics.py -> auto_retrain.sh"

while true; do
  echo "-----"
  date -u +"%Y-%m-%dT%H:%M:%SZ"

  python3 "${REPO_DIR}/src/monitoring/write_metrics.py" || echo "[WARN] write_metrics.py failed"
  "${REPO_DIR}/src/monitoring/auto_retrain.sh" || echo "[WARN] auto_retrain.sh failed"

  echo "Sleeping ${INTERVAL_SECONDS}s..."
  sleep "${INTERVAL_SECONDS}"
done
