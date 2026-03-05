#!/usr/bin/env bash
set -euo pipefail

echo "Checking latest metrics_runs.should_retrain ..."
LATEST=$(docker exec -i fraud_postgres psql -U fraud -d fraud_db -t -A -c "SELECT should_retrain FROM metrics_runs ORDER BY id DESC LIMIT 1;")
LATEST=$(echo "$LATEST" | tr -d '[:space:]')

echo "Latest should_retrain = $LATEST"

if [[ "$LATEST" != "t" && "$LATEST" != "true" ]]; then
  echo "No retrain needed. Exiting."
  exit 0
fi

echo "Retrain triggered -> running trainer..."
docker compose run --rm trainer

echo "Restarting fraud_api to load latest exported model..."
docker compose up -d --build fraud_api

echo "Done."
