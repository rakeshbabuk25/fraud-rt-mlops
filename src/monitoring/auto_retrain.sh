#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/repo}"  # inside container
HOST_REPO_DIR="${HOST_REPO_DIR:-}"  # real host path for Docker Desktop mounts

echo "Checking latest metrics_runs.should_retrain ..."
LATEST="$(docker exec -i fraud_postgres psql -U fraud -d fraud_db -t -A -c "SELECT should_retrain FROM metrics_runs ORDER BY id DESC LIMIT 1;")"
LATEST="$(echo "$LATEST" | tr -d '[:space:]')"
echo "Latest should_retrain = ${LATEST:-<none>}"

if [[ "${LATEST}" != "t" && "${LATEST}" != "true" && "${LATEST}" != "True" ]]; then
  echo "No retrain needed."
  exit 0
fi

# Build docker compose command that works from inside the monitor container on macOS
# Use host path so Docker Desktop allows bind mounts.
if [[ -n "${HOST_REPO_DIR}" ]]; then
  COMPOSE=(docker compose --project-directory "${HOST_REPO_DIR}" -f "${HOST_REPO_DIR}/docker-compose.yml")
else
  # fallback: run from mounted repo (may fail on macOS due to mounts)
  COMPOSE=(docker compose)
fi

echo "Retrain triggered -> running trainer..."
"${COMPOSE[@]}" run --rm --no-deps trainer

echo "Restarting fraud_api to load latest exported model..."
"${COMPOSE[@]}" up -d --no-deps --build fraud_api

echo "Done."
