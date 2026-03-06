This repo is an end-to-end fraud detection pipeline:
- Streaming ingestion (Redpanda)
- Low-latency scoring API (FastAPI)
- Prediction + outcome logging (Postgres)
- Offline training (LightGBM) + model export
- KPI monitoring + auto retrain trigger + redeploy

## API
- Health: http://localhost:8000/health
- MLflow: http://localhost:5001

## Quick start (which terminal?)
- Terminal 2: docker compose up -d
- Terminal 2: curl http://localhost:8000/health
- Terminal 1: run consumer (keeps running)
- Terminal 2: run producer

## Architecture (Mermaid)
```mermaid
flowchart LR
  Producer --> Redpanda --> Consumer -->|POST /score| API --> Postgres
  Trainer -->|exports models/latest| API
  Monitoring --> Postgres
  AutoRetrain --> Trainer
  AutoRetrain --> API