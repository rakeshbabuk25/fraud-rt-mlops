# Real-Time Fraud Detection (Streaming + MLOps)

An end-to-end, production-style fraud detection system:
- Transaction ingestion as a stream
- Real-time scoring behind a low-latency API
- Logging predictions + later outcomes
- Drift + KPI monitoring
- Automated retraining + redeployment when thresholds are breached

## Planned Components
- **Streaming:** Redpanda/Kafka topic for transactions
- **Serving:** FastAPI scoring service (`POST /score`)
- **Storage:** Postgres for predictions, outcomes, and metrics
- **Modeling:** Imbalance-aware model (class weights / scale_pos_weight)
- **Registry:** MLflow for model versioning
- **Monitoring:** Drift + performance checks (AUPRC, precision/recall, latency)

## Repo Structure (high-level)
- `src/` application code (training, serving, streaming, monitoring)
- `infra/` infrastructure config (SQL init scripts, docker assets)
- `tests/` automated tests
- `data/` small samples only (no raw datasets committed)
- `notebooks/` exploration only

## Quick Start
Coming next:
1. `docker compose up` (Postgres + MLflow + streaming + API)
2. Train & register a model
3. Stream transactions and score live
4. Run monitoring + auto retrain trigger
