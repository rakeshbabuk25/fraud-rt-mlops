Your error is because you never close the Mermaid code block. Everything after TR -->|restart| API is still inside the ```mermaid block, and Mermaid can’t parse ---, bullets, or normal prose.

Also: you’re mixing tab bullets (\t•) and a weird divider (⸻) which is fine in Markdown, but must be outside Mermaid.

Below is a fixed README (same content, but properly structured). Replace your README with this.

# Fraud RT MLOps — Real-Time Fraud Detection with Drift-Aware Continuous Training

Production-style, end-to-end **real-time fraud detection** system demonstrating **streaming inference**, **audit-grade logging**, **monitoring + drift checks**, and **automated retraining + redeploy** using an MLOps workflow.

This project is designed to be demoed live (to a tutor/professor) and to read like a professional engineering repo.

---

## What this system does

- **Streams transactions** into a Kafka-compatible broker (Redpanda)
- **Consumes** the stream and calls a **low-latency scoring API** (FastAPI)
- **Logs predictions** + input features + model version to **Postgres** for traceability
- **Logs outcomes/labels** (ground truth) to Postgres and computes online metrics
- Runs periodic **KPI monitoring** (AUPRC, recall) + **drift detection** (PSI)
- If quality or drift breaches thresholds, it **triggers retraining** (LightGBM), exports a new model, and **restarts the API** to serve the new version

---

## Tech stack (tools & technologies)

**Languages**
- Python 3 (training, API, streaming, monitoring)
- SQL (metrics + joins over predictions/outcomes)

**Core components**
- **FastAPI** — scoring service (`/score`, `/health`)
- **Redpanda** — Kafka-compatible streaming broker
- **PostgreSQL** — prediction + outcome store, metrics runs store
- **MLflow** — experiment tracking + model registry (optional but included)
- **LightGBM** — supervised fraud classifier
- **Docker / Docker Compose** — reproducible local environment

**Monitoring & automation**
- KPI computation + drift checks (`src/monitoring/write_metrics.py`)
- Auto-retrain orchestration via a host scheduler (launchd on macOS)

---

## System architecture

```mermaid
flowchart LR
  P[Producer<br/>synthetic tx stream] --> RP[Redpanda<br/>Kafka topic: transactions]
  RP --> C[Consumer<br/>stream worker]
  C -->|POST /score| API[FastAPI Scoring API<br/>fraud_api]
  API --> PG[(Postgres<br/>predictions, outcomes, metrics_runs)]

  TR[Trainer<br/>LightGBM + MLflow] -->|exports /models/latest| API
  MON[Monitoring Job<br/>KPI + PSI drift] --> PG
  MON -->|trigger retrain| TR
  TR -->|restart| API


⸻

Data & ML approach

Dataset
	•	Training uses the credit card fraud dataset (data/creditcard.csv) with features:
	•	Time, V1..V28, Amount
	•	Target label: Class (fraud = 1, normal = 0)

Model
	•	LightGBM (gradient-boosted decision trees) for binary classification.
	•	Handles class imbalance using: scale_pos_weight = neg/pos

Thresholding
	•	Model outputs P(fraud); decision is produced via a configurable threshold.
	•	Training selects a threshold from the validation PR curve, with guardrails:
	•	MIN_THRESHOLD lower bound
	•	optional FORCE_THRESHOLD override (stable demos)

Why these metrics
	•	AUPRC (Average Precision) is preferred over AUROC for imbalanced fraud detection.
	•	Recall is critical: missing fraud is usually more costly than reviewing a false positive.

⸻

Monitoring & drift detection

KPI checks

Monitoring computes metrics over a rolling window by joining:
	•	predictions ↔ outcomes on transaction_id

Computed:
	•	AUPRC (Average Precision)
	•	Precision / Recall / F1

Trigger logic (configurable via env vars):
	•	AUPRC < MIN_AUPRC
	•	RECALL < MIN_RECALL

Drift checks (PSI)
	•	PSI (Population Stability Index) computed for Amount distribution:
	•	reference: last REF_DAYS
	•	current: last CUR_HOURS
	•	If PSI > MAX_PSI, model retraining can be triggered.

Guardrails (important for real-world pipelines)

To avoid noisy retraining decisions on tiny label volume:
	•	Monitoring enforces minimum joined label counts (e.g., MIN_JOINED, MIN_POS_LABELS)
	•	If insufficient labels exist, the run is logged as:
	•	INSUFFICIENT_LABELS(joined=...,pos=...)
	•	Retraining is skipped for that run.

⸻

API endpoints
	•	Health: GET /health
Returns model name/version, threshold, and model path.
	•	Score: POST /score
Accepts a transaction payload and returns score, decision, model_version, latency.

⸻

Repository layout

.
├── docker-compose.yml
├── src/
│   ├── streaming/
│   │   ├── producer.py        # sends synthetic transactions
│   │   └── consumer.py        # reads from Redpanda, calls /score
│   ├── monitoring/
│   │   └── write_metrics.py   # KPIs + PSI drift + retrain decision
│   └── training/
│       └── train.py           # LightGBM training + export artifacts
├── models/latest/             # served artifacts (model.joblib, metadata.json)
├── logs/                      # monitor.out / monitor.err (host scheduler logs)
└── data/                      # training dataset (not pushed if large/private)


⸻

Quickstart (demo flow)

You’ll usually use two terminals:
	•	Terminal A (long-running): consumer tail/logs
	•	Terminal B (commands): compose, producer, queries

1) Start the stack (Terminal B)

cd ~/Projects/fraud-rt-mlops
docker compose up -d
curl -s http://localhost:8000/health | python3 -m json.tool

2) Start the consumer (Terminal A)

docker exec -it fraud_stream_worker sh -lc '
rm -f /work/consumer.log /work/consumer.pid
: > /work/consumer.log
nohup env PYTHONUNBUFFERED=1 KAFKA_BROKER=redpanda:9092 SCORING_URL=http://fraud_api:8000/score \
  python -u /work/consumer.py >> /work/consumer.log 2>&1 &
echo $! > /work/consumer.pid
tail -f /work/consumer.log
'

3) Send transactions (Terminal B)

docker exec -it fraud_stream_worker sh -lc "KAFKA_BROKER=redpanda:9092 python /work/producer.py"

Confirm predictions are landing in Postgres:

docker exec -it fraud_postgres psql -U fraud -d fraud_db -c \
"SELECT COUNT(*) AS n_tx_last2m, MAX(created_at) AS last_time
 FROM predictions
 WHERE transaction_id LIKE 'tx_%'
   AND created_at > now() - interval '2 minutes';"

4) Run monitoring once (manual) (Terminal B)

. .venv/bin/activate
python3 src/monitoring/write_metrics.py

Check last monitoring decisions:

docker exec -it fraud_postgres psql -U fraud -d fraud_db -c \
"SELECT id, run_time, model_version, should_retrain, reason
 FROM metrics_runs
 ORDER BY id DESC
 LIMIT 10;"


⸻

Automated monitoring & retraining (macOS launchd)

A host-scheduled job periodically:
	1.	Runs write_metrics.py
	2.	If should_retrain = true, runs the trainer container
	3.	Restarts fraud_api to load the latest model artifacts

Logs:
	•	logs/monitor.out
	•	logs/monitor.err

⸻

How to present
	1.	Problem framing
Fraud detection is streaming + imbalanced + drifting. A model is only useful if:

	•	it can score in real time
	•	it is monitored continuously
	•	it can be refreshed safely when performance degrades

	2.	Pipeline
Show the architecture diagram and walk through:

	•	Redpanda topic → consumer → FastAPI /score
	•	predictions logged with model_version
	•	outcomes enable supervised KPI computation
	•	monitoring produces a decision
	•	retraining exports a new model and API reloads it

	3.	Model & evaluation

	•	LightGBM works well for tabular data
	•	AUPRC and recall fit class imbalance
	•	threshold controls fraud review rate

	4.	Drift

	•	PSI provides an early warning signal even when labels arrive late

	5.	Operational safety

	•	minimum label guardrails avoid retraining on noise
	•	model versioning ensures auditability

⸻

Notes / known limitations (honest engineering)
	•	Synthetic streaming data is for demonstration; production would use real feature pipelines.
	•	Label availability is a real-world bottleneck; guardrails are implemented to avoid noisy triggers.
	•	MLflow is included for experiment tracking; serving uses locally exported artifacts for simplicity.

⸻

License

MIT (or replace with your preferred license).

### What I changed (so GitHub stops erroring)
- Added the missing closing fence: **```** right after the Mermaid diagram.
- Moved `---` and all prose **outside** Mermaid.
- Converted the weird separators (`⸻`) + tab bullets into normal Markdown headings and lists.
- Wrapped command snippets in proper `bash` code blocks.

If you want, paste your current `README.md` file path and I’ll give you a single `cat > README.md <<'EOF' ... EOF` command to overwrite it cleanly in one shot.
