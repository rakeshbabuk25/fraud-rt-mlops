CREATE TABLE IF NOT EXISTS predictions (
  id SERIAL PRIMARY KEY,
  transaction_id TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL,
  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  score DOUBLE PRECISION NOT NULL,
  decision BOOLEAN NOT NULL,
  latency_ms DOUBLE PRECISION,
  features JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_txid ON predictions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_predictions_event_time ON predictions(event_time);

CREATE TABLE IF NOT EXISTS outcomes (
  id SERIAL PRIMARY KEY,
  transaction_id TEXT NOT NULL UNIQUE,
  label BOOLEAN NOT NULL,
  label_time TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_outcomes_label_time ON outcomes(label_time);

CREATE TABLE IF NOT EXISTS metrics_runs (
  id SERIAL PRIMARY KEY,
  run_time TIMESTAMPTZ NOT NULL,
  window_start TIMESTAMPTZ NOT NULL,
  window_end TIMESTAMPTZ NOT NULL,
  model_name TEXT NOT NULL,
  model_version TEXT NOT NULL,
  auprc DOUBLE PRECISION,
  f1 DOUBLE PRECISION,
  precision DOUBLE PRECISION,
  recall DOUBLE PRECISION,
  latency_p95_ms DOUBLE PRECISION,
  drift_score DOUBLE PRECISION,
  should_retrain BOOLEAN NOT NULL DEFAULT FALSE,
  reason TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_metrics_runs_run_time ON metrics_runs(run_time);
