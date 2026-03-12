import os
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import psycopg
import pandas as pd
import joblib

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "fraud_db")
DB_USER = os.getenv("DB_USER", "fraud")
DB_PASSWORD = os.getenv("DB_PASSWORD", "fraud")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models/latest"))
MODEL_PATH = MODEL_DIR / "model.joblib"
META_PATH = MODEL_DIR / "metadata.json"

app = FastAPI(title="Fraud Scoring API", version="0.3.0")

_model = None
_model_name = "unknown"
_model_version = "local"
_threshold = 0.5
_feature_cols = None


class Transaction(BaseModel):
    transaction_id: str = Field(..., min_length=1)
    event_time: Optional[str] = None
    amount: float = Field(..., ge=0)
    currency: str = Field(default="GBP", min_length=3, max_length=3)
    merchant_category: Optional[str] = None
    country: Optional[str] = None
    features: Dict[str, Any] = Field(default_factory=dict)


class ScoreResponse(BaseModel):
    transaction_id: str
    score: float
    decision: bool
    model_name: str
    model_version: str
    threshold: float
    latency_ms: float


def _connect():
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        autocommit=True,
    )


def _load_local_model():
    global _model, _model_name, _model_version, _threshold, _feature_cols

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    if not META_PATH.exists():
        raise RuntimeError(f"Metadata file not found: {META_PATH}")

    meta = json.loads(META_PATH.read_text())
    _model_name = meta.get("model_name", "fraud_model")
    _model_version = meta.get("run_id", "local")
    _threshold = float(meta.get("threshold", 0.5))
    _feature_cols = meta.get("features")  # expected columns order
    _model = joblib.load(MODEL_PATH)

    return meta


@app.on_event("startup")
def startup_event():
    meta = _load_local_model()
    print(f"Loaded local model: {MODEL_PATH} (run_id={meta.get('run_id')}) threshold={_threshold}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": {"name": _model_name, "version": _model_version},
        "threshold": _threshold,
        "model_path": str(MODEL_PATH),
    }


@app.post("/score", response_model=ScoreResponse)
def score(tx: Transaction):
    global _model, _threshold, _feature_cols
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    t0 = time.perf_counter()

    # Event time parsing
    if tx.event_time:
        try:
            event_dt = datetime.fromisoformat(tx.event_time.replace("Z", "+00:00"))
            if event_dt.tzinfo is None:
                event_dt = event_dt.replace(tzinfo=timezone.utc)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid event_time: {e}")
    else:
        event_dt = datetime.now(timezone.utc)

    # Build feature row aligned to creditcard.csv features (Time, V1..V28, Amount)
    row = {"Time": 0.0, **{f"V{i}": 0.0 for i in range(1, 29)}, "Amount": float(tx.amount)}

    # Allow overrides for Time/V1..V28/Amount via tx.features
    for k, v in tx.features.items():
        if k in row:
            row[k] = float(v)

    # Ensure column order matches training
    if _feature_cols:
        X = pd.DataFrame([[row.get(c, 0.0) for c in _feature_cols]], columns=_feature_cols)
    else:
        X = pd.DataFrame([row])

    try:
        s = float(_model.predict_proba(X)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model predict failed: {e}")

    decision = s >= _threshold
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # Log to Postgres
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions
                      (transaction_id, event_time, model_name, model_version, score, decision, latency_ms, features)
                    VALUES
                      (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        tx.transaction_id,
                        event_dt,
                        _model_name,
                        _model_version,
                        s,
                        decision,
                        latency_ms,
                        psycopg.types.json.Jsonb(
                            {
                                "Amount": tx.amount,
                                "currency": tx.currency,
                                "merchant_category": tx.merchant_category,
                                "country": tx.country,
                                **tx.features,
                            }
                        ),
                    ),
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB insert failed: {e}")

    return ScoreResponse(
        transaction_id=tx.transaction_id,
        score=s,
        decision=decision,
        model_name=_model_name,
        model_version=_model_version,
        threshold=_threshold,
        latency_ms=latency_ms,
    )
