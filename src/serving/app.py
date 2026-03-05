import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import psycopg

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "fraud_db")
DB_USER = os.getenv("DB_USER", "fraud")
DB_PASSWORD = os.getenv("DB_PASSWORD", "fraud")

MODEL_NAME = os.getenv("MODEL_NAME", "baseline_rule")
MODEL_VERSION = os.getenv("MODEL_VERSION", "0")
THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))

app = FastAPI(title="Fraud Scoring API", version="0.1.0")


class Transaction(BaseModel):
    transaction_id: str = Field(..., min_length=1)
    event_time: Optional[str] = None  # ISO8601; if missing we use now()
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


def baseline_score(tx: Transaction) -> float:
    """
    Simple, deterministic baseline so the full online pipeline works.
    You will replace this with a trained model in a later step.
    """
    score = 0.05
    if tx.amount >= 500:
        score += 0.35
    if tx.amount >= 1500:
        score += 0.30
    if tx.country and tx.country.upper() not in ("GB", "UK"):
        score += 0.15
    if tx.merchant_category and tx.merchant_category.lower() in ("crypto", "gambling"):
        score += 0.20
    # clamp
    return max(0.0, min(1.0, score))


@app.get("/health")
def health():
    return {"status": "ok", "model": {"name": MODEL_NAME, "version": MODEL_VERSION}}


@app.post("/score", response_model=ScoreResponse)
def score(tx: Transaction):
    t0 = time.perf_counter()

    # Event time handling
    if tx.event_time:
        try:
            event_dt = datetime.fromisoformat(tx.event_time.replace("Z", "+00:00"))
            if event_dt.tzinfo is None:
                event_dt = event_dt.replace(tzinfo=timezone.utc)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid event_time: {e}")
    else:
        event_dt = datetime.now(timezone.utc)

    s = float(baseline_score(tx))
    decision = s >= THRESHOLD
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
                        MODEL_NAME,
                        MODEL_VERSION,
                        s,
                        decision,
                        latency_ms,
                        psycopg.types.json.Jsonb(
                            {
                                "amount": tx.amount,
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
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        latency_ms=latency_ms,
    )
