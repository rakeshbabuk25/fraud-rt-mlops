import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score, precision_score, recall_score
import lightgbm as lgb
import joblib
import mlflow
import mlflow.lightgbm

DATA_PATH = os.getenv("DATA_PATH", "data/creditcard.csv")
TARGET_COL = os.getenv("TARGET_COL", "Class")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "fraud-training")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud_lgbm")

MIN_THRESHOLD = float(os.getenv("MIN_THRESHOLD", "0.01"))

EXPORT_DIR = Path(os.getenv("EXPORT_DIR", "/export/latest"))
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def best_threshold(y_true, y_score):
    precision, recall, thr = precision_recall_curve(y_true, y_score)
    f1s = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    i = int(np.nanargmax(f1s))
    return float(thr[i]), float(f1s[i])


def main():
    if not Path(DATA_PATH).exists():
        raise SystemExit(f"Missing dataset at {DATA_PATH}.")

    df = pd.read_csv(DATA_PATH)
    y = df[TARGET_COL].astype(int).values
    X = df.drop(columns=[TARGET_COL])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = (neg / max(pos, 1))

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        val_scores = model.predict_proba(X_val)[:, 1]

        auprc = average_precision_score(y_val, val_scores)
        thr, best_f1 = best_threshold(y_val, val_scores)
        thr = max(float(thr), MIN_THRESHOLD)

        FORCE_THRESHOLD = os.getenv("FORCE_THRESHOLD", "").strip()
        if FORCE_THRESHOLD:
            thr = float(FORCE_THRESHOLD)

        y_pred = (val_scores >= thr).astype(int)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        mlflow.log_param("scale_pos_weight", float(scale_pos_weight))
        mlflow.log_param("threshold", float(thr))
        mlflow.log_metric("auprc", float(auprc))
        mlflow.log_metric("precision", float(prec))
        mlflow.log_metric("recall", float(rec))
        mlflow.log_metric("f1", float(f1))

        # Keep registry logging (optional), but not required for serving now
        try:
            mlflow.lightgbm.log_model(
                lgb_model=model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
            )
        except Exception as e:
            print("MLflow model log warning:", e)

        # Export local artifacts for serving
        model_path = EXPORT_DIR / "model.joblib"
        meta_path = EXPORT_DIR / "metadata.json"

        joblib.dump(model, model_path)
        meta = {
            "model_name": MODEL_NAME,
            "run_id": run.info.run_id,
            "threshold": thr,
            "auprc": float(auprc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "features": list(X.columns),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        print("RUN_ID:", run.info.run_id)
        print("AUPRC:", auprc)
        print("THRESHOLD:", thr)
        print("F1:", best_f1)
        print("EXPORTED_MODEL:", str(model_path))
        print("EXPORTED_META:", str(meta_path))


if __name__ == "__main__":
    main()
