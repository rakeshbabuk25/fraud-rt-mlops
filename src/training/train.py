import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score, precision_score, recall_score
import lightgbm as lgb
import mlflow
import mlflow.lightgbm


DATA_PATH = os.getenv("DATA_PATH", "data/creditcard.csv")  # you will download this next
TARGET_COL = os.getenv("TARGET_COL", "Class")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "fraud-training")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud_lgbm")


def best_threshold(y_true, y_score):
    precision, recall, thr = precision_recall_curve(y_true, y_score)
    # thr has len = len(precision)-1
    f1s = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    i = int(np.nanargmax(f1s))
    return float(thr[i]), float(f1s[i])


def main():
    if not Path(DATA_PATH).exists():
        raise SystemExit(f"Missing dataset at {DATA_PATH}. Put creditcard.csv there (Step 6.2).")

    df = pd.read_csv(DATA_PATH)
    if TARGET_COL not in df.columns:
        raise SystemExit(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)[:10]}...")

    y = df[TARGET_COL].astype(int).values
    X = df.drop(columns=[TARGET_COL])

    # Simple split (we'll improve to time-aware later if dataset has time column)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Imbalance handling
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

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)

        val_scores = model.predict_proba(X_val)[:, 1]
        auprc = average_precision_score(y_val, val_scores)
        thr, best_f1 = best_threshold(y_val, val_scores)

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

        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        print("RUN_ID:", run.info.run_id)
        print("AUPRC:", auprc)
        print("THRESHOLD:", thr)
        print("F1:", best_f1)


if __name__ == "__main__":
    main()
