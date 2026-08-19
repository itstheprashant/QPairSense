from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ml.preprocessing import validate_dataset, build_engineered_features, build_clean_texts
from app.services.feature_engineering import FEATURE_NAMES
from app.core.config import BASE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("qpair-train")

def parse_args():
    parser = argparse.ArgumentParser(description="Train QPairSense.")
    parser.add_argument("--data", default=str(BASE_DIR / "data" / "train.csv"))
    parser.add_argument("--sample-size", type=int, default=0,
                        help="0 = use all rows; useful for quick experiments.")
    parser.add_argument("--max-features", type=int, default=3000)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=1000)
    return parser.parse_args()

def main():
    args = parse_args()
    start = time.time()

    logger.info("Reading dataset: %s", args.data)
    df = pd.read_csv(args.data)
    logger.info("Raw shape: %s", df.shape)

    df = validate_dataset(df)
    logger.info("Validated shape: %s", df.shape)

    if args.sample_size > 0 and args.sample_size < len(df):
        df = df.sample(args.sample_size, random_state=args.random_state).reset_index(drop=True)
        logger.info("Using sample: %s", df.shape)

    y = df["is_duplicate"].astype(np.int8).to_numpy()

    logger.info("Building engineered features...")
    engineered = build_engineered_features(df, workers=args.workers)

    logger.info("Cleaning questions...")
    q1, q2 = build_clean_texts(df)

    # Fit vocabulary only on the training split to avoid leakage.
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    train_texts = q1 + q2
    vectorizer = CountVectorizer(
        max_features=args.max_features,
        min_df=2,
        binary=True,
        dtype=np.float32,
    )

    # Fit on training questions only.
    vectorizer.fit([q1[i] for i in train_idx] + [q2[i] for i in train_idx])

    q1_train_bow = vectorizer.transform([q1[i] for i in train_idx])
    q2_train_bow = vectorizer.transform([q2[i] for i in train_idx])
    q1_test_bow = vectorizer.transform([q1[i] for i in test_idx])
    q2_test_bow = vectorizer.transform([q2[i] for i in test_idx])

    X_train = hstack([
        csr_matrix(engineered[train_idx]),
        q1_train_bow,
        q2_train_bow,
    ], format="csr")

    X_test = hstack([
        csr_matrix(engineered[test_idx]),
        q1_test_bow,
        q2_test_bow,
    ], format="csr")

    y_train = y[train_idx]
    y_test = y[test_idx]

    logger.info("Training Logistic Regression on sparse features: %s", X_train.shape)

    model = LogisticRegression(
        max_iter=args.max_iter,
        class_weight="balanced",
        solver="liblinear",
        random_state=args.random_state,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(np.int8)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }

    logger.info("Metrics: %s", json.dumps({
        k: v for k, v in metrics.items()
        if k != "classification_report"
    }, indent=2))

    model_dir = BASE_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "model.joblib", compress=3)
    joblib.dump(vectorizer, model_dir / "vectorizer.joblib", compress=3)

    metadata = {
        "model_name": "QPairSense Logistic Regression",
        "model_version": "1.0.0",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": "LogisticRegression",
        "dataset": str(Path(args.data).name),
        "rows_used": int(len(df)),
        "positive_rate": float(y.mean()),
        "engineered_feature_count": len(FEATURE_NAMES),
        "engineered_features": FEATURE_NAMES,
        "bow_max_features": args.max_features,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "metrics": metrics,
    }

    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=float),
        encoding="utf-8"
    )

    logger.info("Artifacts written to %s", model_dir)
    logger.info("Training completed in %.2fs", time.time() - start)

if __name__ == "__main__":
    main()
