from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from app.core.config import settings
from app.services.feature_engineering import pair_features, clean_pair

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self):
        model_dir = Path(settings.model_dir)
        self.model_path = model_dir / settings.model_file
        self.vectorizer_path = model_dir / settings.vectorizer_file
        self.metadata_path = model_dir / settings.metadata_file

        if not self.model_path.exists() or not self.vectorizer_path.exists():
            raise FileNotFoundError(
                "Model artifacts not found. Run: python -m ml.train"
            )

        self.model = joblib.load(self.model_path)
        self.vectorizer = joblib.load(self.vectorizer_path)

        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        else:
            self.metadata = {"model_version": "unknown"}

        logger.info("Loaded QPairSense model from %s", model_dir)

    def predict(self, question1: str, question2: str) -> dict:
        q1, q2 = clean_pair(question1, question2)

        dense_features = np.asarray(
            pair_features(question1, question2),
            dtype=np.float32
        ).reshape(1, -1)

        bow = self.vectorizer.transform([q1, q2])
        q1_bow = bow[0]
        q2_bow = bow[1]

        X = hstack(
            [csr_matrix(dense_features), q1_bow, q2_bow],
            format="csr"
        )

        probabilities = self.model.predict_proba(X)[0]
        duplicate_probability = float(probabilities[1])
        non_duplicate_probability = float(probabilities[0])

        is_duplicate = duplicate_probability >= settings.similarity_threshold
        confidence = duplicate_probability if is_duplicate else non_duplicate_probability

        if is_duplicate:
            label = "Duplicate"
            message = "The two questions are likely asking for the same information."
        else:
            label = "Not Duplicate"
            message = "The two questions are likely asking for different information."

        return {
            "is_duplicate": bool(is_duplicate),
            "label": label,
            "confidence": round(confidence, 4),
            "duplicate_probability": round(duplicate_probability, 4),
            "non_duplicate_probability": round(non_duplicate_probability, 4),
            "message": message,
            "model_version": self.metadata.get("model_version", "unknown"),
        }

_predictor = None

def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor
