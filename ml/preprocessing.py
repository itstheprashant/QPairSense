from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from app.services.feature_engineering import pair_features, clean_pair, FEATURE_NAMES

REQUIRED_COLUMNS = [
    "id", "qid1", "qid2", "question1", "question2", "is_duplicate"
]

def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    out = df.copy()
    out["question1"] = out["question1"].fillna("").astype(str)
    out["question2"] = out["question2"].fillna("").astype(str)

    out = out[out["question1"].str.strip().ne("") & out["question2"].str.strip().ne("")]
    out = out[out["is_duplicate"].isin([0, 1])].reset_index(drop=True)
    return out

def clean_questions(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    q1 = [clean_pair(x, "")[0] for x in df["question1"].tolist()]
    q2 = [clean_pair(x, "")[0] for x in df["question2"].tolist()]
    return q1, q2

def build_engineered_features(
    df: pd.DataFrame,
    workers: int = 1
) -> np.ndarray:
    pairs = list(zip(df["question1"].tolist(), df["question2"].tolist()))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(lambda p: pair_features(*p), pairs))
    else:
        rows = [pair_features(a, b) for a, b in pairs]

    return np.asarray(rows, dtype=np.float32)

def build_clean_texts(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    q1 = [clean_pair(x, "")[0] for x in df["question1"].tolist()]
    q2 = [clean_pair(x, "")[0] for x in df["question2"].tolist()]
    return q1, q2
