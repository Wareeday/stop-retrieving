"""The stopping classifier: P(evidence is sufficient | features after hop t)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES

MODEL_KINDS = ("logreg", "gbm")


def make_model(kind: str, seed: int = 0):
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0,
            min_samples_leaf=20, early_stopping=False, random_state=seed,
        )
    raise ValueError(f"unknown model kind {kind!r}, expected one of {MODEL_KINDS}")


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, kind: str,
                features: list[str] | None = None, seed: int = 0,
                target: str = "sufficient") -> np.ndarray:
    """target is 'sufficient' (is the evidence enough right now?) or 'eventually' (will it be
    enough by the last hop?)."""
    features = FEATURE_NAMES if features is None else features
    model = make_model(kind, seed)
    model.fit(train[features], train[target])
    return model.predict_proba(test[features])[:, 1]


def cross_val_predict(table: pd.DataFrame, max_hops: int, kind: str,
                      features: list[str] | None = None, n_splits: int = 5,
                      seed: int = 0, target: str = "sufficient") -> np.ndarray:
    """Out-of-fold probabilities. Folds split by question, so no question appears in both
    the training and the test side, and every hop of a question stays together."""
    n_questions = len(table) // max_hops
    out = np.zeros(len(table))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_q, test_q in splitter.split(np.arange(n_questions)):
        train_rows = _rows_of(train_q, max_hops)
        test_rows = _rows_of(test_q, max_hops)
        out[test_rows] = fit_predict(table.iloc[train_rows], table.iloc[test_rows], kind,
                                     features, seed, target)
    return out


def _rows_of(question_idx: np.ndarray, max_hops: int) -> np.ndarray:
    return (question_idx[:, None] * max_hops + np.arange(max_hops)[None, :]).ravel()
