"""Cross-validation and metrics.

`cross_val_oof` produces out-of-fold probabilities, which are the honest basis
for every number we report - each coil is scored only by models that never saw
it during training. `summarise` turns those probabilities into the metrics that
matter for an imbalanced detection task.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from . import config, data
from .model import build_model


def cross_val_oof(X: pd.DataFrame, y: pd.Series, seeds=None) -> np.ndarray:
    """Return out-of-fold probabilities averaged over the given seeds.

    For each fold the imputer and models are fit on the training part only,
    then used to score the held-out part. Averaging several seeds reduces the
    variance of the estimate.
    """
    seeds = seeds or [config.CV_SEED]
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    folds = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.CV_SEED
    )
    oof = np.zeros(len(y))

    for train_idx, valid_idx in folds.split(X, y):
        train_X, valid_X = impute_fold(X, train_idx, valid_idx)
        spw = data.scale_pos_weight(y.iloc[train_idx])

        fold_proba = np.zeros(len(valid_idx))
        for seed in seeds:
            model = build_model(seed, spw)
            model.fit(train_X, y.iloc[train_idx])
            fold_proba += model.predict_proba(valid_X)[:, 1]

        oof[valid_idx] = fold_proba / len(seeds)

    return oof


def impute_fold(X, train_idx, valid_idx):
    """Median-impute one CV fold without leakage."""
    return data.impute(X.iloc[train_idx], X.iloc[valid_idx])


def summarise(y, proba, threshold: float) -> dict:
    """Compute the metrics we care about at a chosen threshold.

    ROC-AUC and PR-AUC are threshold-independent; the rest describe the
    confusion matrix at `threshold`.
    """
    y = np.asarray(y)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()

    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "threshold": threshold,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def best_f1_threshold(y, proba) -> tuple[float, float]:
    """Scan thresholds and return the (threshold, F1) that maximises F1."""
    grid = np.linspace(0.001, 0.95, 300)
    scores = [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])
