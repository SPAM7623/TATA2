"""Step 4 — Model comparison.

Compares the chosen regularised XGBoost against the main alternatives we
evaluated, all on the same leakage-free 5-fold out-of-fold split. This is the
evidence behind picking XGBoost: LightGBM ranks lower on this data, and the
heavier/lighter XGB variants do not improve on the regularised baseline.

Run:  python 04_model_comparison.py
"""

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src import config, data


def oof_for(make_model, X, y) -> np.ndarray:
    """Leakage-free out-of-fold probabilities for an arbitrary model factory."""
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    folds = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.CV_SEED
    )
    oof = np.zeros(len(y))
    for train_idx, valid_idx in folds.split(X, y):
        train_X, valid_X = data.impute(X.iloc[train_idx], X.iloc[valid_idx])
        spw = data.scale_pos_weight(y.iloc[train_idx])
        model = make_model(spw)
        model.fit(train_X, y.iloc[train_idx])
        oof[valid_idx] = model.predict_proba(valid_X)[:, 1]
    return oof


def main() -> None:
    X, y = data.load_train()

    # The candidates we compared. Each is a factory taking scale_pos_weight.
    candidates = {
        "XGB regularised (chosen)": lambda spw: XGBClassifier(
            random_state=config.CV_SEED, scale_pos_weight=spw, **config.XGB_PARAMS
        ),
        "XGB original (d6,150,lr.1)": lambda spw: XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1, scale_pos_weight=spw,
            random_state=config.CV_SEED, eval_metric="logloss", verbosity=0, n_jobs=-1,
        ),
        "LightGBM": lambda spw: LGBMClassifier(
            n_estimators=300, max_depth=3, num_leaves=8, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
            min_child_samples=3, scale_pos_weight=spw,
            random_state=config.CV_SEED, n_jobs=-1, verbose=-1,
        ),
    }

    print("=" * 60)
    print("STEP 4 | MODEL COMPARISON (5-fold OOF)")
    print("=" * 60)
    print(f"\n{'Model':<28}{'ROC-AUC':>9}{'PR-AUC':>9}")
    results = {}
    for name, make in candidates.items():
        oof = oof_for(make, X, y)
        auc = roc_auc_score(y, oof)
        pr = average_precision_score(y, oof)
        results[name] = auc
        print(f"{name:<28}{auc:>9.4f}{pr:>9.4f}")

    best = max(results, key=results.get)
    print(f"\nBest by ROC-AUC: {best}")
    print("Decision: regularised XGBoost is the base model for the ensemble.")


if __name__ == "__main__":
    main()
