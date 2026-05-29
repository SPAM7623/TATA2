"""Cross-validate the model and fit the final 5-seed ensemble.

Two things happen here:

1. We measure honest performance with out-of-fold predictions, both for a
   single seed and for the seed-averaged ensemble, and print the metrics.
2. We refit one model per seed on *all* the training data and store the
   averaged test probabilities, ready for `predict.py`.

The out-of-fold and test probabilities are written to `artifacts/` so the
prediction step never has to retrain.
"""

import numpy as np
import pandas as pd

from src import config, data, evaluation
from src.model import build_model


def report(title: str, y, proba) -> None:
    """Print the standard metric block for a set of predictions."""
    threshold, f1 = evaluation.best_f1_threshold(y, proba)
    metrics = evaluation.summarise(y, proba, threshold)
    print(f"\n{title}")
    print(f"  ROC-AUC          : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC           : {metrics['pr_auc']:.4f}")
    print(f"  best-F1 threshold: {threshold:.4f}  (F1 = {f1:.3f})")
    print(f"  recall / precision: {metrics['recall']:.3f} / {metrics['precision']:.3f}")
    print(f"  MCC              : {metrics['mcc']:.3f}")


def main() -> None:
    X, y = data.load_train()
    test_X, test_ids = data.load_test()

    # --- 1. Honest cross-validation -------------------------------------
    print("=" * 60)
    print("CROSS-VALIDATION (out-of-fold, leakage-free)")
    print("=" * 60)

    single_oof = evaluation.cross_val_oof(X, y, seeds=[config.CV_SEED])
    report("Single seed", y, single_oof)

    ensemble_oof = evaluation.cross_val_oof(X, y, seeds=config.ENSEMBLE_SEEDS)
    report(f"{len(config.ENSEMBLE_SEEDS)}-seed ensemble", y, ensemble_oof)

    # --- 2. Fit the ensemble on all data --------------------------------
    print("\n" + "=" * 60)
    print("FITTING FINAL ENSEMBLE ON ALL TRAINING DATA")
    print("=" * 60)

    train_imp, test_imp = data.impute(X, test_X)
    spw = data.scale_pos_weight(y)

    test_proba = np.zeros(len(test_X))
    for seed in config.ENSEMBLE_SEEDS:
        model = build_model(seed, spw)
        model.fit(train_imp, y)
        test_proba += model.predict_proba(test_imp)[:, 1]
        print(f"  trained seed {seed}")
    test_proba /= len(config.ENSEMBLE_SEEDS)

    # --- 3. Save artifacts ----------------------------------------------
    config.ARTIFACTS.mkdir(exist_ok=True)
    pd.DataFrame({"y": y, "oof_proba": ensemble_oof}).to_csv(
        config.OOF_CSV, index=False
    )
    pd.DataFrame(
        {config.ID_COLUMN: test_ids, "proba": test_proba}
    ).to_csv(config.TEST_PROBA_CSV, index=False)

    print(f"\nSaved out-of-fold predictions -> {config.OOF_CSV.name}")
    print(f"Saved test probabilities      -> {config.TEST_PROBA_CSV.name}")
    print("\nNext: python predict.py --threshold <value>")


if __name__ == "__main__":
    main()
