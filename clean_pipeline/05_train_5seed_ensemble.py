"""Step 5 — Train the final 5-seed ensemble.

Two things happen here:

1. Cross-validate the 5-seed ensemble (out-of-fold) and print the honest
   metrics, so we can see the small, stable gain over the single-seed baseline.
2. Refit one model per seed on ALL training data and average their test
   probabilities. The out-of-fold and test probabilities are cached in
   `artifacts/` so step 6 never has to retrain.

Run:  python 05_train_5seed_ensemble.py
"""

import numpy as np
import pandas as pd

from src import config, data, evaluation
from src.model import build_model


def main() -> None:
    X, y = data.load_train()
    test_X, test_ids = data.load_test()

    # --- 1. Honest cross-validation of the ensemble --------------------
    print("=" * 60)
    print("STEP 5 | FINAL 5-SEED ENSEMBLE")
    print("=" * 60)
    print(f"Seeds: {config.ENSEMBLE_SEEDS}")

    oof = evaluation.cross_val_oof(X, y, seeds=config.ENSEMBLE_SEEDS)
    threshold, f1 = evaluation.best_f1_threshold(y, oof)
    metrics = evaluation.summarise(y, oof, threshold)
    print("\nOut-of-fold metrics:")
    print(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC            : {metrics['pr_auc']:.4f}")
    print(f"  best-F1 threshold : {threshold:.4f}  (F1 = {f1:.3f})")
    print(f"  recall / precision: {metrics['recall']:.3f} / {metrics['precision']:.3f}")

    # --- 2. Fit on all data, one model per seed ------------------------
    print("\nFitting on all training data...")
    train_imp, test_imp = data.impute(X, test_X)
    spw = data.scale_pos_weight(y)

    test_proba = np.zeros(len(test_X))
    for seed in config.ENSEMBLE_SEEDS:
        model = build_model(seed, spw)
        model.fit(train_imp, y)
        test_proba += model.predict_proba(test_imp)[:, 1]
        print(f"  trained seed {seed}")
    test_proba /= len(config.ENSEMBLE_SEEDS)

    # --- 3. Cache artifacts --------------------------------------------
    config.ARTIFACTS.mkdir(exist_ok=True)
    pd.DataFrame({"y": y, "oof_proba": oof}).to_csv(config.OOF_CSV, index=False)
    pd.DataFrame(
        {config.ID_COLUMN: test_ids, "proba": test_proba}
    ).to_csv(config.TEST_PROBA_CSV, index=False)

    print(f"\nSaved out-of-fold predictions -> {config.OOF_CSV.name}")
    print(f"Saved test probabilities      -> {config.TEST_PROBA_CSV.name}")
    print("\nNext: python 06_predict.py --threshold <value>")


if __name__ == "__main__":
    main()
