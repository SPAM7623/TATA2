"""Step 7 — Train the 10-seed ensemble for comparison.

Identical to step 5 (5-seed ensemble) but with 10 seeds:
42, 123, 999, 2025, 7777, 17, 29, 101, 314, 888

This tests whether adding 5 more seeds provides measurable generalization gain.

Run:  python 07_train_10seed_ensemble.py
"""

import numpy as np
import pandas as pd

from src import config, data, evaluation
from src.model import build_model


def main() -> None:
    X, y = data.load_train()
    test_X, test_ids = data.load_test()

    # --- 1. Honest cross-validation of the 10-seed ensemble ---------------
    print("=" * 60)
    print("STEP 7 | EXTENDED 10-SEED ENSEMBLE")
    print("=" * 60)
    print(f"Seeds: {config.ENSEMBLE_SEEDS_10}")

    oof = evaluation.cross_val_oof(X, y, seeds=config.ENSEMBLE_SEEDS_10)
    threshold, f1 = evaluation.best_f1_threshold(y, oof)
    metrics = evaluation.summarise(y, oof, threshold)
    print("\nOut-of-fold metrics:")
    print(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC            : {metrics['pr_auc']:.4f}")
    print(f"  best-F1 threshold : {threshold:.4f}  (F1 = {f1:.3f})")
    print(f"  recall / precision: {metrics['recall']:.3f} / {metrics['precision']:.3f}")

    # --- 2. Fit on all data, one model per seed ----------------------------
    print("\nFitting on all training data...")
    train_imp, test_imp = data.impute(X, test_X)
    spw = data.scale_pos_weight(y)

    test_proba = np.zeros(len(test_X))
    for seed in config.ENSEMBLE_SEEDS_10:
        model = build_model(seed, spw)
        model.fit(train_imp, y)
        test_proba += model.predict_proba(test_imp)[:, 1]
        print(f"  trained seed {seed}")
    test_proba /= len(config.ENSEMBLE_SEEDS_10)

    # --- 3. Cache artifacts ------------------------------------------------
    config.ARTIFACTS.mkdir(exist_ok=True)
    pd.DataFrame({"y": y, "oof_proba_10seed": oof}).to_csv(
        config.ARTIFACTS / "oof_predictions_10seed.csv", index=False
    )
    pd.DataFrame(
        {config.ID_COLUMN: test_ids, "proba_10seed": test_proba}
    ).to_csv(config.ARTIFACTS / "test_probabilities_10seed.csv", index=False)

    print(f"\nSaved 10-seed OOF predictions -> artifacts/oof_predictions_10seed.csv")
    print(f"Saved 10-seed test probs      -> artifacts/test_probabilities_10seed.csv")
    print("\nNext: python 08_compare_5seed_vs_10seed.py")


if __name__ == "__main__":
    main()
