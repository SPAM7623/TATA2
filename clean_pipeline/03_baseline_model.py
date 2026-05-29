"""Step 3 — Single-seed baseline.

Cross-validates one regularised XGBoost model with leakage-free out-of-fold
predictions and prints the honest metrics. This is the reference point that the
ensemble in step 5 has to beat.

Run:  python 03_baseline_model.py
"""

from src import config, data, evaluation


def main() -> None:
    X, y = data.load_train()

    print("=" * 60)
    print("STEP 3 | BASELINE (single-seed regularised XGBoost)")
    print("=" * 60)

    oof = evaluation.cross_val_oof(X, y, seeds=[config.CV_SEED])
    threshold, f1 = evaluation.best_f1_threshold(y, oof)
    metrics = evaluation.summarise(y, oof, threshold)

    print(f"\n5-fold out-of-fold metrics (seed {config.CV_SEED}):")
    print(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC            : {metrics['pr_auc']:.4f}")
    print(f"  best-F1 threshold : {threshold:.4f}  (F1 = {f1:.3f})")
    print(f"  recall / precision: {metrics['recall']:.3f} / {metrics['precision']:.3f}")
    print(f"  MCC               : {metrics['mcc']:.3f}")
    print(f"  balanced accuracy : {metrics['balanced_accuracy']:.3f}")
    print(f"  confusion (tp/fp/fn/tn): "
          f"{metrics['tp']}/{metrics['fp']}/{metrics['fn']}/{metrics['tn']}")


if __name__ == "__main__":
    main()
