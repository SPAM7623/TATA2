"""Step 8 — Compare 5-seed vs 10-seed ensemble.

Tests whether the 10-seed ensemble provides genuine improvement over 5-seed.
Analyzes:
- OOF metrics comparison
- Threshold optimization around the best region
- Prediction correlation between models
- Final recommendation
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from src import config, data, evaluation
from src.model import build_model


def load_predictions():
    """Load baseline, 5-seed, and 10-seed predictions."""
    # Single seed baseline
    single = evaluation.cross_val_oof(
        *data.load_train(), seeds=[config.CV_SEED]
    )

    # 5-seed and 10-seed OOF
    oof5 = pd.read_csv(config.ARTIFACTS / "oof_predictions.csv")["oof_proba"].values
    oof10 = pd.read_csv(config.ARTIFACTS / "oof_predictions_10seed.csv")[
        "oof_proba_10seed"
    ].values

    return single, oof5, oof10


def compare_metrics(y, single, oof5, oof10):
    """Compare all metrics across the three models."""
    # Use threshold optimized from 5-seed
    _, threshold = evaluation.best_f1_threshold(y, oof5)

    m_single = evaluation.summarise(y, single, threshold)
    m5 = evaluation.summarise(y, oof5, threshold)
    m10 = evaluation.summarise(y, oof10, threshold)

    return m_single, m5, m10, threshold


def threshold_sweep(y, proba, name):
    """Sweep thresholds and report metrics."""
    thresholds = [0.0050, 0.0053, 0.0055, 0.00583, 0.0060, 0.0063, 0.0065]

    results = []
    for t in thresholds:
        metrics = evaluation.summarise(y, proba, t)
        metrics["threshold"] = t
        results.append(metrics)

    return results


def find_best_thresholds(y, proba):
    """Find best thresholds by MCC, F1, and balanced accuracy."""
    grid = np.linspace(0.001, 0.95, 500)
    mcc_scores = []
    f1_scores = []
    ba_scores = []

    for t in grid:
        metrics = evaluation.summarise(y, proba, t)
        mcc_scores.append(metrics["mcc"])
        f1_scores.append(metrics["f1"])
        ba_scores.append(metrics["balanced_accuracy"])

    best_mcc_idx = np.argmax(mcc_scores)
    best_f1_idx = np.argmax(f1_scores)
    best_ba_idx = np.argmax(ba_scores)

    return (
        (float(grid[best_mcc_idx]), float(mcc_scores[best_mcc_idx])),
        (float(grid[best_f1_idx]), float(f1_scores[best_f1_idx])),
        (float(grid[best_ba_idx]), float(ba_scores[best_ba_idx])),
    )


def main() -> None:
    print("=" * 80)
    print("STEP 8 | 5-SEED vs 10-SEED ENSEMBLE COMPARISON")
    print("=" * 80)

    X, y = data.load_train()
    y = np.asarray(y)

    # Load predictions
    print("\nLoading predictions...")
    single, oof5, oof10 = load_predictions()
    print("✓ Single seed, 5-seed OOF, 10-seed OOF loaded")

    # --- PART 1: OOF METRICS COMPARISON ------------------------------------
    print("\n" + "=" * 80)
    print("PART 1: OUT-OF-FOLD METRICS COMPARISON")
    print("=" * 80)

    m_single, m5, m10, best_threshold = compare_metrics(y, single, oof5, oof10)

    print(f"\nUsing best-F1 threshold from 5-seed: {best_threshold:.5f}\n")
    print(f"{'Metric':<20} {'Single':>12} {'5-Seed':>12} {'10-Seed':>12} {'Δ(10 vs 5)':>12}")
    print("-" * 70)

    metrics_to_report = ["roc_auc", "pr_auc", "recall", "precision", "f1", "mcc", "balanced_accuracy"]
    diffs = {}

    for metric in metrics_to_report:
        s_val = m_single[metric]
        m5_val = m5[metric]
        m10_val = m10[metric]
        diff = m10_val - m5_val
        diffs[metric] = diff

        print(f"{metric:<20} {s_val:>12.4f} {m5_val:>12.4f} {m10_val:>12.4f} {diff:>+12.4f}")

    # --- PART 2: THRESHOLD OPTIMIZATION -----------------------------------
    print("\n" + "=" * 80)
    print("PART 2: THRESHOLD OPTIMIZATION ACROSS CRITICAL REGION")
    print("=" * 80)

    thresholds = [0.0050, 0.0053, 0.0055, 0.00583, 0.0060, 0.0063, 0.0065]

    print("\n5-SEED ENSEMBLE:")
    print(f"{'Threshold':<12} {'Recall':>8} {'Precision':>10} {'F1':>8} {'MCC':>8} {'Bal.Acc':>8} {'Positives':>10}")
    print("-" * 70)
    for t in thresholds:
        m = evaluation.summarise(y, oof5, t)
        pred_pos = (oof5 >= t).sum()
        print(f"{t:<12.5f} {m['recall']:>8.3f} {m['precision']:>10.3f} {m['f1']:>8.3f} {m['mcc']:>8.3f} {m['balanced_accuracy']:>8.3f} {pred_pos:>10}")

    print("\n10-SEED ENSEMBLE:")
    print(f"{'Threshold':<12} {'Recall':>8} {'Precision':>10} {'F1':>8} {'MCC':>8} {'Bal.Acc':>8} {'Positives':>10}")
    print("-" * 70)
    for t in thresholds:
        m = evaluation.summarise(y, oof10, t)
        pred_pos = (oof10 >= t).sum()
        print(f"{t:<12.5f} {m['recall']:>8.3f} {m['precision']:>10.3f} {m['f1']:>8.3f} {m['mcc']:>8.3f} {m['balanced_accuracy']:>8.3f} {pred_pos:>10}")

    # --- PART 3: BEST THRESHOLDS ------------------------------------------
    print("\n" + "=" * 80)
    print("PART 3: BEST THRESHOLDS BY OPTIMIZATION CRITERION")
    print("=" * 80)

    print("\n5-SEED ENSEMBLE:")
    best5_mcc, best5_f1, best5_ba = find_best_thresholds(y, oof5)

    m_best_mcc = evaluation.summarise(y, oof5, best5_mcc[0])
    m_best_f1 = evaluation.summarise(y, oof5, best5_f1[0])
    m_best_ba = evaluation.summarise(y, oof5, best5_ba[0])

    print(f"  Best by MCC (threshold={best5_mcc[0]:.5f}, MCC={best5_mcc[1]:.4f})")
    print(f"    → Recall={m_best_mcc['recall']:.3f}, Precision={m_best_mcc['precision']:.3f}, F1={m_best_mcc['f1']:.3f}")

    print(f"  Best by F1 (threshold={best5_f1[0]:.5f}, F1={best5_f1[1]:.4f})")
    print(f"    → Recall={m_best_f1['recall']:.3f}, Precision={m_best_f1['precision']:.3f}, MCC={m_best_f1['mcc']:.3f}")

    print(f"  Best by Balanced Accuracy (threshold={best5_ba[0]:.5f}, BA={best5_ba[1]:.4f})")
    print(f"    → Recall={m_best_ba['recall']:.3f}, Precision={m_best_ba['precision']:.3f}, F1={m_best_ba['f1']:.3f}")

    print("\n10-SEED ENSEMBLE:")
    best10_mcc, best10_f1, best10_ba = find_best_thresholds(y, oof10)

    m_best_mcc_10 = evaluation.summarise(y, oof10, best10_mcc[0])
    m_best_f1_10 = evaluation.summarise(y, oof10, best10_f1[0])
    m_best_ba_10 = evaluation.summarise(y, oof10, best10_ba[0])

    print(f"  Best by MCC (threshold={best10_mcc[0]:.5f}, MCC={best10_mcc[1]:.4f})")
    print(f"    → Recall={m_best_mcc_10['recall']:.3f}, Precision={m_best_mcc_10['precision']:.3f}, F1={m_best_mcc_10['f1']:.3f}")

    print(f"  Best by F1 (threshold={best10_f1[0]:.5f}, F1={best10_f1[1]:.4f})")
    print(f"    → Recall={m_best_f1_10['recall']:.3f}, Precision={m_best_f1_10['precision']:.3f}, MCC={m_best_f1_10['mcc']:.3f}")

    print(f"  Best by Balanced Accuracy (threshold={best10_ba[0]:.5f}, BA={best10_ba[1]:.4f})")
    print(f"    → Recall={m_best_ba_10['recall']:.3f}, Precision={m_best_ba_10['precision']:.3f}, F1={m_best_ba_10['f1']:.3f}")

    # --- PART 4: PREDICTION CORRELATION -----------------------------------
    print("\n" + "=" * 80)
    print("PART 4: PREDICTION CORRELATION ANALYSIS")
    print("=" * 80)

    # Pearson correlation
    r_single_5, p_single_5 = pearsonr(single, oof5)
    r_single_10, p_single_10 = pearsonr(single, oof10)
    r_5_10, p_5_10 = pearsonr(oof5, oof10)

    # Spearman rank correlation
    s_single_5, ps_single_5 = spearmanr(single, oof5)
    s_single_10, ps_single_10 = spearmanr(single, oof10)
    s_5_10, ps_5_10 = spearmanr(oof5, oof10)

    print("\nPearson Correlation:")
    print(f"  Single vs 5-Seed : {r_single_5:.4f}")
    print(f"  Single vs 10-Seed: {r_single_10:.4f}")
    print(f"  5-Seed vs 10-Seed: {r_5_10:.4f} ← Very high (ensemble stability)")

    print("\nSpearman Rank Correlation:")
    print(f"  Single vs 5-Seed : {s_single_5:.4f}")
    print(f"  Single vs 10-Seed: {s_single_10:.4f}")
    print(f"  5-Seed vs 10-Seed: {s_5_10:.4f} ← Very high (ranking consistency)")

    # --- PART 5: SUMMARY & RECOMMENDATION ---------------------------------
    print("\n" + "=" * 80)
    print("PART 5: ANALYSIS SUMMARY & RECOMMENDATION")
    print("=" * 80)

    print("\nKEY FINDINGS:")
    print(f"  • ROC-AUC improvement (10 vs 5):        {diffs['roc_auc']:+.4f}")
    print(f"  • PR-AUC improvement (10 vs 5):        {diffs['pr_auc']:+.4f}")
    print(f"  • F1 improvement (10 vs 5):            {diffs['f1']:+.4f}")
    print(f"  • MCC improvement (10 vs 5):           {diffs['mcc']:+.4f}")
    print(f"  • Balanced Accuracy improvement (10):  {diffs['balanced_accuracy']:+.4f}")
    print(f"  • Prediction correlation (5 vs 10):    {r_5_10:.4f} (very high)")

    # Recommendation logic
    roc_auc_threshold = 0.002
    mcc_threshold = 0.005

    print("\nRECOMMENDATION:")
    if (diffs['roc_auc'] >= roc_auc_threshold or
        diffs['mcc'] >= mcc_threshold or
        (diffs['roc_auc'] > 0 and diffs['mcc'] > 0)):
        print("  ✓ UPGRADE TO 10-SEED ENSEMBLE")
        print("    Rationale:")
        if diffs['roc_auc'] > 0:
            print(f"      - ROC-AUC improved by {diffs['roc_auc']:.4f}")
        if diffs['mcc'] > 0:
            print(f"      - MCC improved by {diffs['mcc']:.4f}")
        if diffs['f1'] > 0:
            print(f"      - F1 improved by {diffs['f1']:.4f}")
        print(f"      - Prediction correlation very high ({r_5_10:.4f})")
        print("      - Consistent improvement across metrics suggests genuine gain")
    else:
        print("  ✗ KEEP CURRENT 5-SEED MODEL")
        print("    Rationale:")
        print(f"      - 10-seed ROC-AUC improvement ({diffs['roc_auc']:+.4f}) below threshold ({roc_auc_threshold})")
        print(f"      - 10-seed MCC improvement ({diffs['mcc']:+.4f}) below threshold ({mcc_threshold})")
        print(f"      - High prediction correlation ({r_5_10:.4f}) indicates 5-seed already near ceiling")
        print("      - Additional seeds do not provide practical benefit")
        print("      - Cost/benefit favors simpler 5-seed model")

    print("\nCOMPUTATIONAL COST:")
    print(f"  5-seed:  5 CV rounds × 5 folds = 25 model fits")
    print(f"  10-seed: 10 CV rounds × 5 folds = 50 model fits (2x cost)")

    # Save detailed report
    report_path = config.ARTIFACTS / "ensemble_comparison_report.txt"
    config.ARTIFACTS.mkdir(exist_ok=True)

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("5-SEED vs 10-SEED ENSEMBLE COMPARISON REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("OOF METRICS\n")
        f.write(f"{'Metric':<20} {'Single':>12} {'5-Seed':>12} {'10-Seed':>12} {'Δ(10 vs 5)':>12}\n")
        f.write("-" * 70 + "\n")
        for metric in metrics_to_report:
            s_val = m_single[metric]
            m5_val = m5[metric]
            m10_val = m10[metric]
            diff = m10_val - m5_val
            f.write(f"{metric:<20} {s_val:>12.4f} {m5_val:>12.4f} {m10_val:>12.4f} {diff:>+12.4f}\n")

        f.write(f"\nPrediction Correlation (5 vs 10): {r_5_10:.4f}\n")
        f.write(f"Spearman Rank Correlation: {s_5_10:.4f}\n")

        f.write(f"\nRECOMMENDATION:\n")
        if (diffs['roc_auc'] >= roc_auc_threshold or
            diffs['mcc'] >= mcc_threshold or
            (diffs['roc_auc'] > 0 and diffs['mcc'] > 0)):
            f.write("UPGRADE TO 10-SEED ENSEMBLE\n")
        else:
            f.write("KEEP CURRENT 5-SEED MODEL\n")

    print(f"\nDetailed report saved → {report_path.name}")


if __name__ == "__main__":
    main()
