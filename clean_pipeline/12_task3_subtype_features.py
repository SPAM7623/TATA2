"""TASK 3: DEFECT SUBTYPE DISCOVERY & DISTANCE-TO-SUBTYPE FEATURE ENGINEERING
Create subtype-aware features and test if they improve model performance.

Only execute if Task 2 finds clear, meaningful positive clusters.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)
import shap

from src import config, data, evaluation
from src.model import build_model


def create_subtype_features():
    """Create distance-to-subtype features from cluster analysis."""
    print("=" * 80)
    print("TASK 3: DEFECT SUBTYPE DISCOVERY & FEATURE ENGINEERING")
    print("=" * 80)

    # Load original data
    X, y = data.load_train()
    test_X, test_ids = data.load_test()
    features = data.feature_columns(X)
    y = np.asarray(y).flatten()

    # Load cluster assignments
    cluster_assignment_df = pd.read_csv(config.ARTIFACTS / "defect_cluster_assignments.csv")
    defect_scaled = pd.read_csv(config.ARTIFACTS / "defect_scaled_features.csv")

    # Get defect indices
    defect_indices = cluster_assignment_df["defect_index"].values
    cluster_labels = cluster_assignment_df["cluster"].values

    n_clusters = len(np.unique(cluster_labels))
    print(f"\nFound {n_clusters} defect clusters")

    # --- CHECK IF CLUSTERING IS MEANINGFUL ---
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"Cluster sizes: {dict(zip(unique, counts))}")

    # Proceed only if meaningful structure found
    if n_clusters == 1:
        print("\n⚠ WARNING: Only 1 cluster found. Skipping Task 3.")
        print("Recommendation: Stay with baseline 5-seed model.")
        return None

    if counts.min() < 2:
        print(f"\n⚠ WARNING: Cluster too imbalanced (min size {counts.min()}). Skipping Task 3.")
        return None

    print("\n✓ Meaningful clustering found. Proceeding with feature engineering...")

    # --- CREATE SUBTYPE DISTANCE FEATURES ---
    print("\n" + "=" * 80)
    print("PART 1: CREATING SUBTYPE DISTANCE FEATURES")
    print("=" * 80)

    # Prepare all data (scaled)
    X_imp = X[features].fillna(X[features].median())
    test_imp = test_X[features].fillna(test_X[features].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    test_scaled = scaler.transform(test_imp)

    # Compute cluster centroids
    centroids = []
    for c in range(n_clusters):
        c_mask = cluster_labels == c
        defect_scaled_arr = defect_scaled.values[c_mask]
        centroid = defect_scaled_arr.mean(axis=0)
        centroids.append(centroid)

    print(f"\nComputed {n_clusters} cluster centroids")

    # Create distance features for all samples
    dist_features = {}

    for c in range(n_clusters):
        centroid = np.array(centroids[c])
        distances = np.linalg.norm(X_scaled - centroid, axis=1)
        test_distances = np.linalg.norm(test_scaled - centroid, axis=1)

        dist_features[f"dist_cluster_{c}"] = distances
        print(f"  Created dist_cluster_{c}")

    # Create new feature matrix
    X_new = X_imp.copy()
    X_test_new = test_imp.copy()

    for feat_name, feat_vals in dist_features.items():
        X_new[feat_name] = feat_vals

    for feat_name in dist_features.keys():
        c = int(feat_name.split("_")[-1])
        centroid = np.array(centroids[c])
        test_distances = np.linalg.norm(test_scaled - centroid, axis=1)
        X_test_new[feat_name] = test_distances

    print(f"\nOriginal feature count: {len(features)}")
    print(f"New feature count: {len(X_new.columns)}")

    new_features = list(X_new.columns)

    # --- TRAIN BASELINE (FROZEN 5-SEED) ---
    print("\n" + "=" * 80)
    print("PART 2: FROZEN 5-SEED BASELINE (NO SUBTYPE FEATURES)")
    print("=" * 80)

    # Use original features only
    oof_baseline = evaluation.cross_val_oof(X[features], y, seeds=config.ENSEMBLE_SEEDS)
    threshold_baseline, f1_baseline = evaluation.best_f1_threshold(y, oof_baseline)
    metrics_baseline = evaluation.summarise(y, oof_baseline, threshold_baseline)

    print(f"\nBaseline 5-Seed (original features):")
    print(f"  ROC-AUC: {metrics_baseline['roc_auc']:.4f}")
    print(f"  PR-AUC: {metrics_baseline['pr_auc']:.4f}")
    print(f"  F1: {metrics_baseline['f1']:.4f}")
    print(f"  MCC: {metrics_baseline['mcc']:.4f}")
    print(f"  Balanced Accuracy: {metrics_baseline['balanced_accuracy']:.4f}")

    # --- TRAIN WITH SUBTYPE FEATURES ---
    print("\n" + "=" * 80)
    print("PART 3: ENHANCED MODEL WITH SUBTYPE DISTANCE FEATURES")
    print("=" * 80)

    oof_enhanced = evaluation.cross_val_oof(X_new, y, seeds=config.ENSEMBLE_SEEDS)
    threshold_enhanced, f1_enhanced = evaluation.best_f1_threshold(y, oof_enhanced)
    metrics_enhanced = evaluation.summarise(y, oof_enhanced, threshold_enhanced)

    print(f"\nEnhanced 5-Seed (with {n_clusters} subtype features):")
    print(f"  ROC-AUC: {metrics_enhanced['roc_auc']:.4f}")
    print(f"  PR-AUC: {metrics_enhanced['pr_auc']:.4f}")
    print(f"  F1: {metrics_enhanced['f1']:.4f}")
    print(f"  MCC: {metrics_enhanced['mcc']:.4f}")
    print(f"  Balanced Accuracy: {metrics_enhanced['balanced_accuracy']:.4f}")

    # --- COMPARISON ---
    print("\n" + "=" * 80)
    print("PART 4: PERFORMANCE COMPARISON")
    print("=" * 80)

    print(f"\n{'Metric':<20} {'Baseline':>12} {'Enhanced':>12} {'Δ':>12} {'Winner':<10}")
    print("-" * 60)

    metrics_list = ["roc_auc", "pr_auc", "f1", "mcc", "balanced_accuracy", "recall", "precision"]
    improvements = {}

    for metric in metrics_list:
        baseline_val = metrics_baseline[metric]
        enhanced_val = metrics_enhanced[metric]
        delta = enhanced_val - baseline_val
        improvements[metric] = delta

        winner = "Enhanced" if delta > 0.001 else ("Baseline" if delta < -0.001 else "Tie")
        print(f"{metric:<20} {baseline_val:>12.4f} {enhanced_val:>12.4f} {delta:>+12.4f}  {winner:<10}")

    # --- DECISION CRITERION ---
    print("\n" + "=" * 80)
    print("PART 5: DECISION ANALYSIS")
    print("=" * 80)

    roc_improvement = improvements["roc_auc"]
    mcc_improvement = improvements["mcc"]
    f1_improvement = improvements["f1"]

    print(f"\nImprovement analysis:")
    print(f"  ROC-AUC improvement: {roc_improvement:+.4f}")
    print(f"  MCC improvement: {mcc_improvement:+.4f}")
    print(f"  F1 improvement: {f1_improvement:+.4f}")

    # Thresholds for meaningful improvement
    ROC_THRESHOLD = 0.002
    MCC_THRESHOLD = 0.005

    if roc_improvement >= ROC_THRESHOLD or mcc_improvement >= MCC_THRESHOLD:
        recommend_enhanced = True
        reason = "Meaningful improvement in primary metrics"
    elif roc_improvement > 0 and mcc_improvement > 0 and f1_improvement > 0:
        recommend_enhanced = True
        reason = "Consistent improvement across all metrics (but below threshold)"
    elif roc_improvement > 0:
        recommend_enhanced = True
        reason = "Positive ROC-AUC improvement"
    else:
        recommend_enhanced = False
        reason = "ROC-AUC degradation or no meaningful improvement"

    print(f"\nRecommendation: {'UPGRADE' if recommend_enhanced else 'STAY WITH BASELINE'}")
    print(f"Reason: {reason}")

    # --- FALSE NEGATIVE ANALYSIS ---
    print("\n" + "=" * 80)
    print("PART 6: FALSE NEGATIVE IMPROVEMENT")
    print("=" * 80)

    threshold = 0.00583

    # Baseline FN
    baseline_pred = (oof_baseline >= threshold).astype(int)
    baseline_fn = (y == 1) & (baseline_pred == 0)
    baseline_fn_count = baseline_fn.sum()

    # Enhanced FN
    enhanced_pred = (oof_enhanced >= threshold).astype(int)
    enhanced_fn = (y == 1) & (enhanced_pred == 0)
    enhanced_fn_count = enhanced_fn.sum()

    print(f"\nAt threshold {threshold}:")
    print(f"  Baseline false negatives: {baseline_fn_count}")
    print(f"  Enhanced false negatives: {enhanced_fn_count}")
    print(f"  Improvement: {baseline_fn_count - enhanced_fn_count} fewer FN")

    if enhanced_fn_count < baseline_fn_count:
        # Show which FN were recovered
        recovered = baseline_fn & (~enhanced_fn)
        recovered_indices = np.where(recovered)[0]
        print(f"  Recovered samples: {list(recovered_indices)}")

    # Save results
    config.ARTIFACTS.mkdir(exist_ok=True)

    results_summary = {
        "Model": ["Baseline 5-Seed (Frozen)", "Enhanced with Subtype Features"],
        "ROC-AUC": [metrics_baseline["roc_auc"], metrics_enhanced["roc_auc"]],
        "PR-AUC": [metrics_baseline["pr_auc"], metrics_enhanced["pr_auc"]],
        "F1": [metrics_baseline["f1"], metrics_enhanced["f1"]],
        "MCC": [metrics_baseline["mcc"], metrics_enhanced["mcc"]],
        "Balanced Accuracy": [metrics_baseline["balanced_accuracy"], metrics_enhanced["balanced_accuracy"]],
        "False Negatives (@ 0.00583)": [baseline_fn_count, enhanced_fn_count],
    }

    results_df = pd.DataFrame(results_summary)
    results_df.to_csv(config.ARTIFACTS / "subtype_features_comparison.csv", index=False)

    print(f"\nSaved comparison → subtype_features_comparison.csv")

    # Save enhanced predictions
    pd.DataFrame({"y": y, "oof_proba_enhanced": oof_enhanced}).to_csv(
        config.ARTIFACTS / "oof_predictions_enhanced.csv", index=False
    )

    # Return for potential later use
    return {
        "baseline_metrics": metrics_baseline,
        "enhanced_metrics": metrics_enhanced,
        "oof_baseline": oof_baseline,
        "oof_enhanced": oof_enhanced,
        "recommendation": recommend_enhanced,
        "roc_improvement": roc_improvement,
        "mcc_improvement": mcc_improvement,
    }


def main():
    result = create_subtype_features()

    if result is None:
        print("\nTask 3 skipped due to insufficient clustering structure.")
        return

    # --- FINAL SUMMARY ---
    print("\n" + "=" * 80)
    print("TASK 3 SUMMARY")
    print("=" * 80)

    print(f"\nFinal Question: Do hidden defect subtypes provide additional signal?")
    print(f"\nAnswer:")

    if result["recommendation"]:
        print(f"  ✓ YES - Subtype-aware features improve performance")
        print(f"  ROC-AUC improvement: {result['roc_improvement']:+.4f}")
        print(f"  MCC improvement: {result['mcc_improvement']:+.4f}")
        print(f"  → Consider using enhanced model for production")
    else:
        print(f"  ✗ NO - Subtype features do not improve performance")
        print(f"  ROC-AUC change: {result['roc_improvement']:+.4f}")
        print(f"  MCC change: {result['mcc_improvement']:+.4f}")
        print(f"  → Stay with frozen baseline 5-seed model")

    print(f"\nTask 3 complete.")


if __name__ == "__main__":
    main()
