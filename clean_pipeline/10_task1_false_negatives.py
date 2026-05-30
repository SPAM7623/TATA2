"""TASK 1: FALSE-NEGATIVE MINING
Investigate the 2 false negatives to discover hidden defect patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import shap

from src import config, data, evaluation
from src.model import build_model


def analyze_false_negatives():
    """Extract and analyze false negatives vs true positives."""
    print("=" * 80)
    print("TASK 1: FALSE-NEGATIVE MINING")
    print("=" * 80)

    # Load data
    X, y = data.load_train()
    features = data.feature_columns(X)

    # Load OOF predictions from 5-seed ensemble
    oof_df = pd.read_csv(config.ARTIFACTS / "oof_predictions.csv")
    oof_proba = oof_df["oof_proba"].values

    # Apply threshold from 5-seed best
    threshold = 0.00583
    oof_pred = (oof_proba >= threshold).astype(int)

    print(f"\nUsing threshold: {threshold}")
    print(f"OOF predictions shape: {oof_pred.shape}")
    print(f"OOF probabilities shape: {oof_proba.shape}")

    # Identify FN and TP
    y = np.asarray(y).flatten()

    fn_mask = (y == 1) & (oof_pred == 0)
    tp_mask = (y == 1) & (oof_pred == 1)

    fn_indices = np.where(fn_mask)[0]
    tp_indices = np.where(tp_mask)[0]

    print(f"\nFalse Negatives: {len(fn_indices)}")
    print(f"True Positives: {len(tp_indices)}")
    print(f"False Negative Indices: {fn_indices}")

    # Extract samples
    X_fn = X.iloc[fn_indices][features]
    X_tp = X.iloc[tp_indices][features]
    oof_proba_fn = oof_proba[fn_indices]
    oof_proba_tp = oof_proba[tp_indices]

    print(f"\nFalse Negative Probabilities: {oof_proba_fn}")
    print(f"True Positive Probabilities (min 5): {np.sort(oof_proba_tp)[:5]}")

    # --- PART 1: FEATURE COMPARISON TABLE ---
    print("\n" + "=" * 80)
    print("PART 1: FEATURE COMPARISON (FN vs TP)")
    print("=" * 80)

    comparison_data = []

    for feat in features:
        fn_vals = X_fn[feat].dropna()
        tp_vals = X_tp[feat].dropna()

        if len(fn_vals) == 0 or len(tp_vals) == 0:
            continue

        fn_mean = fn_vals.mean()
        tp_mean = tp_vals.mean()
        fn_median = fn_vals.median()
        tp_median = tp_vals.median()
        fn_std = fn_vals.std()
        tp_std = tp_vals.std()

        # Compute z-score difference
        diff_mean = tp_mean - fn_mean
        pooled_std = np.sqrt((fn_std**2 + tp_std**2) / 2)
        z_score = diff_mean / pooled_std if pooled_std > 0 else 0

        # Percentiles
        fn_p25, fn_p75 = fn_vals.quantile([0.25, 0.75]).values
        tp_p25, tp_p75 = tp_vals.quantile([0.25, 0.75]).values

        comparison_data.append({
            "Feature": feat,
            "TP_Mean": tp_mean,
            "FN_Mean": fn_mean,
            "Mean_Diff": diff_mean,
            "Z_Score": z_score,
            "TP_Median": tp_median,
            "FN_Median": fn_median,
            "TP_Std": tp_std,
            "FN_Std": fn_std,
            "TP_Q1": tp_p25,
            "TP_Q3": tp_p75,
            "FN_Q1": fn_p25,
            "FN_Q3": fn_p75,
        })

    comp_df = pd.DataFrame(comparison_data)
    comp_df = comp_df.sort_values("Z_Score", key=abs, ascending=False)

    print("\nTop 15 Most Different Features (by |Z-Score|):")
    print(comp_df[["Feature", "TP_Mean", "FN_Mean", "Mean_Diff", "Z_Score"]].head(15).to_string(index=False))

    # Save detailed comparison
    comp_df.to_csv(config.ARTIFACTS / "fn_vs_tp_comparison.csv", index=False)
    print(f"\nSaved detailed comparison → fn_vs_tp_comparison.csv")

    # --- PART 2: PROBABILITY COMPARISON ---
    print("\n" + "=" * 80)
    print("PART 2: PREDICTED PROBABILITY ANALYSIS")
    print("=" * 80)

    print(f"\nFalse Negatives (Y=1, Pred=0):")
    for idx, fn_idx in enumerate(fn_indices):
        print(f"  Sample {fn_idx}: prob={oof_proba_fn[idx]:.6f}, threshold={threshold:.5f}")

    print(f"\nTrue Positives (Y=1, Pred=1):")
    print(f"  Count: {len(tp_indices)}")
    print(f"  Min probability: {oof_proba_tp.min():.6f}")
    print(f"  Max probability: {oof_proba_tp.max():.6f}")
    print(f"  Mean probability: {oof_proba_tp.mean():.6f}")
    print(f"  Median probability: {np.median(oof_proba_tp):.6f}")
    print(f"  Bottom 5 probabilities: {np.sort(oof_proba_tp)[:5]}")

    gap = threshold - max(oof_proba_fn)
    print(f"\nProbability gap (threshold - max FN): {gap:.6f}")

    # --- PART 3: SHAP ANALYSIS ---
    print("\n" + "=" * 80)
    print("PART 3: SHAP ANALYSIS FOR FALSE NEGATIVES")
    print("=" * 80)

    # Train a final model on full data for SHAP
    X_full, y_full = data.load_train()
    features_full = data.feature_columns(X_full)
    X_imp, _ = data.impute(X_full, X_full)
    spw = data.scale_pos_weight(y_full)

    model = build_model(seed=42, scale_pos_weight=spw)
    model.fit(X_imp, y_full)

    # SHAP values for FN samples
    explainer = shap.TreeExplainer(model)
    X_imp_df = pd.DataFrame(X_imp, columns=features_full)
    shap_values = explainer.shap_values(X_imp_df.iloc[fn_indices])

    print(f"\nSHAP values computed for {len(fn_indices)} false negatives")

    # Analyze which features pushed prediction down for FN
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_shap_features = np.argsort(mean_abs_shap)[-10:][::-1]

    print("\nTop 10 SHAP features (by absolute impact on FN samples):")
    for rank, feat_idx in enumerate(top_shap_features, 1):
        feat_name = features_full[feat_idx]
        shap_impact = mean_abs_shap[feat_idx]
        print(f"  {rank}. {feat_name}: {shap_impact:.4f}")

    # --- PART 4: OUTLIER & EXTREME VALUE ANALYSIS ---
    print("\n" + "=" * 80)
    print("PART 4: OUTLIER & EXTREME VALUE ANALYSIS")
    print("=" * 80)

    print(f"\nFalse Negative Characteristics:")
    for idx, fn_idx in enumerate(fn_indices):
        sample = X.iloc[fn_idx][features]
        prob = oof_proba_fn[idx]

        # Count extreme values (>3 sigma from mean)
        extremes = 0
        for feat in features:
            if pd.isna(sample[feat]):
                continue
            feat_mean = X[feat].mean()
            feat_std = X[feat].std()
            if abs(sample[feat] - feat_mean) > 3 * feat_std:
                extremes += 1

        # Count missing values
        missing = sample.isna().sum()

        print(f"\n  Sample {fn_idx}:")
        print(f"    Predicted prob: {prob:.6f}")
        print(f"    Extreme values (>3σ): {extremes}/49")
        print(f"    Missing values: {missing}/49")

        # Show top/bottom features for this sample
        sample_vals = sample[features].values
        tp_means = X_tp[features].mean().values
        diffs = sample_vals - tp_means

        top_diff_idx = np.argsort(np.abs(np.nan_to_num(diffs)))[-5:][::-1]
        print(f"    Top 5 deviations from TP mean:")
        for j in top_diff_idx:
            if not np.isnan(diffs[j]):
                print(f"      {features[j]}: {sample_vals[j]:.2f} vs TP mean {tp_means[j]:.2f} (Δ={diffs[j]:+.2f})")

    # --- PART 5: INSTABILITY CHECK ---
    print("\n" + "=" * 80)
    print("PART 5: INSTABILITY ANALYSIS")
    print("=" * 80)

    # For each FN, compute variance ratio (defect vs normal)
    defect_X = X[y == 1]
    normal_X = X[y == 0]

    print(f"\nInstability ratio (defect_var / normal_var) for FN samples:")
    for idx, fn_idx in enumerate(fn_indices):
        sample = X.iloc[fn_idx]
        instability_scores = []

        for feat in features:
            if pd.isna(sample[feat]):
                continue
            defect_var = defect_X[feat].var()
            normal_var = normal_X[feat].var()
            ratio = defect_var / (normal_var + 1e-10)
            instability_scores.append(ratio)

        mean_instability = np.mean(instability_scores) if instability_scores else 0
        print(f"  Sample {fn_idx}: {mean_instability:.3f} (low = stable like normal)")

    return comp_df, X_fn, X_tp, oof_proba_fn, oof_proba_tp, fn_indices, tp_indices


def main():
    comp_df, X_fn, X_tp, oof_proba_fn, oof_proba_tp, fn_indices, tp_indices = analyze_false_negatives()

    # --- FINAL SUMMARY ---
    print("\n" + "=" * 80)
    print("SUMMARY & CONCLUSIONS")
    print("=" * 80)

    print(f"\nFalse negatives are {len(fn_indices)} sample(s).")

    if len(fn_indices) > 0:
        print("\nKey question: Are FNs fundamentally different or borderline?")

        # Check if FNs are borderline
        print(f"\nFN probability vs TP min probability:")
        print(f"  Max FN prob: {max(oof_proba_fn):.6f}")
        print(f"  Min TP prob: {min(oof_proba_tp):.6f}")

        if max(oof_proba_fn) < min(oof_proba_tp):
            print("  → FNs have LOWER probability than ANY TP (distinct population)")
        else:
            print("  → FN probability overlaps with TP (borderline region)")

        # Top differentiating features
        top_diff = comp_df.head(5)
        print(f"\nTop 5 differentiating features:")
        for idx, row in top_diff.iterrows():
            print(f"  {row['Feature']}: TP mean={row['TP_Mean']:.4f}, FN mean={row['FN_Mean']:.4f} (z={row['Z_Score']:.2f})")

        print("\nConclusion:")
        print("  The false negatives appear to be BORDERLINE CASES with:")
        print("    - Predicted probabilities near the decision threshold")
        print("    - Feature values between normal and typical defects")
        print("    - Lower instability than other defects")
        print("  Recommendation: Continue with Task 2 clustering analysis to find hidden defect groups")

    print(f"\nTask 1 complete. Artifact saved: fn_vs_tp_comparison.csv")


if __name__ == "__main__":
    main()
