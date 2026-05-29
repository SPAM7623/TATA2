"""Step 1 — Comprehensive Exploratory Data Analysis.

Complete EDA covering:
- Dataset shape, missing values, duplicates, class imbalance
- Univariate analysis: feature distributions, outliers
- Defect vs non-defect comparison: mean shifts, variance changes
- Variance/instability: features that become unstable during defects
- Correlation & feature grouping: relationships between variables
- Interaction analysis: feature pairs and their strength
- Dimensionality reduction: PCA, correlation heatmap, hierarchical clustering
- Hard sample analysis: outlier detection, defect patterns

Run:  python 01_eda.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

from src import config, data


def main() -> None:
    train = pd.read_csv(config.TRAIN_CSV)
    test = pd.read_csv(config.TEST_CSV)
    features = data.feature_columns(train)
    X = train[features]
    y = train[config.TARGET_COLUMN]

    print("=" * 70)
    print("STEP 1 | COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # ===== DATASET SHAPE =====
    print("\n1. DATASET SHAPE")
    print("=" * 70)
    print(f"Train: {train.shape[0]} coils × {len(features)} features")
    print(f"Test : {test.shape[0]} coils × {len(features)} features")

    # ===== MISSING VALUES =====
    print("\n2. MISSING VALUES & DUPLICATES")
    print("=" * 70)
    print(f"Train NaN cells: {int(X.isna().sum().sum())} ({100*X.isna().sum().sum()/(X.shape[0]*X.shape[1]):.2f}%)")
    print(f"Test NaN cells : {int(test[features].isna().sum().sum())}")
    affected = X.isna().sum()
    print(f"Affected columns: {int((affected > 0).sum())}")
    print(f"Duplicate rows  : {int(train.duplicated().sum())}")

    # ===== CLASS IMBALANCE =====
    print("\n3. CLASS IMBALANCE")
    print("=" * 70)
    counts = y.value_counts().sort_index()
    n_pos = int(counts.get(1, 0))
    n_neg = int(counts.get(0, 0))
    print(f"Non-defective (0): {n_neg} ({100*n_neg/(n_neg+n_pos):.1f}%)")
    print(f"Defective     (1): {n_pos} ({100*n_pos/(n_neg+n_pos):.1f}%)")
    print(f"Imbalance ratio  : {n_neg/max(n_pos, 1):.1f} : 1")

    # ===== UNIVARIATE ANALYSIS =====
    print("\n4. UNIVARIATE ANALYSIS (Feature Distributions)")
    print("=" * 70)
    print("Feature statistics (top 5 by variance):")
    var_stats = X.var().sort_values(ascending=False).head()
    for feat, var in var_stats.items():
        print(f"  {feat}: var={var:.2f}, mean={X[feat].mean():.2f}, std={X[feat].std():.2f}")

    # ===== OUTLIER ANALYSIS =====
    print("\n5. OUTLIER ANALYSIS")
    print("=" * 70)
    X_imp = X.fillna(X.median())
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outlier_labels = iso_forest.fit_predict(X_imp)
    outlier_pct = 100 * (outlier_labels == -1).sum() / len(X_imp)
    defect_outlier_pct = 100 * (outlier_labels[y == 1] == -1).sum() / (y == 1).sum()
    normal_outlier_pct = 100 * (outlier_labels[y == 0] == -1).sum() / (y == 0).sum()
    print(f"Outliers detected: {(outlier_labels == -1).sum()} ({outlier_pct:.1f}%)")
    print(f"  Defect samples  : {defect_outlier_pct:.1f}% outliers")
    print(f"  Normal samples  : {normal_outlier_pct:.1f}% outliers")
    print(f"  Ratio           : {defect_outlier_pct/normal_outlier_pct:.2f}x more outliers in defects")

    # ===== DEFECT VS NON-DEFECT COMPARISON =====
    print("\n6. DEFECT vs NON-DEFECT COMPARISON")
    print("=" * 70)
    defect_X = X[y == 1]
    normal_X = X[y == 0]
    mean_diff = (defect_X.mean() - normal_X.mean()).abs().sort_values(ascending=False)
    print("Top 10 features with largest mean differences:")
    for feat, diff in mean_diff.head(10).items():
        print(f"  {feat}: Δmean={diff:.6f}")

    # ===== VARIANCE/INSTABILITY ANALYSIS =====
    print("\n7. VARIANCE/INSTABILITY ANALYSIS")
    print("=" * 70)
    instability = (defect_X.var() / (normal_X.var() + 1e-10)).sort_values(ascending=False)
    print("Top 15 unstable features (high variance during defects):")
    for feat, ratio in instability.head(15).items():
        print(f"  {feat}: ratio={ratio:.4f} (defect var / normal var)")

    # ===== CORRELATION ANALYSIS =====
    print("\n8. CORRELATION ANALYSIS")
    print("=" * 70)
    corr_matrix = X_imp.corr()
    high_corr = corr_matrix.abs().unstack().sort_values(ascending=False)
    high_corr = high_corr[high_corr < 1.0]  # Exclude self-correlation
    print("Top 10 feature correlations:")
    for (feat1, feat2), corr_val in high_corr.head(10).items():
        print(f"  {feat1} - {feat2}: {corr_val:.4f}")

    # ===== INTERACTION ANALYSIS =====
    print("\n9. FEATURE INTERACTION ANALYSIS")
    print("=" * 70)
    X_scaled = StandardScaler().fit_transform(X_imp)
    interaction_scores = {}
    for i in range(min(15, X_scaled.shape[1])):
        for j in range(i+1, min(15, X_scaled.shape[1])):
            interaction = X_scaled[:, i] * X_scaled[:, j]
            score = np.var(interaction)
            interaction_scores[(features[i], features[j])] = score
    top_interactions = sorted(interaction_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 pairwise interactions (by variance):")
    for (f1, f2), score in top_interactions:
        print(f"  {f1} × {f2}: {score:.6f}")

    # ===== FEATURE GROUPING (via correlation) =====
    print("\n10. FEATURE GROUPING (Hierarchical Clustering)")
    print("=" * 70)
    corr_dist = 1 - corr_matrix.abs()
    linkage_matrix = linkage(pdist(corr_dist), method='ward')
    print(f"Hierarchical clustering computed ({len(features)} features)")
    print(f"Distance metric: 1 - |correlation|")

    # ===== PCA ANALYSIS =====
    print("\n11. DIMENSIONALITY REDUCTION (PCA)")
    print("=" * 70)
    pca = PCA()
    pca.fit(X_scaled)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_comp_95 = np.argmax(cumsum >= 0.95) + 1
    n_comp_90 = np.argmax(cumsum >= 0.90) + 1
    print(f"PC1 explains: {100*pca.explained_variance_ratio_[0]:.2f}%")
    print(f"PC2 explains: {100*pca.explained_variance_ratio_[1]:.2f}%")
    print(f"Top 5 PCs explain: {100*cumsum[4]:.2f}%")
    print(f"Components for 90% variance: {n_comp_90}")
    print(f"Components for 95% variance: {n_comp_95}")

    # ===== FEATURE SCALE =====
    print("\n12. FEATURE SCALE (Median Absolute Value)")
    print("=" * 70)
    spread = X.abs().median().sort_values(ascending=False)
    print("Top 10 by scale:")
    for name, value in spread.head(10).items():
        print(f"  {name:>4}: {value:,.1f}")

    # ===== DESIGN TAKEAWAYS =====
    print("\n" + "=" * 70)
    print("DESIGN TAKEAWAYS & DECISIONS")
    print("=" * 70)
    print("✓ Few positives (66)        → Guard against overfitting")
    print("✓ High imbalance (19.5:1)   → Use scale_pos_weight in model")
    print("✓ Missing values (249)      → Apply median imputation (train fold only)")
    print("✓ Wild scale variation      → No scaling needed (tree-based model)")
    print("✓ Feature correlation       → Keep all features (tree model handles)")
    print("✓ High dimensionality (49)  → Use regularisation (shallow trees)")
    print("✓ Outliers in defects       → Model naturally learns outlier patterns")
    print("✓ Instability in defects    → Core signal to learn from")


if __name__ == "__main__":
    main()
