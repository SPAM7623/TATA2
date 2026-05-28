"""
PART 1: INDUSTRIAL EDA (Focused & Insight-Driven)
Alpha Defect Prediction in Hot Rolling Mills

Goal: Understand unstable process behavior and defect-state characteristics

Key Insights To Extract:
- Which variables become unstable during defects?
- Are defects clustered or anomaly-like?
- Are defects threshold-driven?
- Which parameter combinations look risky?
- Which variables move together?
- Which regions behave like unstable operating zones?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# PART 1 CHECKLIST
# =====================================================================
CHECKLIST = {
    "1.1_data_loading": False,
    "1.2_basic_statistics": False,
    "1.3_missing_duplicates": False,
    "1.4_class_imbalance": False,
    "1.5_low_variance_features": False,
    "1.6_defect_vs_normal_comparison": False,
    "1.7_variance_instability_analysis": False,
    "1.8_outlier_analysis": False,
    "1.9_correlation_analysis": False,
    "1.10_correlation_difference": False,
    "1.11_pairwise_interactions": False,
    "1.12_defect_density_regions": False,
    "1.13_pca_visualization": False,
    "1.14_tsne_visualization": False,
    "1.15_hidden_regime_exploration": False,
    "1.16_threshold_stability_analysis": False,
    "1.17_anomaly_behavior_clustering": False,
}

class IndustrialEDA:
    """Comprehensive EDA for Alpha defect prediction"""

    def __init__(self, train_path='train.csv', test_path='test.csv'):
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        self.insights = {}

    def load_data(self):
        """1.1: Load training and test data"""
        print("\n" + "="*70)
        print("1.1 DATA LOADING")
        print("="*70)

        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)

        print(f"Train shape: {self.train_df.shape}")
        print(f"Test shape: {self.test_df.shape}")
        print(f"\nColumns: {list(self.train_df.columns)}")

        CHECKLIST["1.1_data_loading"] = True
        return self.train_df, self.test_df

    def basic_statistics(self):
        """1.2: Basic statistics and feature understanding"""
        print("\n" + "="*70)
        print("1.2 BASIC STATISTICS")
        print("="*70)

        print("\nTrain Dataset Info:")
        print(self.train_df.info())
        print("\nTrain Descriptive Stats:")
        print(self.train_df.describe())

        CHECKLIST["1.2_basic_statistics"] = True

    def missing_and_duplicates(self):
        """1.3: Check missing values and duplicates"""
        print("\n" + "="*70)
        print("1.3 MISSING VALUES & DUPLICATES")
        print("="*70)

        missing = self.train_df.isnull().sum()
        print(f"\nMissing values:\n{missing[missing > 0] if missing.sum() > 0 else 'None'}")

        duplicates = self.train_df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicates}")

        # Check for constant features
        constant_features = []
        for col in self.train_df.columns:
            if col != 'Y':
                if self.train_df[col].nunique() == 1:
                    constant_features.append(col)

        print(f"\nConstant features: {constant_features if constant_features else 'None'}")

        CHECKLIST["1.3_missing_duplicates"] = True

    def class_imbalance_analysis(self):
        """1.4: Class imbalance analysis"""
        print("\n" + "="*70)
        print("1.4 CLASS IMBALANCE ANALYSIS")
        print("="*70)

        class_dist = self.train_df['Y'].value_counts()
        class_pct = self.train_df['Y'].value_counts(normalize=True) * 100

        print("\nClass Distribution:")
        print(f"  0 (No Defect): {class_dist[0]} samples ({class_pct[0]:.2f}%)")
        print(f"  1 (Defect): {class_dist[1]} samples ({class_pct[1]:.2f}%)")
        print(f"\nImbalance Ratio: {class_dist[0]/class_dist[1]:.2f}:1")

        self.insights['class_imbalance_ratio'] = class_dist[0]/class_dist[1]

        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        self.train_df['Y'].value_counts().plot(kind='bar', ax=axes[0])
        axes[0].set_title('Class Distribution')
        axes[0].set_ylabel('Count')

        self.train_df['Y'].value_counts(normalize=True).plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
        axes[1].set_title('Class Proportion')

        plt.tight_layout()
        plt.savefig('01_class_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 01_class_distribution.png")

        CHECKLIST["1.4_class_imbalance"] = True

    def low_variance_features(self):
        """1.5: Identify low-variance features"""
        print("\n" + "="*70)
        print("1.5 LOW-VARIANCE FEATURE DETECTION")
        print("="*70)

        X = self.train_df.drop(['CoilID', 'Y'], axis=1)
        variances = X.var()

        # Features with very low variance
        low_var_threshold = variances.quantile(0.05)
        low_var_features = variances[variances < low_var_threshold].sort_values()

        print(f"\nLow variance features (bottom 5%):")
        for feat, var in low_var_features.items():
            print(f"  {feat}: variance={var:.6f}")

        self.insights['low_variance_features'] = list(low_var_features.index)

        CHECKLIST["1.5_low_variance_features"] = True

    def defect_vs_normal_comparison(self):
        """1.6: Statistical comparison between defect and non-defect samples"""
        print("\n" + "="*70)
        print("1.6 DEFECT vs NON-DEFECT COMPARISON")
        print("="*70)

        X = self.train_df.drop(['CoilID', 'Y'], axis=1)
        y = self.train_df['Y']

        defect_df = self.train_df[y == 1][X.columns]
        normal_df = self.train_df[y == 0][X.columns]

        # Feature-wise mean comparison
        mean_diff = (defect_df.mean() - normal_df.mean()).abs().sort_values(ascending=False)

        print("\nTop 10 features with largest mean differences:")
        for feat, diff in mean_diff.head(10).items():
            print(f"  {feat}: {diff:.6f}")

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Distribution of means
        axes[0, 0].scatter(normal_df.mean(), defect_df.mean(), alpha=0.6)
        axes[0, 0].plot([normal_df.mean().min(), normal_df.mean().max()],
                       [normal_df.mean().min(), normal_df.mean().max()], 'r--')
        axes[0, 0].set_xlabel('Non-Defect Mean')
        axes[0, 0].set_ylabel('Defect Mean')
        axes[0, 0].set_title('Mean Shift: Defect vs Non-Defect')
        axes[0, 0].grid(True, alpha=0.3)

        # Top differences
        mean_diff.head(15).plot(kind='barh', ax=axes[0, 1])
        axes[0, 1].set_title('Top 15 Mean Differences')
        axes[0, 1].set_xlabel('Absolute Difference')

        # Variance comparison
        var_normal = normal_df.var()
        var_defect = defect_df.var()
        axes[1, 0].scatter(var_normal, var_defect, alpha=0.6)
        axes[1, 0].set_xlabel('Non-Defect Variance')
        axes[1, 0].set_ylabel('Defect Variance')
        axes[1, 0].set_title('Variance Shift: Defect vs Non-Defect')
        axes[1, 0].grid(True, alpha=0.3)

        # Std comparison
        (defect_df.std() / (normal_df.std() + 1e-10)).sort_values(ascending=False).head(15).plot(kind='barh', ax=axes[1, 1])
        axes[1, 1].set_title('Top 15 Std Ratio (Defect/Normal)')
        axes[1, 1].set_xlabel('Ratio')

        plt.tight_layout()
        plt.savefig('02_defect_vs_normal.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 02_defect_vs_normal.png")

        CHECKLIST["1.6_defect_vs_normal_comparison"] = True

    def variance_instability_analysis(self):
        """1.7: Identify unstable features during defects"""
        print("\n" + "="*70)
        print("1.7 VARIANCE/INSTABILITY ANALYSIS")
        print("="*70)

        X = self.train_df.drop(['CoilID', 'Y'], axis=1)
        y = self.train_df['Y']

        defect_df = self.train_df[y == 1][X.columns]
        normal_df = self.train_df[y == 0][X.columns]

        # Instability ratio: variance during defect / variance during normal
        instability_ratio = (defect_df.var() / (normal_df.var() + 1e-10)).sort_values(ascending=False)

        print("\nTop 15 Unstable Features (High variance during defects):")
        for feat, ratio in instability_ratio.head(15).items():
            print(f"  {feat}: {ratio:.4f}")

        self.insights['unstable_features'] = list(instability_ratio.head(10).index)

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        instability_ratio.head(20).plot(kind='barh', ax=axes[0])
        axes[0].set_title('Feature Instability During Defects\n(Variance Ratio: Defect/Normal)')
        axes[0].set_xlabel('Ratio')

        # Scatter plot
        axes[1].scatter(normal_df.std(), defect_df.std(), alpha=0.6)
        axes[1].plot([0, normal_df.std().max()], [0, normal_df.std().max()], 'r--', label='Equal')
        axes[1].set_xlabel('Non-Defect Std Dev')
        axes[1].set_ylabel('Defect Std Dev')
        axes[1].set_title('Std Dev Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('03_instability_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 03_instability_analysis.png")

        CHECKLIST["1.7_variance_instability_analysis"] = True

    def outlier_analysis(self):
        """1.8: Outlier analysis (preserve meaningful industrial outliers)"""
        print("\n" + "="*70)
        print("1.8 OUTLIER ANALYSIS")
        print("="*70)

        X = self.train_df.drop(['CoilID', 'Y'], axis=1)
        y = self.train_df['Y']

        # Use Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        outlier_scores = iso_forest.fit_predict(X)
        outlier_probs = -iso_forest.score_samples(X)  # Negative scores -> positive anomaly scores

        self.train_df['outlier_score'] = outlier_probs

        # Outlier distribution by class
        normal_outlier_pct = (outlier_scores[y == 0] == -1).sum() / (y == 0).sum() * 100
        defect_outlier_pct = (outlier_scores[y == 1] == -1).sum() / (y == 1).sum() * 100

        print(f"\nOutlier Percentage (Isolation Forest):")
        print(f"  Non-Defect: {normal_outlier_pct:.2f}%")
        print(f"  Defect: {defect_outlier_pct:.2f}%")
        print(f"  Ratio: {defect_outlier_pct/normal_outlier_pct:.2f}x more outliers in defects")

        self.insights['outlier_rate_defect'] = defect_outlier_pct

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(outlier_probs[y == 0], bins=50, alpha=0.6, label='Non-Defect')
        axes[0].hist(outlier_probs[y == 1], bins=50, alpha=0.6, label='Defect')
        axes[0].set_xlabel('Anomaly Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Outlier Score Distribution')
        axes[0].legend()

        pd.DataFrame({'Non-Defect': [normal_outlier_pct], 'Defect': [defect_outlier_pct]}).T.plot(
            kind='bar', ax=axes[1], legend=False
        )
        axes[1].set_title('Outlier Percentage by Class')
        axes[1].set_ylabel('Percentage')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

        plt.tight_layout()
        plt.savefig('04_outlier_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 04_outlier_analysis.png")

        CHECKLIST["1.8_outlier_analysis"] = True

    def correlation_analysis(self):
        """1.9: Global correlation analysis"""
        print("\n" + "="*70)
        print("1.9 CORRELATION ANALYSIS")
        print("="*70)

        X = self.train_df.drop(['CoilID', 'Y'], axis=1)
        corr_matrix = X.corr()

        # High correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        print(f"\nTop 20 High Correlations (>0.7):")
        for feat1, feat2, corr in high_corr_pairs[:20]:
            print(f"  {feat1} <-> {feat2}: {corr:.4f}")

        self.insights['high_corr_pairs'] = high_corr_pairs[:10]

        # Visualization
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True)
        plt.title('Correlation Heatmap - All Features')
        plt.tight_layout()
        plt.savefig('05_correlation_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 05_correlation_heatmap.png")

        CHECKLIST["1.9_correlation_analysis"] = True

    def correlation_difference_analysis(self):
        """1.10: Correlation differences between defect/non-defect"""
        print("\n" + "="*70)
        print("1.10 CORRELATION DIFFERENCE ANALYSIS")
        print("="*70)

        X = self.train_df.drop(['CoilID', 'Y'], axis=1)
        y = self.train_df['Y']

        corr_normal = self.train_df[y == 0][X.columns].corr()
        corr_defect = self.train_df[y == 1][X.columns].corr()

        # Correlation difference
        corr_diff = corr_defect - corr_normal

        # Extract largest changes
        corr_diff_abs = corr_diff.abs()

        # Find top differences
        top_diffs = []
        for i in range(len(corr_diff.columns)):
            for j in range(i+1, len(corr_diff.columns)):
                top_diffs.append((
                    corr_diff.columns[i],
                    corr_diff.columns[j],
                    corr_diff.iloc[i, j]
                ))

        top_diffs.sort(key=lambda x: abs(x[2]), reverse=True)

        print(f"\nTop 15 Correlation Changes (Defect - Normal):")
        for feat1, feat2, diff in top_diffs[:15]:
            print(f"  {feat1} <-> {feat2}: {diff:.4f}")

        self.insights['correlation_changes'] = top_diffs[:5]

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 12))

        sns.heatmap(corr_normal, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                   square=True, ax=axes[0])
        axes[0].set_title('Correlation Matrix - Non-Defect Samples')

        sns.heatmap(corr_defect, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                   square=True, ax=axes[1])
        axes[1].set_title('Correlation Matrix - Defect Samples')

        plt.tight_layout()
        plt.savefig('06_correlation_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 06_correlation_comparison.png")

        CHECKLIST["1.10_correlation_difference"] = True

    def pairwise_interaction_exploration(self):
        """1.11: Explore pairwise interactions"""
        print("\n" + "="*70)
        print("1.11 PAIRWISE INTERACTION EXPLORATION")
        print("="*70)

        X = self.train_df.drop(['CoilID', 'Y'], axis=1).iloc[:1000]  # Sample for speed
        y = self.train_df['Y'].iloc[:1000]

        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X)

        # Interaction scores based on variance increase
        n_features = X_scaled.shape[1]
        interaction_scores = {}

        for i in range(min(10, n_features)):  # Check top 10 features
            for j in range(i+1, min(10, n_features)):
                interaction = X_scaled[:, i] * X_scaled[:, j]
                # Variance of interaction is proxy for interaction strength
                score = np.var(interaction)
                interaction_scores[(X.columns[i], X.columns[j])] = score

        top_interactions = sorted(interaction_scores.items(),
                                 key=lambda x: x[1], reverse=True)[:10]

        print(f"\nTop 10 Pairwise Interactions (by variance):")
        for (feat1, feat2), score in top_interactions:
            print(f"  {feat1} * {feat2}: {score:.6f}")

        CHECKLIST["1.11_pairwise_interactions"] = True

    def defect_density_analysis(self):
        """1.12: Defect density in different regions"""
        print("\n" + "="*70)
        print("1.12 DEFECT DENSITY REGION ANALYSIS")
        print("="*70)

        X = self.train_df.drop(['CoilID', 'Y'], axis=1)
        y = self.train_df['Y']

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Create density regions based on Euclidean distance from origin
        distances = np.linalg.norm(X_scaled, axis=1)

        # Quantile-based regions
        quantiles = [0, 0.25, 0.5, 0.75, 1.0]
        labels = [f'Region {i}' for i in range(len(quantiles)-1)]

        distance_bins = pd.qcut(distances, q=quantiles, labels=labels, duplicates='drop')

        defect_density = pd.DataFrame({
            'Region': distance_bins,
            'Y': y
        }).groupby('Region')['Y'].agg(['mean', 'count'])

        defect_density['mean'] = defect_density['mean'] * 100

        print(f"\nDefect Density by Distance Region:")
        print(defect_density)

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        defect_density['mean'].plot(kind='bar', ax=axes[0], color='coral')
        axes[0].set_title('Defect Density by Distance Region')
        axes[0].set_ylabel('Defect %')
        axes[0].set_xlabel('Region')

        defect_density['count'].plot(kind='bar', ax=axes[1], color='skyblue')
        axes[1].set_title('Sample Count by Distance Region')
        axes[1].set_ylabel('Count')
        axes[1].set_xlabel('Region')

        plt.tight_layout()
        plt.savefig('07_defect_density.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 07_defect_density.png")

        CHECKLIST["1.12_defect_density_regions"] = True

    def pca_visualization(self):
        """1.13: PCA visualization"""
        print("\n" + "="*70)
        print("1.13 PCA VISUALIZATION")
        print("="*70)

        X = self.train_df.drop(['CoilID', 'Y'], axis=1)
        y = self.train_df['Y']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        print(f"\nExplained Variance Ratio:")
        print(f"  PC1: {pca.explained_variance_ratio_[0]:.4f}")
        print(f"  PC2: {pca.explained_variance_ratio_[1]:.4f}")
        print(f"  Total: {pca.explained_variance_ratio_.sum():.4f}")

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', alpha=0.6)
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        axes[0].set_title('PCA Visualization - Defect vs Normal')
        plt.colorbar(scatter, ax=axes[0])

        # Cumulative variance
        cumsum_var = np.cumsum(PCA().fit(X_scaled).explained_variance_ratio_)
        axes[1].plot(range(1, len(cumsum_var)+1), cumsum_var, 'b-o')
        axes[1].axhline(y=0.95, color='r', linestyle='--', label='95%')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('PCA Cumulative Variance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('08_pca_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 08_pca_visualization.png")

        CHECKLIST["1.13_pca_visualization"] = True

    def tsne_visualization(self):
        """1.14: t-SNE visualization"""
        print("\n" + "="*70)
        print("1.14 t-SNE VISUALIZATION")
        print("="*70)

        X = self.train_df.drop(['CoilID', 'Y'], axis=1).iloc[:500]  # Sample for speed
        y = self.train_df['Y'].iloc[:500]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print("Computing t-SNE (this may take a moment)...")
        tsne = TSNE(n_components=2, random_state=42, n_iter=1000)
        X_tsne = tsne.fit_transform(X_scaled)

        # Visualization
        fig, axes = plt.subplots(1, 1, figsize=(10, 8))

        scatter = axes.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='coolwarm', alpha=0.6)
        axes.set_title('t-SNE Visualization - Defect Detection')
        axes.set_xlabel('t-SNE 1')
        axes.set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=axes)

        plt.tight_layout()
        plt.savefig('09_tsne_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 09_tsne_visualization.png")

        CHECKLIST["1.14_tsne_visualization"] = True

    def hidden_regime_exploration(self):
        """1.15: Explore hidden operating regimes"""
        print("\n" + "="*70)
        print("1.15 HIDDEN OPERATING REGIME EXPLORATION")
        print("="*70)

        X = self.train_df.drop(['CoilID', 'Y'], axis=1)
        y = self.train_df['Y']

        # Use row statistics as regime indicators
        row_means = X.mean(axis=1)
        row_stds = X.std(axis=1)
        row_ranges = X.max(axis=1) - X.min(axis=1)

        regime_data = pd.DataFrame({
            'row_mean': row_means,
            'row_std': row_stds,
            'row_range': row_ranges,
            'Y': y
        })

        # Regime classification by quantiles
        regime_data['mean_regime'] = pd.qcut(regime_data['row_mean'], q=3, labels=['Low', 'Mid', 'High'], duplicates='drop')
        regime_data['std_regime'] = pd.qcut(regime_data['row_std'], q=3, labels=['Low', 'Mid', 'High'], duplicates='drop')

        defect_by_regime = regime_data.groupby('mean_regime')['Y'].agg(['mean', 'count'])

        print(f"\nDefect Rate by Mean-Value Regime:")
        print(defect_by_regime)

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Mean vs Y
        axes[0, 0].scatter(regime_data[regime_data['Y']==0]['row_mean'],
                          regime_data[regime_data['Y']==0]['row_std'],
                          alpha=0.5, label='Normal', s=20)
        axes[0, 0].scatter(regime_data[regime_data['Y']==1]['row_mean'],
                          regime_data[regime_data['Y']==1]['row_std'],
                          alpha=0.5, label='Defect', s=20, color='red')
        axes[0, 0].set_xlabel('Row Mean')
        axes[0, 0].set_ylabel('Row Std Dev')
        axes[0, 0].set_title('Operating Regime: Mean vs Std')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Regime distribution
        defect_by_regime['mean'].plot(kind='bar', ax=axes[0, 1], color='coral')
        axes[0, 1].set_title('Defect Rate by Mean Regime')
        axes[0, 1].set_ylabel('Defect Rate')
        axes[0, 1].set_xlabel('Mean Regime')

        # Range analysis
        axes[1, 0].hist(regime_data[regime_data['Y']==0]['row_range'], bins=50, alpha=0.6, label='Normal')
        axes[1, 0].hist(regime_data[regime_data['Y']==1]['row_range'], bins=50, alpha=0.6, label='Defect')
        axes[1, 0].set_xlabel('Row Range')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Row Range Distribution')
        axes[1, 0].legend()

        # 3D regime
        axes[1, 1].scatter(regime_data[regime_data['Y']==0]['row_mean'],
                          regime_data[regime_data['Y']==0]['row_range'],
                          alpha=0.5, label='Normal', s=20)
        axes[1, 1].scatter(regime_data[regime_data['Y']==1]['row_mean'],
                          regime_data[regime_data['Y']==1]['row_range'],
                          alpha=0.5, label='Defect', s=20, color='red')
        axes[1, 1].set_xlabel('Row Mean')
        axes[1, 1].set_ylabel('Row Range')
        axes[1, 1].set_title('Operating Regime: Mean vs Range')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('10_hidden_regimes.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 10_hidden_regimes.png")

        CHECKLIST["1.15_hidden_regime_exploration"] = True

    def threshold_stability_analysis(self):
        """1.16: Threshold and stability boundary analysis"""
        print("\n" + "="*70)
        print("1.16 THRESHOLD/STABILITY BOUNDARY ANALYSIS")
        print("="*70)

        X = self.train_df.drop(['CoilID', 'Y'], axis=1)
        y = self.train_df['Y']

        # Stability features
        stability_features = {
            'row_cv': X.std(axis=1) / (X.mean(axis=1).abs() + 1e-10),
            'row_max_min': (X.max(axis=1) - X.min(axis=1)) / (X.mean(axis=1).abs() + 1e-10),
            'row_skew': X.skew(axis=1),
        }

        stability_df = pd.DataFrame(stability_features)
        stability_df['Y'] = y.values

        print("\nStability Metrics by Class:")
        print(stability_df.groupby('Y').describe().round(4))

        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        for idx, (feat, ax) in enumerate(zip(list(stability_features.keys()), axes)):
            ax.hist(stability_df[stability_df['Y']==0][feat], bins=50, alpha=0.6, label='Normal')
            ax.hist(stability_df[stability_df['Y']==1][feat], bins=50, alpha=0.6, label='Defect')
            ax.set_xlabel(feat)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{feat} Distribution')
            ax.legend()

        plt.tight_layout()
        plt.savefig('11_threshold_stability.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 11_threshold_stability.png")

        CHECKLIST["1.16_threshold_stability_analysis"] = True

    def anomaly_clustering_analysis(self):
        """1.17: Anomaly behavior and light clustering"""
        print("\n" + "="*70)
        print("1.17 ANOMALY BEHAVIOR & CLUSTERING ANALYSIS")
        print("="*70)

        X = self.train_df.drop(['CoilID', 'Y'], axis=1)
        y = self.train_df['Y']

        # Already computed outlier scores
        if 'outlier_score' not in self.train_df.columns:
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            outlier_scores = iso_forest.fit_predict(X)
            outlier_probs = -iso_forest.score_samples(X)
            self.train_df['outlier_score'] = outlier_probs

        # Anomaly-defect correlation
        anomaly_df = pd.DataFrame({
            'outlier_score': self.train_df['outlier_score'],
            'Y': y
        })

        print("\nAnomaly Score Statistics:")
        print(anomaly_df.groupby('Y')['outlier_score'].describe().round(4))

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(anomaly_df[anomaly_df['Y']==0]['outlier_score'], bins=50, alpha=0.6, label='Normal')
        axes[0].hist(anomaly_df[anomaly_df['Y']==1]['outlier_score'], bins=50, alpha=0.6, label='Defect')
        axes[0].set_xlabel('Anomaly Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Anomaly Score by Class')
        axes[0].legend()

        # Anomaly vs Y
        axes[1].scatter(range(len(anomaly_df)), anomaly_df['outlier_score'],
                       c=anomaly_df['Y'], cmap='coolwarm', alpha=0.5, s=10)
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Anomaly Score')
        axes[1].set_title('Anomaly Scores Across All Samples')

        plt.tight_layout()
        plt.savefig('12_anomaly_behavior.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 12_anomaly_behavior.png")

        CHECKLIST["1.17_anomaly_behavior_clustering"] = True

    def generate_insights_report(self):
        """Generate comprehensive insights summary"""
        print("\n" + "="*70)
        print("PART 1: INSIGHTS SUMMARY")
        print("="*70)

        report = f"""
KEY INSIGHTS FROM INDUSTRIAL EDA
================================

1. CLASS IMBALANCE:
   - Imbalance Ratio: {self.insights.get('class_imbalance_ratio', 'N/A'):.2f}:1
   - Defects are minority class
   - Requires imbalance handling strategies

2. UNSTABLE FEATURES DURING DEFECTS:
   - Top unstable features: {self.insights.get('unstable_features', 'N/A')[:3]}
   - These features show high variance during defect states
   - Candidates for interaction features

3. ANOMALY CHARACTERISTICS:
   - Defect samples contain {self.insights.get('outlier_rate_defect', 'N/A'):.2f}% anomalies
   - Defects may be anomaly-like or threshold-driven

4. HIGH CORRELATIONS:
   - Found {len(self.insights.get('high_corr_pairs', []))} highly correlated pairs (>0.7)
   - Multicollinearity present
   - Feature grouping recommended

5. RECOMMENDATIONS FOR NEXT STEPS:
   ✓ Investigate identified unstable features
   ✓ Create interaction features from correlated groups
   ✓ Consider anomaly-based features
   ✓ Prepare for imbalance handling in later steps
   ✓ Use PCA/correlated groups for dimensionality reduction
"""

        print(report)

        with open('PART1_INSIGHTS.txt', 'w') as f:
            f.write(report)

        print("✓ Saved: PART1_INSIGHTS.txt")

    def run_complete_eda(self):
        """Execute all EDA steps"""
        self.load_data()
        self.basic_statistics()
        self.missing_and_duplicates()
        self.class_imbalance_analysis()
        self.low_variance_features()
        self.defect_vs_normal_comparison()
        self.variance_instability_analysis()
        self.outlier_analysis()
        self.correlation_analysis()
        self.correlation_difference_analysis()
        self.pairwise_interaction_exploration()
        self.defect_density_analysis()
        self.pca_visualization()
        self.tsne_visualization()
        self.hidden_regime_exploration()
        self.threshold_stability_analysis()
        self.anomaly_clustering_analysis()
        self.generate_insights_report()

        print("\n" + "="*70)
        print("PART 1 COMPLETION CHECKLIST")
        print("="*70)
        for step, completed in CHECKLIST.items():
            status = "✓" if completed else "✗"
            print(f"{status} {step}")

        return self.insights

if __name__ == "__main__":
    from insights_manager import InsightsManager

    eda = IndustrialEDA(train_path='train.csv', test_path='test.csv')
    insights = eda.run_complete_eda()

    # Save insights for downstream parts
    manager = InsightsManager()
    manager.set_part1_insights(insights)

    print("\n" + "="*70)
    print("PART 1 COMPLETE")
    print("="*70)
    print("\nKey insights extracted and saved.")
    print("✓ Insights propagated to downstream parts")
    print("Ready for Part 2: Baseline Model Training")
