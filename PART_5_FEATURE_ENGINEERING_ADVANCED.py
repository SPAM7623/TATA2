"""
PART 5: ADVANCED FEATURE ENGINEERING (Phase 4)
Alpha Defect Prediction in Hot Rolling Mills

OPTIMIZATION FOCUS:
- SHAP-guided interaction features
- Regime detection (risky operating conditions)
- Instability/change features
- Anomaly detection features
- Group-level features
- Strategic feature selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# INTERACTION FEATURE ENGINEER
# =====================================================================

class SHAPGuidedFeatureEngineer:
    """Engineer features based on SHAP insights"""

    def __init__(self, X_data, y_data, interaction_pairs=None):
        self.X_data = X_data.copy()
        self.y_data = y_data
        self.interaction_pairs = interaction_pairs or []
        self.engineered_features = {}

    def engineer_interaction_features(self, max_features=15):
        """Create interaction features from top SHAP pairs"""
        print("\n" + "-"*70)
        print("ENGINEER INTERACTION FEATURES")
        print("-"*70)

        interactions_added = 0
        X_engineered = self.X_data.copy()

        # Default interaction pairs if not provided
        if not self.interaction_pairs:
            # Common patterns in industrial data
            numeric_cols = self.X_data.select_dtypes(include=[np.number]).columns.tolist()
            self.interaction_pairs = [
                (numeric_cols[i], numeric_cols[j])
                for i in range(min(5, len(numeric_cols)))
                for j in range(i+1, min(5, len(numeric_cols)))
            ]

        print(f"Creating {len(self.interaction_pairs)} interaction features...")

        for feat1, feat2 in self.interaction_pairs[:max_features]:
            try:
                # Multiplicative interaction
                feat_name = f"{feat1}_x_{feat2}_mul"
                X_engineered[feat_name] = self.X_data[feat1] * self.X_data[feat2]
                self.engineered_features[feat_name] = 'multiplicative'
                interactions_added += 1

                # Ratio interaction (avoid division by zero)
                feat_name = f"{feat1}_{feat2}_ratio"
                denom = self.X_data[feat2] + 1e-6
                X_engineered[feat_name] = self.X_data[feat1] / denom
                self.engineered_features[feat_name] = 'ratio'
                interactions_added += 1

                # Polynomial interaction (squared)
                if interactions_added < max_features:
                    feat_name = f"{feat1}_x_{feat2}_poly2"
                    X_engineered[feat_name] = (self.X_data[feat1] * self.X_data[feat2]) ** 0.5
                    self.engineered_features[feat_name] = 'polynomial'
                    interactions_added += 1

            except Exception as e:
                pass

        print(f"✓ Created {interactions_added} interaction features")
        return X_engineered

    def engineer_statistical_interactions(self):
        """Create statistical aggregation features"""
        print("\n" + "-"*70)
        print("ENGINEER STATISTICAL FEATURES")
        print("-"*70)

        X_engineered = self.X_data.copy()

        numeric_cols = self.X_data.select_dtypes(include=[np.number]).columns.tolist()

        if numeric_cols:
            # Row-wise statistics
            X_engineered['row_mean'] = self.X_data[numeric_cols].mean(axis=1)
            X_engineered['row_std'] = self.X_data[numeric_cols].std(axis=1)
            X_engineered['row_max'] = self.X_data[numeric_cols].max(axis=1)
            X_engineered['row_min'] = self.X_data[numeric_cols].min(axis=1)
            X_engineered['row_range'] = X_engineered['row_max'] - X_engineered['row_min']

            for feat in ['row_mean', 'row_std', 'row_max', 'row_min', 'row_range']:
                self.engineered_features[feat] = 'statistical'

            print(f"✓ Created 5 row-level statistical features")

        return X_engineered


# =====================================================================
# REGIME DETECTOR
# =====================================================================

class RegimeDetector:
    """Detect risky operating regimes"""

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        self.regimes = {}
        self.engineered_features = {}

    def identify_regimes_kmeans(self, n_regimes=5):
        """Identify regimes using K-means clustering"""
        print("\n" + "-"*70)
        print("IDENTIFY OPERATING REGIMES")
        print("-"*70)

        # Standardize for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_data.select_dtypes(include=[np.number]))

        # Cluster
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        regime_labels = kmeans.fit_predict(X_scaled)

        # Analyze defect rate per regime
        regime_analysis = []
        for regime in range(n_regimes):
            mask = regime_labels == regime
            defect_rate = self.y_data[mask].mean()
            count = mask.sum()

            regime_analysis.append({
                'regime': regime,
                'count': count,
                'defect_rate': defect_rate,
                'risk_score': defect_rate * np.log1p(count)
            })

        regime_df = pd.DataFrame(regime_analysis).sort_values('defect_rate', ascending=False)

        print(f"\nRegimes identified (n={n_regimes}):")
        for idx, row in regime_df.iterrows():
            risk_level = 'HIGH' if row['defect_rate'] > self.y_data.mean() * 1.5 else 'NORMAL'
            print(f"  Regime {row['regime']}: defect_rate={row['defect_rate']:.2%}, count={row['count']}, risk={risk_level}")

        self.regimes = regime_df
        return regime_labels

    def create_regime_features(self, regime_labels):
        """Create binary regime flags"""
        print("\n" + "-"*70)
        print("CREATE REGIME FEATURES")
        print("-"*70)

        X_engineered = self.X_data.copy()

        high_risk_threshold = self.y_data.mean() * 1.5

        high_risk_regimes = self.regimes[self.regimes['defect_rate'] > high_risk_threshold]['regime'].tolist()

        for regime in range(regime_labels.max() + 1):
            feat_name = f"regime_{regime}"
            X_engineered[feat_name] = (regime_labels == regime).astype(int)
            self.engineered_features[feat_name] = 'regime'

            if regime in high_risk_regimes:
                feat_name = f"high_risk_regime_{regime}"
                X_engineered[feat_name] = (regime_labels == regime).astype(int)
                self.engineered_features[feat_name] = 'regime_high_risk'

        print(f"✓ Created {len(self.engineered_features)} regime features")
        return X_engineered


# =====================================================================
# INSTABILITY FEATURES
# =====================================================================

class InstabilityFeatures:
    """Create instability/change detection features"""

    def __init__(self, X_data):
        self.X_data = X_data
        self.engineered_features = {}

    def create_variance_features(self):
        """Create variance-based instability features"""
        print("\n" + "-"*70)
        print("CREATE VARIANCE FEATURES")
        print("-"*70)

        X_engineered = self.X_data.copy()

        numeric_cols = self.X_data.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            # Coefficient of variation
            mean_val = self.X_data[col].mean()
            std_val = self.X_data[col].std()
            if mean_val != 0:
                X_engineered[f"{col}_cv"] = std_val / abs(mean_val)
                self.engineered_features[f"{col}_cv"] = 'instability'

            # Outlier count
            q1 = self.X_data[col].quantile(0.25)
            q3 = self.X_data[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((self.X_data[col] < q1 - 1.5*iqr) | (self.X_data[col] > q3 + 1.5*iqr)).astype(int)
            X_engineered[f"{col}_outlier"] = outliers
            self.engineered_features[f"{col}_outlier"] = 'instability'

        print(f"✓ Created {len([k for k, v in self.engineered_features.items() if v == 'instability'])} instability features")
        return X_engineered


# =====================================================================
# ANOMALY FEATURES
# =====================================================================

class AnomalyFeatures:
    """Create anomaly detection features"""

    def __init__(self, X_data):
        self.X_data = X_data
        self.engineered_features = {}

    def isolation_forest_score(self, contamination=0.1):
        """Create Isolation Forest anomaly scores"""
        print("\n" + "-"*70)
        print("CREATE ISOLATION FOREST ANOMALY SCORES")
        print("-"*70)

        X_engineered = self.X_data.copy()

        numeric_data = self.X_data.select_dtypes(include=[np.number])

        if len(numeric_data.columns) > 0:
            iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
            anomaly_scores = iso_forest.fit_predict(numeric_data)
            anomaly_proba = iso_forest.score_samples(numeric_data)

            X_engineered['anomaly_score_if'] = -anomaly_proba  # Higher = more anomalous
            X_engineered['is_anomaly'] = (anomaly_scores == -1).astype(int)

            self.engineered_features['anomaly_score_if'] = 'anomaly'
            self.engineered_features['is_anomaly'] = 'anomaly'

            print(f"✓ Created anomaly features")
            print(f"  Anomalies detected: {X_engineered['is_anomaly'].sum()} ({100*X_engineered['is_anomaly'].mean():.2f}%)")

        return X_engineered

    def mahalanobis_distance(self):
        """Create Mahalanobis distance features"""
        print("\n" + "-"*70)
        print("CREATE MAHALANOBIS DISTANCE FEATURES")
        print("-"*70)

        X_engineered = self.X_data.copy()

        numeric_data = self.X_data.select_dtypes(include=[np.number])

        if len(numeric_data.columns) > 1:
            try:
                # Compute covariance
                mean = numeric_data.mean()
                cov = numeric_data.cov()

                # Regularize covariance
                cov_inv = np.linalg.inv(cov + np.eye(len(cov)) * 1e-6)

                # Compute Mahalanobis distance
                diff = numeric_data - mean
                mahal_dist = np.sqrt((diff @ cov_inv * diff).sum(axis=1))

                X_engineered['mahal_distance'] = mahal_dist
                self.engineered_features['mahal_distance'] = 'anomaly'

                print(f"✓ Created Mahalanobis distance feature")

            except:
                print("⚠ Could not compute Mahalanobis distance")

        return X_engineered


# =====================================================================
# GROUP FEATURES
# =====================================================================

class GroupFeatures:
    """Create features based on group membership"""

    def __init__(self, X_data, y_data, group_col=None):
        self.X_data = X_data
        self.y_data = y_data
        self.group_col = group_col
        self.engineered_features = {}

    def create_group_statistics(self):
        """Create features based on group statistics"""
        print("\n" + "-"*70)
        print("CREATE GROUP STATISTICS FEATURES")
        print("-"*70)

        X_engineered = self.X_data.copy()

        if self.group_col is None:
            # Try to create pseudo-groups from clustering
            print("Creating pseudo-groups from clustering...")

            numeric_cols = self.X_data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(self.X_data[numeric_cols])

                kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                groups = kmeans.fit_predict(X_scaled)
            else:
                return X_engineered
        else:
            groups = self.X_data[self.group_col]

        # Create group features
        numeric_cols = self.X_data.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols[:5]:  # Limit to first 5 for speed
            group_mean = self.y_data.groupby(groups).transform('mean')
            group_std = self.y_data.groupby(groups).transform('std')

            X_engineered[f"{col}_group_mean"] = group_mean
            X_engineered[f"{col}_group_std"] = group_std
            X_engineered[f"{col}_group_deviation"] = self.X_data[col] - self.X_data[col].groupby(groups).transform('mean')

            self.engineered_features[f"{col}_group_mean"] = 'group'
            self.engineered_features[f"{col}_group_std"] = 'group'
            self.engineered_features[f"{col}_group_deviation"] = 'group'

        print(f"✓ Created {len([k for k, v in self.engineered_features.items() if v == 'group'])} group features")

        return X_engineered


# =====================================================================
# FEATURE SELECTION
# =====================================================================

class FeatureSelector:
    """Select best engineered features"""

    def __init__(self, X_original, X_engineered, y_data):
        self.X_original = X_original
        self.X_engineered = X_engineered
        self.y_data = y_data

    def test_feature_importance(self, n_features_to_select=30):
        """Test which engineered features improve model"""
        print("\n" + "-"*70)
        print("TEST ENGINEERED FEATURE IMPORTANCE")
        print("-"*70)

        # Train with original features
        model_original = XGBClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )

        model_original.fit(self.X_original, self.y_data)
        y_pred_original = model_original.predict_proba(self.X_original)[:, 1]
        auc_original = roc_auc_score(self.y_data, y_pred_original)

        print(f"\nOriginal features AUC: {auc_original:.4f}")

        # Train with all features
        model_all = XGBClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )

        model_all.fit(self.X_engineered, self.y_data)
        y_pred_all = model_all.predict_proba(self.X_engineered)[:, 1]
        auc_all = roc_auc_score(self.y_data, y_pred_all)

        print(f"All features AUC: {auc_all:.4f}")
        print(f"Improvement: {auc_all - auc_original:+.4f}")

        # Feature importance from full model
        importances = pd.DataFrame({
            'feature': self.X_engineered.columns,
            'importance': model_all.feature_importances_
        }).sort_values('importance', ascending=False)

        engineered_cols = [col for col in self.X_engineered.columns if col not in self.X_original.columns]
        engineered_importances = importances[importances['feature'].isin(engineered_cols)]

        print(f"\nTop 10 engineered features:")
        for idx, row in engineered_importances.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.6f}")

        # Select top features
        top_features = importances.head(n_features_to_select)['feature'].tolist()

        print(f"\n✓ Selected {n_features_to_select} features for final model")

        return top_features, importances


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def run_part5_feature_engineering(train_path='train.csv', interaction_pairs=None):
    """Execute Part 5: Advanced Feature Engineering"""

    print("\n" + "="*70)
    print("PART 5: ADVANCED FEATURE ENGINEERING")
    print("="*70)

    # Load data
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(['CoilID', 'Y'], axis=1)
    y_train = train_df['Y']

    print(f"Data loaded: {X_train.shape}")

    # Step 1: Interaction features
    print("\n" + "="*70)
    print("STEP 1: INTERACTION FEATURES")
    print("="*70)

    engineer = SHAPGuidedFeatureEngineer(X_train, y_train, interaction_pairs)
    X_with_interactions = engineer.engineer_interaction_features()
    X_with_interactions = engineer.engineer_statistical_interactions()

    print(f"Features after interactions: {X_with_interactions.shape[1]}")

    # Step 2: Regime features
    print("\n" + "="*70)
    print("STEP 2: REGIME FEATURES")
    print("="*70)

    regime_detector = RegimeDetector(X_train, y_train)
    regime_labels = regime_detector.identify_regimes_kmeans(n_regimes=5)
    X_with_regimes = regime_detector.create_regime_features(regime_labels)

    print(f"Features after regimes: {X_with_regimes.shape[1]}")

    # Step 3: Instability features
    print("\n" + "="*70)
    print("STEP 3: INSTABILITY FEATURES")
    print("="*70)

    instability = InstabilityFeatures(X_train)
    X_with_instability = instability.create_variance_features()

    print(f"Features after instability: {X_with_instability.shape[1]}")

    # Step 4: Anomaly features
    print("\n" + "="*70)
    print("STEP 4: ANOMALY FEATURES")
    print("="*70)

    anomaly = AnomalyFeatures(X_train)
    X_with_anomaly = anomaly.isolation_forest_score()
    X_with_anomaly = anomaly.mahalanobis_distance()

    print(f"Features after anomaly: {X_with_anomaly.shape[1]}")

    # Step 5: Group features
    print("\n" + "="*70)
    print("STEP 5: GROUP FEATURES")
    print("="*70)

    groups = GroupFeatures(X_train, y_train)
    X_engineered_all = groups.create_group_statistics()

    # Combine all
    X_engineered_all = pd.concat([
        X_with_interactions,
        X_with_regimes.drop(X_train.columns, axis=1, errors='ignore'),
        X_with_instability.drop(X_train.columns, axis=1, errors='ignore'),
        X_with_anomaly.drop(X_train.columns, axis=1, errors='ignore'),
        X_engineered_all.drop(X_train.columns, axis=1, errors='ignore')
    ], axis=1)

    # Remove duplicates
    X_engineered_all = X_engineered_all.loc[:, ~X_engineered_all.columns.duplicated()]

    print(f"\nTotal features after engineering: {X_engineered_all.shape[1]}")
    print(f"  Original: {X_train.shape[1]}")
    print(f"  Engineered: {X_engineered_all.shape[1] - X_train.shape[1]}")

    # Step 6: Feature selection
    print("\n" + "="*70)
    print("STEP 6: FEATURE SELECTION")
    print("="*70)

    selector = FeatureSelector(X_train, X_engineered_all, y_train)
    top_features, importances = selector.test_feature_importance(n_features_to_select=40)

    # Save results
    X_engineered_all.to_csv('X_engineered_full.csv', index=False)
    X_engineered_all[top_features].to_csv('X_engineered_selected.csv', index=False)

    print(f"\n✓ Engineered features saved to X_engineered_full.csv")
    print(f"✓ Selected features saved to X_engineered_selected.csv")

    summary = {
        'total_engineered': X_engineered_all.shape[1] - X_train.shape[1],
        'total_features': X_engineered_all.shape[1],
        'selected_features': len(top_features),
        'top_features': top_features[:10],
    }

    return summary


if __name__ == "__main__":
    run_part5_feature_engineering()
