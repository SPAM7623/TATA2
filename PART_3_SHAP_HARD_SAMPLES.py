"""
PART 3: SHAP ERROR ANALYSIS (Phase 3 & 5 - Enhanced Hard Sample Analysis)
Alpha Defect Prediction in Hot Rolling Mills

OPTIMIZATION FOCUS:
- Feature interaction detection via SHAP
- Hard sample analysis (boundary, FN, FP)
- Feature importance consistency across folds
- Error pattern profiling
- Targeted improvement recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
import shap
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# SHAP INTERACTION ANALYZER
# =====================================================================

class SHAPInteractionAnalyzer:
    """Analyze feature interactions via SHAP"""

    def __init__(self, model, X_data, y_data=None):
        self.model = model
        self.X_data = X_data
        self.y_data = y_data
        self.explainer = None
        self.shap_values = None

    def compute_shap_values(self, sample_size=100):
        """Compute SHAP values for understanding"""
        print("\n" + "-"*70)
        print("COMPUTE SHAP VALUES")
        print("-"*70)

        # Use sample for speed if data is large
        if len(self.X_data) > sample_size:
            indices = np.random.choice(len(self.X_data), sample_size, replace=False)
            X_sample = self.X_data.iloc[indices]
            print(f"Using sample of {sample_size} instances (full data: {len(self.X_data)})")
        else:
            X_sample = self.X_data

        try:
            # Use TreeExplainer for XGBoost
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(X_sample)

            # Handle XGBoost binary classification
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]  # Use positive class

            print(f"✓ SHAP values computed: {self.shap_values.shape}")
            return self.shap_values

        except Exception as e:
            print(f"⚠ Error computing SHAP values: {e}")
            return None

    def compute_interaction_indices(self, top_n=20):
        """Compute SHAP interaction indices"""
        print("\n" + "-"*70)
        print("COMPUTE FEATURE INTERACTIONS (SHAP)")
        print("-"*70)

        if self.shap_values is None:
            print("✗ SHAP values not computed yet")
            return []

        # Compute pairwise feature interactions
        interactions = []

        features = self.X_data.columns

        # Use absolute SHAP values for importance
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        important_features = np.argsort(mean_abs_shap)[-top_n:][::-1]

        print(f"\nAnalyzing interactions among top {len(important_features)} features...")

        for i in range(len(important_features)):
            for j in range(i+1, len(important_features)):
                feat_i = important_features[i]
                feat_j = important_features[j]

                # Correlation between SHAP values of feature i and values of feature j
                feat_i_shap = self.shap_values[:, feat_i]
                feat_j_values = self.X_data.iloc[:, feat_j]

                # Check if interaction exists
                try:
                    correlation = np.corrcoef(feat_i_shap, feat_j_values)[0, 1]
                    interaction_strength = abs(correlation)

                    if interaction_strength > 0.1:  # Threshold
                        interactions.append({
                            'feature_1': features[feat_i],
                            'feature_2': features[feat_j],
                            'interaction_strength': interaction_strength,
                            'correlation': correlation
                        })
                except:
                    pass

        interactions_df = pd.DataFrame(interactions).sort_values('interaction_strength', ascending=False)

        print(f"\nTop feature interactions:")
        for idx, row in interactions_df.head(10).iterrows():
            print(f"  {row['feature_1']} × {row['feature_2']}: {row['interaction_strength']:.4f}")

        return interactions_df


# =====================================================================
# HARD SAMPLE ANALYZER
# =====================================================================

class HardSampleAnalyzer:
    """Detect and analyze hard/ambiguous samples"""

    def __init__(self, X_data, y_true, y_pred_proba):
        self.X_data = X_data
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.y_pred = (y_pred_proba >= 0.5).astype(int)
        self.hard_samples_report = {}

    def identify_boundary_samples(self, margin=0.1):
        """Identify samples near decision boundary"""
        print("\n" + "-"*70)
        print("IDENTIFY BOUNDARY SAMPLES (Uncertainty = |Prob - 0.5|)")
        print("-"*70)

        uncertainty = np.abs(self.y_pred_proba - 0.5)
        boundary_mask = uncertainty < margin

        boundary_indices = np.where(boundary_mask)[0]

        print(f"\nBoundary samples (uncertainty < {margin}): {len(boundary_indices)} ({100*len(boundary_indices)/len(self.y_pred_proba):.2f}%)")

        if len(boundary_indices) > 0:
            boundary_uncertainty = uncertainty[boundary_indices]
            print(f"  Mean uncertainty: {boundary_uncertainty.mean():.4f}")
            print(f"  Std uncertainty: {boundary_uncertainty.std():.4f}")

            # Analyze characteristics
            boundary_data = self.X_data.iloc[boundary_indices]
            print(f"\nBoundary sample characteristics:")
            print(f"  Mean prediction prob: {self.y_pred_proba[boundary_indices].mean():.4f}")
            print(f"  Defect rate: {self.y_true.iloc[boundary_indices].mean():.2%}")

        self.hard_samples_report['boundary_samples'] = {
            'count': len(boundary_indices),
            'indices': boundary_indices,
            'avg_uncertainty': uncertainty[boundary_indices].mean() if len(boundary_indices) > 0 else 0
        }

        return boundary_indices

    def identify_false_negatives(self):
        """Identify escaped defects (y=1, pred=0)"""
        print("\n" + "-"*70)
        print("IDENTIFY FALSE NEGATIVES (Escaped Defects)")
        print("-"*70)

        fn_mask = (self.y_true == 1) & (self.y_pred == 0)
        fn_indices = np.where(fn_mask)[0]

        print(f"\nFalse negatives: {len(fn_indices)} ({100*len(fn_indices)/np.sum(self.y_true):.2f}% of defects)")

        if len(fn_indices) > 0:
            fn_probs = self.y_pred_proba[fn_indices]
            print(f"  Predicted prob range: [{fn_probs.min():.4f}, {fn_probs.max():.4f}]")
            print(f"  Mean predicted prob: {fn_probs.mean():.4f}")
            print(f"  Std predicted prob: {fn_probs.std():.4f}")

            fn_data = self.X_data.iloc[fn_indices]
            print(f"\nFN characteristics (compared to true positives):")
            tp_data = self.X_data.iloc[self.y_true == 1]

            for col in self.X_data.columns[:5]:  # Show first 5 features
                fn_mean = fn_data[col].mean()
                tp_mean = tp_data[col].mean()
                print(f"  {col}: FN={fn_mean:.4f}, TP_avg={tp_mean:.4f}, diff={fn_mean-tp_mean:.4f}")

        self.hard_samples_report['false_negatives'] = {
            'count': len(fn_indices),
            'indices': fn_indices,
            'rate': len(fn_indices) / np.sum(self.y_true) if np.sum(self.y_true) > 0 else 0,
            'avg_pred_prob': self.y_pred_proba[fn_indices].mean() if len(fn_indices) > 0 else 0
        }

        return fn_indices

    def identify_false_positives(self):
        """Identify false alarms (y=0, pred=1)"""
        print("\n" + "-"*70)
        print("IDENTIFY FALSE POSITIVES (False Alarms)")
        print("-"*70)

        fp_mask = (self.y_true == 0) & (self.y_pred == 1)
        fp_indices = np.where(fp_mask)[0]

        print(f"\nFalse positives: {len(fp_indices)} ({100*len(fp_indices)/(len(self.y_true)-np.sum(self.y_true)):.2f}% of normals)")

        if len(fp_indices) > 0:
            fp_probs = self.y_pred_proba[fp_indices]
            print(f"  Predicted prob range: [{fp_probs.min():.4f}, {fp_probs.max():.4f}]")
            print(f"  Mean predicted prob: {fp_probs.mean():.4f}")
            print(f"  Std predicted prob: {fp_probs.std():.4f}")

            fp_data = self.X_data.iloc[fp_indices]
            print(f"\nFP characteristics (compared to true negatives):")
            tn_data = self.X_data.iloc[self.y_true == 0]

            for col in self.X_data.columns[:5]:  # Show first 5 features
                fp_mean = fp_data[col].mean()
                tn_mean = tn_data[col].mean()
                print(f"  {col}: FP={fp_mean:.4f}, TN_avg={tn_mean:.4f}, diff={fp_mean-tn_mean:.4f}")

        self.hard_samples_report['false_positives'] = {
            'count': len(fp_indices),
            'indices': fp_indices,
            'rate': len(fp_indices) / (len(self.y_true) - np.sum(self.y_true)) if (len(self.y_true) - np.sum(self.y_true)) > 0 else 0,
            'avg_pred_prob': self.y_pred_proba[fp_indices].mean() if len(fp_indices) > 0 else 0
        }

        return fp_indices

    def analyze_hard_sample_patterns(self):
        """Analyze patterns in hard samples"""
        print("\n" + "-"*70)
        print("ANALYZE HARD SAMPLE PATTERNS")
        print("-"*70)

        fn_indices = self.hard_samples_report.get('false_negatives', {}).get('indices', [])
        fp_indices = self.hard_samples_report.get('false_positives', {}).get('indices', [])
        boundary_indices = self.hard_samples_report.get('boundary_samples', {}).get('indices', [])

        print(f"\nHard sample counts:")
        print(f"  False negatives (FN): {len(fn_indices)}")
        print(f"  False positives (FP): {len(fp_indices)}")
        print(f"  Boundary samples (B): {len(boundary_indices)}")
        print(f"  Total hard samples: {len(set(fn_indices) | set(fp_indices) | set(boundary_indices))}")

        # Overlap analysis
        fn_and_b = len(set(fn_indices) & set(boundary_indices))
        fp_and_b = len(set(fp_indices) & set(boundary_indices))

        print(f"\nOverlap:")
        print(f"  FN ∩ Boundary: {fn_and_b}")
        print(f"  FP ∩ Boundary: {fp_and_b}")

        return {
            'fn_count': len(fn_indices),
            'fp_count': len(fp_indices),
            'boundary_count': len(boundary_indices)
        }


# =====================================================================
# CONSISTENCY ANALYZER
# =====================================================================

class FeatureConsistencyAnalyzer:
    """Analyze feature importance consistency across folds"""

    def __init__(self, X_train, y_train, n_splits=5):
        self.X_train = X_train
        self.y_train = y_train
        self.n_splits = n_splits
        self.fold_importances = []

    def compute_importance_across_folds(self):
        """Train models on each fold and get importances"""
        print("\n" + "-"*70)
        print("COMPUTE FEATURE IMPORTANCE ACROSS FOLDS")
        print("-"*70)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            X_fold_train = self.X_train.iloc[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]

            # Train model
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )

            model.fit(X_fold_train, y_fold_train)

            importances = pd.Series(model.feature_importances_, index=self.X_train.columns)
            self.fold_importances.append(importances)

            print(f"Fold {fold_idx+1}: Top 3 features: {importances.nlargest(3).index.tolist()}")

        # Aggregate
        importances_df = pd.concat(self.fold_importances, axis=1)
        importances_df.columns = [f'fold_{i+1}' for i in range(self.n_splits)]
        importances_df['mean'] = importances_df.mean(axis=1)
        importances_df['std'] = importances_df.std(axis=1)
        importances_df['cv'] = importances_df['std'] / (importances_df['mean'] + 1e-6)
        importances_df = importances_df.sort_values('mean', ascending=False)

        print(f"\n✓ Feature consistency across folds:")
        print(f"Top 10 stable features:")
        for feature, row in importances_df.head(10).iterrows():
            print(f"  {feature}: mean={row['mean']:.4f}, cv={row['cv']:.4f}")

        return importances_df

    def identify_stable_vs_noisy_features(self, cv_threshold=0.5):
        """Separate stable from noisy features"""
        importances_df = pd.concat(self.fold_importances, axis=1)
        importances_df.columns = [f'fold_{i+1}' for i in range(self.n_splits)]
        importances_df['mean'] = importances_df.mean(axis=1)
        importances_df['std'] = importances_df.std(axis=1)
        importances_df['cv'] = importances_df['std'] / (importances_df['mean'] + 1e-6)

        stable = importances_df[importances_df['cv'] < cv_threshold].index.tolist()
        noisy = importances_df[importances_df['cv'] >= cv_threshold].index.tolist()

        print(f"\nStable features (cv < {cv_threshold}): {len(stable)}")
        print(f"Noisy features (cv >= {cv_threshold}): {len(noisy)}")

        return stable, noisy


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def run_part3_shap_hard_samples(train_path='train.csv', use_sample=True):
    """Execute Part 3: SHAP + Hard Sample Analysis"""

    print("\n" + "="*70)
    print("PART 3: SHAP ERROR ANALYSIS (Phase 3 & 5)")
    print("="*70)

    # Load data
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(['CoilID', 'Y'], axis=1)
    y_train = train_df['Y']

    print(f"\nData loaded: {X_train.shape}")
    print(f"Class distribution: {y_train.value_counts().to_dict()}")

    # Use sample for faster analysis if specified
    if use_sample and len(X_train) > 500:
        sample_size = 500
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_work = X_train.iloc[indices]
        y_work = y_train.iloc[indices]
        print(f"\nUsing sample of {sample_size} for SHAP analysis")
    else:
        X_work = X_train
        y_work = y_train

    # Train baseline model for analysis
    print("\n" + "-"*70)
    print("Train baseline model for analysis")
    print("-"*70)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_work, y_work)
    y_pred_proba = model.predict_proba(X_work)[:, 1]

    print("Model trained")

    # Step 1: SHAP interactions
    print("\n" + "="*70)
    print("STEP 1: FEATURE INTERACTIONS")
    print("="*70)

    shap_analyzer = SHAPInteractionAnalyzer(model, X_work, y_work)
    shap_analyzer.compute_shap_values(sample_size=100)
    interactions_df = shap_analyzer.compute_interaction_indices(top_n=10)

    # Step 2: Hard sample analysis
    print("\n" + "="*70)
    print("STEP 2: HARD SAMPLE ANALYSIS")
    print("="*70)

    hard_analyzer = HardSampleAnalyzer(X_work, y_work, y_pred_proba)
    hard_analyzer.identify_boundary_samples(margin=0.1)
    hard_analyzer.identify_false_negatives()
    hard_analyzer.identify_false_positives()
    hard_patterns = hard_analyzer.analyze_hard_sample_patterns()

    # Step 3: Consistency analysis
    print("\n" + "="*70)
    print("STEP 3: FEATURE CONSISTENCY")
    print("="*70)

    consistency_analyzer = FeatureConsistencyAnalyzer(X_train, y_train)
    importances_df = consistency_analyzer.compute_importance_across_folds()
    stable_features, noisy_features = consistency_analyzer.identify_stable_vs_noisy_features()

    # Summary
    print("\n" + "="*70)
    print("PART 3 SUMMARY")
    print("="*70)

    summary = {
        'shap_interactions': interactions_df.head(10).to_dict('records') if len(interactions_df) > 0 else [],
        'hard_samples': hard_patterns,
        'stable_features': stable_features[:10],
        'noisy_features': noisy_features[:10],
        'total_interactions': len(interactions_df),
    }

    print(f"\n✓ PART 3 ANALYSIS COMPLETE")
    print(f"  Feature interactions found: {len(interactions_df)}")
    print(f"  Hard samples: {hard_patterns.get('fn_count', 0)} FN + {hard_patterns.get('fp_count', 0)} FP + {hard_patterns.get('boundary_count', 0)} boundary")
    print(f"  Stable features: {len(stable_features)}")
    print(f"  Noisy features: {len(noisy_features)}")

    # Save to InsightsManager
    try:
        from insights_manager import InsightsManager
        manager = InsightsManager()

        # Prepare insights dict
        insights_dict = {
            'shap_interactions': interactions_df.head(5).to_dict('records') if len(interactions_df) > 0 else [],
            'fn_rate': hard_analyzer.hard_samples_report.get('false_negatives', {}).get('rate', 0) * 100,
            'fn_characteristics': [],
            'fp_rate': hard_analyzer.hard_samples_report.get('false_positives', {}).get('rate', 0) * 100,
            'fp_characteristics': [],
        }

        manager.set_part3_insights(insights_dict)
    except:
        print("\n⚠ Could not update InsightsManager")

    return summary


if __name__ == "__main__":
    run_part3_shap_hard_samples()
