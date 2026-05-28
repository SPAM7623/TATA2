"""
PART 2: QUICK BASELINE XGBOOST/LIGHTGBM
Alpha Defect Prediction in Hot Rolling Mills

Goal: Establish realistic deployment baseline quickly

Key Insights To Extract:
- Is defect signal learnable?
- Does strong overlap exist?
- Are probabilities compressed?
- Is threshold sensitivity high?
- Are predictions unstable across folds?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (roc_auc_score, pr_auc_score, precision_score,
                            recall_score, f1_score, roc_curve, precision_recall_curve,
                            confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# PART 2 CHECKLIST
# =====================================================================
CHECKLIST = {
    "2.1_data_loading": False,
    "2.2_data_preparation": False,
    "2.3_xgboost_baseline": False,
    "2.4_lightgbm_baseline": False,
    "2.5_cross_validation": False,
    "2.6_probability_analysis": False,
    "2.7_threshold_sensitivity": False,
    "2.8_baseline_comparison": False,
}

class BaselineModeling:
    """Quick baseline models for Alpha defect prediction"""

    def __init__(self, train_path='train.csv', test_path='test.csv'):
        self.train_path = train_path
        self.test_path = test_path
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.models = {}
        self.insights = {}

    def load_and_prepare_data(self):
        """2.1-2.2: Load and prepare data"""
        print("\n" + "="*70)
        print("2.1-2.2 DATA LOADING & PREPARATION")
        print("="*70)

        # Load Part 1 insights
        from insights_manager import InsightsManager
        manager = InsightsManager()
        part1_insights = manager.get_part1_insights()

        if part1_insights:
            print("\n✓ Using Part 1 insights:")
            print(f"  Class imbalance ratio: {part1_insights.get('class_imbalance_ratio', 'N/A'):.2f}:1")
            print(f"  Unstable features to monitor: {part1_insights.get('unstable_features', [])[:3]}")
            self.insights['part1_context'] = part1_insights

        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)

        # Separate features and target
        self.X_train = self.train_df.drop(['CoilID', 'Y'], axis=1)
        self.y_train = self.train_df['Y']
        self.X_test = self.test_df.drop(['CoilID'], axis=1)

        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Target distribution: {self.y_train.value_counts().to_dict()}")

        CHECKLIST["2.1_data_loading"] = True
        CHECKLIST["2.2_data_preparation"] = True

    def train_xgboost_baseline(self):
        """2.3: Train XGBoost baseline"""
        print("\n" + "="*70)
        print("2.3 XGBOOST BASELINE MODEL")
        print("="*70)

        # Basic XGBoost - no class weighting yet
        xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            n_jobs=-1
        )

        xgb_model.fit(self.X_train, self.y_train)
        self.models['xgboost'] = xgb_model

        print("XGBoost model trained")
        print(f"Feature importance (top 10):")

        importances = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        for idx, row in importances.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.6f}")

        self.insights['xgb_importances'] = importances

        CHECKLIST["2.3_xgboost_baseline"] = True

    def train_lightgbm_baseline(self):
        """2.4: Train LightGBM baseline"""
        print("\n" + "="*70)
        print("2.4 LIGHTGBM BASELINE MODEL")
        print("="*70)

        lgb_model = LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )

        lgb_model.fit(self.X_train, self.y_train)
        self.models['lightgbm'] = lgb_model

        print("LightGBM model trained")
        print(f"Feature importance (top 10):")

        importances = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        for idx, row in importances.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.6f}")

        self.insights['lgb_importances'] = importances

        CHECKLIST["2.4_lightgbm_baseline"] = True

    def cross_validation_evaluation(self):
        """2.5: Stratified cross-validation evaluation"""
        print("\n" + "="*70)
        print("2.5 CROSS-VALIDATION EVALUATION")
        print("="*70)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = {
            'xgboost': [],
            'lightgbm': []
        }

        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            # XGBoost
            xgb_cv = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                                  verbosity=0, n_jobs=-1)
            xgb_cv.fit(X_fold_train, y_fold_train)
            xgb_proba = xgb_cv.predict_proba(X_fold_val)[:, 1]

            # LightGBM
            lgb_cv = LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42,
                                   verbosity=-1, n_jobs=-1)
            lgb_cv.fit(X_fold_train, y_fold_train)
            lgb_proba = lgb_cv.predict_proba(X_fold_val)[:, 1]

            # Metrics
            for model_name, proba in [('xgboost', xgb_proba), ('lightgbm', lgb_proba)]:
                metrics = {
                    'fold': fold + 1,
                    'roc_auc': roc_auc_score(y_fold_val, proba),
                    'pr_auc': pr_auc_score(y_fold_val, proba),
                    'precision': precision_score(y_fold_val, (proba > 0.5).astype(int), zero_division=0),
                    'recall': recall_score(y_fold_val, (proba > 0.5).astype(int), zero_division=0),
                    'f1': f1_score(y_fold_val, (proba > 0.5).astype(int), zero_division=0),
                }
                fold_results[model_name].append(metrics)

        # Summary
        for model_name in ['xgboost', 'lightgbm']:
            results_df = pd.DataFrame(fold_results[model_name])
            print(f"\n{model_name.upper()} Cross-Validation Results:")
            print(results_df)
            print(f"\nMean Metrics:")
            print(results_df.drop('fold', axis=1).mean())

            self.insights[f'{model_name}_cv_results'] = results_df

        CHECKLIST["2.5_cross_validation"] = True

    def probability_distribution_analysis(self):
        """2.6: Analyze probability predictions"""
        print("\n" + "="*70)
        print("2.6 PROBABILITY DISTRIBUTION ANALYSIS")
        print("="*70)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        xgb_proba_all = []
        lgb_proba_all = []
        y_val_all = []

        for train_idx, val_idx in skf.split(self.X_train, self.y_train):
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

            # XGBoost
            xgb_cv = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                                  verbosity=0, n_jobs=-1)
            xgb_cv.fit(X_fold_train, y_fold_train)
            xgb_proba = xgb_cv.predict_proba(X_fold_val)[:, 1]
            xgb_proba_all.extend(xgb_proba)

            # LightGBM
            lgb_cv = LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8, random_state=42,
                                   verbosity=-1, n_jobs=-1)
            lgb_cv.fit(X_fold_train, y_fold_train)
            lgb_proba = lgb_cv.predict_proba(X_fold_val)[:, 1]
            lgb_proba_all.extend(lgb_proba)

            y_val_all.extend(y_fold_val)

        xgb_proba_all = np.array(xgb_proba_all)
        lgb_proba_all = np.array(lgb_proba_all)
        y_val_all = np.array(y_val_all)

        # Analysis
        print("\nXGBoost Probability Statistics:")
        print(f"  Normal class (0) mean: {xgb_proba_all[y_val_all == 0].mean():.4f}")
        print(f"  Defect class (1) mean: {xgb_proba_all[y_val_all == 1].mean():.4f}")
        print(f"  Separation: {abs(xgb_proba_all[y_val_all == 1].mean() - xgb_proba_all[y_val_all == 0].mean()):.4f}")

        print("\nLightGBM Probability Statistics:")
        print(f"  Normal class (0) mean: {lgb_proba_all[y_val_all == 0].mean():.4f}")
        print(f"  Defect class (1) mean: {lgb_proba_all[y_val_all == 1].mean():.4f}")
        print(f"  Separation: {abs(lgb_proba_all[y_val_all == 1].mean() - lgb_proba_all[y_val_all == 0].mean()):.4f}")

        self.insights['xgb_proba'] = xgb_proba_all
        self.insights['lgb_proba'] = lgb_proba_all
        self.insights['y_val'] = y_val_all

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # XGBoost distributions
        axes[0, 0].hist(xgb_proba_all[y_val_all == 0], bins=50, alpha=0.6, label='Normal')
        axes[0, 0].hist(xgb_proba_all[y_val_all == 1], bins=50, alpha=0.6, label='Defect')
        axes[0, 0].set_xlabel('Probability')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('XGBoost - Probability Distribution')
        axes[0, 0].legend()

        # LightGBM distributions
        axes[0, 1].hist(lgb_proba_all[y_val_all == 0], bins=50, alpha=0.6, label='Normal')
        axes[0, 1].hist(lgb_proba_all[y_val_all == 1], bins=50, alpha=0.6, label='Defect')
        axes[0, 1].set_xlabel('Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('LightGBM - Probability Distribution')
        axes[0, 1].legend()

        # ROC curves
        fpr_xgb, tpr_xgb, _ = roc_curve(y_val_all, xgb_proba_all)
        fpr_lgb, tpr_lgb, _ = roc_curve(y_val_all, lgb_proba_all)

        axes[1, 0].plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={roc_auc_score(y_val_all, xgb_proba_all):.4f})')
        axes[1, 0].plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC={roc_auc_score(y_val_all, lgb_proba_all):.4f})')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curves')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # PR curves
        precision_xgb, recall_xgb, _ = precision_recall_curve(y_val_all, xgb_proba_all)
        precision_lgb, recall_lgb, _ = precision_recall_curve(y_val_all, lgb_proba_all)

        axes[1, 1].plot(recall_xgb, precision_xgb, label=f'XGBoost (PR-AUC={pr_auc_score(y_val_all, xgb_proba_all):.4f})')
        axes[1, 1].plot(recall_lgb, precision_lgb, label=f'LightGBM (PR-AUC={pr_auc_score(y_val_all, lgb_proba_all):.4f})')
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision-Recall Curves')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('13_probability_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 13_probability_analysis.png")

        CHECKLIST["2.6_probability_analysis"] = True

    def threshold_sensitivity_analysis(self):
        """2.7: Threshold sensitivity analysis"""
        print("\n" + "="*70)
        print("2.7 THRESHOLD SENSITIVITY ANALYSIS")
        print("="*70)

        xgb_proba = self.insights['xgb_proba']
        lgb_proba = self.insights['lgb_proba']
        y_val = self.insights['y_val']

        thresholds = np.linspace(0, 1, 101)
        results = {'threshold': [], 'xgb_precision': [], 'xgb_recall': [], 'xgb_f1': [],
                   'lgb_precision': [], 'lgb_recall': [], 'lgb_f1': []}

        for threshold in thresholds:
            xgb_pred = (xgb_proba >= threshold).astype(int)
            lgb_pred = (lgb_proba >= threshold).astype(int)

            results['threshold'].append(threshold)
            results['xgb_precision'].append(precision_score(y_val, xgb_pred, zero_division=0))
            results['xgb_recall'].append(recall_score(y_val, xgb_pred, zero_division=0))
            results['xgb_f1'].append(f1_score(y_val, xgb_pred, zero_division=0))
            results['lgb_precision'].append(precision_score(y_val, lgb_pred, zero_division=0))
            results['lgb_recall'].append(recall_score(y_val, lgb_pred, zero_division=0))
            results['lgb_f1'].append(f1_score(y_val, lgb_pred, zero_division=0))

        results_df = pd.DataFrame(results)
        self.insights['threshold_analysis'] = results_df

        # Find optimal thresholds
        best_f1_xgb = results_df.loc[results_df['xgb_f1'].idxmax()]
        best_f1_lgb = results_df.loc[results_df['lgb_f1'].idxmax()]

        print(f"\nOptimal F1 Threshold - XGBoost: {best_f1_xgb['threshold']:.3f} (F1={best_f1_xgb['xgb_f1']:.4f})")
        print(f"Optimal F1 Threshold - LightGBM: {best_f1_lgb['threshold']:.3f} (F1={best_f1_lgb['lgb_f1']:.4f})")

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(results_df['threshold'], results_df['xgb_precision'], label='Precision')
        axes[0].plot(results_df['threshold'], results_df['xgb_recall'], label='Recall')
        axes[0].plot(results_df['threshold'], results_df['xgb_f1'], label='F1')
        axes[0].axvline(best_f1_xgb['threshold'], color='r', linestyle='--', label=f"Optimal={best_f1_xgb['threshold']:.3f}")
        axes[0].set_xlabel('Threshold')
        axes[0].set_ylabel('Score')
        axes[0].set_title('XGBoost - Threshold Sensitivity')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(results_df['threshold'], results_df['lgb_precision'], label='Precision')
        axes[1].plot(results_df['threshold'], results_df['lgb_recall'], label='Recall')
        axes[1].plot(results_df['threshold'], results_df['lgb_f1'], label='F1')
        axes[1].axvline(best_f1_lgb['threshold'], color='r', linestyle='--', label=f"Optimal={best_f1_lgb['threshold']:.3f}")
        axes[1].set_xlabel('Threshold')
        axes[1].set_ylabel('Score')
        axes[1].set_title('LightGBM - Threshold Sensitivity')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('14_threshold_sensitivity.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 14_threshold_sensitivity.png")

        CHECKLIST["2.7_threshold_sensitivity"] = True

    def generate_baseline_report(self):
        """Generate baseline model report"""
        print("\n" + "="*70)
        print("PART 2: BASELINE INSIGHTS SUMMARY")
        print("="*70)

        xgb_importances = self.insights['xgb_importances']
        y_val = self.insights['y_val']
        xgb_proba = self.insights['xgb_proba']

        report = f"""
BASELINE MODEL INSIGHTS
=======================

1. DEFECT SIGNAL LEARNABILITY:
   - ROC-AUC Score: {roc_auc_score(y_val, xgb_proba):.4f}
   - PR-AUC Score: {pr_auc_score(y_val, xgb_proba):.4f}
   - Signal is {'STRONG' if roc_auc_score(y_val, xgb_proba) > 0.8 else 'MODERATE' if roc_auc_score(y_val, xgb_proba) > 0.6 else 'WEAK'}

2. PROBABILITY COMPRESSION:
   - Normal class mean probability: {xgb_proba[y_val == 0].mean():.4f}
   - Defect class mean probability: {xgb_proba[y_val == 1].mean():.4f}
   - Separation: {abs(xgb_proba[y_val == 1].mean() - xgb_proba[y_val == 0].mean()):.4f}
   - Overlap Status: {'HIGH OVERLAP' if abs(xgb_proba[y_val == 1].mean() - xgb_proba[y_val == 0].mean()) < 0.3 else 'MODERATE' if abs(xgb_proba[y_val == 1].mean() - xgb_proba[y_val == 0].mean()) < 0.6 else 'GOOD SEPARATION'}

3. TOP PREDICTIVE FEATURES:
   {xgb_importances.head(10).to_string()}

4. THRESHOLD SENSITIVITY:
   - Threshold highly influences precision-recall tradeoff
   - Optimization needed for deployment

5. RECOMMENDATIONS FOR IMPROVEMENT:
   ✓ Investigate feature engineering (Part 5)
   ✓ Analyze prediction errors (Part 3)
   ✓ Consider imbalance handling (Part 6)
   ✓ Optimize threshold for deployment (Part 7)
"""

        print(report)

        with open('PART2_INSIGHTS.txt', 'w') as f:
            f.write(report)

        print("✓ Saved: PART2_INSIGHTS.txt")

        CHECKLIST["2.8_baseline_comparison"] = True

    def run_baseline_modeling(self):
        """Execute all baseline modeling steps"""
        self.load_and_prepare_data()
        self.train_xgboost_baseline()
        self.train_lightgbm_baseline()
        self.cross_validation_evaluation()
        self.probability_distribution_analysis()
        self.threshold_sensitivity_analysis()
        self.generate_baseline_report()

        print("\n" + "="*70)
        print("PART 2 COMPLETION CHECKLIST")
        print("="*70)
        for step, completed in CHECKLIST.items():
            status = "✓" if completed else "✗"
            print(f"{status} {step}")

        return self.insights

if __name__ == "__main__":
    from insights_manager import InsightsManager

    baseline = BaselineModeling(train_path='train.csv', test_path='test.csv')
    insights = baseline.run_baseline_modeling()

    # Save insights for downstream parts
    manager = InsightsManager()
    manager.set_part2_insights(insights)

    print("\n" + "="*70)
    print("PART 2 COMPLETE")
    print("="*70)
    print("\nBaseline models trained and evaluated.")
    print("✓ Insights propagated to downstream parts")
    print("Ready for Part 3: SHAP + Error Analysis")
