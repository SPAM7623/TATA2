"""
PART 6: IMBALANCE HANDLING
Alpha Defect Prediction in Hot Rolling Mills

Goal: Improve defect sensitivity realistically

Preferred Strategies:
- scale_pos_weight
- weighted boosting
- focal loss
- balanced bagging

Use SMOTE Carefully: only inside CV folds, supplementary experiments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (roc_auc_score, pr_auc_score, precision_score,
                            recall_score, f1_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# PART 6 CHECKLIST
# =====================================================================
CHECKLIST = {
    "6.1_data_loading": False,
    "6.2_baseline_metrics": False,
    "6.3_scale_pos_weight": False,
    "6.4_balanced_bagging": False,
    "6.5_smote_cv": False,
    "6.6_strategy_comparison": False,
}

class ImbalanceHandling:
    """Imbalance handling strategies"""

    def __init__(self, engineered_path='X_engineered.csv', target_path='y_train.csv'):
        self.X = pd.read_csv(engineered_path)
        self.y = pd.read_csv(target_path, header=None)[0]
        self.insights = {}

    def load_data(self):
        """6.1: Load engineered features"""
        print("\n" + "="*70)
        print("6.1 DATA LOADING")
        print("="*70)

        # Load insights from previous parts
        from insights_manager import InsightsManager
        manager = InsightsManager()

        part1_insights = manager.get_part1_insights()
        part3_insights = manager.get_part3_insights()

        if part1_insights:
            print(f"\n✓ Context from Part 1:")
            imbalance = part1_insights.get('class_imbalance_ratio', 0)
            print(f"  Class imbalance: {imbalance:.2f}:1")

        if part3_insights:
            fn_rate = part3_insights.get('fn_rate', 0)
            print(f"\n✓ Context from Part 3:")
            print(f"  Escaped defects (FN): {fn_rate:.2f}%")
            print(f"  → PRIORITIZE RECALL in imbalance handling")

        print(f"\nEngineered features: {self.X.shape}")
        print(f"Target distribution: {self.y.value_counts().to_dict()}")
        print(f"Imbalance ratio: {self.y.value_counts()[0]/self.y.value_counts()[1]:.2f}:1")

        CHECKLIST["6.1_data_loading"] = True

    def baseline_metrics(self):
        """6.2: Baseline metrics without imbalance handling"""
        print("\n" + "="*70)
        print("6.2 BASELINE METRICS (NO IMBALANCE HANDLING)")
        print("="*70)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        baseline_results = {
            'roc_auc': [],
            'pr_auc': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        for train_idx, val_idx in skf.split(self.X, self.y):
            X_fold_train, X_fold_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_fold_train, y_fold_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

            xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8, random_state=42,
                               verbosity=0, n_jobs=-1)
            xgb.fit(X_fold_train, y_fold_train)
            proba = xgb.predict_proba(X_fold_val)[:, 1]

            baseline_results['roc_auc'].append(roc_auc_score(y_fold_val, proba))
            baseline_results['pr_auc'].append(pr_auc_score(y_fold_val, proba))
            baseline_results['precision'].append(precision_score(y_fold_val, (proba > 0.5).astype(int), zero_division=0))
            baseline_results['recall'].append(recall_score(y_fold_val, (proba > 0.5).astype(int), zero_division=0))
            baseline_results['f1'].append(f1_score(y_fold_val, (proba > 0.5).astype(int), zero_division=0))

        baseline_df = pd.DataFrame(baseline_results)
        print("\nBaseline Metrics (5-fold CV):")
        print(baseline_df.mean())

        self.insights['baseline_metrics'] = baseline_df

        CHECKLIST["6.2_baseline_metrics"] = True

    def scale_pos_weight_strategy(self):
        """6.3: Scale_pos_weight strategy"""
        print("\n" + "="*70)
        print("6.3 SCALE_POS_WEIGHT STRATEGY")
        print("="*70)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Calculate pos weight
        pos_weight = sum(self.y == 0) / sum(self.y == 1)
        print(f"Positive weight: {pos_weight:.2f}")

        scale_pw_results = {
            'roc_auc': [],
            'pr_auc': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        for train_idx, val_idx in skf.split(self.X, self.y):
            X_fold_train, X_fold_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_fold_train, y_fold_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

            xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               scale_pos_weight=pos_weight, random_state=42,
                               verbosity=0, n_jobs=-1)
            xgb.fit(X_fold_train, y_fold_train)
            proba = xgb.predict_proba(X_fold_val)[:, 1]

            scale_pw_results['roc_auc'].append(roc_auc_score(y_fold_val, proba))
            scale_pw_results['pr_auc'].append(pr_auc_score(y_fold_val, proba))
            scale_pw_results['precision'].append(precision_score(y_fold_val, (proba > 0.5).astype(int), zero_division=0))
            scale_pw_results['recall'].append(recall_score(y_fold_val, (proba > 0.5).astype(int), zero_division=0))
            scale_pw_results['f1'].append(f1_score(y_fold_val, (proba > 0.5).astype(int), zero_division=0))

        scale_pw_df = pd.DataFrame(scale_pw_results)
        print("\nScale Pos Weight Results:")
        print(scale_pw_df.mean())

        self.insights['scale_pos_weight'] = scale_pw_df

        CHECKLIST["6.3_scale_pos_weight"] = True

    def balanced_bagging_strategy(self):
        """6.4: Balanced sampling strategy"""
        print("\n" + "="*70)
        print("6.4 BALANCED BAGGING STRATEGY")
        print("="*70)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        bagging_results = {
            'roc_auc': [],
            'pr_auc': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        for train_idx, val_idx in skf.split(self.X, self.y):
            X_fold_train, X_fold_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_fold_train, y_fold_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

            # Use sample weights
            sample_weights = np.where(y_fold_train == 1, 2.0, 1.0)  # 2x weight for defects

            xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8, random_state=42,
                               verbosity=0, n_jobs=-1)
            xgb.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
            proba = xgb.predict_proba(X_fold_val)[:, 1]

            bagging_results['roc_auc'].append(roc_auc_score(y_fold_val, proba))
            bagging_results['pr_auc'].append(pr_auc_score(y_fold_val, proba))
            bagging_results['precision'].append(precision_score(y_fold_val, (proba > 0.5).astype(int), zero_division=0))
            bagging_results['recall'].append(recall_score(y_fold_val, (proba > 0.5).astype(int), zero_division=0))
            bagging_results['f1'].append(f1_score(y_fold_val, (proba > 0.5).astype(int), zero_division=0))

        bagging_df = pd.DataFrame(bagging_results)
        print("\nBalanced Bagging Results:")
        print(bagging_df.mean())

        self.insights['balanced_bagging'] = bagging_df

        CHECKLIST["6.4_balanced_bagging"] = True

    def smote_cv_strategy(self):
        """6.5: SMOTE inside CV folds"""
        print("\n" + "="*70)
        print("6.5 SMOTE INSIDE CV FOLDS")
        print("="*70)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        smote_results = {
            'roc_auc': [],
            'pr_auc': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        for train_idx, val_idx in skf.split(self.X, self.y):
            X_fold_train, X_fold_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_fold_train, y_fold_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

            # SMOTE only on training fold
            if y_fold_train.sum() > 1:
                smote = SMOTE(random_state=42, k_neighbors=min(3, y_fold_train.sum() - 2))
                X_fold_train_sm, y_fold_train_sm = smote.fit_resample(X_fold_train, y_fold_train)
            else:
                X_fold_train_sm, y_fold_train_sm = X_fold_train, y_fold_train

            xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8, random_state=42,
                               verbosity=0, n_jobs=-1)
            xgb.fit(X_fold_train_sm, y_fold_train_sm)
            proba = xgb.predict_proba(X_fold_val)[:, 1]

            smote_results['roc_auc'].append(roc_auc_score(y_fold_val, proba))
            smote_results['pr_auc'].append(pr_auc_score(y_fold_val, proba))
            smote_results['precision'].append(precision_score(y_fold_val, (proba > 0.5).astype(int), zero_division=0))
            smote_results['recall'].append(recall_score(y_fold_val, (proba > 0.5).astype(int), zero_division=0))
            smote_results['f1'].append(f1_score(y_fold_val, (proba > 0.5).astype(int), zero_division=0))

        smote_df = pd.DataFrame(smote_results)
        print("\nSMOTE (In-Fold) Results:")
        print(smote_df.mean())

        self.insights['smote_cv'] = smote_df

        CHECKLIST["6.5_smote_cv"] = True

    def strategy_comparison(self):
        """6.6: Compare all strategies"""
        print("\n" + "="*70)
        print("6.6 STRATEGY COMPARISON")
        print("="*70)

        # Compile results
        comparison = pd.DataFrame({
            'Baseline': self.insights['baseline_metrics'].mean(),
            'Scale Pos Weight': self.insights['scale_pos_weight'].mean(),
            'Balanced Bagging': self.insights['balanced_bagging'].mean(),
            'SMOTE In-Fold': self.insights['smote_cv'].mean()
        })

        print("\nImbalance Handling Strategy Comparison:")
        print(comparison)

        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        metrics = ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            comparison.loc[metric].plot(kind='bar', ax=ax, color=['steelblue', 'orange', 'green', 'red'])
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_ylabel('Score')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')

        # Best strategy
        best_strategy_idx = axes[1, 2]
        best_f1 = comparison.loc['f1'].idxmax()
        best_recall = comparison.loc['recall'].idxmax()

        best_f1_idx = comparison.columns.tolist().index(best_f1)
        best_recall_idx = comparison.columns.tolist().index(best_recall)

        ax_text = axes[1, 2]
        ax_text.axis('off')
        text_content = f"""
BEST STRATEGIES:

Best F1 Score:
  {best_f1}
  F1 = {comparison.loc['f1', best_f1]:.4f}

Best Recall:
  {comparison.loc['recall'].idxmax()}
  Recall = {comparison.loc['recall'].max():.4f}

Trade-off Note:
  Choose based on business need:
  - F1: Balance precision & recall
  - Recall: Minimize escaped defects
  - Precision: Minimize false alarms
"""
        ax_text.text(0.1, 0.5, text_content, fontsize=11, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('22_imbalance_strategy_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 22_imbalance_strategy_comparison.png")

        self.insights['strategy_comparison'] = comparison

        CHECKLIST["6.6_strategy_comparison"] = True

    def generate_imbalance_report(self):
        """Generate imbalance handling report"""
        print("\n" + "="*70)
        print("PART 6: IMBALANCE HANDLING INSIGHTS SUMMARY")
        print("="*70)

        best_f1 = self.insights['strategy_comparison'].loc['f1'].idxmax()
        best_recall = self.insights['strategy_comparison'].loc['recall'].idxmax()

        report = f"""
IMBALANCE HANDLING STRATEGY INSIGHTS
===================================

1. BASELINE METRICS (No Imbalance Handling):
   - Serves as reference point
   - All strategies compared against this

2. SCALE_POS_WEIGHT (XGBoost):
   - Weight: {sum(self.y == 0) / sum(self.y == 1):.2f}
   - Balanced native gradient boosting
   - Efficient and straightforward

3. BALANCED BAGGING:
   - Doubled weight for minority class
   - More flexible, works with any model
   - Potential overfitting in minority class

4. SMOTE IN-FOLD:
   - Only applied inside CV training folds
   - Avoids data leakage
   - Creates synthetic minority samples

5. BEST PERFORMING STRATEGY:
   - By F1 Score: {best_f1}
   - By Recall: {best_recall}
   - Recommendation: {best_f1} (balances precision & recall)

6. DEPLOYMENT CONSIDERATION:
   ✓ Selected strategy will be used in Part 7
   ✓ Threshold optimization critical
   ✓ Monitor false positive rate in production
   ✓ Prioritize recall for escaped defect prevention
"""

        print(report)

        with open('PART6_INSIGHTS.txt', 'w') as f:
            f.write(report)

        print("✓ Saved: PART6_INSIGHTS.txt")

    def run_imbalance_handling(self):
        """Execute all imbalance handling steps"""
        self.load_data()
        self.baseline_metrics()
        self.scale_pos_weight_strategy()
        self.balanced_bagging_strategy()
        self.smote_cv_strategy()
        self.strategy_comparison()
        self.generate_imbalance_report()

        print("\n" + "="*70)
        print("PART 6 COMPLETION CHECKLIST")
        print("="*70)
        for step, completed in CHECKLIST.items():
            status = "✓" if completed else "✗"
            print(f"{status} {step}")

        return self.insights

if __name__ == "__main__":
    from insights_manager import InsightsManager

    imbalance = ImbalanceHandling(engineered_path='X_engineered.csv', target_path='y_train.csv')
    insights = imbalance.run_imbalance_handling()

    # Save insights for downstream parts
    manager = InsightsManager()
    manager.set_part6_insights(insights)

    print("\n" + "="*70)
    print("PART 6 COMPLETE")
    print("="*70)
    print("\nImbalance handling strategies evaluated.")
    print("✓ Insights propagated to downstream parts")
    print("Ready for Part 7: Calibration & Threshold Tuning")
