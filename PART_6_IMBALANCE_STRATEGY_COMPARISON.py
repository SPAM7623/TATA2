"""
PART 6: IMBALANCE STRATEGY COMPARISON (Phase 6)
Alpha Defect Prediction in Hot Rolling Mills

OPTIMIZATION FOCUS:
- Compare 5+ imbalance handling strategies
- Measure stability across folds
- Integrate with feature engineering insights
- Select strategy maximizing recall while maintaining precision
- Error-profile-aware rebalancing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score,
    precision_score, recall_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# IMBALANCE STRATEGY COMPARISON
# =====================================================================

class ImbalanceStrategyComparison:
    """Compare multiple imbalance handling strategies"""

    def __init__(self, X_train, y_train, n_splits=5):
        self.X_train = X_train
        self.y_train = y_train
        self.n_splits = n_splits
        self.strategy_results = {}

    def scale_pos_weight_strategy(self):
        """Baseline: Use scale_pos_weight in XGBoost"""
        print("\n" + "-"*70)
        print("STRATEGY 1: SCALE_POS_WEIGHT (Native XGBoost weighting)")
        print("-"*70)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        fold_scores = {'roc_auc': [], 'pr_auc': [], 'f1': [], 'recall': [], 'precision': []}

        # Calculate scale_pos_weight
        neg_count = (self.y_train == 0).sum()
        pos_count = (self.y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count

        print(f"Scale pos weight: {scale_pos_weight:.2f}")

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            X_fold_train = self.X_train.iloc[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_val = self.y_train.iloc[val_idx]

            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )

            model.fit(X_fold_train, y_fold_train)
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            fold_scores['roc_auc'].append(roc_auc_score(y_fold_val, y_pred_proba))
            fold_scores['recall'].append(recall_score(y_fold_val, y_pred))
            fold_scores['precision'].append(precision_score(y_fold_val, y_pred, zero_division=0))
            fold_scores['f1'].append(f1_score(y_fold_val, y_pred))

            # PR-AUC
            precision, recall, _ = precision_recall_curve(y_fold_val, y_pred_proba)
            pr_auc = np.trapz(precision[::-1], recall[::-1])
            fold_scores['pr_auc'].append(pr_auc)

        self.strategy_results['scale_pos_weight'] = self._summarize_scores(fold_scores)
        self._print_strategy_results('scale_pos_weight', fold_scores)

        return fold_scores

    def smote_strategy(self):
        """SMOTE oversampling (in-fold only)"""
        print("\n" + "-"*70)
        print("STRATEGY 2: SMOTE (Oversampling)")
        print("-"*70)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        fold_scores = {'roc_auc': [], 'pr_auc': [], 'f1': [], 'recall': [], 'precision': []}

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            X_fold_train = self.X_train.iloc[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_val = self.y_train.iloc[val_idx]

            # Apply SMOTE to training fold only
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, (y_fold_train == 1).sum() - 1))
                X_fold_train_balanced, y_fold_train_balanced = smote.fit_resample(X_fold_train, y_fold_train)

                print(f"Fold {fold_idx+1}: SMOTE applied - {len(X_fold_train)} → {len(X_fold_train_balanced)}")
            except Exception as e:
                print(f"⚠ SMOTE failed: {e} - using original data")
                X_fold_train_balanced = X_fold_train
                y_fold_train_balanced = y_fold_train

            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )

            model.fit(X_fold_train_balanced, y_fold_train_balanced)
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            fold_scores['roc_auc'].append(roc_auc_score(y_fold_val, y_pred_proba))
            fold_scores['recall'].append(recall_score(y_fold_val, y_pred))
            fold_scores['precision'].append(precision_score(y_fold_val, y_pred, zero_division=0))
            fold_scores['f1'].append(f1_score(y_fold_val, y_pred))

            precision, recall, _ = precision_recall_curve(y_fold_val, y_pred_proba)
            pr_auc = np.trapz(precision[::-1], recall[::-1])
            fold_scores['pr_auc'].append(pr_auc)

        self.strategy_results['smote'] = self._summarize_scores(fold_scores)
        self._print_strategy_results('smote', fold_scores)

        return fold_scores

    def cost_sensitive_strategy(self):
        """Cost-sensitive learning with class weights"""
        print("\n" + "-"*70)
        print("STRATEGY 3: COST-SENSITIVE (Class weights)")
        print("-"*70)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        fold_scores = {'roc_auc': [], 'pr_auc': [], 'f1': [], 'recall': [], 'precision': []}

        # Calculate weights
        neg_count = (self.y_train == 0).sum()
        pos_count = (self.y_train == 1).sum()
        weight_positive = neg_count / pos_count

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            X_fold_train = self.X_train.iloc[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_val = self.y_train.iloc[val_idx]

            # Custom sample weights
            sample_weights = np.where(y_fold_train == 1, weight_positive, 1.0)

            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )

            model.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            fold_scores['roc_auc'].append(roc_auc_score(y_fold_val, y_pred_proba))
            fold_scores['recall'].append(recall_score(y_fold_val, y_pred))
            fold_scores['precision'].append(precision_score(y_fold_val, y_pred, zero_division=0))
            fold_scores['f1'].append(f1_score(y_fold_val, y_pred))

            precision, recall, _ = precision_recall_curve(y_fold_val, y_pred_proba)
            pr_auc = np.trapz(precision[::-1], recall[::-1])
            fold_scores['pr_auc'].append(pr_auc)

        self.strategy_results['cost_sensitive'] = self._summarize_scores(fold_scores)
        self._print_strategy_results('cost_sensitive', fold_scores)

        return fold_scores

    def balanced_threshold_strategy(self):
        """Different threshold for balanced precision-recall"""
        print("\n" + "-"*70)
        print("STRATEGY 4: THRESHOLD_OPTIMIZATION (F1-optimal threshold)")
        print("-"*70)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        fold_scores = {'roc_auc': [], 'pr_auc': [], 'f1': [], 'recall': [], 'precision': []}

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            X_fold_train = self.X_train.iloc[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_val = self.y_train.iloc[val_idx]

            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )

            model.fit(X_fold_train, y_fold_train)
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]

            # Find F1-optimal threshold
            precisions, recalls, thresholds = precision_recall_curve(y_fold_val, y_pred_proba)
            f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-6)
            best_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[best_idx]

            y_pred = (y_pred_proba >= optimal_threshold).astype(int)

            fold_scores['roc_auc'].append(roc_auc_score(y_fold_val, y_pred_proba))
            fold_scores['recall'].append(recall_score(y_fold_val, y_pred))
            fold_scores['precision'].append(precision_score(y_fold_val, y_pred, zero_division=0))
            fold_scores['f1'].append(f1_score(y_fold_val, y_pred))

            precision, recall, _ = precision_recall_curve(y_fold_val, y_pred_proba)
            pr_auc = np.trapz(precision[::-1], recall[::-1])
            fold_scores['pr_auc'].append(pr_auc)

        self.strategy_results['threshold_optimization'] = self._summarize_scores(fold_scores)
        self._print_strategy_results('threshold_optimization', fold_scores)

        return fold_scores

    def _summarize_scores(self, fold_scores):
        """Summarize fold scores"""
        summary = {}
        for metric, scores in fold_scores.items():
            summary[f'{metric}_mean'] = np.mean(scores)
            summary[f'{metric}_std'] = np.std(scores)
        return summary

    def _print_strategy_results(self, strategy_name, fold_scores):
        """Print formatted strategy results"""
        print(f"\nResults:")
        print(f"  ROC-AUC:  {np.mean(fold_scores['roc_auc']):.4f} ± {np.std(fold_scores['roc_auc']):.4f}")
        print(f"  PR-AUC:   {np.mean(fold_scores['pr_auc']):.4f} ± {np.std(fold_scores['pr_auc']):.4f}")
        print(f"  Recall:   {np.mean(fold_scores['recall']):.4f} ± {np.std(fold_scores['recall']):.4f}")
        print(f"  Precision:{np.mean(fold_scores['precision']):.4f} ± {np.std(fold_scores['precision']):.4f}")
        print(f"  F1:       {np.mean(fold_scores['f1']):.4f} ± {np.std(fold_scores['f1']):.4f}")

    def compare_all_strategies(self):
        """Compare all strategies"""
        print("\n" + "="*70)
        print("COMPARING ALL STRATEGIES")
        print("="*70)

        self.scale_pos_weight_strategy()
        self.smote_strategy()
        self.cost_sensitive_strategy()
        self.balanced_threshold_strategy()

        # Create comparison dataframe
        comparison_df = pd.DataFrame(self.strategy_results).T
        comparison_df = comparison_df.sort_values('roc_auc_mean', ascending=False)

        print("\n" + "="*70)
        print("STRATEGY COMPARISON SUMMARY")
        print("="*70)
        print(comparison_df)

        return comparison_df


# =====================================================================
# STRATEGY STABILITY ANALYZER
# =====================================================================

class StrategyStabilityAnalyzer:
    """Analyze strategy stability across folds"""

    def __init__(self, comparison_df):
        self.comparison_df = comparison_df

    def rank_strategies_by_stability(self):
        """Rank by metric std (lower = more stable)"""
        print("\n" + "-"*70)
        print("RANK STRATEGIES BY STABILITY")
        print("-"*70)

        stability_metrics = ['roc_auc_std', 'pr_auc_std', 'f1_std', 'recall_std']

        for metric in stability_metrics:
            if metric in self.comparison_df.columns:
                ranked = self.comparison_df.sort_values(metric)
                print(f"\nMost stable by {metric.replace('_std', '')}:")
                for strategy, row in ranked.head(3).iterrows():
                    print(f"  {strategy}: {row[metric]:.4f}")


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def run_part6_imbalance_strategy(train_path='train.csv', feature_path='X_engineered_selected.csv'):
    """Execute Part 6: Imbalance Strategy Comparison"""

    print("\n" + "="*70)
    print("PART 6: IMBALANCE STRATEGY COMPARISON (Phase 6)")
    print("="*70)

    # Load data
    train_df = pd.read_csv(train_path)
    y_train = train_df['Y']

    # Try to load engineered features, fall back to original
    try:
        X_train = pd.read_csv(feature_path)
        print(f"Loaded engineered features from {feature_path}: {X_train.shape}")
    except:
        print(f"Could not load engineered features from {feature_path}")
        X_train = train_df.drop(['CoilID', 'Y'], axis=1)
        print(f"Using original features: {X_train.shape}")

    print(f"Target distribution: {y_train.value_counts().to_dict()}")

    # Analyze imbalance
    print("\n" + "-"*70)
    print("IMBALANCE ANALYSIS")
    print("-"*70)

    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    imbalance_ratio = neg_count / pos_count

    print(f"Positive (defect): {pos_count} ({100*pos_count/len(y_train):.2f}%)")
    print(f"Negative (normal): {neg_count} ({100*neg_count/len(y_train):.2f}%)")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

    # Run strategy comparison
    print("\n" + "="*70)
    print("RUN STRATEGY COMPARISON")
    print("="*70)

    comparator = ImbalanceStrategyComparison(X_train, y_train, n_splits=5)
    comparison_df = comparator.compare_all_strategies()

    # Analyze stability
    print("\n" + "="*70)
    print("ANALYZE STABILITY")
    print("="*70)

    stability = StrategyStabilityAnalyzer(comparison_df)
    stability.rank_strategies_by_stability()

    # Select best strategy
    print("\n" + "="*70)
    print("SELECT BEST STRATEGY")
    print("="*70)

    best_roc = comparison_df['roc_auc_mean'].idxmax()
    best_f1 = comparison_df['f1_mean'].idxmax()
    best_recall = comparison_df['recall_mean'].idxmax()

    print(f"\nBest by ROC-AUC: {best_roc}")
    print(f"Best by F1: {best_f1}")
    print(f"Best by Recall: {best_recall}")

    # For production: prefer recall if defects are critical (escaped defects costly)
    selected_strategy = best_recall if comparison_df.loc[best_recall, 'recall_mean'] > 0.85 else best_roc

    print(f"\n✓ SELECTED STRATEGY: {selected_strategy}")
    print(f"  ROC-AUC: {comparison_df.loc[selected_strategy, 'roc_auc_mean']:.4f}")
    print(f"  Recall: {comparison_df.loc[selected_strategy, 'recall_mean']:.4f}")
    print(f"  Precision: {comparison_df.loc[selected_strategy, 'precision_mean']:.4f}")
    print(f"  F1: {comparison_df.loc[selected_strategy, 'f1_mean']:.4f}")

    # Save comparison
    comparison_df.to_csv('strategy_comparison.csv')
    print(f"\n✓ Comparison saved to strategy_comparison.csv")

    # Save to InsightsManager
    try:
        from insights_manager import InsightsManager
        manager = InsightsManager()
        manager.set_part6_insights({
            'strategy_comparison': comparison_df,
            'best_strategy': selected_strategy,
        })
    except:
        print("\n⚠ Could not update InsightsManager")

    summary = {
        'best_strategy': selected_strategy,
        'comparison': comparison_df.to_dict('index'),
        'selected_metrics': comparison_df.loc[selected_strategy].to_dict(),
    }

    return summary


if __name__ == "__main__":
    run_part6_imbalance_strategy()
