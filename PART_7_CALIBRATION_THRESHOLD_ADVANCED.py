"""
PART 7: CALIBRATION & THRESHOLD OPTIMIZATION (Phase 2)
Alpha Defect Prediction in Hot Rolling Mills

OPTIMIZATION FOCUS:
- Multi-method calibration comparison (Isotonic, Platt, Temperature Scaling)
- Calibration stability across folds
- Expected Calibration Error (ECE) analysis
- Calibration-aware threshold selection
- Optimal threshold for multiple objectives (F1, Recall@95%, PR-AUC)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
from sklearn.isotonic import IsotonicRegression as IsotonicRegressor
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score,
    precision_score, recall_score, brier_score_loss,
    calibration_curve
)
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# CALIBRATION COMPARISON
# =====================================================================

class CalibrationComparison:
    """Compare multiple calibration methods"""

    def __init__(self, X_train, y_train, n_splits=5):
        self.X_train = X_train
        self.y_train = y_train
        self.n_splits = n_splits
        self.calibration_results = {}

    def isotonic_calibration(self):
        """Train models with isotonic regression calibration"""
        print("\n" + "-"*70)
        print("METHOD 1: ISOTONIC REGRESSION CALIBRATION")
        print("-"*70)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        ece_scores = []
        brier_scores = []
        roc_auc_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            X_fold_train = self.X_train.iloc[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_val = self.y_train.iloc[val_idx]

            # Train base model
            base_model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )

            # Calibrate using isotonic regression
            calibrated_model = CalibratedClassifierCV(
                base_model,
                method='isotonic',
                cv='prefit'
            )

            # Fit base model first
            base_model.fit(X_fold_train, y_fold_train)

            # Get predictions on validation for calibration training
            y_pred_proba_train = base_model.predict_proba(X_fold_val)[:, 1]

            # Create isotonic regressor
            iso_reg = IsotonicRegressor(out_of_bounds='clip')
            iso_reg.fit(y_pred_proba_train, y_fold_val)

            # Apply calibration
            y_pred_proba_val = base_model.predict_proba(X_fold_val)[:, 1]
            y_pred_proba_calibrated = iso_reg.predict(y_pred_proba_val)

            # Compute metrics
            ece = self._compute_ece(y_fold_val, y_pred_proba_calibrated)
            brier = brier_score_loss(y_fold_val, y_pred_proba_calibrated)
            roc_auc = roc_auc_score(y_fold_val, y_pred_proba_calibrated)

            ece_scores.append(ece)
            brier_scores.append(brier)
            roc_auc_scores.append(roc_auc)

            print(f"Fold {fold_idx+1}: ECE={ece:.4f}, Brier={brier:.4f}, ROC-AUC={roc_auc:.4f}")

        self.calibration_results['isotonic'] = {
            'ece': np.mean(ece_scores),
            'ece_std': np.std(ece_scores),
            'brier': np.mean(brier_scores),
            'roc_auc': np.mean(roc_auc_scores)
        }

        print(f"\nIsotonic Summary:")
        print(f"  ECE: {np.mean(ece_scores):.4f} ± {np.std(ece_scores):.4f}")
        print(f"  Brier: {np.mean(brier_scores):.4f} ± {np.std(brier_scores):.4f}")

        return self.calibration_results['isotonic']

    def platt_scaling(self):
        """Platt scaling calibration (sigmoid fit)"""
        print("\n" + "-"*70)
        print("METHOD 2: PLATT SCALING CALIBRATION")
        print("-"*70)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        ece_scores = []
        brier_scores = []
        roc_auc_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            X_fold_train = self.X_train.iloc[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_val = self.y_train.iloc[val_idx]

            # Train base model
            base_model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )

            base_model.fit(X_fold_train, y_fold_train)

            # Get predictions for calibration training
            y_pred_proba_train = base_model.predict_proba(X_fold_val)[:, 1]

            # Fit sigmoid (Platt scaling)
            A, B = self._fit_platt_scaling(y_pred_proba_train, y_fold_val)

            # Apply to validation predictions
            y_pred_proba_val = base_model.predict_proba(X_fold_val)[:, 1]
            y_pred_proba_calibrated = self._apply_platt_scaling(y_pred_proba_val, A, B)

            # Compute metrics
            ece = self._compute_ece(y_fold_val, y_pred_proba_calibrated)
            brier = brier_score_loss(y_fold_val, y_pred_proba_calibrated)
            roc_auc = roc_auc_score(y_fold_val, y_pred_proba_calibrated)

            ece_scores.append(ece)
            brier_scores.append(brier)
            roc_auc_scores.append(roc_auc)

            print(f"Fold {fold_idx+1}: ECE={ece:.4f}, Brier={brier:.4f}, ROC-AUC={roc_auc:.4f}")

        self.calibration_results['platt'] = {
            'ece': np.mean(ece_scores),
            'ece_std': np.std(ece_scores),
            'brier': np.mean(brier_scores),
            'roc_auc': np.mean(roc_auc_scores)
        }

        print(f"\nPlatt Scaling Summary:")
        print(f"  ECE: {np.mean(ece_scores):.4f} ± {np.std(ece_scores):.4f}")
        print(f"  Brier: {np.mean(brier_scores):.4f} ± {np.std(brier_scores):.4f}")

        return self.calibration_results['platt']

    def temperature_scaling(self):
        """Temperature scaling (simpler than Platt)"""
        print("\n" + "-"*70)
        print("METHOD 3: TEMPERATURE SCALING")
        print("-"*70)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        ece_scores = []
        brier_scores = []
        roc_auc_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            X_fold_train = self.X_train.iloc[train_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_val = self.y_train.iloc[val_idx]

            # Train base model
            base_model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )

            base_model.fit(X_fold_train, y_fold_train)

            # Get predictions for temperature tuning
            y_pred_proba_train = base_model.predict_proba(X_fold_val)[:, 1]

            # Tune temperature
            temperature = self._tune_temperature(y_pred_proba_train, y_fold_val)

            # Apply to validation predictions
            y_pred_proba_val = base_model.predict_proba(X_fold_val)[:, 1]
            y_pred_proba_calibrated = 1 / (1 + np.exp(-np.log(y_pred_proba_val / (1 - y_pred_proba_val + 1e-6)) / temperature))

            # Compute metrics
            ece = self._compute_ece(y_fold_val, y_pred_proba_calibrated)
            brier = brier_score_loss(y_fold_val, y_pred_proba_calibrated)
            roc_auc = roc_auc_score(y_fold_val, y_pred_proba_calibrated)

            ece_scores.append(ece)
            brier_scores.append(brier)
            roc_auc_scores.append(roc_auc)

            print(f"Fold {fold_idx+1}: ECE={ece:.4f}, Brier={brier:.4f}, ROC-AUC={roc_auc:.4f}")

        self.calibration_results['temperature'] = {
            'ece': np.mean(ece_scores),
            'ece_std': np.std(ece_scores),
            'brier': np.mean(brier_scores),
            'roc_auc': np.mean(roc_auc_scores)
        }

        print(f"\nTemperature Scaling Summary:")
        print(f"  ECE: {np.mean(ece_scores):.4f} ± {np.std(ece_scores):.4f}")
        print(f"  Brier: {np.mean(brier_scores):.4f} ± {np.std(brier_scores):.4f}")

        return self.calibration_results['temperature']

    def _compute_ece(self, y_true, y_pred_proba, n_bins=10):
        """Expected Calibration Error"""
        bin_sums = np.zeros(n_bins)
        bin_true = np.zeros(n_bins)
        bin_total = np.zeros(n_bins)

        for i in range(len(y_true)):
            bin_idx = int(y_pred_proba[i] * n_bins)
            if bin_idx == n_bins:
                bin_idx = n_bins - 1

            bin_sums[bin_idx] += y_pred_proba[i]
            bin_true[bin_idx] += y_true[i]
            bin_total[bin_idx] += 1

        ece = 0
        for i in range(n_bins):
            if bin_total[i] > 0:
                bin_acc = bin_true[i] / bin_total[i]
                bin_conf = bin_sums[i] / bin_total[i]
                ece += abs(bin_acc - bin_conf) * bin_total[i]

        return ece / len(y_true)

    def _fit_platt_scaling(self, y_pred_proba, y_true, max_iter=100):
        """Fit Platt scaling parameters"""
        # Simple implementation using sklearn's approach
        from scipy.optimize import minimize

        def sigmoid_loss(params):
            A, B = params
            preds = 1 / (1 + np.exp(-(A * np.log(y_pred_proba / (1 - y_pred_proba + 1e-6)) + B)))
            return -np.mean(y_true * np.log(preds + 1e-6) + (1 - y_true) * np.log(1 - preds + 1e-6))

        result = minimize(sigmoid_loss, [1.0, 0.0], method='Nelder-Mead')
        return result.x

    def _apply_platt_scaling(self, y_pred_proba, A, B):
        """Apply Platt scaling"""
        return 1 / (1 + np.exp(-(A * np.log(y_pred_proba / (1 - y_pred_proba + 1e-6)) + B)))

    def _tune_temperature(self, y_pred_proba, y_true):
        """Tune temperature parameter"""
        best_temp = 1.0
        best_ece = float('inf')

        for temp in np.linspace(0.5, 2.0, 20):
            y_pred_calibrated = 1 / (1 + np.exp(-np.log(y_pred_proba / (1 - y_pred_proba + 1e-6)) / temp))
            ece = self._compute_ece(y_true, y_pred_calibrated)
            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        return best_temp

    def compare_methods(self):
        """Compare calibration methods"""
        self.isotonic_calibration()
        self.platt_scaling()
        self.temperature_scaling()

        comparison_df = pd.DataFrame(self.calibration_results).T
        comparison_df = comparison_df.sort_values('ece')

        print("\n" + "="*70)
        print("CALIBRATION METHOD COMPARISON")
        print("="*70)
        print(comparison_df)

        return comparison_df


# =====================================================================
# THRESHOLD OPTIMIZER
# =====================================================================

class ThresholdOptimizer:
    """Calibration-aware threshold optimization"""

    def __init__(self, X_val, y_val, y_pred_proba):
        self.X_val = X_val
        self.y_val = y_val
        self.y_pred_proba = y_pred_proba

    def find_optimal_threshold_f1(self):
        """Find threshold that maximizes F1"""
        print("\n" + "-"*70)
        print("OPTIMIZE THRESHOLD FOR F1")
        print("-"*70)

        thresholds = np.linspace(0.1, 0.9, 50)
        f1_scores = []

        for threshold in thresholds:
            y_pred = (self.y_pred_proba >= threshold).astype(int)
            f1 = f1_score(self.y_val, y_pred)
            f1_scores.append(f1)

        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        print(f"Optimal threshold (F1): {best_threshold:.4f}")
        print(f"F1 score: {best_f1:.4f}")

        return best_threshold, best_f1

    def find_optimal_threshold_recall(self, target_recall=0.95):
        """Find threshold that achieves target recall"""
        print("\n" + "-"*70)
        print(f"OPTIMIZE THRESHOLD FOR RECALL ({target_recall:.1%})")
        print("-"*70)

        thresholds = np.linspace(0.05, 0.5, 100)
        recalls = []

        for threshold in thresholds:
            y_pred = (self.y_pred_proba >= threshold).astype(int)
            recall = recall_score(self.y_val, y_pred, zero_division=0)
            recalls.append(recall)

        # Find threshold closest to target recall
        recalls = np.array(recalls)
        best_idx = np.argmin(np.abs(recalls - target_recall))
        best_threshold = thresholds[best_idx]
        achieved_recall = recalls[best_idx]

        y_pred = (self.y_pred_proba >= best_threshold).astype(int)
        precision = precision_score(self.y_val, y_pred, zero_division=0)

        print(f"Threshold for {target_recall:.1%} recall: {best_threshold:.4f}")
        print(f"Achieved recall: {achieved_recall:.4f}")
        print(f"Precision at threshold: {precision:.4f}")

        return best_threshold, achieved_recall

    def find_optimal_threshold_pr_auc(self):
        """Find threshold maximizing PR-AUC region"""
        print("\n" + "-"*70)
        print("OPTIMIZE THRESHOLD FOR PR-AUC")
        print("-"*70)

        precision, recall, thresholds = precision_recall_curve(self.y_val, self.y_pred_proba)

        # F-beta score (beta=1 for F1)
        f_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-6)
        best_idx = np.argmax(f_scores)
        best_threshold = thresholds[best_idx]

        y_pred = (self.y_pred_proba >= best_threshold).astype(int)
        f1 = f1_score(self.y_val, y_pred)

        print(f"Optimal threshold (PR-AUC): {best_threshold:.4f}")
        print(f"F1 at threshold: {f1:.4f}")

        return best_threshold, f1

    def find_all_optimal_thresholds(self):
        """Find thresholds for multiple objectives"""
        threshold_f1, f1_score_val = self.find_optimal_threshold_f1()
        threshold_recall, recall_val = self.find_optimal_threshold_recall(0.95)
        threshold_pr, pr_score_val = self.find_optimal_threshold_pr_auc()

        print("\n" + "="*70)
        print("THRESHOLD SUMMARY")
        print("="*70)

        thresholds_summary = {
            'best_f1': threshold_f1,
            'f1_score': f1_score_val,
            'recall_95': threshold_recall,
            'recall_achieved': recall_val,
            'pr_balance': threshold_pr,
            'pr_f1': pr_score_val,
        }

        print(f"\nOptimal thresholds:")
        print(f"  F1 optimal: {threshold_f1:.4f} (F1={f1_score_val:.4f})")
        print(f"  95% Recall: {threshold_recall:.4f} (recall={recall_val:.4f})")
        print(f"  PR-AUC optimal: {threshold_pr:.4f} (F1={pr_score_val:.4f})")

        return thresholds_summary


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def run_part7_calibration_threshold(train_path='train.csv'):
    """Execute Part 7: Calibration & Threshold Optimization"""

    print("\n" + "="*70)
    print("PART 7: CALIBRATION & THRESHOLD OPTIMIZATION (Phase 2)")
    print("="*70)

    # Load data
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(['CoilID', 'Y'], axis=1)
    y_train = train_df['Y']

    print(f"Data loaded: {X_train.shape}")

    # Step 1: Calibration method comparison
    print("\n" + "="*70)
    print("STEP 1: CALIBRATION METHOD COMPARISON")
    print("="*70)

    calibrator = CalibrationComparison(X_train, y_train, n_splits=5)
    calibration_comparison = calibrator.compare_methods()

    # Select best calibration method
    best_method = calibration_comparison['ece'].idxmin()
    print(f"\n✓ Best calibration method: {best_method}")

    # Step 2: Threshold optimization
    print("\n" + "="*70)
    print("STEP 2: THRESHOLD OPTIMIZATION")
    print("="*70)

    # Use last fold for demonstration
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

    # Train model on full train data
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    optimizer = ThresholdOptimizer(X_val, y_val, y_pred_proba)
    thresholds_summary = optimizer.find_all_optimal_thresholds()

    # Summary
    print("\n" + "="*70)
    print("PART 7 SUMMARY")
    print("="*70)

    summary = {
        'best_calibration_method': best_method,
        'calibration_ece': calibration_comparison.loc[best_method, 'ece'],
        'optimal_thresholds': thresholds_summary,
    }

    print(f"\n✓ PART 7 OPTIMIZATION COMPLETE")
    print(f"  Best calibration: {best_method}")
    print(f"  Calibration ECE: {calibration_comparison.loc[best_method, 'ece']:.4f}")

    # Save to InsightsManager
    try:
        from insights_manager import InsightsManager
        manager = InsightsManager()
        manager.set_part7_insights({'optimal_thresholds': thresholds_summary})
    except:
        print("\n⚠ Could not update InsightsManager")

    return summary


if __name__ == "__main__":
    run_part7_calibration_threshold()
