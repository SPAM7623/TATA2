"""
PART 8: MODEL STABILITY & ENSEMBLE (Phase 7)
Alpha Defect Prediction in Hot Rolling Mills

OPTIMIZATION FOCUS:
- Train multiple models with different random seeds
- Implement ensemble averaging for prediction stability
- Analyze prediction disagreement across seeds
- Measure ensemble calibration and stability
- Create final ensemble configuration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, brier_score_loss
)
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# MULTI-SEED ENSEMBLE
# =====================================================================

class MultiSeedEnsemble:
    """Train and manage ensemble from multiple seeds"""

    def __init__(self, X_train, y_train, seeds=None):
        self.X_train = X_train
        self.y_train = y_train
        self.seeds = seeds or [42, 123, 456, 789, 999]
        self.models = []
        self.predictions = {}

    def train_models_multiple_seeds(self, n_estimators=100, max_depth=6):
        """Train models with different random seeds"""
        print("\n" + "-"*70)
        print("TRAIN MODELS WITH MULTIPLE SEEDS")
        print("-"*70)

        for seed in self.seeds:
            print(f"\nTraining model with seed={seed}...")

            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=-1,
                verbosity=0
            )

            model.fit(self.X_train, self.y_train)
            self.models.append((seed, model))

            # Get training predictions for later analysis
            y_pred_proba = model.predict_proba(self.X_train)[:, 1]
            self.predictions[f'seed_{seed}'] = y_pred_proba

            # Quick metric on training data
            auc = roc_auc_score(self.y_train, y_pred_proba)
            print(f"  AUC (train): {auc:.4f}")

        print(f"\n✓ Trained {len(self.models)} models")
        return self.models

    def compute_prediction_agreement(self):
        """Compute agreement rate across models"""
        print("\n" + "-"*70)
        print("COMPUTE PREDICTION AGREEMENT")
        print("-"*70)

        if len(self.models) < 2:
            print("Need at least 2 models for agreement analysis")
            return 0

        # Get predictions from all models
        all_predictions = np.array([pred for pred in self.predictions.values()])

        # Hard predictions at 0.5 threshold
        hard_preds = (all_predictions >= 0.5).astype(int)

        # Agreement rate: samples where all models agree
        agreement = np.mean(np.sum(np.abs(hard_preds - hard_preds[0, :]), axis=0) == 0)

        print(f"Agreement rate: {agreement:.2%}")
        print(f"  (Percentage of samples where all models make same prediction)")

        return agreement

    def identify_hard_samples(self):
        """Identify samples with high disagreement across models"""
        print("\n" + "-"*70)
        print("IDENTIFY HARD SAMPLES (High Disagreement)")
        print("-"*70)

        all_predictions = np.array([pred for pred in self.predictions.values()])

        # Variance across model predictions
        pred_std = np.std(all_predictions, axis=0)
        pred_mean = np.mean(all_predictions, axis=0)
        pred_uncertainty = np.abs(pred_mean - 0.5)  # Distance from boundary

        # Hard samples: high std (disagreement)
        hard_idx = np.argsort(pred_std)[-50:]

        print(f"Top 5 most disagreed samples:")
        for idx in hard_idx[-5:]:
            print(f"  Sample {idx}: std={pred_std[idx]:.4f}, mean_pred={pred_mean[idx]:.4f}, y_true={self.y_train[idx]}")

        return hard_idx, pred_std, pred_mean

    def ensemble_average_predictions(self, X_test=None):
        """Simple average of probabilities"""
        print("\n" + "-"*70)
        print("ENSEMBLE AVERAGE PREDICTIONS")
        print("-"*70)

        if X_test is None:
            X_test = self.X_train
            print("Using training data for ensemble predictions")

        ensemble_preds = np.zeros(len(X_test))

        for seed, model in self.models:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            ensemble_preds += y_pred_proba

        ensemble_preds /= len(self.models)

        print(f"✓ Ensemble predictions computed")
        print(f"  Mean: {ensemble_preds.mean():.4f}")
        print(f"  Std: {ensemble_preds.std():.4f}")

        return ensemble_preds

    def weighted_ensemble_predictions(self, X_test=None, weights=None):
        """Weighted average using individual model AUCs"""
        print("\n" + "-"*70)
        print("WEIGHTED ENSEMBLE PREDICTIONS")
        print("-"*70)

        if X_test is None:
            X_test = self.X_train

        if weights is None:
            # Use AUC as weights
            weights = []
            for seed, model in self.models:
                y_pred_proba = model.predict_proba(self.X_train)[:, 1]
                auc = roc_auc_score(self.y_train, y_pred_proba)
                weights.append(auc)

            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
        else:
            weights = np.array(weights)

        print(f"Weights: {weights}")

        ensemble_preds = np.zeros(len(X_test))

        for (seed, model), weight in zip(self.models, weights):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            ensemble_preds += weight * y_pred_proba

        print(f"✓ Weighted ensemble predictions computed")
        print(f"  Mean: {ensemble_preds.mean():.4f}")
        print(f"  Std: {ensemble_preds.std():.4f}")

        return ensemble_preds

    def get_prediction_uncertainty(self, X_test=None):
        """Get std of predictions (uncertainty measure)"""
        print("\n" + "-"*70)
        print("COMPUTE PREDICTION UNCERTAINTY")
        print("-"*70)

        if X_test is None:
            X_test = self.X_train

        predictions = []

        for seed, model in self.models:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            predictions.append(y_pred_proba)

        predictions = np.array(predictions)
        pred_std = np.std(predictions, axis=0)
        pred_mean = np.mean(predictions, axis=0)

        print(f"Uncertainty statistics:")
        print(f"  Mean std: {pred_std.mean():.4f}")
        print(f"  Max std: {pred_std.max():.4f}")
        print(f"  Min std: {pred_std.min():.4f}")

        return pred_std, pred_mean


# =====================================================================
# ENSEMBLE STABILITY ANALYZER
# =====================================================================

class EnsembleStabilityAnalyzer:
    """Analyze ensemble stability"""

    def __init__(self, ensemble_predictions, y_true, individual_predictions=None):
        self.ensemble_predictions = ensemble_predictions
        self.y_true = y_true
        self.individual_predictions = individual_predictions or {}

    def compute_prediction_std(self):
        """Compute standard deviation across ensemble members"""
        if self.individual_predictions is None or len(self.individual_predictions) == 0:
            print("Individual predictions not available")
            return None

        preds = np.array([p for p in self.individual_predictions.values()])
        pred_std = np.std(preds, axis=0)

        print(f"\nPrediction std statistics:")
        print(f"  Mean: {pred_std.mean():.4f}")
        print(f"  Std: {pred_std.std():.4f}")
        print(f"  Min: {pred_std.min():.4f}")
        print(f"  Max: {pred_std.max():.4f}")

        return pred_std

    def compute_agreement_rate(self, threshold=0.5):
        """Compute agreement rate at given threshold"""
        if self.individual_predictions is None or len(self.individual_predictions) == 0:
            print("Individual predictions not available")
            return 0

        preds = np.array([p for p in self.individual_predictions.values()])
        hard_preds = (preds >= threshold).astype(int)

        # All models agree on same prediction
        agreement = np.mean(hard_preds.var(axis=0) == 0)

        print(f"\nAgreement rate (threshold={threshold}): {agreement:.2%}")
        return agreement

    def identify_confident_predictions(self, uncertainty_threshold=0.05):
        """Identify predictions with low uncertainty"""
        if self.individual_predictions is None or len(self.individual_predictions) == 0:
            print("Individual predictions not available")
            return []

        preds = np.array([p for p in self.individual_predictions.values()])
        pred_std = np.std(preds, axis=0)

        confident_idx = np.where(pred_std < uncertainty_threshold)[0]

        print(f"\nConfident predictions (std < {uncertainty_threshold}): {len(confident_idx)} ({100*len(confident_idx)/len(self.y_true):.1f}%)")
        print(f"  Accuracy on confident: {np.mean(self.y_true[confident_idx] == (self.ensemble_predictions[confident_idx] >= 0.5).astype(int)):.2%}")

        return confident_idx

    def identify_uncertain_predictions(self, uncertainty_threshold=0.08):
        """Identify predictions with high uncertainty"""
        if self.individual_predictions is None or len(self.individual_predictions) == 0:
            print("Individual predictions not available")
            return []

        preds = np.array([p for p in self.individual_predictions.values()])
        pred_std = np.std(preds, axis=0)

        uncertain_idx = np.where(pred_std >= uncertainty_threshold)[0]

        print(f"\nUncertain predictions (std >= {uncertainty_threshold}): {len(uncertain_idx)} ({100*len(uncertain_idx)/len(self.y_true):.1f}%)")
        print(f"  Accuracy on uncertain: {np.mean(self.y_true[uncertain_idx] == (self.ensemble_predictions[uncertain_idx] >= 0.5).astype(int)):.2%}")

        return uncertain_idx


# =====================================================================
# ENSEMBLE CALIBRATION
# =====================================================================

class EnsembleCalibration:
    """Calibrate ensemble predictions"""

    def __init__(self, ensemble_predictions, y_true):
        self.ensemble_predictions = ensemble_predictions
        self.y_true = y_true

    def compute_ece(self, n_bins=10):
        """Expected Calibration Error"""
        bin_sums = np.zeros(n_bins)
        bin_true = np.zeros(n_bins)
        bin_total = np.zeros(n_bins)

        for i in range(len(self.y_true)):
            bin_idx = int(self.ensemble_predictions[i] * n_bins)
            if bin_idx == n_bins:
                bin_idx = n_bins - 1

            bin_sums[bin_idx] += self.ensemble_predictions[i]
            bin_true[bin_idx] += self.y_true[i]
            bin_total[bin_idx] += 1

        ece = 0
        for i in range(n_bins):
            if bin_total[i] > 0:
                bin_acc = bin_true[i] / bin_total[i]
                bin_conf = bin_sums[i] / bin_total[i]
                ece += abs(bin_acc - bin_conf) * bin_total[i]

        return ece / len(self.y_true)

    def compute_brier_score(self):
        """Brier score (MSE of calibration)"""
        return brier_score_loss(self.y_true, self.ensemble_predictions)


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def run_part8_model_stability(train_path='train.csv', feature_path='X_engineered_selected.csv'):
    """Execute Part 8: Model Stability & Ensemble"""

    print("\n" + "="*70)
    print("PART 8: MODEL STABILITY & ENSEMBLE (Phase 7)")
    print("="*70)

    # Load data
    train_df = pd.read_csv(train_path)
    y_train = train_df['Y']

    try:
        X_train = pd.read_csv(feature_path)
        print(f"Loaded engineered features from {feature_path}: {X_train.shape}")
    except:
        X_train = train_df.drop(['CoilID', 'Y'], axis=1)
        print(f"Using original features: {X_train.shape}")

    print(f"Target distribution: {y_train.value_counts().to_dict()}")

    # Step 1: Train multiple seed models
    print("\n" + "="*70)
    print("STEP 1: TRAIN MODELS WITH MULTIPLE SEEDS")
    print("="*70)

    ensemble = MultiSeedEnsemble(X_train, y_train)
    models = ensemble.train_models_multiple_seeds(n_estimators=100, max_depth=6)

    # Step 2: Analyze agreement and hard samples
    print("\n" + "="*70)
    print("STEP 2: ANALYZE PREDICTION AGREEMENT")
    print("="*70)

    agreement = ensemble.compute_prediction_agreement()
    hard_idx, pred_std, pred_mean = ensemble.identify_hard_samples()

    # Step 3: Create ensemble predictions
    print("\n" + "="*70)
    print("STEP 3: CREATE ENSEMBLE PREDICTIONS")
    print("="*70)

    ensemble_simple = ensemble.ensemble_average_predictions()
    ensemble_weighted = ensemble.weighted_ensemble_predictions()

    # Step 4: Stability analysis
    print("\n" + "="*70)
    print("STEP 4: STABILITY ANALYSIS")
    print("="*70)

    stability = EnsembleStabilityAnalyzer(ensemble_simple, y_train, ensemble.predictions)
    pred_std_vals = stability.compute_prediction_std()
    agreement_rate = stability.compute_agreement_rate()
    confident_idx = stability.identify_confident_predictions(uncertainty_threshold=0.05)
    uncertain_idx = stability.identify_uncertain_predictions(uncertainty_threshold=0.08)

    # Step 5: Ensemble calibration
    print("\n" + "="*70)
    print("STEP 5: ENSEMBLE CALIBRATION")
    print("="*70)

    calibration = EnsembleCalibration(ensemble_simple, y_train)
    ece = calibration.compute_ece()
    brier = calibration.compute_brier_score()

    print(f"\nEnsemble calibration metrics:")
    print(f"  ECE: {ece:.4f}")
    print(f"  Brier: {brier:.4f}")

    # Evaluate both ensemble methods
    print("\n" + "="*70)
    print("ENSEMBLE COMPARISON")
    print("="*70)

    auc_simple = roc_auc_score(y_train, ensemble_simple)
    auc_weighted = roc_auc_score(y_train, ensemble_weighted)

    print(f"\nSimple average AUC: {auc_simple:.4f}")
    print(f"Weighted average AUC: {auc_weighted:.4f}")

    # Verify individual model AUCs
    print(f"\nIndividual model AUCs:")
    for seed, model in models:
        y_pred = model.predict_proba(X_train)[:, 1]
        auc = roc_auc_score(y_train, y_pred)
        print(f"  Seed {seed}: {auc:.4f}")

    # Summary
    print("\n" + "="*70)
    print("PART 8 SUMMARY")
    print("="*70)

    summary = {
        'ensemble_method': 'weighted_average' if auc_weighted > auc_simple else 'simple_average',
        'num_models': len(models),
        'seeds': [seed for seed, _ in models],
        'simple_average_auc': auc_simple,
        'weighted_average_auc': auc_weighted,
        'agreement_rate': agreement,
        'confident_predictions': len(confident_idx),
        'uncertain_predictions': len(uncertain_idx),
        'calibration_ece': ece,
        'calibration_brier': brier,
    }

    print(f"\n✓ PART 8 OPTIMIZATION COMPLETE")
    print(f"  Best ensemble method: {summary['ensemble_method']}")
    print(f"  Ensemble AUC: {max(auc_simple, auc_weighted):.4f}")
    print(f"  Agreement rate: {agreement:.2%}")
    print(f"  Calibration ECE: {ece:.4f}")

    # Save model predictions for final evaluation
    np.save('ensemble_predictions_simple.npy', ensemble_simple)
    np.save('ensemble_predictions_weighted.npy', ensemble_weighted)

    print(f"\n✓ Ensemble predictions saved")

    # Save to InsightsManager
    try:
        from insights_manager import InsightsManager
        manager = InsightsManager()
        manager.set_part8_insights({
            'final_model_config': {
                'ensemble_method': summary['ensemble_method'],
                'num_models': len(models),
                'seeds': [seed for seed, _ in models],
            },
            'consistency_analysis': {
                'agreement_rate': agreement,
                'confident_predictions': len(confident_idx),
                'uncertain_predictions': len(uncertain_idx),
            },
            'calibration_stability': {
                'ece': ece,
                'brier': brier,
            }
        })
    except:
        print("\n⚠ Could not update InsightsManager")

    return summary


if __name__ == "__main__":
    run_part8_model_stability()
