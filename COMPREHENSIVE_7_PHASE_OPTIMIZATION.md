# COMPREHENSIVE 7-PHASE OPTIMIZATION FOR ALPHA DEFECT DETECTION

## Executive Summary
This document outlines a complete 7-phase optimization strategy for the Alpha Defect Detection workflow (Parts 1-8), designed to improve leaderboard score from 68.50 to 75+.

---

# PHASE 1: VERIFY PIPELINE (No Data Leakage, Reproducible Baseline)

## PART 1: Industrial EDA - OPTIMIZATION CHANGES

### Summary of Changes Needed
- **Add data split verification** to ensure no target-aware transformations before CV split
- **Track baseline metrics** across all features with 5-fold CV
- **Identify problematic features** that might leak information
- **Build reproducible baseline** with strict random seeds
- **Create feature stability assessment** to prioritize engineering efforts

### Key Modifications
1. **Add CV-aware feature statistics tracking**
   - Compute statistics WITHIN each fold, never on full dataset
   - Track feature distributions by fold to detect stability issues
   
2. **Implement leakage detection**
   - Check for features with >95% correlation to target before any transformation
   - Flag features with non-normal distributions that correlate with target
   - Verify no test data statistics were used in feature computation

3. **Create baseline stability analysis**
   - 5-fold CV on raw features with simple model
   - Document AUC variability across folds
   - Identify which features are most stable

### New Functions/Classes to Add
```python
class DataLeakageDetector:
    """Verify no information leakage in pipeline"""
    - verify_cv_stratification()
    - detect_target_correlated_features()
    - check_feature_computation_timing()
    - validate_train_test_separation()

class BaselineStabilityTracker:
    """Track reproducibility and stability"""
    - compute_cv_statistics_safely()  # Within fold only
    - assess_feature_stability_across_folds()
    - track_random_seed_reproducibility()
    - generate_baseline_stability_report()
```

### Data Structures to Track
```python
{
    'data_integrity': {
        'train_test_overlap': boolean,
        'target_aware_transformations': list,
        'feature_leakage_score': float
    },
    'baseline_stability': {
        'fold_auc_scores': [0.65, 0.66, 0.64, ...],
        'fold_auc_std': float,
        'feature_stability': {'feature1': 0.95, ...},
        'random_seed_reproducible': boolean
    },
    'feature_quality': {
        'high_variance': list,
        'low_variance': list,
        'correlated_with_target': list,
        'missing_pattern_analysis': dict
    }
}
```

### Integration Points
- **Input**: Raw train.csv, test.csv
- **Output**: Verified dataset, baseline metrics stored in InsightsManager
- **Next Part Integration**: Part 2 uses baseline metrics to set expectations
- **InsightsManager Updates**: 
  - `set_part1_insights()` includes leakage_score and cv_stability

---

# PHASE 2: PROBABILITY CALIBRATION (Isotonic + Platt Scaling)

## PART 7: Calibration & Threshold - OPTIMIZATION CHANGES

### Summary of Changes Needed
- **Add multi-method calibration comparison** (Isotonic vs Platt vs Sigmoid)
- **Track calibration stability** across folds
- **Implement calibration-aware threshold selection**
- **Create Expected Calibration Error (ECE) analysis**
- **Validate calibration on held-out data**

### Key Modifications
1. **Implement calibration comparison framework**
   - Train calibrator on fold calibration set (not validation)
   - Apply to validation set
   - Measure ECE, MCE, Brier score
   
2. **Add temperature scaling** as another method
   - Simpler than Platt, often as effective
   - Single hyperparameter
   
3. **Create fold-level calibration tracking**
   - Store calibrated probabilities for each fold
   - Track confidence interval per prediction

4. **Integrate with threshold selection**
   - Once calibrated, find optimal threshold on validation curves
   - Account for calibration uncertainty in threshold selection

### New Functions/Classes to Add
```python
class CalibrationComparison:
    """Compare multiple calibration methods"""
    - isotonic_calibration()
    - platt_scaling()
    - temperature_scaling()
    - sigmoid_calibration()
    - compute_ece()  # Expected Calibration Error
    - compute_mce()  # Maximum Calibration Error
    - compare_methods()

class ThresholdOptimizer:
    """Calibration-aware threshold optimization"""
    - find_optimal_threshold_f1()
    - find_optimal_threshold_recall()
    - find_optimal_threshold_pr_auc()
    - validate_threshold_stability()
```

### Data Structures to Track
```python
{
    'calibration_analysis': {
        'method_comparison': {
            'isotonic': {'ece': 0.05, 'mce': 0.12, 'brier': 0.15},
            'platt': {'ece': 0.06, 'mce': 0.14, 'brier': 0.16},
            'temperature': {'ece': 0.05, 'mce': 0.11, 'brier': 0.14}
        },
        'best_method': 'isotonic',
        'calibration_per_fold': {...}
    },
    'threshold_optimization': {
        'optimal_threshold_f1': 0.45,
        'optimal_threshold_recall95': 0.30,
        'optimal_threshold_pr_auc': 0.42,
        'threshold_stability': {'std': 0.03}
    }
}
```

### Integration Points
- **Input**: Predictions from Part 6, Part 2 baseline
- **Output**: Calibrated probabilities, optimal threshold
- **Dependency**: Must have multiple model predictions from Part 6
- **InsightsManager Updates**: 
  - `set_part7_insights()` with calibration method + threshold metrics

---

# PHASE 3: SHAP ANALYSIS (Enhanced - Feature Importance, Interactions, Hard Samples)

## PART 3: SHAP Error Analysis - OPTIMIZATION CHANGES

### Summary of Changes Needed
- **Add feature interaction detection** via SHAP interactions
- **Implement hard sample analysis** (boundary samples, FN/FP patterns)
- **Track feature importance consistency** across folds
- **Create decision boundary visualization**
- **Identify feature combinations** that trigger false negatives

### Key Modifications
1. **Add SHAP Interactions module**
   - Compute SHAP interaction indices for top feature pairs
   - Identify which combinations cause prediction uncertainty
   - Rank by interaction strength
   
2. **Implement hard sample detection**
   - Identify samples with high uncertainty (prob near 0.5)
   - Separate FN samples (actual 1, pred 0) - escaped defects
   - Separate FP samples (actual 0, pred 1) - false alarms
   - Characterize each group with SHAP values
   
3. **Add fold-level consistency analysis**
   - Train model on each fold
   - Compare feature importance across folds
   - Identify consistently important features vs noisy ones
   
4. **Create boundary sample analysis**
   - Identify samples closest to decision boundary
   - Understand what makes them ambiguous
   - Check if feature engineering can clarify them

### New Functions/Classes to Add
```python
class SHAPInteractionAnalyzer:
    """Analyze feature interactions via SHAP"""
    - compute_interaction_indices()
    - find_top_interactions()
    - visualize_interaction_effects()
    - get_interaction_recommendations()

class HardSampleAnalyzer:
    """Detect and analyze hard samples"""
    - identify_boundary_samples()
    - identify_false_negatives()
    - identify_false_positives()
    - characterize_sample_group()
    - get_feature_recommendations_for_group()

class ConsistencyAnalyzer:
    """Analyze feature importance consistency"""
    - compute_importance_across_folds()
    - rank_by_consistency()
    - identify_stable_features()
    - identify_noisy_features()
```

### Data Structures to Track
```python
{
    'shap_interactions': {
        'top_interactions': [
            {'feature1': 'X', 'feature2': 'Y', 'interaction_strength': 0.12},
            ...
        ],
        'interaction_by_class': {
            'defect': [...],
            'normal': [...]
        }
    },
    'hard_samples': {
        'boundary_samples': {
            'count': 50,
            'avg_uncertainty': 0.08,
            'top_confusing_features': ['X', 'Y', 'Z']
        },
        'false_negatives': {
            'count': 15,
            'characteristics': {...},
            'common_patterns': [...]
        },
        'false_positives': {
            'count': 25,
            'characteristics': {...},
            'common_patterns': [...]
        }
    },
    'feature_consistency': {
        'stable_features': ['X', 'Y'],
        'noisy_features': ['A', 'B'],
        'consistency_scores': {...}
    }
}
```

### Integration Points
- **Input**: Baseline model predictions, training data
- **Output**: Feature interaction insights, hard sample characterization
- **Feedback to Part 5**: Informs which features to engineer
- **Feedback to Part 6**: Informs imbalance strategy (which samples to oversample)
- **InsightsManager Updates**: 
  - `set_part3_insights()` with shap_interactions, hard_sample_analysis

---

# PHASE 4: FEATURE ENGINEERING (Interactions, Instability, Groups, Anomaly, Regime Features)

## PART 5: Feature Engineering - OPTIMIZATION CHANGES

### Summary of Changes Needed
- **Engineer features guided by Phase 3 insights** (SHAP interactions, hard samples)
- **Add regime detection** (normal vs risky operating conditions)
- **Create instability scoring** (rate of change, variance in small windows)
- **Build anomaly features** informed by hard sample analysis
- **Generate group-level features** that stabilize group behavior
- **Track feature importance after engineering**

### Key Modifications
1. **SHAP-guided interaction features**
   - Only engineer top interaction pairs from Phase 3
   - Test polynomial, multiplicative, and ratio features
   - Select those that improve hard sample predictions

2. **Regime detection features**
   - Identify "risky" combinations of features
   - Create binary flags for risky regimes
   - Weighted by false negative rate in that regime

3. **Instability features**
   - Rolling std over small windows (5-10 samples)
   - Rate of change detection
   - Sudden shift detection
   - Variance in recent history

4. **Anomaly features**
   - Isolation Forest anomaly scores (per group)
   - Mahalanobis distance from normal cluster
   - Local density features
   - Deviation from group mean

5. **Group-level features**
   - Group mean, std, min, max (per sample's group)
   - Deviation from group baseline
   - Group trend indicators

### New Functions/Classes to Add
```python
class SHAPGuidedFeatureEngineer:
    """Engineer features based on SHAP insights"""
    - engineer_interaction_features()  # From Part 3 interactions
    - engineer_hard_sample_features()  # From Part 3 hard samples
    - test_feature_impact()  # Does it help hard samples?

class RegimeDetector:
    """Detect risky operating regimes"""
    - identify_regimes()
    - rank_regimes_by_defect_rate()
    - create_regime_features()

class InstabilityFeatures:
    """Create instability/change features"""
    - rolling_variance()
    - rate_of_change()
    - sudden_shift_detection()
    - trend_strength()

class AnomalyFeatures:
    """Create anomaly detection features"""
    - isolation_forest_score()
    - mahalanobis_distance()
    - local_density()
    - group_deviation()

class GroupFeatures:
    """Create features based on group membership"""
    - group_statistics()
    - group_deviation()
    - group_trend()
```

### Data Structures to Track
```python
{
    'engineered_features': {
        'interaction_features': {
            'count': 15,
            'avg_importance': 0.08,
            'improvement': 0.02  # AUC improvement
        },
        'regime_features': {
            'count': 5,
            'regimes_identified': 10,
            'avg_importance': 0.06
        },
        'instability_features': {
            'count': 8,
            'avg_importance': 0.05
        },
        'anomaly_features': {
            'count': 6,
            'avg_importance': 0.07
        },
        'group_features': {
            'count': 12,
            'avg_importance': 0.06
        }
    },
    'feature_selection': {
        'total_engineered': 46,
        'selected_for_model': 30,
        'selection_ratio': 0.65,
        'importance_threshold': 0.03
    }
}
```

### Integration Points
- **Input**: Part 3 insights (SHAP interactions, hard samples), Part 4 feature groups
- **Output**: Engineered features saved to X_engineered.csv
- **Validation**: Test on Part 2 baseline model to ensure improvement
- **InsightsManager Updates**: 
  - `set_part5_insights()` with engineered_features, feature_selection

---

# PHASE 5: HARD SAMPLE ANALYSIS (Boundary Samples, FN/FP Patterns)

## PART 3 (Enhanced): SHAP Error Analysis - HARD SAMPLE ANALYSIS SECTION

### Summary of Changes Needed
- **Deep-dive into false negatives** (escaped defects)
- **Deep-dive into false positives** (false alarms)
- **Identify boundary samples** (high uncertainty)
- **Create targeted feature recommendations** for each error type
- **Track error patterns** by defect characteristics

### Key Modifications
1. **False Negative Analysis**
   - Extract all FN samples (y=1, pred=0)
   - Compute SHAP values for each FN
   - Identify common feature patterns
   - Check if new features would separate them
   - Recommend specific engineered features

2. **False Positive Analysis**
   - Extract all FP samples (y=0, pred=1)
   - Compute SHAP values for each FP
   - Identify feature combinations causing false alarms
   - Recommend features to reduce false alarms

3. **Boundary Sample Analysis**
   - Find samples with P(defect) in [0.4, 0.6]
   - These are most uncertain predictions
   - Analyze why model is uncertain
   - Check if clearer features exist

4. **Error Pattern Clustering**
   - Group FNs by similar error patterns
   - Group FPs by similar error patterns
   - Create error profiles: "High Temperature FN", "Low Speed FP", etc.
   - Recommend targeted improvements per profile

### New Functions/Classes to Add
```python
class FalseNegativeAnalyzer:
    """Deep-dive into escaped defects"""
    - identify_false_negatives()
    - analyze_fn_characteristics()
    - compute_fn_shap_values()
    - identify_fn_patterns()
    - get_engineering_recommendations()

class FalsePositiveAnalyzer:
    """Deep-dive into false alarms"""
    - identify_false_positives()
    - analyze_fp_characteristics()
    - compute_fp_shap_values()
    - identify_fp_patterns()
    - get_engineering_recommendations()

class BoundaryAnalyzer:
    """Analyze high-uncertainty predictions"""
    - identify_boundary_samples()
    - analyze_boundary_characteristics()
    - compute_uncertainty_drivers()
    - get_clarifying_features()

class ErrorPatternProfiler:
    """Create error profiles for targeted fixes"""
    - cluster_fn_patterns()
    - cluster_fp_patterns()
    - create_error_profiles()
    - rank_profiles_by_frequency()
```

### Data Structures to Track
```python
{
    'false_negatives': {
        'count': 15,
        'rate': 0.10,  # FN / (FN + TP)
        'characteristics': {
            'avg_feature_X': 150.0,
            'avg_feature_Y': 50.0,
            ...
        },
        'common_patterns': [
            'High Temperature + Medium Pressure',
            'Low Speed + High Load',
            ...
        ],
        'shap_drivers': ['X', 'Y', 'Z'],  # Features causing FN
        'engineering_recommendations': [...]
    },
    'false_positives': {
        'count': 25,
        'rate': 0.05,  # FP / (FP + TN)
        'characteristics': {...},
        'common_patterns': [...],
        'shap_drivers': [...],
        'engineering_recommendations': [...]
    },
    'boundary_samples': {
        'count': 50,
        'avg_uncertainty': 0.08,
        'uncertainty_drivers': [...],
        'clarifying_features': [...]
    },
    'error_profiles': [
        {
            'name': 'High Temp FN',
            'count': 8,
            'characteristics': {...},
            'recommended_fixes': [...]
        },
        ...
    ]
}
```

### Integration Points
- **Input**: Baseline model predictions, SHAP values
- **Output**: Error profiles, feature recommendations
- **Feedback to Part 5**: Specific engineered features to create
- **Feedback to Part 6**: Imbalance handling should oversample problematic profiles
- **Feedback to Part 7**: Threshold should account for error patterns
- **InsightsManager Updates**: 
  - Extend `set_part3_insights()` with fn_analysis, fp_analysis, error_profiles

---

# PHASE 6: IMPROVE IMBALANCE STRATEGY (Compare Multiple Strategies)

## PART 6: Imbalance Handling - OPTIMIZATION CHANGES

### Summary of Changes Needed
- **Compare 5+ imbalance strategies** on engineered features
- **Use error profiles from Phase 5** to guide strategy
- **Measure stability of strategy** across folds
- **Select strategy based on recall target** (minimize escape rate)
- **Validate on held-out fold**

### Key Modifications
1. **Strategy Comparison Framework**
   - Scale_pos_weight: baseline weighted approach
   - SMOTE (in-fold only): oversample minority
   - Cost-sensitive learning: loss weighting
   - Focal loss: dynamic class weighting
   - Balanced bagging: ensemble of balanced samples
   - Hybrid: combine multiple strategies
   
2. **Evaluation on engineered features**
   - Use features from Part 5
   - Test each strategy with 5-fold CV
   - Measure: ROC-AUC, PR-AUC, F1, Recall, Precision
   - Track stability (std across folds)
   
3. **Error-profile-aware strategy**
   - Oversample error profiles with high FN rate
   - Apply different strategies per profile if needed
   - Validate improvement on hard samples

4. **Threshold-strategy interaction**
   - Each strategy produces different probability distribution
   - Test optimal threshold for each strategy
   - Select strategy+threshold combo that best meets recall target

### New Functions/Classes to Add
```python
class ImbalanceStrategyComparison:
    """Compare multiple imbalance strategies"""
    - scale_pos_weight_strategy()
    - smote_strategy()
    - cost_sensitive_strategy()
    - focal_loss_strategy()
    - balanced_bagging_strategy()
    - hybrid_strategy()
    - compare_all_strategies()  # Returns DataFrame with metrics

class StrategyStabilityAnalyzer:
    """Analyze strategy stability across folds"""
    - compute_fold_stability()
    - compute_metric_confidence_intervals()
    - rank_strategies_by_stability()

class ErrorProfileAwareBalancer:
    """Apply strategy with error profile awareness"""
    - oversample_error_profiles()
    - apply_profile_specific_strategy()
```

### Data Structures to Track
```python
{
    'strategy_comparison': {
        'scale_pos_weight': {
            'roc_auc': [0.70, 0.71, 0.69, 0.72, 0.70],  # Per-fold
            'roc_auc_mean': 0.704,
            'roc_auc_std': 0.012,
            'recall': 0.85,
            'precision': 0.45,
            'f1': 0.59
        },
        'smote': {
            'roc_auc': [...],
            'roc_auc_mean': 0.715,
            'roc_auc_std': 0.015,
            'recall': 0.88,
            'precision': 0.42,
            'f1': 0.57
        },
        'focal_loss': {...},
        'balanced_bagging': {...},
        'hybrid': {...}
    },
    'best_strategy': 'focal_loss',
    'best_strategy_metrics': {...},
    'strategy_threshold_pairs': {
        'scale_pos_weight': 0.45,
        'smote': 0.40,
        'focal_loss': 0.38,
        ...
    }
}
```

### Integration Points
- **Input**: Engineered features (Part 5), error profiles (Phase 5)
- **Output**: Selected strategy, strategy-specific model
- **Validation**: Test on hold-out fold to estimate real performance
- **Next Step**: Use selected strategy in Part 7 + 8
- **InsightsManager Updates**: 
  - `set_part6_insights()` with strategy_comparison, best_strategy, stability_metrics

---

# PHASE 7: MODEL STABILITY (Multiple Seeds, Ensemble Averaging)

## PART 8: Final Refinement - OPTIMIZATION CHANGES

### Summary of Changes Needed
- **Train multiple models** with different random seeds
- **Implement ensemble averaging** for prediction stability
- **Analyze prediction disagreement** across seeds
- **Measure ensemble calibration** and stability
- **Track variance in calibration** across seeds
- **Create final ensemble** that combines stability insights

### Key Modifications
1. **Multi-seed training**
   - Train 5-7 models with different random seeds (42, 123, 456, ...)
   - Keep all hyperparameters constant (from Part 8)
   - Measure prediction agreement
   - Identify consistently hard samples
   
2. **Ensemble averaging**
   - Simple average of probabilities
   - Weighted average (weight by individual model AUC)
   - Stacking meta-learner
   - Test which works best
   
3. **Stability analysis**
   - For each test sample, measure std of predictions across seeds
   - High std = uncertain, low std = confident
   - Use uncertainty for decision-making (flag for manual review)
   - Threshold on uncertainty to separate "confident" from "uncertain" predictions
   
4. **Calibration per seed**
   - Calibrate each seed model independently
   - Check if calibrated ensemble is better calibrated
   - Track ECE before/after ensemble
   
5. **Final model specification**
   - Document ensemble configuration
   - Store all seed models + calibrators
   - Create inference pipeline

### New Functions/Classes to Add
```python
class MultiSeedEnsemble:
    """Train and manage ensemble from multiple seeds"""
    - train_models_multiple_seeds()
    - compute_prediction_agreement()
    - identify_hard_samples()  # High disagreement
    - ensemble_average_predictions()
    - weighted_ensemble()
    - stacking_ensemble()
    - get_prediction_uncertainty()

class EnsembleStabilityAnalyzer:
    """Analyze ensemble stability"""
    - compute_prediction_std()
    - compute_agreement_rate()
    - identify_confident_predictions()
    - identify_uncertain_predictions()
    - visualize_prediction_distribution()

class EnsembleCalibration:
    """Calibrate ensemble predictions"""
    - calibrate_ensemble()
    - compute_ensemble_ece()
    - compare_individual_vs_ensemble_calibration()
```

### Data Structures to Track
```python
{
    'ensemble': {
        'num_models': 5,
        'random_seeds': [42, 123, 456, 789, 999],
        'ensemble_method': 'weighted_average',
        'individual_auc': [0.705, 0.710, 0.708, 0.712, 0.707],
        'ensemble_auc': 0.715,
        'auc_improvement': 0.010
    },
    'stability_analysis': {
        'prediction_std': {
            'mean': 0.08,
            'std': 0.05,
            'min': 0.01,
            'max': 0.15
        },
        'agreement_rate': 0.88,  # % samples where all 5 models agree on top-1 choice
        'confident_predictions': 0.65,  # % with std < threshold
        'uncertain_predictions': 0.35   # % with std >= threshold
    },
    'calibration': {
        'individual_ece': [0.05, 0.06, 0.05, 0.07, 0.05],
        'ensemble_ece': 0.04,
        'calibration_improvement': 0.01
    },
    'final_specification': {
        'ensemble_method': 'weighted_average',
        'uncertainty_threshold': 0.08,
        'deployment_config': {...}
    }
}
```

### Integration Points
- **Input**: Engineered features (Part 5), best imbalance strategy (Part 6), optimal threshold (Part 7)
- **Output**: Final ensemble model, deployment config
- **Validation**: Test on test set with uncertainty estimates
- **InsightsManager Updates**: 
  - `set_part8_insights()` with ensemble specs, stability metrics, calibration metrics

---

# INTEGRATION SUMMARY

## Cross-Phase Dependencies

```
PHASE 1 (Verify)
  ↓ (Baseline metrics, feature quality)
PHASE 3 (SHAP) ← Uses baseline, identifies hard samples
  ↓ (Feature importance, interactions, error patterns)
PHASE 5 (Feature Engineering) ← Uses SHAP insights
  ↓ (Engineered features)
PHASE 4 (Hard Sample Analysis) ← Runs in parallel with Phase 3, uses error patterns
  ↓ (Error profiles for rebalancing)
PHASE 6 (Imbalance) ← Uses engineered features + error profiles
  ↓ (Best strategy)
PHASE 2 (Calibration) ← Uses imbalanced model
  ↓ (Calibration method + optimal threshold)
PHASE 7 (Stability) ← Uses all above
  ↓ (Final ensemble)
```

## Expected Improvements

- **Phase 1**: Baseline stability +0% (establishes sanity)
- **Phase 3**: Feature understanding (enables better engineering)
- **Phase 5**: Feature quality +2-3% AUC (targeted features)
- **Phase 4**: Hard sample understanding (enables imbalance tuning)
- **Phase 6**: Recall improvement +5-8% (better strategy)
- **Phase 2**: Calibration +1-2% AUC (better threshold)
- **Phase 7**: Overall stability +1-2% AUC (ensemble)

**Total Expected Improvement: 9-17% → Target 75+ from 68.50**

## InsightsManager Integration

Each phase saves to InsightsManager:
- Phase 1 → `set_part1_insights()` (data quality, baseline stability)
- Phase 3 → `set_part3_insights()` (SHAP, error patterns, hard samples)
- Phase 5 → `set_part5_insights()` (engineered features, importance)
- Phase 4 → `set_part3_insights()` (error profiles)
- Phase 6 → `set_part6_insights()` (strategy comparison, selection)
- Phase 2 → `set_part7_insights()` (calibration method, threshold)
- Phase 7 → `set_part8_insights()` (ensemble specs, stability)

---

# IMPLEMENTATION ROADMAP

1. **Week 1**: Phase 1 (Verify) + Phase 3 (SHAP - basic)
2. **Week 2**: Phase 4 (Hard samples) + Phase 5 (Feature engineering)
3. **Week 3**: Phase 6 (Imbalance) + refine strategy
4. **Week 4**: Phase 2 (Calibration) + Phase 7 (Ensemble)
5. **Week 5**: Integration testing + leaderboard submission

## Success Metrics

- **Leaderboard Score**: 75+ (from 68.50)
- **Cross-validation Stability**: σ(AUC) < 0.015
- **Calibration**: ECE < 0.05
- **Recall@95%**: Escape rate < 5%

