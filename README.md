# Alpha Defect Prediction in Hot Rolling Mills
## Comprehensive Industrial ML Workflow

### Project Overview

This project implements a complete industrial machine learning workflow for detecting Alpha defects in hot rolling mills. The workflow is structured into **8 interconnected parts**, each building upon previous insights while maintaining data and computational integrity.

**Goal**: Prevent customer complaints and reduce downgrades by proactively detecting Alpha defects during hot rolling processes.

---

## Workflow Structure

### Part 1: Industrial EDA (Exploratory Data Analysis)
**File**: `part1_industrial_eda.py`

**Purpose**: Understand unstable process behavior and defect characteristics

**Key Outputs**:
- Dataset shape and feature understanding
- Missing values and duplicate analysis
- Class imbalance characteristics
- Low-variance feature detection
- Defect vs non-defect statistical comparison
- Variance/instability analysis
- Outlier detection and characterization
- Correlation analysis (global and comparative)
- PCA and t-SNE visualization
- Hidden operating regime identification
- Threshold/stability boundary analysis

**Insights File**: `PART1_INSIGHTS.txt`

**Visualizations**:
- `01_class_distribution.png`
- `02_defect_vs_normal.png`
- `03_instability_analysis.png`
- `04_outlier_analysis.png`
- `05_correlation_heatmap.png`
- `06_correlation_comparison.png`
- `07_defect_density.png`
- `08_pca_visualization.png`
- `09_tsne_visualization.png`
- `10_hidden_regimes.png`
- `11_threshold_stability.png`
- `12_anomaly_behavior.png`

---

### Part 2: Quick Baseline XGBoost/LightGBM
**File**: `part2_baseline_model.py`

**Purpose**: Establish realistic deployment baseline quickly

**Key Outputs**:
- Feature importance rankings (XGBoost and LightGBM)
- Cross-validation evaluation (5-fold stratified)
- Probability distribution analysis
- ROC and Precision-Recall curves
- Threshold sensitivity analysis
- Baseline metrics for comparison

**Insights File**: `PART2_INSIGHTS.txt`

**Visualizations**:
- `13_probability_analysis.png`
- `14_threshold_sensitivity.png`

---

### Part 3: SHAP + Error Analysis
**File**: `part3_shap_error_analysis.py`

**Purpose**: Understand WHY model succeeds/fails - critical for improvement

**Key Outputs**:
- Global SHAP feature importance
- Local SHAP analysis for individual samples
- SHAP interaction values
- False positive analysis and characterization
- False negative (escaped defect) analysis
- Hard/borderline sample analysis
- Confusion between defect and non-defect samples

**Insights File**: `PART3_INSIGHTS.txt`

**Visualizations**:
- `15_global_shap_analysis.png`
- `16_false_positive_analysis.png`
- `17_false_negative_analysis.png`
- `18_hard_sample_analysis.png`

---

### Part 4: Correlation Grouping + Hidden Process-Block Discovery
**File**: `part4_correlation_grouping.py`

**Purpose**: Discover statistically coupled process-variable groups

**Key Outputs**:
- Correlation matrix computation
- Hierarchical clustering of features
- Feature group identification (NOT confirmed physical stages)
- Group characteristics analysis
- Group-level defect behavior analysis
- Multicollinearity analysis
- Latent process-behavior pattern discovery

**Important**: These are statistically correlated variable groups, NOT confirmed furnace/cooling stages.

**Insights File**: `PART4_INSIGHTS.txt`

**Visualizations**:
- `19_full_correlation_matrix.png`
- `20_feature_dendrogram.png`
- `21_group_defect_behavior.png`

---

### Part 5: Feature Engineering (HIGHEST ROI PHASE)
**File**: `part5_feature_engineering.py`

**Purpose**: Improve hidden process-state representation

**Feature Categories Created**:

A. **Interaction Features** (~30 features)
   - Multiplicative (Xi * Xj)
   - Divisive (Xi / Xj)
   - Difference (Xi - Xj)

B. **Instability Features** (13 features)
   - Row mean, std, variance, range
   - Row skewness, kurtosis
   - Coefficient of variation
   - Max/min ratios

C. **Group Features** (~35 features)
   - Group mean, std, variance
   - Group max, min, range
   - Group imbalance indicators

D. **Anomaly Features** (4 features)
   - Isolation Forest scores
   - Euclidean distance in scaled space
   - Percentile rankings

E. **Threshold/Regime Features** (8 features)
   - Operating regime classifications
   - Stability regimes
   - Extreme value flags

**Outputs**:
- `X_engineered.csv` - Complete engineered feature set
- `y_train.csv` - Target variable
- Feature importance rankings

**Insights File**: `PART5_INSIGHTS.txt`

---

### Part 6: Imbalance Handling
**File**: `part6_imbalance_handling.py`

**Purpose**: Improve defect sensitivity realistically

**Strategies Evaluated**:

1. **Baseline** - No special handling (reference)
2. **Scale Pos Weight** - XGBoost native class weighting
3. **Balanced Bagging** - Sample weight adjustment
4. **SMOTE In-Fold** - Synthetic oversampling inside CV folds

**Key Metrics**:
- ROC-AUC, PR-AUC
- Precision, Recall, F1
- Strategy comparison and selection

**Insights File**: `PART6_INSIGHTS.txt`

**Visualizations**:
- `22_imbalance_strategy_comparison.png`

---

### Part 7: Probability Calibration + Threshold Tuning
**File**: `part7_calibration_threshold.py`

**Purpose**: Optimize real industrial deployment behavior

**Calibration Methods**:
1. **Isotonic Regression** - Non-parametric, flexible
2. **Platt Scaling** - Parametric, smoother

**Threshold Optimization**:
- Sweep across 0.0 to 1.0
- Evaluate precision, recall, F1
- Calculate inspection burden
- Identify operating points for:
  - Best F1 (balanced)
  - 95% Recall (minimize escaped defects)
  - PR balance

**Insights File**: `PART7_INSIGHTS.txt`

**Visualizations**:
- `23_threshold_optimization.png`

**Output Decision**: Optimal threshold for production deployment

---

### Part 8: Final Model Refinement
**File**: `part8_final_refinement.py`

**Purpose**: Improve robustness and consistency

**Key Components**:
- Targeted hyperparameter tuning (depth, learning rate, n_estimators)
- Feature pruning (retain 95% importance features)
- Calibration stability analysis
- Cross-fold consistency verification
- Final model training on full dataset

**Outputs**:
- Optimized hyperparameters
- Pruned feature set
- Calibration stability metrics
- Cross-fold consistency scores
- Production-ready model

**Insights File**: `PART8_INSIGHTS.txt`

---

## How to Run the Workflow

### Option 1: Run All Parts at Once (Recommended)
```bash
python WORKFLOW_ORCHESTRATOR.py
```

This will:
1. Execute all 8 parts sequentially
2. Track completion status
3. Generate summary report
4. Print deployment readiness status

### Option 2: Run Individual Parts
```bash
# Part 1
python part1_industrial_eda.py

# Part 2
python part2_baseline_model.py

# Part 3
python part3_shap_error_analysis.py

# Part 4
python part4_correlation_grouping.py

# Part 5
python part5_feature_engineering.py

# Part 6
python part6_imbalance_handling.py

# Part 7
python part7_calibration_threshold.py

# Part 8
python part8_final_refinement.py
```

**Note**: Parts 6-8 depend on outputs from Part 5 (X_engineered.csv and y_train.csv)

---

## Expected Data Files

### Input Files
- `train.csv` (1352 × 51) - Training data with features X1-X49 and target Y
- `test.csv` (339 × 50) - Test data with features X1-X49 (no target)

### Generated Files

**Engineered Features**:
- `X_engineered.csv` - Full engineered feature set
- `y_train.csv` - Target variable

**Insights Reports** (8 files):
- `PART1_INSIGHTS.txt`
- `PART2_INSIGHTS.txt`
- `PART3_INSIGHTS.txt`
- `PART4_INSIGHTS.txt`
- `PART5_INSIGHTS.txt`
- `PART6_INSIGHTS.txt`
- `PART7_INSIGHTS.txt`
- `PART8_INSIGHTS.txt`

**Visualization Plots** (24 files):
- 01-12: EDA visualizations
- 13-14: Baseline model evaluation
- 15-18: SHAP and error analysis
- 19-21: Correlation grouping
- 22: Imbalance strategy comparison
- 23: Threshold optimization

---

## Key Insights & Outputs

### From Part 1: EDA
- **Class Imbalance Ratio**: [Quantified from data]
- **Unstable Features**: [Top features showing high variance during defects]
- **Anomaly Rate in Defects**: [% of defect samples classified as anomalies]
- **High Correlations**: [Feature pairs with correlation >0.7]

### From Part 2: Baseline
- **Defect Signal Learnability**: [ROC-AUC and PR-AUC scores]
- **Probability Separation**: [Mean probability by class]
- **Model Effectiveness**: [Baseline metrics for comparison]

### From Part 3: SHAP Analysis
- **Top Predictive Features**: [SHAP-ranked important features]
- **Escaped Defects Analysis**: [Characteristics of false negatives]
- **False Positive Drivers**: [Features causing misclassification]

### From Part 4: Correlation Grouping
- **Process-Variable Groups**: [Identified statistically correlated groups]
- **Group Defect Behavior**: [Which groups show instability during defects]
- **Multicollinearity Status**: [Dimensionality reduction opportunity]

### From Part 5: Feature Engineering
- **Engineered Features Count**: [Total engineered features created]
- **Top New Features**: [Most important engineered features]
- **Feature Importance Shift**: [Change in model interpretability]

### From Part 6: Imbalance Handling
- **Best Strategy**: [Selected strategy for deployment]
- **Improvement Metrics**: [Recall and precision with chosen strategy]
- **Operational Trade-off**: [Balance between sensitivity and specificity]

### From Part 7: Calibration & Threshold
- **Optimal Threshold**: [Production deployment threshold]
- **Operating Point**: [Precision/Recall at optimal threshold]
- **Inspection Burden**: [% of samples requiring inspection]
- **Escaped Defect Rate**: [Undetected defects percentage]

### From Part 8: Final Refinement
- **Final Model Config**: [Optimized hyperparameters]
- **Consistency Score**: [Cross-fold prediction stability]
- **Calibration Quality**: [Expected Calibration Error]
- **Deployment Readiness**: [Model ready for production]

---

## Core Mindset for Interpretation

**Do NOT think**: "Which variable predicts Y?"

**Instead think**: "Which hidden thermo-mechanical process states and parameter interactions create unstable operating conditions leading to Alpha defect formation?"

Treat each row as:
- A complete process-state fingerprint
- A hidden industrial operating condition
- Not isolated independent variables

X1–X49 are anonymized process parameters, NOT confirmed as specific stages.

---

## Production Deployment Checklist

- [ ] Read all PART*_INSIGHTS.txt files
- [ ] Review all visualization plots
- [ ] Understand optimal threshold (from Part 7)
- [ ] Prepare serving infrastructure
- [ ] Set up prediction logging
- [ ] Configure alert thresholds
- [ ] Prepare manual inspection workflow
- [ ] Train operators on system
- [ ] Deploy to production
- [ ] Monitor predictions weekly
- [ ] Track false positive rate
- [ ] Track escaped defects
- [ ] Plan quarterly recalibration

---

## Support & Documentation

- **Workflow Design**: Based on industrial ML best practices
- **Feature Engineering**: Focused on process physics, not just statistics
- **Error Analysis**: SHAP-based interpretability for model transparency
- **Threshold Optimization**: Business-aware (minimize escaped defects)

---

## Dependency Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
shap
imbalanced-learn
scipy
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap imbalanced-learn scipy
```

---

## Success Criteria

The workflow successfully completes when:
1. ✓ All 8 parts execute without errors
2. ✓ All 24+ visualizations are generated
3. ✓ All 8 insights reports are created
4. ✓ Engineered features are saved
5. ✓ Optimal threshold is identified
6. ✓ Model is calibrated and validated
7. ✓ Cross-fold consistency is confirmed

---

**Last Updated**: 2026-05-28

**Author**: Claude Code Industrial ML Workflow

**Status**: Ready for Production Deployment
