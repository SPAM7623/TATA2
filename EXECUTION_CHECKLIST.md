# Alpha Defect Prediction Workflow - Execution Checklist

## Pre-Execution Setup

- [ ] **Data Files Ready**
  - [ ] `train.csv` present (1352 × 51)
  - [ ] `test.csv` present (339 × 50)
  - [ ] Both files verified and clean

- [ ] **Environment Setup**
  - [ ] Python 3.7+ installed
  - [ ] All dependencies installed (see README.md)
  - [ ] Working directory is `/home/user/TATA2/`

- [ ] **Documentation Ready**
  - [ ] Problem statement understood
  - [ ] Workflow structure reviewed
  - [ ] Roles and responsibilities clear

---

## Part 1: Industrial EDA

**File**: `part1_industrial_eda.py`

**Execution**: `python part1_industrial_eda.py`

### Sub-tasks Checklist

- [ ] **1.1 Data Loading** (part1_industrial_eda.py:load_data)
  - [ ] Train data loaded
  - [ ] Test data loaded
  - [ ] Column names verified
  - [ ] Dimensions confirmed

- [ ] **1.2 Basic Statistics** (part1_industrial_eda.py:basic_statistics)
  - [ ] Data info printed
  - [ ] Descriptive statistics generated
  - [ ] Data types verified

- [ ] **1.3 Missing Values & Duplicates** (part1_industrial_eda.py:missing_and_duplicates)
  - [ ] Missing values checked
  - [ ] Duplicate rows identified
  - [ ] Constant features listed

- [ ] **1.4 Class Imbalance Analysis** (part1_industrial_eda.py:class_imbalance_analysis)
  - [ ] Class distribution quantified
  - [ ] Imbalance ratio calculated
  - [ ] Visualization: `01_class_distribution.png` ✓

- [ ] **1.5 Low-Variance Features** (part1_industrial_eda.py:low_variance_features)
  - [ ] Low variance features identified
  - [ ] Bottom 5% features noted

- [ ] **1.6 Defect vs Normal Comparison** (part1_industrial_eda.py:defect_vs_normal_comparison)
  - [ ] Mean differences calculated
  - [ ] Visualization: `02_defect_vs_normal.png` ✓

- [ ] **1.7 Instability Analysis** (part1_industrial_eda.py:variance_instability_analysis)
  - [ ] Unstable features identified
  - [ ] Variance ratio computed
  - [ ] Visualization: `03_instability_analysis.png` ✓

- [ ] **1.8 Outlier Analysis** (part1_industrial_eda.py:outlier_analysis)
  - [ ] Isolation Forest applied
  - [ ] Outlier rates by class
  - [ ] Visualization: `04_outlier_analysis.png` ✓

- [ ] **1.9 Correlation Analysis** (part1_industrial_eda.py:correlation_analysis)
  - [ ] Correlation matrix computed
  - [ ] High correlations identified
  - [ ] Visualization: `05_correlation_heatmap.png` ✓

- [ ] **1.10 Correlation Difference** (part1_industrial_eda.py:correlation_difference_analysis)
  - [ ] Defect vs normal correlations compared
  - [ ] Visualization: `06_correlation_comparison.png` ✓

- [ ] **1.11 Pairwise Interactions** (part1_industrial_eda.py:pairwise_interaction_exploration)
  - [ ] Top interactions identified
  - [ ] Interaction strength calculated

- [ ] **1.12 Defect Density Analysis** (part1_industrial_eda.py:defect_density_analysis)
  - [ ] Density regions defined
  - [ ] Visualization: `07_defect_density.png` ✓

- [ ] **1.13 PCA Visualization** (part1_industrial_eda.py:pca_visualization)
  - [ ] PCA variance explained quantified
  - [ ] Visualization: `08_pca_visualization.png` ✓

- [ ] **1.14 t-SNE Visualization** (part1_industrial_eda.py:tsne_visualization)
  - [ ] t-SNE computed
  - [ ] Visualization: `09_tsne_visualization.png` ✓

- [ ] **1.15 Hidden Regime Exploration** (part1_industrial_eda.py:hidden_regime_exploration)
  - [ ] Operating regimes identified
  - [ ] Visualization: `10_hidden_regimes.png` ✓

- [ ] **1.16 Threshold/Stability Analysis** (part1_industrial_eda.py:threshold_stability_analysis)
  - [ ] Stability metrics computed
  - [ ] Visualization: `11_threshold_stability.png` ✓

- [ ] **1.17 Anomaly Behavior** (part1_industrial_eda.py:anomaly_clustering_analysis)
  - [ ] Anomaly scores analyzed
  - [ ] Visualization: `12_anomaly_behavior.png` ✓

- [ ] **PART 1 OUTPUT**
  - [ ] `PART1_INSIGHTS.txt` generated ✓
  - [ ] 12 visualizations created ✓

---

## Part 2: Quick Baseline XGBoost/LightGBM

**File**: `part2_baseline_model.py`

**Execution**: `python part2_baseline_model.py`

**Dependency**: Requires train.csv and test.csv

### Sub-tasks Checklist

- [ ] **2.1-2.2 Data Loading & Preparation**
  - [ ] Features extracted
  - [ ] Target variable separated
  - [ ] Data shapes confirmed

- [ ] **2.3 XGBoost Baseline**
  - [ ] Model trained
  - [ ] Feature importances extracted
  - [ ] Top 10 features identified

- [ ] **2.4 LightGBM Baseline**
  - [ ] Model trained
  - [ ] Feature importances extracted
  - [ ] Compared with XGBoost

- [ ] **2.5 Cross-Validation Evaluation**
  - [ ] 5-fold stratified CV executed
  - [ ] ROC-AUC scores computed
  - [ ] PR-AUC scores computed
  - [ ] Precision, Recall, F1 calculated

- [ ] **2.6 Probability Analysis**
  - [ ] Probability distributions examined
  - [ ] Separation between classes quantified
  - [ ] Visualization: `13_probability_analysis.png` ✓

- [ ] **2.7 Threshold Sensitivity**
  - [ ] Threshold sweep (0.0-1.0) executed
  - [ ] Optimal F1 threshold identified
  - [ ] Visualization: `14_threshold_sensitivity.png` ✓

- [ ] **PART 2 OUTPUT**
  - [ ] `PART2_INSIGHTS.txt` generated ✓
  - [ ] 2 visualizations created ✓
  - [ ] Baseline metrics established ✓

---

## Part 3: SHAP + Error Analysis

**File**: `part3_shap_error_analysis.py`

**Execution**: `python part3_shap_error_analysis.py`

**Dependency**: Requires train.csv

### Sub-tasks Checklist

- [ ] **3.1 Data Loading**
  - [ ] Training data loaded
  - [ ] Features and target separated

- [ ] **3.2 Model Training**
  - [ ] XGBoost model trained for SHAP
  - [ ] Predictions made
  - [ ] Accuracy baseline established

- [ ] **3.3 Global SHAP Analysis**
  - [ ] TreeExplainer initialized
  - [ ] SHAP values computed
  - [ ] Feature importance from SHAP extracted
  - [ ] Visualization: `15_global_shap_analysis.png` ✓

- [ ] **3.4 Local SHAP Analysis**
  - [ ] Individual sample explanations computed
  - [ ] Interesting samples identified

- [ ] **3.5 SHAP Interactions**
  - [ ] Interaction values computed
  - [ ] Top interactions identified
  - [ ] Process interactions understood

- [ ] **3.6 False Positive Analysis**
  - [ ] FP samples identified
  - [ ] FP characteristics extracted
  - [ ] Distinguishing features identified
  - [ ] Visualization: `16_false_positive_analysis.png` ✓

- [ ] **3.7 False Negative Analysis** ⚠️ **CRITICAL**
  - [ ] FN (escaped defect) samples identified
  - [ ] FN rate calculated
  - [ ] FN characteristics analyzed
  - [ ] Visualization: `17_false_negative_analysis.png` ✓

- [ ] **3.8 Hard Sample Analysis**
  - [ ] Borderline samples identified
  - [ ] Hard vs easy sample comparison
  - [ ] Visualization: `18_hard_sample_analysis.png` ✓

- [ ] **PART 3 OUTPUT**
  - [ ] `PART3_INSIGHTS.txt` generated ✓
  - [ ] 4 visualizations created ✓
  - [ ] Error patterns understood ✓

---

## Part 4: Correlation Grouping + Hidden Process-Block Discovery

**File**: `part4_correlation_grouping.py`

**Execution**: `python part4_correlation_grouping.py`

**Dependency**: Requires train.csv

### Sub-tasks Checklist

- [ ] **4.1 Data Loading**
  - [ ] Training data loaded
  - [ ] Features extracted

- [ ] **4.2 Correlation Matrix**
  - [ ] Correlation matrix computed
  - [ ] High correlations (>0.6) identified
  - [ ] Count of correlation pairs noted
  - [ ] Visualization: `19_full_correlation_matrix.png` ✓

- [ ] **4.3 Hierarchical Clustering**
  - [ ] Distance matrix computed (1 - |correlation|)
  - [ ] Ward linkage applied
  - [ ] Dendrogram created
  - [ ] Visualization: `20_feature_dendrogram.png` ✓

- [ ] **4.4 Feature Group Identification**
  - [ ] Clusters cut at threshold=1.5
  - [ ] Groups enumerated
  - [ ] Features per group listed
  - [ ] Group names assigned

- [ ] **4.5 Group Characteristics**
  - [ ] Within-group statistics computed
  - [ ] Mean values per group
  - [ ] Std deviation per group
  - [ ] Internal correlations analyzed

- [ ] **4.6 Defect Behavior by Group** ⚠️ **IMPORTANT**
  - [ ] Group means during defects computed
  - [ ] Group means during normal computed
  - [ ] Mean shifts calculated
  - [ ] Std shifts calculated
  - [ ] Visualization: `21_group_defect_behavior.png` ✓

- [ ] **4.7 Multicollinearity Analysis**
  - [ ] Condition number computed
  - [ ] Matrix rank calculated
  - [ ] Per-group multicollinearity assessed

- [ ] **PART 4 OUTPUT**
  - [ ] `PART4_INSIGHTS.txt` generated ✓
  - [ ] 3 visualizations created ✓
  - [ ] Feature groups identified ✓

---

## Part 5: Feature Engineering (HIGHEST ROI)

**File**: `part5_feature_engineering.py`

**Execution**: `python part5_feature_engineering.py`

**Dependency**: Requires train.csv

### Sub-tasks Checklist

- [ ] **5.1 Data Loading**
  - [ ] Training data loaded
  - [ ] Features and target separated

- [ ] **5.2 Interaction Features** (~30 features)
  - [ ] Top 10 features selected
  - [ ] Multiplication interactions created
  - [ ] Division interactions created
  - [ ] Difference interactions created

- [ ] **5.3 Instability Features** (13 features)
  - [ ] Row mean created
  - [ ] Row std created
  - [ ] Row variance created
  - [ ] Row range (max-min) created
  - [ ] Row median created
  - [ ] Row IQR created
  - [ ] Row skewness created
  - [ ] Row kurtosis created
  - [ ] Coefficient of variation created
  - [ ] Max ratio created
  - [ ] Min ratio created

- [ ] **5.4 Group Features** (~35 features)
  - [ ] Feature groups confirmed
  - [ ] Group mean per group created
  - [ ] Group std per group created
  - [ ] Group variance per group created
  - [ ] Group max per group created
  - [ ] Group min per group created
  - [ ] Group range per group created
  - [ ] Group imbalance per group created

- [ ] **5.5 Anomaly Features** (4 features)
  - [ ] Isolation Forest trained
  - [ ] Anomaly scores computed
  - [ ] Anomaly binary flag created
  - [ ] Euclidean distance computed
  - [ ] Percentile rank computed

- [ ] **5.6 Threshold/Regime Features** (8 features)
  - [ ] Mean-based regimes created (Low/Mid/High)
  - [ ] Stability regimes created (High/Medium/Low)
  - [ ] Extreme value flags created

- [ ] **5.7 Feature Importance & Selection**
  - [ ] XGBoost trained on engineered features
  - [ ] Feature importances extracted
  - [ ] Top 20 features ranked
  - [ ] X_engineered.csv saved ✓
  - [ ] y_train.csv saved ✓

- [ ] **PART 5 OUTPUT**
  - [ ] `PART5_INSIGHTS.txt` generated ✓
  - [ ] `X_engineered.csv` created ✓
  - [ ] `y_train.csv` created ✓
  - [ ] ~90-110 total engineered features ✓

---

## Part 6: Imbalance Handling

**File**: `part6_imbalance_handling.py`

**Execution**: `python part6_imbalance_handling.py`

**Dependency**: Requires X_engineered.csv and y_train.csv (from Part 5)

### Sub-tasks Checklist

- [ ] **6.1 Data Loading**
  - [ ] X_engineered.csv loaded
  - [ ] y_train.csv loaded
  - [ ] Class distribution confirmed

- [ ] **6.2 Baseline Metrics** (No imbalance handling)
  - [ ] 5-fold CV executed
  - [ ] ROC-AUC scores computed
  - [ ] PR-AUC scores computed
  - [ ] Precision, Recall, F1 computed

- [ ] **6.3 Scale Pos Weight Strategy**
  - [ ] Pos weight calculated
  - [ ] XGBoost with scale_pos_weight trained
  - [ ] 5-fold CV executed
  - [ ] Metrics compared with baseline

- [ ] **6.4 Balanced Bagging Strategy**
  - [ ] Sample weights assigned (2x for defects)
  - [ ] XGBoost with weighted training
  - [ ] 5-fold CV executed
  - [ ] Metrics compared with baseline

- [ ] **6.5 SMOTE In-Fold Strategy**
  - [ ] SMOTE applied only inside CV training folds
  - [ ] XGBoost trained on SMOTE-generated data
  - [ ] Validation done on original data
  - [ ] 5-fold CV executed
  - [ ] Metrics compared with baseline

- [ ] **6.6 Strategy Comparison**
  - [ ] All 4 strategies compared
  - [ ] Best F1 strategy identified
  - [ ] Best Recall strategy identified
  - [ ] Visualization: `22_imbalance_strategy_comparison.png` ✓

- [ ] **PART 6 OUTPUT**
  - [ ] `PART6_INSIGHTS.txt` generated ✓
  - [ ] 1 visualization created ✓
  - [ ] Best imbalance strategy selected ✓

---

## Part 7: Probability Calibration + Threshold Tuning

**File**: `part7_calibration_threshold.py`

**Execution**: `python part7_calibration_threshold.py`

**Dependency**: Requires X_engineered.csv and y_train.csv (from Part 5)

### Sub-tasks Checklist

- [ ] **7.1 Data Loading**
  - [ ] X_engineered.csv loaded
  - [ ] y_train.csv loaded

- [ ] **7.2 Base Model Training**
  - [ ] Model with scale_pos_weight trained
  - [ ] Full dataset training completed

- [ ] **7.3 Isotonic Regression Calibration**
  - [ ] 5-fold CV calibration applied
  - [ ] ECE (Expected Calibration Error) computed
  - [ ] ECE improvement calculated
  - [ ] Calibrated probabilities generated

- [ ] **7.4 Platt Scaling Calibration**
  - [ ] Sigmoid calibration applied
  - [ ] Alternative calibration computed

- [ ] **7.5 Threshold Optimization** ⚠️ **CRITICAL**
  - [ ] 101 thresholds evaluated (0.0-1.0)
  - [ ] Precision vs threshold computed
  - [ ] Recall vs threshold computed
  - [ ] F1 vs threshold computed
  - [ ] Inspection burden computed
  - [ ] Optimal F1 threshold identified
  - [ ] 95% Recall threshold identified
  - [ ] PR-balance threshold identified
  - [ ] Visualization: `23_threshold_optimization.png` ✓

- [ ] **7.6 Operating Point Analysis**
  - [ ] Confusion matrix at optimal threshold
  - [ ] True/False Positive/Negative counts
  - [ ] Sensitivity, Specificity calculated
  - [ ] Escaped defect rate computed
  - [ ] **Production threshold selected** ✓

- [ ] **PART 7 OUTPUT**
  - [ ] `PART7_INSIGHTS.txt` generated ✓
  - [ ] 1 visualization created ✓
  - [ ] **Optimal threshold identified** ✓
  - [ ] Operating metrics documented ✓

---

## Part 8: Final Model Refinement

**File**: `part8_final_refinement.py`

**Execution**: `python part8_final_refinement.py`

**Dependency**: Requires X_engineered.csv and y_train.csv (from Part 5)

### Sub-tasks Checklist

- [ ] **8.1 Data Loading**
  - [ ] X_engineered.csv loaded
  - [ ] y_train.csv loaded

- [ ] **8.2 Hyperparameter Tuning** ⚠️ **IMPORTANT**
  - [ ] Grid search executed (depth, learning_rate, n_estimators)
  - [ ] 5-fold CV for each combination
  - [ ] F1 scores computed
  - [ ] Best parameters identified
  - [ ] Top 10 combinations ranked

- [ ] **8.3 Feature Pruning**
  - [ ] Feature importances extracted
  - [ ] 95% importance threshold calculated
  - [ ] Reduced feature set identified
  - [ ] Performance impact assessed
  - [ ] Dimensionality reduction verified

- [ ] **8.4 Calibration Stability**
  - [ ] ECE computed across 5 folds
  - [ ] Mean ECE reported
  - [ ] Std ECE reported
  - [ ] Stability verified

- [ ] **8.5 Consistency Analysis** ⚠️ **CRITICAL**
  - [ ] 5-fold consistency evaluation
  - [ ] ROC-AUC per fold computed
  - [ ] PR-AUC per fold computed
  - [ ] Precision, Recall per fold computed
  - [ ] Coefficient of Variation calculated
  - [ ] Cross-fold consistency confirmed

- [ ] **8.6 Final Model Selection**
  - [ ] Best model trained on full dataset
  - [ ] Final hyperparameters confirmed
  - [ ] Feature set finalized
  - [ ] Model configuration documented

- [ ] **PART 8 OUTPUT**
  - [ ] `PART8_INSIGHTS.txt` generated ✓
  - [ ] Final model trained ✓
  - [ ] Hyperparameters optimized ✓
  - [ ] Consistency verified ✓

---

## Post-Execution Validation

- [ ] **All Files Generated**
  - [ ] 8 insights reports (PART1-8_INSIGHTS.txt)
  - [ ] 24+ visualization plots (PNG files)
  - [ ] Engineered features (X_engineered.csv)
  - [ ] Target variable (y_train.csv)

- [ ] **Insights Review**
  - [ ] Read PART1_INSIGHTS.txt ✓
  - [ ] Read PART2_INSIGHTS.txt ✓
  - [ ] Read PART3_INSIGHTS.txt ✓
  - [ ] Read PART4_INSIGHTS.txt ✓
  - [ ] Read PART5_INSIGHTS.txt ✓
  - [ ] Read PART6_INSIGHTS.txt ✓
  - [ ] Read PART7_INSIGHTS.txt ✓
  - [ ] Read PART8_INSIGHTS.txt ✓

- [ ] **Key Metrics Documented**
  - [ ] Class imbalance ratio noted
  - [ ] ROC-AUC baseline documented
  - [ ] Best feature engineering impact noted
  - [ ] Optimal imbalance strategy identified
  - [ ] **Production threshold identified** ⚠️
  - [ ] Cross-fold consistency verified
  - [ ] Model ready for deployment ✓

---

## Production Deployment Readiness

- [ ] **Model Configuration**
  - [ ] Optimal hyperparameters documented
  - [ ] Feature engineering pipeline defined
  - [ ] Calibration method selected
  - [ ] Optimal threshold identified

- [ ] **Data Pipeline**
  - [ ] Input data format defined
  - [ ] Feature engineering reproducible
  - [ ] Prediction output format defined
  - [ ] Probability score included

- [ ] **Monitoring Plan**
  - [ ] Prediction logging enabled
  - [ ] False positive rate tracking
  - [ ] Escaped defect rate tracking
  - [ ] Probability distribution monitoring
  - [ ] Quarterly recalibration scheduled

- [ ] **Documentation**
  - [ ] README.md reviewed ✓
  - [ ] All insights reports filed
  - [ ] Visualizations cataloged
  - [ ] Model card prepared

- [ ] **Deployment**
  - [ ] Serving infrastructure ready
  - [ ] Model serialized and versioned
  - [ ] Threshold configuration locked
  - [ ] Alert thresholds set
  - [ ] Fallback procedures defined

---

## Final Sign-Off

- [ ] **Workflow Complete**: All 8 parts executed successfully
- [ ] **Quality Assured**: Outputs reviewed and validated
- [ ] **Deployment Ready**: Model ready for production use
- [ ] **Documentation Complete**: All insights documented
- [ ] **Monitoring Plan**: Post-deployment tracking defined

**Workflow Status**: ✅ **READY FOR PRODUCTION DEPLOYMENT**

---

**Last Updated**: 2026-05-28

**Completion Date**: [To be filled]

**Approved By**: [To be filled]
