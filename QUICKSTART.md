# Quick Start Guide - Alpha Defect Prediction Workflow

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap imbalanced-learn scipy
```

### 2. Verify Data
```bash
# Check that these files exist:
ls -lh train.csv test.csv

# Expected sizes:
# train.csv: ~1.1 MB (1352 rows × 51 cols)
# test.csv: ~264 KB (339 rows × 50 cols)
```

### 3. Run Complete Workflow
```bash
python WORKFLOW_ORCHESTRATOR.py
```

This runs all 8 parts sequentially and generates:
- **8 insights reports** (PART1-8_INSIGHTS.txt)
- **24+ visualizations** (PNG files)
- **Engineered features** (X_engineered.csv)
- **Calibrated model with optimal threshold**

---

## Workflow Overview

| Part | File | Purpose | Outputs | Time |
|------|------|---------|---------|------|
| 1 | part1_industrial_eda.py | Understand data patterns | 12 plots, insights | 3-5 min |
| 2 | part2_baseline_model.py | Establish baseline | 2 plots, metrics | 2-3 min |
| 3 | part3_shap_error_analysis.py | Interpret errors | 4 plots, SHAP insights | 5-8 min |
| 4 | part4_correlation_grouping.py | Discover process groups | 3 plots, feature groups | 2-3 min |
| 5 | part5_feature_engineering.py | Create new features | Engineered features | 2-3 min |
| 6 | part6_imbalance_handling.py | Handle class imbalance | 1 plot, best strategy | 5-7 min |
| 7 | part7_calibration_threshold.py | Optimize threshold | 1 plot, optimal threshold | 3-5 min |
| 8 | part8_final_refinement.py | Final optimization | Model ready | 5-7 min |

**Total Time**: ~30-45 minutes

---

## What Each Part Does

### Part 1: EDA (Industrial Data Exploration)
- Analyzes dataset characteristics
- Identifies class imbalance (critical!)
- Finds unstable features during defects
- Visualizes defect patterns

**Key Insight**: Understand which features become unstable when defects occur

### Part 2: Baseline (Quick Model)
- Trains XGBoost and LightGBM
- Evaluates with 5-fold CV
- Checks if defect signal is learnable
- Establishes comparison baseline

**Key Insight**: ROC-AUC tells us how learnable the problem is

### Part 3: Error Analysis (Why does model fail?)
- SHAP-based feature importance
- Analyzes false positives (wrong alarms)
- Analyzes false negatives (escaped defects) ⚠️ **CRITICAL**
- Finds borderline samples

**Key Insight**: False negatives = Escaped defects = Customer complaints

### Part 4: Grouping (Find process blocks)
- Clusters correlated features
- NOT confirmed physical stages (X1-X49 are anonymized)
- Identifies latent process-behavior groups
- Analyzes group instability during defects

**Key Insight**: Some feature groups become unstable during defects

### Part 5: Feature Engineering (Build new features)
- Creates 90-110 new features from original 49
- Types:
  - Interaction features (X_i * X_j)
  - Instability features (row variance, range, etc.)
  - Group features (aggregate statistics)
  - Anomaly features (outlier scores)
  - Regime features (operating conditions)

**Key Insight**: Engineered features improve model by capturing process states

### Part 6: Imbalance Handling (Fix class imbalance)
- Compares 4 strategies:
  - Baseline (no handling)
  - Scale pos weight (class weighting)
  - Balanced bagging (sample weights)
  - SMOTE (synthetic oversampling)
- Selects best strategy

**Key Insight**: Right imbalance handling improves recall for escaped defects

### Part 7: Calibration & Threshold (Deploy decision)
- Calibrates probabilities (makes them trustworthy)
- Tests 101 thresholds
- Identifies optimal threshold for production
- Shows precision/recall trade-off

**Key Insight**: Threshold determines operationalinspection burden vs safety

### Part 8: Final Refinement (Polish model)
- Tunes hyperparameters
- Prunes unnecessary features
- Verifies consistency across folds
- Confirms model ready for production

**Key Insight**: Final model is stable and reproducible

---

## Expected Output Structure

After running workflow:

```
/home/user/TATA2/
├── train.csv                          # Input data
├── test.csv                           # Test data
├── README.md                          # Full documentation
├── QUICKSTART.md                      # This file
├── EXECUTION_CHECKLIST.md             # Task checklist
├── WORKFLOW_ORCHESTRATOR.py           # Master runner
├── part1_industrial_eda.py            # EDA
├── part2_baseline_model.py            # Baseline
├── part3_shap_error_analysis.py       # Error analysis
├── part4_correlation_grouping.py      # Grouping
├── part5_feature_engineering.py       # Feature engineering
├── part6_imbalance_handling.py        # Imbalance
├── part7_calibration_threshold.py     # Calibration
├── part8_final_refinement.py          # Refinement
│
├── X_engineered.csv                   # Engineered features (from Part 5)
├── y_train.csv                        # Target variable (from Part 5)
│
├── PART1_INSIGHTS.txt                 # EDA insights
├── PART2_INSIGHTS.txt                 # Baseline insights
├── PART3_INSIGHTS.txt                 # Error analysis insights
├── PART4_INSIGHTS.txt                 # Grouping insights
├── PART5_INSIGHTS.txt                 # Feature engineering insights
├── PART6_INSIGHTS.txt                 # Imbalance strategy choice
├── PART7_INSIGHTS.txt                 # **PRODUCTION THRESHOLD HERE**
├── PART8_INSIGHTS.txt                 # Final model config
│
├── 01_class_distribution.png          # Class imbalance
├── 02_defect_vs_normal.png            # Mean/var differences
├── 03_instability_analysis.png        # Unstable features
├── 04_outlier_analysis.png            # Anomaly rates
├── 05_correlation_heatmap.png         # Feature correlations
├── 06_correlation_comparison.png      # Defect vs normal correlations
├── 07_defect_density.png              # Where defects cluster
├── 08_pca_visualization.png           # PCA 2D view
├── 09_tsne_visualization.png          # t-SNE embedding
├── 10_hidden_regimes.png              # Operating conditions
├── 11_threshold_stability.png         # Stability metrics
├── 12_anomaly_behavior.png            # Outlier characteristics
├── 13_probability_analysis.png        # Model probabilities
├── 14_threshold_sensitivity.png       # Threshold tuning
├── 15_global_shap_analysis.png        # Feature importance
├── 16_false_positive_analysis.png     # FP characteristics
├── 17_false_negative_analysis.png     # **Escaped defects** ⚠️
├── 18_hard_sample_analysis.png        # Borderline samples
├── 19_full_correlation_matrix.png     # All correlations
├── 20_feature_dendrogram.png          # Feature clustering
├── 21_group_defect_behavior.png       # Group instability
├── 22_imbalance_strategy_comparison.png # Strategy comparison
└── 23_threshold_optimization.png      # **Threshold curves** ⚠️
```

---

## Critical Files to Review

### For Understanding
1. **README.md** - Complete documentation (start here)
2. **EXECUTION_CHECKLIST.md** - Task tracking

### For Insights (Read these!)
1. **PART1_INSIGHTS.txt** - Data patterns
2. **PART3_INSIGHTS.txt** - Why does model fail? ⚠️
3. **PART5_INSIGHTS.txt** - What new features were created
4. **PART7_INSIGHTS.txt** - **PRODUCTION THRESHOLD** ⚠️
5. **PART8_INSIGHTS.txt** - Final model config

### For Visuals
- **23_threshold_optimization.png** - Shows precision/recall tradeoff
- **17_false_negative_analysis.png** - Shows escaped defects
- **15_global_shap_analysis.png** - Shows most important features

---

## Production Deployment

### Step 1: Extract Key Information
From **PART7_INSIGHTS.txt**, find:
- Optimal threshold (e.g., 0.25)
- Expected precision at that threshold
- Expected recall at that threshold
- Inspection burden (% of samples flagged)

### Step 2: Implement Prediction
```python
# Pseudocode
def predict_defect(process_data):
    # Apply engineered features (from Part 5)
    features = engineer_features(process_data)
    
    # Get probability from model
    probability = final_model.predict_proba(features)[0][1]
    
    # Apply threshold
    THRESHOLD = 0.25  # From Part 7
    is_defect = probability >= THRESHOLD
    
    return {
        'defect': is_defect,
        'confidence': probability,
        'action': 'INSPECT' if is_defect else 'PASS'
    }
```

### Step 3: Monitor Production
- Log all predictions
- Track false positive rate (should stay low)
- Track escaped defects (should be <5%)
- Recalibrate threshold if drift detected

---

## Quick Troubleshooting

### Issue: "SHAP computation is slow"
- Part 3 uses SHAP which is computationally intensive
- This is normal, may take 5-10 minutes
- Run with smaller sample if needed

### Issue: "Out of memory during SMOTE"
- SMOTE can be memory-intensive
- Reduce sample size in Part 6 if needed
- Or skip SMOTE and use other strategies

### Issue: "Different results on re-run"
- Some randomness in cross-validation
- Use fixed random_state (already set to 42)
- Results should be reproducible

---

## Next Steps

1. **Run the workflow**: `python WORKFLOW_ORCHESTRATOR.py`
2. **Review outputs**: Read all PART*_INSIGHTS.txt files
3. **Understand threshold**: Find optimal threshold in PART7_INSIGHTS.txt
4. **Plan deployment**: Use model configuration from PART8_INSIGHTS.txt
5. **Set up monitoring**: Implement performance tracking
6. **Deploy to production**: Use optimal threshold and calibrated model

---

## Key Concepts

**Alpha Defect**: Specific quality issue in hot rolling that:
- Can't be detected inline (coil under tension)
- Only caught by sample inspection (time-intensive)
- Causes customer complaints when missed

**Our Solution**: Use process parameters (X1-X49) to predict defects before manual inspection

**Why 8 Parts?**: Each part builds understanding:
1. Understand data → 2. Build baseline → 3. Analyze errors → 4. Group features → 5. Engineer features → 6. Handle imbalance → 7. Optimize threshold → 8. Finalize model

---

**For full details, see README.md**

**Questions? Review the insights reports - they explain everything!**
