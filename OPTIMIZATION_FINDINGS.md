# 7-Phase Optimization — Honest Measured Findings

## TL;DR

All configurations were evaluated with **5-fold stratified cross-validation
using out-of-fold predictions** (no data leakage). Results are real, not
assumed.

**Best validated config: Regularized XGBoost, CV AUC = 0.8655** (vs baseline 0.8641).
The improvement is small but real. The elaborate feature engineering and
multi-seed ensembling **did not help** on this dataset and were dropped.

---

## Dataset Reality Check

| Property | Value |
|----------|-------|
| Train rows | 1352 |
| Defects (positives) | 66 |
| Non-defects | 1286 |
| Imbalance ratio | 19.48 : 1 |
| Missing values | 249 NaN |
| Features | 49 (X1–X49) |

With only **66 positive examples**, this dataset is highly prone to
overfitting. Adding many engineered features increases variance faster than
it adds signal.

---

## Measured Results (5-fold OOF AUC)

### Phase 4 — Feature Engineering (REJECTED)

| Configuration | AUC | PR-AUC |
|---------------|-----|--------|
| Baseline (raw + scale_pos_weight) | **0.8641** | 0.3162 |
| + Feature Engineering (instability/anomaly/regime/distance) | 0.8481 | 0.2992 |
| + Calibration (Isotonic) on engineered | 0.8531 | 0.2937 |
| + Multi-seed Ensemble on engineered | 0.8522 | 0.2961 |

➡️ **Feature engineering REDUCED AUC by ~0.016.** Dropped.

### Phases 1/6 — Model & Hyperparameter Search

| Configuration | AUC | PR-AUC |
|---------------|-----|--------|
| XGB original (d6, 150, lr.1, spw10) | 0.8618 | 0.3240 |
| XGB current (d4, 200, lr.05, spw_auto) | 0.8641 | 0.3162 |
| **XGB regularized (d3, 300, lr.03, reg)** | **0.8655** | 0.3217 |
| XGB heavy-reg (d3, 400, lr.02) | 0.8614 | 0.3102 |
| LightGBM (d3, 300, balanced) | 0.8486 | 0.2973 |
| LightGBM (d4, 400, spw) | 0.8453 | 0.3042 |

➡️ **Regularized XGBoost wins.** LightGBM underperforms here.

### Phase 7 — Multi-seed Ensemble (REJECTED)

| Configuration | AUC | PR-AUC |
|---------------|-----|--------|
| Regularized XGB single-seed | **0.8655** | 0.3217 |
| Regularized XGB 5-seed ensemble | 0.8648 | 0.3167 |

➡️ Ensemble slightly **hurt** AUC. Single seed retained.

---

## What Was Adopted

✅ **Regularized XGBoost** hyperparameters (the only real improvement):
```python
XGBClassifier(
    n_estimators=300, max_depth=3, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=2.0, min_child_weight=3,
    scale_pos_weight=(neg/pos),  # ~19.5
    random_state=42
)
```
✅ **Median imputation** for the 249 NaN values (fit on train fold only).
✅ **Leakage-free 5-fold CV** evaluation harness (`integrated_pipeline.py`).

## What Was Rejected (with evidence)

❌ Row-statistic / instability features — added noise (AUC −0.016)
❌ Isolation Forest / distance-to-centroid features — no signal gain
❌ Isotonic calibration — did not improve ranking AUC
❌ Multi-seed ensemble — slightly worse (AUC −0.0007)
❌ LightGBM — consistently below XGBoost here

---

## Honest Assessment of the "+5 to +15 points" Expectation

The roadmap projected large gains from feature engineering and ensembling.
**Measured on this data, those gains did not appear** — the dataset is too
small and imbalanced for additional features to help; they overfit instead.
The realistic, validated gain is **+0.0014 AUC** from better regularization.

If a larger labeled dataset becomes available, feature engineering is worth
revisiting — the infrastructure is in place (`PART_5_FEATURE_ENGINEERING_ADVANCED.py`).

---

## Reproduce

```bash
python integrated_pipeline.py   # feature-engineering comparison
python tune_experiment.py       # hyperparameter / model search
python final_experiment.py      # confirm best + ensemble
python generate_predictions.py 0.28   # generate predictions with best model
```
