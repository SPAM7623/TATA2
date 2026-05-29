# Pipeline Checklist — EDA → 5-Seed Ensemble

A complete, honest record of every part, sub-part, and model in the
`clean_pipeline/`. Status legend:

- ✅ done & validated
- ⚙️ implemented (utility / infrastructure)
- ❌ tried and rejected (with measured evidence)

---

## Part 1 — Exploratory Data Analysis (`eda.py`)

- ✅ Load train (1352 × 49) and test (339 × 49)
- ✅ Class balance: 1286 non-defective vs 66 defective (19.5 : 1)
- ✅ Missing-value audit: 249 cells (train), 68 cells (test), 12 affected columns
- ✅ Feature-scale inspection (X35 ~14M down to small ratios)
- ✅ Design takeaways recorded:
  - ✅ Few positives → guard against overfitting
  - ✅ Missing values → median imputation
  - ✅ Strong imbalance → `scale_pos_weight`
  - ✅ Wild scales → no scaling needed (tree model)

## Part 2 — Configuration (`src/config.py`)

- ⚙️ Centralised paths (train/test/artifacts)
- ⚙️ Column names (`CoilID`, `Y`)
- ⚙️ CV settings: 5 folds, seed 42
- ⚙️ Ensemble seeds: `[42, 123, 999, 2025, 7777]`
- ✅ Validated XGBoost hyperparameters (single source of truth)
- ⚙️ Default operating threshold (0.0068)

## Part 3 — Data Preparation (`src/data.py`)

- ⚙️ `feature_columns` — select X1…X49
- ⚙️ `load_train` / `load_test`
- ✅ `impute` — median imputation, **fit on train fold only** (no leakage)
- ✅ `scale_pos_weight` — negative/positive ratio (~19.5)

## Part 4 — Model (`src/model.py`)

- ✅ `build_model(seed, scale_pos_weight)` — one factory, identical config in CV
  and final fit; only the seed changes

## Part 5 — Evaluation (`src/evaluation.py`)

- ✅ `cross_val_oof` — leakage-free out-of-fold probabilities, seed-averaged
- ✅ `impute_fold` — per-fold imputation helper
- ✅ `summarise` — metric block at a chosen threshold
- ✅ `best_f1_threshold` — threshold scan for the best F1

## Part 6 — Training (`train.py`)

- ✅ Cross-validate **single seed** → OOF metrics
- ✅ Cross-validate **5-seed ensemble** → OOF metrics
- ✅ Fit final ensemble (one model per seed) on all training data
- ✅ Cache out-of-fold predictions → `artifacts/oof_predictions.csv`
- ✅ Cache test probabilities → `artifacts/test_probabilities.csv`

## Part 7 — Prediction (`predict.py`)

- ✅ Load cached test probabilities (no retraining)
- ✅ Apply chosen threshold (`--threshold`, default 0.0068)
- ✅ Write `artifacts/submission.csv` (CoilID, Y)
- ✅ Report defect / clean counts

---

## Models evaluated (leakage-free 5-fold OOF)

| Model | ROC-AUC | PR-AUC | Decision |
|---|---|---|---|
| ✅ XGBoost — regularised (final) | 0.8655 | 0.3217 | **Adopted (base)** |
| ✅ XGBoost — 5-seed ensemble (final) | 0.8664–0.8668 | ~0.31 | **Adopted (production)** |
| ❌ XGBoost — original (d6, 150, lr0.1) | 0.8618 | 0.3240 | Replaced by regularised |
| ❌ XGBoost — heavy-reg (d3, 400, lr0.02) | 0.8614 | 0.3102 | No gain |
| ❌ LightGBM | 0.8500 | 0.2855 | Weaker; dropped |
| ❌ CatBoost | 0.8646 | 0.3282 | Strong PR-AUC but not better overall |
| ❌ Ensemble (equal XGB+LGBM+CAT) | 0.8647 | 0.3247 | Did not beat single XGB |
| ❌ Ensemble (0.4/0.3/0.3) | 0.8649 | 0.3227 | Did not beat single XGB |
| ❌ XGB + Isotonic calibration | 0.8435 | 0.2636 | Hurt ranking |
| ❌ XGB + Platt calibration | 0.8443 | 0.2929 | Best Brier only, hurt ranking |

## Techniques evaluated

- ✅ Median imputation — adopted
- ✅ `scale_pos_weight` imbalance handling — adopted
- ✅ Multi-seed averaging — adopted (small, free variance reduction)
- ❌ Hand-built interaction features (X34–X38) — every candidate hurt AUC
- ❌ Instability / row-statistic features — added noise
- ❌ Probability recalibration — improved Brier/LogLoss, hurt ranking
- ❌ Hyperparameter re-tuning beyond the regularised config — no stable gain

## Validation & stability

- ✅ Leakage-free 5-fold stratified CV throughout
- ✅ Stability across CV seeds: AUC 0.8626 ± 0.0040
- ✅ Stability across model seeds: AUC 0.8646 ± 0.0019
- ✅ Threshold sensitivity: F1 swing only 0.004 for ±10% shift (robust)
- ✅ Overfitting check: train vs OOF gap documented (PR-AUC gap is the binding
  limit, driven by only 66 positives)

## Known limits

- ✅ Documented: model is saturated at ROC-AUC ≈ 0.866 on this data
- ✅ Documented: the binding constraint is the 66 positive examples, not the
  modelling choices
