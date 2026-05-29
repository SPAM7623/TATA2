# Hot-Rolling Mill Defect Detection

A leakage-free machine-learning pipeline that predicts surface defects on steel
coils from process-sensor measurements.

## The problem

Each coil is described by 49 anonymised process variables (`X1`–`X49`). The
target `Y` marks whether the coil came out defective. The data is small and
heavily imbalanced:

| | Coils | Defects | Ratio | Missing values |
|---|------:|--------:|:-----:|---------------:|
| Train | 1352 | 66 | ~19.5 : 1 | 249 |
| Test  |  339 |  — |   —   |  68 |

With only 66 positive examples the dominant risk is overfitting, so every
modelling choice here is checked with cross-validation rather than assumed.

## Layout

The pipeline is a numbered sequence — run the steps in order, 01 → 06. They
share a small library in `src/`.

```
clean_pipeline/
├── src/
│   ├── config.py              paths, hyperparameters, random seeds
│   ├── data.py                loading + median imputation
│   ├── model.py               the validated XGBoost configuration
│   └── evaluation.py          cross-validation + metrics
├── 01_eda.py                  explore the data
├── 02_preprocessing.py        leakage-free median imputation check
├── 03_baseline_model.py       single-seed XGBoost, out-of-fold metrics
├── 04_model_comparison.py     XGB vs alternatives (why XGB wins)
├── 05_train_5seed_ensemble.py final ensemble: CV + fit + cache probs
└── 06_predict.py              score the test set -> submission.csv
```

## How the final model was chosen

Everything below was measured with 5-fold stratified cross-validation on
out-of-fold predictions (no leakage):

- A **regularised XGBoost** (shallow trees, strong L1/L2) gave the best and most
  stable ranking, ROC-AUC ≈ 0.866.
- **Median imputation** fills the missing values; it is fit on the training fold
  only so the validation fold stays untouched.
- **`scale_pos_weight`** compensates for the 19.5:1 imbalance.
- **Averaging five seeds** trims run-to-run variance for a small, free gain.

Approaches that were tried and dropped because they did not help on this data:
hand-built feature interactions, LightGBM/CatBoost blends, and probability
recalibration (it improved Brier score but hurt ranking).

## Usage

Run from inside `clean_pipeline/`, with `train.csv` and `test.csv` in the
repository root. The steps are meant to be run in order:

```bash
python 01_eda.py                          # data summary
python 02_preprocessing.py                # imputation check
python 03_baseline_model.py               # single-seed OOF metrics
python 04_model_comparison.py             # XGB vs alternatives
python 05_train_5seed_ensemble.py         # CV + fit + cache probabilities
python 06_predict.py --threshold 0.00583  # submission.csv at your threshold
```

Only steps 05 and 06 are required to produce a submission; 01–04 document and
justify the choices behind it.
