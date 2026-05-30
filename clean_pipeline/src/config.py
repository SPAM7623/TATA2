"""Central configuration for the defect-detection pipeline.

Keeping paths, hyperparameters and seeds in one place means the training and
prediction scripts can never drift out of sync.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
# train.csv / test.csv live in the repository root, one level above this
# package's parent directory.
ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
OOF_CSV = ARTIFACTS / "oof_predictions.csv"
TEST_PROBA_CSV = ARTIFACTS / "test_probabilities.csv"
SUBMISSION_CSV = ARTIFACTS / "submission.csv"

# --- Columns ---------------------------------------------------------------
ID_COLUMN = "CoilID"
TARGET_COLUMN = "Y"

# --- Cross-validation ------------------------------------------------------
N_FOLDS = 5
CV_SEED = 42

# Five seeds for the final ensemble. Averaging across them smooths out the
# run-to-run noise that a single seed shows on such a small positive class.
ENSEMBLE_SEEDS = [42, 123, 999, 2025, 7777]

# Ten seeds for extended ensemble testing.
ENSEMBLE_SEEDS_10 = [42, 123, 999, 2025, 7777, 17, 29, 101, 314, 888]

# --- Model hyperparameters -------------------------------------------------
# Validated by cross-validation: shallow, strongly regularised trees gave the
# best out-of-fold ranking without overfitting the 66 positive examples.
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "min_child_weight": 3,
    "eval_metric": "logloss",
    "n_jobs": -1,
    "verbosity": 0,
}

# Default operating point. The probabilities sit very low because of the class
# weighting, so the useful threshold is also low.
DEFAULT_THRESHOLD = 0.0068
