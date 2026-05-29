"""Loading and preparing the coil dataset.

The only preprocessing the model needs is median imputation for the missing
sensor readings. To stay leakage-free the imputer is always fit on training
data and merely applied to validation/test data, so it lives here as a small
helper rather than being baked into the loaders.
"""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer

from . import config


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the process-variable columns (X1 ... X49)."""
    return [c for c in df.columns if c.startswith("X")]


def load_train() -> tuple[pd.DataFrame, pd.Series]:
    """Load the training features and target.

    Returns (X, y) where X holds the X-columns and y is the integer target.
    """
    df = pd.read_csv(config.TRAIN_CSV)
    X = df[feature_columns(df)]
    y = df[config.TARGET_COLUMN].astype(int)
    return X, y


def load_test() -> tuple[pd.DataFrame, pd.Series]:
    """Load the test features and the coil IDs needed for submission."""
    df = pd.read_csv(config.TEST_CSV)
    X = df[feature_columns(df)]
    ids = df[config.ID_COLUMN]
    return X, ids


def impute(train_X, valid_X=None):
    """Median-impute features.

    The imputer is fit on `train_X` only. If `valid_X` is given it is
    transformed with that same imputer and both are returned; otherwise only
    the transformed training matrix comes back.
    """
    imputer = SimpleImputer(strategy="median")
    train_out = imputer.fit_transform(train_X)
    if valid_X is None:
        return train_out
    return train_out, imputer.transform(valid_X)


def scale_pos_weight(y) -> float:
    """Negative-to-positive ratio used to balance the loss."""
    y = pd.Series(y)
    positives = max((y == 1).sum(), 1)
    return (y == 0).sum() / positives
