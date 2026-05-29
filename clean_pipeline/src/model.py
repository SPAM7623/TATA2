"""The defect-detection model.

A single function builds an XGBoost classifier from the validated
configuration. Keeping it in one place guarantees that cross-validation and
the final fit use identical settings - only the random seed changes.
"""

from __future__ import annotations

from xgboost import XGBClassifier

from . import config


def build_model(seed: int, scale_pos_weight: float) -> XGBClassifier:
    """Create an XGBoost classifier with the validated hyperparameters.

    Parameters
    ----------
    seed:
        Random seed for this instance. The ensemble trains one model per seed.
    scale_pos_weight:
        Negative/positive ratio that rebalances the loss for the rare defects.
    """
    return XGBClassifier(
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
        **config.XGB_PARAMS,
    )
