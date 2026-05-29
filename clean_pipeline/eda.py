"""Exploratory data analysis.

Prints the handful of facts that actually shaped the modelling decisions:
the class imbalance, how much data is missing, and how wildly the feature
scales differ. Run it first to understand why the pipeline looks the way it
does.
"""

import pandas as pd

from src import config, data


def main() -> None:
    train = pd.read_csv(config.TRAIN_CSV)
    test = pd.read_csv(config.TEST_CSV)
    features = data.feature_columns(train)

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Train: {train.shape[0]} coils x {len(features)} features")
    print(f"Test : {test.shape[0]} coils x {len(features)} features")

    print("\nClass balance (train):")
    counts = train[config.TARGET_COLUMN].value_counts().sort_index()
    n_pos = int(counts.get(1, 0))
    n_neg = int(counts.get(0, 0))
    print(f"  non-defective (0): {n_neg}")
    print(f"  defective     (1): {n_pos}")
    print(f"  imbalance ratio  : {n_neg / max(n_pos, 1):.1f} : 1")

    print("\nMissing values:")
    print(f"  train: {int(train[features].isna().sum().sum())} cells")
    print(f"  test : {int(test[features].isna().sum().sum())} cells")
    cols_with_na = train[features].isna().sum()
    cols_with_na = cols_with_na[cols_with_na > 0]
    print(f"  affected feature columns: {len(cols_with_na)}")

    print("\nFeature scale (median absolute value, top 5 by spread):")
    spread = train[features].abs().median().sort_values(ascending=False)
    for name, value in spread.head().items():
        print(f"  {name:>4}: {value:,.1f}")
    print("  -> scales vary by orders of magnitude; tree models handle this,")
    print("     so no scaling is required.")

    print("\nTakeaways that drive the pipeline:")
    print("  * Few positives  -> guard hard against overfitting")
    print("  * Missing values -> median imputation (fit on train folds only)")
    print("  * Strong imbalance -> scale_pos_weight in the model")


if __name__ == "__main__":
    main()
