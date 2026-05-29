"""Step 1 — Exploratory Data Analysis.

Prints the facts that shape every later decision: the class imbalance, how much
data is missing, and how wildly the feature scales differ. Nothing here is
saved; it is a read-only look at the data.

Run:  python 01_eda.py
"""

import pandas as pd

from src import config, data


def main() -> None:
    train = pd.read_csv(config.TRAIN_CSV)
    test = pd.read_csv(config.TEST_CSV)
    features = data.feature_columns(train)

    print("=" * 60)
    print("STEP 1 | EXPLORATORY DATA ANALYSIS")
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
    affected = train[features].isna().sum()
    print(f"  affected columns: {int((affected > 0).sum())}")

    print("\nFeature scale (median absolute value, top 5 by spread):")
    spread = train[features].abs().median().sort_values(ascending=False)
    for name, value in spread.head().items():
        print(f"  {name:>4}: {value:,.1f}")

    print("\nDesign takeaways:")
    print("  * Few positives    -> guard against overfitting (shallow, regularised)")
    print("  * Missing values   -> median imputation, fit on train folds only")
    print("  * Strong imbalance -> scale_pos_weight in the model")
    print("  * Wild scales      -> no scaling needed (tree model)")


if __name__ == "__main__":
    main()
