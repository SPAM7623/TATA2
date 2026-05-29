"""Step 2 — Preprocessing check.

The only preprocessing the model needs is median imputation for the missing
sensor readings. This step demonstrates the imputation contract used everywhere
else: the imputer is fit on training data and merely applied to held-out data,
so no information leaks from validation/test into training.

Run:  python 02_preprocessing.py
"""

import numpy as np

from src import data


def main() -> None:
    X, y = data.load_train()
    test_X, _ = data.load_test()

    print("=" * 60)
    print("STEP 2 | PREPROCESSING (leakage-free median imputation)")
    print("=" * 60)
    print(f"Raw train missing cells: {int(X.isna().sum().sum())}")
    print(f"Raw test  missing cells: {int(test_X.isna().sum().sum())}")
    print(f"Imbalance weight (neg/pos): {data.scale_pos_weight(y):.2f}")

    # Impute: fit on train, apply to test. This is the exact call the training
    # and CV code uses.
    train_imp, test_imp = data.impute(X, test_X)

    print("\nAfter imputation:")
    print(f"  train NaNs: {int(np.isnan(train_imp).sum())}")
    print(f"  test  NaNs: {int(np.isnan(test_imp).sum())}")
    print("\nContract: imputer.fit() sees training rows only; test rows are")
    print("transformed with the training medians. Inside cross-validation the")
    print("same rule is applied per fold (see src/evaluation.py).")


if __name__ == "__main__":
    main()
