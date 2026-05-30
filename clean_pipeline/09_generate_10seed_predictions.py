"""Generate predictions at optimal thresholds for both 5-seed and 10-seed."""

import pandas as pd
from src import config

# Load test probabilities
test5 = pd.read_csv(config.ARTIFACTS / "test_probabilities.csv")
test10 = pd.read_csv(config.ARTIFACTS / "test_probabilities_10seed.csv")

# Generate predictions at key thresholds
thresholds = {
    "0.00583": 0.00583,
    "0.0053": 0.0053,
    "0.00530_optimal": 0.00530,  # 10-seed best for early detection
}

print("Generating test predictions at key thresholds...")
print("=" * 60)

for name, threshold in thresholds.items():
    # 5-seed predictions
    pred5 = (test5["proba"] >= threshold).astype(int)
    defects5 = (pred5 == 1).sum()

    # 10-seed predictions
    pred10 = (test10["proba_10seed"] >= threshold).astype(int)
    defects10 = (pred10 == 1).sum()

    print(f"\nThreshold: {threshold:.5f}")
    print(f"  5-seed:  {defects5} defects predicted")
    print(f"  10-seed: {defects10} defects predicted")

    # Save 5-seed submission
    sub5 = pd.DataFrame({
        config.ID_COLUMN: test5[config.ID_COLUMN],
        "Y": pred5
    })
    path5 = config.ARTIFACTS / f"submission_5seed_{name}.csv"
    sub5.to_csv(path5, index=False)
    print(f"  Saved 5-seed → {path5.name}")

    # Save 10-seed submission
    sub10 = pd.DataFrame({
        config.ID_COLUMN: test10[config.ID_COLUMN],
        "Y": pred10
    })
    path10 = config.ARTIFACTS / f"submission_10seed_{name}.csv"
    sub10.to_csv(path10, index=False)
    print(f"  Saved 10-seed → {path10.name}")

print("\n✓ All predictions generated")
