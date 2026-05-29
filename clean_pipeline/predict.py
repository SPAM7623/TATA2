"""Turn the saved test probabilities into a submission file.

`train.py` already produced the averaged test probabilities, so scoring at a
new threshold is instant - no retraining. Pass `--threshold` to pick your
operating point; the default comes from `config`.
"""

import argparse

import pandas as pd

from src import config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the submission file.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.DEFAULT_THRESHOLD,
        help=f"probability cut-off (default {config.DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args()

    if not config.TEST_PROBA_CSV.exists():
        raise SystemExit("Run train.py first - test probabilities are missing.")

    proba = pd.read_csv(config.TEST_PROBA_CSV)
    labels = (proba["proba"] >= args.threshold).astype(int)

    submission = pd.DataFrame(
        {config.ID_COLUMN: proba[config.ID_COLUMN], config.TARGET_COLUMN: labels}
    )
    submission.to_csv(config.SUBMISSION_CSV, index=False)

    n_defects = int(labels.sum())
    total = len(labels)
    print(f"Threshold {args.threshold}")
    print(f"  defects flagged: {n_defects} ({100 * n_defects / total:.1f}%)")
    print(f"  clean          : {total - n_defects} ({100 * (total - n_defects) / total:.1f}%)")
    print(f"  written to     : {config.SUBMISSION_CSV}")


if __name__ == "__main__":
    main()
