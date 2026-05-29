"""
Build a submission from pre-computed test probabilities — instant, no retrain.

Usage:
    python make_submission.py <variant> <threshold>
    python make_submission.py xgb_seedavg 0.0068
    python make_submission.py ens_weighted 0.0068

Variants come from test_probabilities.csv produced by alpha_ensemble.py:
    xgb | xgb_seedavg | ens_equal | ens_weighted

Output: test_predictions.csv  (CoilID, Y)
"""
import pandas as pd, sys

variant = sys.argv[1] if len(sys.argv) > 1 else 'xgb_seedavg'
thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0068

probs = pd.read_csv('test_probabilities.csv')
if variant not in probs.columns:
    print(f"Unknown variant '{variant}'. Choose from: {list(probs.columns[1:])}")
    sys.exit(1)

p = probs[variant].values
y = (p >= thresh).astype(int)
out = pd.DataFrame({'CoilID': probs['CoilID'].values, 'Y': y})
out.to_csv('test_predictions.csv', index=False)

n = len(y); d = int(y.sum())
print(f"variant={variant}  threshold={thresh}")
print(f"  defects={d} ({100*d/n:.1f}%)  clean={n-d} ({100*(n-d)/n:.1f}%)")
print(f"✓ wrote test_predictions.csv ({n} rows)")
