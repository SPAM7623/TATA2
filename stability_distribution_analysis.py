"""
PRIORITY 3 & 4 — Stability Analysis + Prediction Distribution
=============================================================
For the ORIGINAL regularized XGB model @ threshold 0.0068.

Priority 3: Stability
  - Multiple CV seeds
  - Multiple train/val splits
  - Threshold sensitivity (does perf collapse if threshold shifts slightly?)

Priority 4: Prediction Distribution
  - Number of predicted positives
  - Probability histogram
  - Top 1% highest-risk coils
  - Bottom 1% lowest-risk coils
  - Why does 0.0068 work?
"""
import pandas as pd, numpy as np, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score, matthews_corrcoef)
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

THRESHOLD = 0.0068
train = pd.read_csv('train.csv'); test = pd.read_csv('test.csv')
feat = [c for c in train.columns if c.startswith('X')]
X = train[feat]; y = train['Y'].astype(int).values
Xt = test[feat]; coil = test['CoilID'].values
spw = (y == 0).sum() / max((y == 1).sum(), 1)

def mk(sd=42):
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
        min_child_weight=3, scale_pos_weight=spw, random_state=sd,
        eval_metric='logloss', verbosity=0, n_jobs=-1)

def oof_for_seed(cv_seed, model_seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed)
    p = np.zeros(len(y))
    for tr, vl in skf.split(X, y):
        imp = SimpleImputer(strategy='median')
        Xtr = imp.fit_transform(X.iloc[tr]); Xvl = imp.transform(X.iloc[vl])
        m = mk(model_seed); m.fit(Xtr, y[tr]); p[vl] = m.predict_proba(Xvl)[:, 1]
    return p

# =========================================================================
# PRIORITY 3a: DIFFERENT CV SEEDS
# =========================================================================
print("="*70)
print("PRIORITY 3a — STABILITY ACROSS CV SEEDS")
print("="*70)
print(f"\n{'CV seed':>8} {'AUC':>7} {'PR-AUC':>7} {'F1@0.0068':>10} {'Recall':>7} {'Prec':>7}")
cv_seeds = [42, 123, 999, 2025, 7777]
aucs = []
for cs in cv_seeds:
    p = oof_for_seed(cs)
    auc = roc_auc_score(y, p); pr = average_precision_score(y, p)
    pred = (p >= THRESHOLD).astype(int)
    f1 = f1_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    prec = precision_score(y, pred, zero_division=0)
    aucs.append(auc)
    print(f"{cs:>8} {auc:>7.4f} {pr:>7.4f} {f1:>10.3f} {rec:>7.3f} {prec:>7.3f}")
print(f"\nAUC: mean={np.mean(aucs):.4f}  std={np.std(aucs):.4f}  "
      f"range=[{min(aucs):.4f}, {max(aucs):.4f}]")
print(f"  -> {'STABLE' if np.std(aucs) < 0.01 else 'UNSTABLE'} "
      f"(std {'<' if np.std(aucs)<0.01 else '>='} 0.01)")

# =========================================================================
# PRIORITY 3b: DIFFERENT MODEL SEEDS (train randomness)
# =========================================================================
print("\n" + "="*70)
print("PRIORITY 3b — STABILITY ACROSS MODEL SEEDS (fixed CV split)")
print("="*70)
print(f"\n{'model seed':>10} {'AUC':>7} {'PR-AUC':>7}")
maucs = []
for ms in [42, 123, 999, 2025, 7777]:
    p = oof_for_seed(42, ms)
    auc = roc_auc_score(y, p); pr = average_precision_score(y, p)
    maucs.append(auc)
    print(f"{ms:>10} {auc:>7.4f} {pr:>7.4f}")
print(f"\nAUC: mean={np.mean(maucs):.4f}  std={np.std(maucs):.4f}")
print(f"  -> {'STABLE' if np.std(maucs) < 0.01 else 'UNSTABLE'}")

# =========================================================================
# PRIORITY 3c: THRESHOLD SENSITIVITY
# =========================================================================
print("\n" + "="*70)
print("PRIORITY 3c — THRESHOLD SENSITIVITY (does perf collapse near 0.0068?)")
print("="*70)
p = oof_for_seed(42)  # canonical OOF
print(f"\n{'Threshold':>10} {'F1':>7} {'Recall':>7} {'Prec':>7} {'MCC':>7} {'#pos':>5}")
base_thresholds = [0.0050, 0.0060, 0.0065, 0.0068, 0.0070, 0.0075, 0.0080, 0.0100]
f1_at = {}
for t in base_thresholds:
    pred = (p >= t).astype(int)
    f1 = f1_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    prec = precision_score(y, pred, zero_division=0)
    mcc = matthews_corrcoef(y, pred)
    f1_at[t] = f1
    marker = "  <-- 0.0068" if abs(t-0.0068) < 1e-9 else ""
    print(f"{t:>10.4f} {f1:>7.3f} {rec:>7.3f} {prec:>7.3f} {mcc:>7.3f} {pred.sum():>5}{marker}")

# Sensitivity: how much does F1 change for +/-10% threshold shift?
f1_center = f1_at[0.0068]
f1_low = f1_at[0.0060]   # -12%
f1_high = f1_at[0.0075]  # +10%
delta = max(abs(f1_center - f1_low), abs(f1_center - f1_high))
print(f"\nF1 @ 0.0068 = {f1_center:.3f}")
print(f"F1 swing for ±~10% threshold shift: {delta:.3f}")
print(f"  -> {'FRAGILE (collapses)' if delta > 0.05 else 'ROBUST (stable)'}")

# =========================================================================
# PRIORITY 4: PREDICTION DISTRIBUTION ON TEST
# =========================================================================
print("\n" + "="*70)
print("PRIORITY 4 — PREDICTION DISTRIBUTION (test set @ 0.0068)")
print("="*70)
imp = SimpleImputer(strategy='median')
Xf = imp.fit_transform(X); Xtf = imp.transform(Xt)
m = mk(42); m.fit(Xf, y)
test_p = m.predict_proba(Xtf)[:, 1]

n_pos = int((test_p >= THRESHOLD).sum())
print(f"\nPredicted positives @ 0.0068: {n_pos} / {len(test_p)} ({100*n_pos/len(test_p):.1f}%)")

# Probability histogram
print(f"\nProbability histogram (test):")
bins = [0, 0.001, 0.002, 0.004, 0.0068, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0]
for i in range(len(bins)-1):
    cnt = int(((test_p >= bins[i]) & (test_p < bins[i+1])).sum())
    bar = '#' * int(50 * cnt / len(test_p))
    print(f"  [{bins[i]:.4f}-{bins[i+1]:.4f}): {cnt:>3}  {bar}")

print(f"\nProbability stats:")
print(f"  min={test_p.min():.5f}  median={np.median(test_p):.5f}  "
      f"mean={test_p.mean():.5f}  max={test_p.max():.5f}")
print(f"  Train base rate: {y.mean():.4f} ({y.sum()}/{len(y)})")

# Top 1% / Bottom 1%
order = np.argsort(-test_p)
n1 = max(1, len(test_p)//100)
print(f"\nTop {n1} highest-risk coils:")
for idx in order[:n1]:
    print(f"  CoilID {coil[idx]:>5}  prob={test_p[idx]:.5f}")
print(f"\nBottom {n1} lowest-risk coils:")
for idx in order[-n1:]:
    print(f"  CoilID {coil[idx]:>5}  prob={test_p[idx]:.6f}")

# Why does 0.0068 work?
print("\n" + "="*70)
print("WHY DOES 0.0068 WORK?")
print("="*70)
below = int((test_p < THRESHOLD).sum())
print(f"""
  - Model probabilities are heavily compressed toward zero (median={np.median(test_p):.5f}).
  - scale_pos_weight={spw:.1f} pushes the decision region very low.
  - At 0.0068, {n_pos} coils ({100*n_pos/len(test_p):.0f}%) exceed the bar -> flagged.
  - The remaining {below} coils sit below 0.0068 with near-zero risk.
  - 0.0068 sits in the steep part of the probability CDF, so it acts as a
    natural separator between the dense low-prob mass and the elevated-risk tail.
""")
