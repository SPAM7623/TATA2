"""
Multi-Seed Averaging — Production Implementation
===============================================
Train 5 XGBoost models (seeds 42, 123, 999, 2025, 7777) on full training data.
Average test probabilities.
Retune threshold on OOF ensemble.
Report metrics and optimal threshold.
"""
import pandas as pd, numpy as np, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score, matthews_corrcoef)
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

SEEDS = [42, 123, 999, 2025, 7777]
train = pd.read_csv('train.csv'); test = pd.read_csv('test.csv')
feat = [c for c in train.columns if c.startswith('X')]
X = train[feat]; y = train['Y'].astype(int).values
Xt = test[feat]; coil = test['CoilID'].values
spw = (y == 0).sum() / max((y == 1).sum(), 1)

def mk(sd):
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
        min_child_weight=3, scale_pos_weight=spw, random_state=sd,
        eval_metric='logloss', verbosity=0, n_jobs=-1)

# =========================================================================
# STEP 1: OOF ensemble (for threshold tuning)
# =========================================================================
print("="*70)
print("MULTI-SEED AVERAGING — Production Training")
print("="*70)
print(f"\nTraining 5 models with seeds: {SEEDS}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_ensemble = np.zeros(len(y))
test_ensemble = np.zeros(len(Xt))

for sd in SEEDS:
    print(f"\nSeed {sd}:")

    # OOF on this seed
    oof_seed = np.zeros(len(y))
    for tr, vl in skf.split(X, y):
        imp = SimpleImputer(strategy='median')
        Xtr = imp.fit_transform(X.iloc[tr])
        Xvl = imp.transform(X.iloc[vl])
        m = mk(sd)
        m.fit(Xtr, y[tr])
        oof_seed[vl] = m.predict_proba(Xvl)[:, 1]
    oof_ensemble += oof_seed

    # Test on this seed (trained on full data)
    imp = SimpleImputer(strategy='median')
    Xf = imp.fit_transform(X)
    Xtf = imp.transform(Xt)
    m = mk(sd)
    m.fit(Xf, y)
    test_seed = m.predict_proba(Xtf)[:, 1]
    test_ensemble += test_seed
    print(f"  ✓ OOF + test predictions added")

# Average
oof_ensemble /= len(SEEDS)
test_ensemble /= len(SEEDS)
print(f"\n✓ Averaged OOF and test predictions across {len(SEEDS)} seeds")

# =========================================================================
# STEP 2: Evaluate ensemble
# =========================================================================
print("\n" + "="*70)
print("ENSEMBLE EVALUATION (OOF)")
print("="*70)

auc = roc_auc_score(y, oof_ensemble)
pr = average_precision_score(y, oof_ensemble)
print(f"\n  AUC = {auc:.4f}")
print(f"  PR-AUC = {pr:.4f}")

# =========================================================================
# STEP 3: Retune threshold
# =========================================================================
print("\n" + "="*70)
print("THRESHOLD TUNING (OOF ensemble)")
print("="*70)

ts = np.linspace(0.001, 0.95, 300)
f1s = [f1_score(y, (oof_ensemble >= t).astype(int), zero_division=0) for t in ts]
bi = int(np.argmax(f1s))
opt_t = ts[bi]
opt_f1 = f1s[bi]

pred_opt = (oof_ensemble >= opt_t).astype(int)
rec = recall_score(y, pred_opt)
prec = precision_score(y, pred_opt, zero_division=0)
mcc = matthews_corrcoef(y, pred_opt)

print(f"\nOptimal F1 threshold: {opt_t:.4f}")
print(f"  Best F1 = {opt_f1:.3f}")
print(f"  Recall = {rec:.4f}")
print(f"  Precision = {prec:.4f}")
print(f"  MCC = {mcc:.4f}")
print(f"  Predicted positives = {pred_opt.sum()} / {len(y)}")

# Also show performance at original 0.0068 for comparison
print(f"\nFor reference, @ 0.0068:")
pred_0068 = (oof_ensemble >= 0.0068).astype(int)
f1_0068 = f1_score(y, pred_0068, zero_division=0)
rec_0068 = recall_score(y, pred_0068)
prec_0068 = precision_score(y, pred_0068, zero_division=0)
mcc_0068 = matthews_corrcoef(y, pred_0068)
print(f"  F1 = {f1_0068:.3f}  Recall = {rec_0068:.4f}  Prec = {prec_0068:.4f}  MCC = {mcc_0068:.4f}")
print(f"  Predicted positives = {pred_0068.sum()} / {len(y)}")

# =========================================================================
# STEP 4: Generate test predictions at optimal threshold
# =========================================================================
print("\n" + "="*70)
print("TEST PREDICTIONS")
print("="*70)

n_pos = int((test_ensemble >= opt_t).sum())
print(f"\nAt optimal threshold {opt_t:.4f}:")
print(f"  Predicted defects: {n_pos} / {len(Xt)} ({100*n_pos/len(Xt):.1f}%)")

# Also show at 0.0068 for comparison
n_pos_0068 = int((test_ensemble >= 0.0068).sum())
print(f"\nAt 0.0068 (original best threshold):")
print(f"  Predicted defects: {n_pos_0068} / {len(Xt)} ({100*n_pos_0068/len(Xt):.1f}%)")

# Save both
df_opt = pd.DataFrame({'CoilID': coil, 'Y': (test_ensemble >= opt_t).astype(int)})
df_opt.to_csv('test_predictions_multiseed_optimal.csv', index=False)
df_0068 = pd.DataFrame({'CoilID': coil, 'Y': (test_ensemble >= 0.0068).astype(int)})
df_0068.to_csv('test_predictions_multiseed_0068.csv', index=False)

print(f"\n✓ Saved test_predictions_multiseed_optimal.csv (threshold {opt_t:.4f})")
print(f"✓ Saved test_predictions_multiseed_0068.csv (threshold 0.0068)")

# =========================================================================
# STEP 5: Threshold sweep
# =========================================================================
print("\n" + "="*70)
print("THRESHOLD SWEEP (test predictions)")
print("="*70)
ts_sweep = [0.001, 0.003, 0.005, 0.0068, 0.007, 0.008, 0.009, 0.010, 0.015, 0.020, opt_t]
print(f"\n{'Threshold':>10} {'Defects':>8} {'Pct':>6}")
for t in sorted(set(ts_sweep)):
    d = int((test_ensemble >= t).sum())
    pct = 100 * d / len(Xt)
    marker = f"  <-- OPTIMAL" if abs(t - opt_t) < 1e-6 else ""
    print(f"{t:>10.4f} {d:>8} {pct:>5.1f}%{marker}")
