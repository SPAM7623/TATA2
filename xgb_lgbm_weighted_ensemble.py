"""
Weighted Ensemble: 0.7*XGB + 0.3*LGBM
======================================
Leakage-free 5-fold CV evaluation with honest metrics.
If no improvement over baseline XGB, stop here.
"""
import pandas as pd, numpy as np, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef, brier_score_loss)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore')

SEED = 42; N = 5
train = pd.read_csv('train.csv'); test = pd.read_csv('test.csv')
feat = [c for c in train.columns if c.startswith('X')]
X = train[feat]; y = train['Y'].astype(int).values
Xt = test[feat]; coil = test['CoilID'].values
spw = (y == 0).sum() / max((y == 1).sum(), 1)

def mk_xgb(sd):
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
        min_child_weight=3, scale_pos_weight=spw, random_state=sd,
        eval_metric='logloss', verbosity=0, n_jobs=-1)

def mk_lgbm(sd):
    return LGBMClassifier(n_estimators=300, max_depth=3, num_leaves=8,
        learning_rate=0.03, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=2.0, min_child_samples=3,
        scale_pos_weight=spw, random_state=sd, n_jobs=-1, verbose=-1)

# =========================================================================
# OOF evaluation
# =========================================================================
print("="*70)
print("WEIGHTED ENSEMBLE: 0.7*XGB + 0.3*LGBM")
print("="*70)

skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=SEED)
oof_xgb = np.zeros(len(y))
oof_lgbm = np.zeros(len(y))
oof_ens = np.zeros(len(y))

for tr, vl in skf.split(X, y):
    imp = SimpleImputer(strategy='median')
    Xtr = imp.fit_transform(X.iloc[tr])
    Xvl = imp.transform(X.iloc[vl])

    # XGB
    m_xgb = mk_xgb(SEED)
    m_xgb.fit(Xtr, y[tr])
    oof_xgb[vl] = m_xgb.predict_proba(Xvl)[:, 1]

    # LGBM
    m_lgbm = mk_lgbm(SEED)
    m_lgbm.fit(Xtr, y[tr])
    oof_lgbm[vl] = m_lgbm.predict_proba(Xvl)[:, 1]

    # Ensemble
    oof_ens[vl] = 0.7 * oof_xgb[vl] + 0.3 * oof_lgbm[vl]

# =========================================================================
# Metrics
# =========================================================================
def evaluate(name, p):
    auc = roc_auc_score(y, p)
    pr = average_precision_score(y, p)
    brier = brier_score_loss(y, p)
    ts = np.linspace(0.001, 0.95, 300)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in ts]
    bi = int(np.argmax(f1s))
    bf1 = f1s[bi]
    bt = ts[bi]
    mcc = matthews_corrcoef(y, (p >= bt).astype(int))
    print(f"  {name:22s} AUC={auc:.4f}  PR-AUC={pr:.4f}  "
          f"BestF1={bf1:.3f}@{bt:.4f}  MCC={mcc:.3f}  Brier={brier:.4f}")
    return auc, pr, bf1, bt, mcc

print("\nOOF Results (5-fold stratified CV, leakage-free):\n")
auc_xgb, pr_xgb, f1_xgb, t_xgb, mcc_xgb = evaluate('XGB (baseline)', oof_xgb)
auc_lgbm, pr_lgbm, f1_lgbm, t_lgbm, mcc_lgbm = evaluate('LGBM', oof_lgbm)
auc_ens, pr_ens, f1_ens, t_ens, mcc_ens = evaluate('0.7*XGB + 0.3*LGBM', oof_ens)

print("\n" + "="*70)
print("COMPARISON vs BASELINE XGB")
print("="*70)
print(f"  AUC:   {auc_ens:.4f} vs {auc_xgb:.4f}  ({auc_ens - auc_xgb:+.4f})")
print(f"  PR-AUC:{pr_ens:.4f} vs {pr_xgb:.4f}  ({pr_ens - pr_xgb:+.4f})")
print(f"  F1:    {f1_ens:.3f} vs {f1_xgb:.3f}  ({f1_ens - f1_xgb:+.3f})")
print(f"  MCC:   {mcc_ens:.3f} vs {mcc_xgb:.3f}  ({mcc_ens - mcc_xgb:+.3f})")

if auc_ens <= auc_xgb:
    print("\n❌ ENSEMBLE DID NOT IMPROVE AUC. STOPPING HERE.")
    print(f"   (Ensemble {auc_ens:.4f} ≤ XGB {auc_xgb:.4f})")
    import sys; sys.exit(0)

# =========================================================================
# If we reach here, ensemble improved. Generate test predictions.
# =========================================================================
print("\n✓ Ensemble improved AUC. Generating test predictions...")

imp = SimpleImputer(strategy='median')
Xf = imp.fit_transform(X)
Xtf = imp.transform(Xt)

m_xgb = mk_xgb(SEED)
m_xgb.fit(Xf, y)
test_xgb = m_xgb.predict_proba(Xtf)[:, 1]

m_lgbm = mk_lgbm(SEED)
m_lgbm.fit(Xf, y)
test_lgbm = m_lgbm.predict_proba(Xtf)[:, 1]

test_ens = 0.7 * test_xgb + 0.3 * test_lgbm

# Save to test_probabilities.csv
pdf = pd.DataFrame({'CoilID': coil, 'ens_07xgb_03lgbm': test_ens})
pdf.to_csv('test_probabilities.csv', index=False)

# Threshold sweep
print("\n" + "="*70)
print("THRESHOLD SWEEP (test predictions)")
print("="*70)
print(f"\n  Optimal OOF threshold: {t_ens:.4f}")
print(f"  Defects @ {t_ens:.4f}: {int((test_ens >= t_ens).sum())} ({100*(test_ens >= t_ens).sum()/len(test_ens):.1f}%)\n")

ts = [0.001, 0.003, 0.005, 0.0068, 0.007, 0.008, 0.009, 0.010, 0.015, 0.020]
for t in ts:
    d = int((test_ens >= t).sum())
    pct = 100 * d / len(test_ens)
    print(f"  {t:.4f} -> {d:3d} defects ({pct:5.1f}%)")

print(f"\n✓ Saved test probabilities to test_probabilities.csv")
print(f"✓ Use: python make_submission.py ens_07xgb_03lgbm <threshold>")
