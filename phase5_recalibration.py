"""
PHASE 5A — Probability Recalibration
====================================
Phase 4 showed the model is well-calibrated below 0.2 but breaks down above
0.5 (the [0.5-0.6) bin had only 32% positives, not ~55%). The decision
boundary is mis-scaled.

We fix this WITHOUT retraining the model by learning a monotonic mapping
from raw probability -> calibrated probability.

Leakage-free protocol:
  - OUTER 5-fold CV. In each outer fold:
      * INNER CV on the training part produces OOF probabilities
      * fit the calibrator on those inner-OOF probs (never sees outer test)
      * apply calibrator to the outer-test fold's raw probs
  - Compare raw vs calibrated on the same honest metrics from Phase 4.

Two calibrators compared: Isotonic and Platt (sigmoid).
If calibration helps, we refit on full train and recalibrate TEST probs.
"""
import pandas as pd, numpy as np, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef, brier_score_loss, log_loss)
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

SEED = 42; N_FOLDS = 5
train = pd.read_csv('train.csv'); test = pd.read_csv('test.csv')
feat = [c for c in train.columns if c.startswith('X')]
X = train[feat]; y = train['Y'].astype(int).values
Xt = test[feat]; coil = test['CoilID'].values
spw = (y == 0).sum() / max((y == 1).sum(), 1)

def mk():
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
        min_child_weight=3, scale_pos_weight=spw, random_state=SEED,
        eval_metric='logloss', verbosity=0, n_jobs=-1)

def inner_oof(Xpart, ypart):
    """OOF probabilities on a training partition (for fitting calibrator)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED+1)
    p = np.zeros(len(ypart))
    for tr, vl in skf.split(Xpart, ypart):
        imp = SimpleImputer(strategy='median')
        Xtr = imp.fit_transform(Xpart.iloc[tr]); Xvl = imp.transform(Xpart.iloc[vl])
        m = mk(); m.fit(Xtr, ypart[tr]); p[vl] = m.predict_proba(Xvl)[:, 1]
    return p

def fit_iso(p, t):
    ir = IsotonicRegression(out_of_bounds='clip'); ir.fit(p, t); return ir
def fit_platt(p, t):
    lr = LogisticRegression(); lr.fit(p.reshape(-1, 1), t); return lr
def apply_platt(lr, p):
    return lr.predict_proba(p.reshape(-1, 1))[:, 1]

# ============ OUTER CV: honest raw vs calibrated comparison
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_raw = np.zeros(len(y)); oof_iso = np.zeros(len(y)); oof_platt = np.zeros(len(y))

print("Running nested CV (this trains many models)...")
for fold, (tr, vl) in enumerate(skf.split(X, y)):
    Xtr_part, ytr_part = X.iloc[tr], y[tr]

    # raw probs for outer-test fold (model trained on full outer-train)
    imp = SimpleImputer(strategy='median')
    Xtr_imp = imp.fit_transform(Xtr_part); Xvl_imp = imp.transform(X.iloc[vl])
    m = mk(); m.fit(Xtr_imp, ytr_part)
    raw_vl = m.predict_proba(Xvl_imp)[:, 1]
    oof_raw[vl] = raw_vl

    # calibrators fit on inner-OOF of the outer-train part
    inner_p = inner_oof(Xtr_part, ytr_part)
    iso = fit_iso(inner_p, ytr_part)
    platt = fit_platt(inner_p, ytr_part)
    oof_iso[vl] = iso.predict(raw_vl)
    oof_platt[vl] = apply_platt(platt, raw_vl)

def metrics(name, p):
    auc = roc_auc_score(y, p); pr = average_precision_score(y, p)
    brier = brier_score_loss(y, p); ll = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
    ts = np.linspace(0.001, 0.95, 300)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in ts]
    bi = int(np.argmax(f1s)); bf1 = f1s[bi]; bt = ts[bi]
    mcc = matthews_corrcoef(y, (p >= bt).astype(int))
    print(f"  {name:10s} AUC={auc:.4f} PR={pr:.4f} Brier={brier:.4f} "
          f"LogLoss={ll:.4f} bestF1={bf1:.3f}@{bt:.4f} MCC={mcc:.3f}")
    return bf1

print("\n" + "="*78)
print("RAW vs CALIBRATED — honest nested-CV metrics")
print("="*78)
metrics('raw', oof_raw)
metrics('isotonic', oof_iso)
metrics('platt', oof_platt)

# ============ Calibration table (raw vs isotonic) to show the fix
print("\n" + "="*78)
print("CALIBRATION CHECK (bin -> actual positive rate)")
print("="*78)
bins = np.linspace(0, 1, 11)
print(f"  {'bin':12s} {'n':>4s} {'raw_posrate':>12s} {'iso_posrate':>12s}")
for i in range(len(bins)-1):
    mr = (oof_raw >= bins[i]) & (oof_raw < bins[i+1])
    mi = (oof_iso >= bins[i]) & (oof_iso < bins[i+1])
    rr = y[mr].mean() if mr.sum() else float('nan')
    ri = y[mi].mean() if mi.sum() else float('nan')
    print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}) {mr.sum():>4d} {rr:>12.3f} {ri:>12.3f}")

# ============ Decide & produce TEST probabilities
# Refit model on full train, fit calibrator on full-train inner-OOF, recalibrate test.
print("\n" + "="*78)
print("Producing calibrated TEST probabilities")
print("="*78)
imp = SimpleImputer(strategy='median')
Xf = imp.fit_transform(X); Xtf = imp.transform(Xt)
m = mk(); m.fit(Xf, y)
test_raw = m.predict_proba(Xtf)[:, 1]

inner_p_full = inner_oof(X, y)
iso_full = fit_iso(inner_p_full, y)
platt_full = fit_platt(inner_p_full, y)
test_iso = iso_full.predict(test_raw)
test_platt = apply_platt(platt_full, test_raw)

prob_df = pd.read_csv('test_probabilities.csv') if __import__('os').path.exists('test_probabilities.csv') else pd.DataFrame({'CoilID': coil})
prob_df['xgb_raw'] = test_raw
prob_df['xgb_isotonic'] = test_iso
prob_df['xgb_platt'] = test_platt
prob_df.to_csv('test_probabilities.csv', index=False)
print("✓ Added xgb_raw, xgb_isotonic, xgb_platt columns to test_probabilities.csv")
print("\nUse: python make_submission.py xgb_isotonic <threshold>")
