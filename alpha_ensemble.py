"""
ALPHA DEFECT PROJECT — Robustness & Ensemble Pipeline
=====================================================
Implements roadmap Phases 2 (multi-seed averaging) and 3 (XGB+LGBM+CatBoost
ensemble), with honest leakage-free 5-fold OOF evaluation.

For every variant it:
  1. Computes out-of-fold (OOF) probabilities on TRAIN  -> honest metrics
  2. Refits on FULL train and predicts TEST probabilities
  3. Saves all TEST probabilities to test_probabilities.csv

You then pick a variant + threshold and build a submission instantly with
    python make_submission.py <variant> <threshold>
without retraining anything.

Variants produced:
    xgb            single regularized XGBoost (current best, seed 42)
    xgb_seedavg    XGBoost averaged over 5 seeds   (Phase 2)
    ens_equal      (XGB + LGBM + CAT) / 3          (Phase 3A)
    ens_weighted   0.4*XGB + 0.3*LGBM + 0.3*CAT    (Phase 3B)
"""
import pandas as pd, numpy as np, warnings, json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, matthews_corrcoef)
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5
SEEDS = [42, 123, 999, 2025, 7777]      # Phase 2 seeds
WEIGHTS = {'xgb': 0.4, 'lgbm': 0.3, 'cat': 0.3}  # Phase 3B weights

# ---------------------------------------------------------------- load
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
feat = [c for c in train.columns if c.startswith('X')]
X = train[feat]; y = train['Y'].astype(int).values
Xt = test[feat]; coil = test['CoilID'].values
spw = (y == 0).sum() / max((y == 1).sum(), 1)
print(f"train={X.shape}  test={Xt.shape}  pos={y.sum()}  spw={spw:.2f}")

# ---------------------------------------------------------------- models
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

def mk_cat(sd):
    return CatBoostClassifier(iterations=300, depth=3, learning_rate=0.03,
        l2_leaf_reg=2.0, subsample=0.7, scale_pos_weight=spw,
        random_seed=sd, verbose=0, allow_writing_files=False)

# ---------------------------------------------------------------- helpers
def oof_and_test(make, seeds):
    """Return (oof_probs_on_train, test_probs) averaged over given seeds."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, vl in skf.split(X, y):
        imp = SimpleImputer(strategy='median')
        Xtr = imp.fit_transform(X.iloc[tr]); Xvl = imp.transform(X.iloc[vl])
        p = np.zeros(len(vl))
        for sd in seeds:
            m = make(sd); m.fit(Xtr, y[tr]); p += m.predict_proba(Xvl)[:, 1]
        oof[vl] = p / len(seeds)
    # full-train refit -> test probs
    imp = SimpleImputer(strategy='median')
    Xf = imp.fit_transform(X); Xtf = imp.transform(Xt)
    pt = np.zeros(len(Xt))
    for sd in seeds:
        m = make(sd); m.fit(Xf, y); pt += m.predict_proba(Xtf)[:, 1]
    return oof, pt / len(seeds)

def report(name, oof):
    auc = roc_auc_score(y, oof); pr = average_precision_score(y, oof)
    ts = np.linspace(0.001, 0.95, 200)
    f1s = [f1_score(y, (oof >= t).astype(int), zero_division=0) for t in ts]
    bi = int(np.argmax(f1s)); bf1 = f1s[bi]; bt = ts[bi]
    mcc = matthews_corrcoef(y, (oof >= bt).astype(int))
    print(f"  {name:13s} AUC={auc:.4f}  PR-AUC={pr:.4f}  "
          f"bestF1={bf1:.3f}@{bt:.4f}  MCC={mcc:.3f}")
    return {'auc': auc, 'pr_auc': pr, 'best_f1': bf1, 'best_f1_thresh': bt}

# ---------------------------------------------------------------- run
print("\nBuilding base model OOF + test probabilities (this trains several models)...")
oof_xgb1, test_xgb1 = oof_and_test(mk_xgb, [SEED])
oof_xgb5, test_xgb5 = oof_and_test(mk_xgb, SEEDS)
oof_lgbm, test_lgbm = oof_and_test(mk_lgbm, [SEED])
oof_cat,  test_cat  = oof_and_test(mk_cat,  [SEED])

# Phase 3 ensembles (average of per-model probabilities)
oof_eq  = (oof_xgb1 + oof_lgbm + oof_cat) / 3
test_eq = (test_xgb1 + test_lgbm + test_cat) / 3
oof_wt  = WEIGHTS['xgb']*oof_xgb1 + WEIGHTS['lgbm']*oof_lgbm + WEIGHTS['cat']*oof_cat
test_wt = WEIGHTS['xgb']*test_xgb1 + WEIGHTS['lgbm']*test_lgbm + WEIGHTS['cat']*test_cat

print("\n" + "="*70)
print("HONEST 5-FOLD OOF METRICS (no leakage)")
print("="*70)
metrics = {}
metrics['xgb']          = report('xgb',          oof_xgb1)
report('lgbm', oof_lgbm); report('cat', oof_cat)   # components, for context
metrics['xgb_seedavg']  = report('xgb_seedavg',  oof_xgb5)
metrics['ens_equal']    = report('ens_equal',    oof_eq)
metrics['ens_weighted'] = report('ens_weighted', oof_wt)

# ---------------------------------------------------------------- save
out = pd.DataFrame({
    'CoilID': coil,
    'xgb': test_xgb1,
    'xgb_seedavg': test_xgb5,
    'ens_equal': test_eq,
    'ens_weighted': test_wt,
})
out.to_csv('test_probabilities.csv', index=False)
with open('alpha_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("\n✓ Saved test_probabilities.csv (one column per variant)")
print("✓ Saved alpha_metrics.json")
print("\nNext: python make_submission.py <variant> <threshold>")
print("      variants:", list(out.columns[1:]))
