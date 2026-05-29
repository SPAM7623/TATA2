"""
CONSOLIDATED MODEL RANKING
==========================
Ranks every candidate model/ensemble/calibration on identical leakage-free
5-fold OOF predictions, across all evaluation metrics. Produces a single
ranked leaderboard so the best config is unambiguous.
"""
import pandas as pd, numpy as np, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef, brier_score_loss, log_loss, recall_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')

SEED = 42; N = 5; SEEDS = [42, 123, 999, 2025, 7777]
train = pd.read_csv('train.csv')
feat = [c for c in train.columns if c.startswith('X')]
X = train[feat]; y = train['Y'].astype(int).values
spw = (y == 0).sum() / max((y == 1).sum(), 1)

def mk_xgb(sd): return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
    min_child_weight=3, scale_pos_weight=spw, random_state=sd,
    eval_metric='logloss', verbosity=0, n_jobs=-1)
def mk_lgbm(sd): return LGBMClassifier(n_estimators=300, max_depth=3, num_leaves=8,
    learning_rate=0.03, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5,
    reg_lambda=2.0, min_child_samples=3, scale_pos_weight=spw,
    random_state=sd, n_jobs=-1, verbose=-1)
def mk_cat(sd): return CatBoostClassifier(iterations=300, depth=3, learning_rate=0.03,
    l2_leaf_reg=2.0, subsample=0.7, scale_pos_weight=spw, random_seed=sd,
    verbose=0, allow_writing_files=False)

def oof(make, seeds):
    skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=SEED)
    p = np.zeros(len(y))
    for tr, vl in skf.split(X, y):
        imp = SimpleImputer(strategy='median')
        Xtr = imp.fit_transform(X.iloc[tr]); Xvl = imp.transform(X.iloc[vl])
        acc = np.zeros(len(vl))
        for sd in seeds:
            m = make(sd); m.fit(Xtr, y[tr]); acc += m.predict_proba(Xvl)[:, 1]
        p[vl] = acc / len(seeds)
    return p

print("Training all candidates (leakage-free OOF)...")
P = {}
P['XGB (single)']        = oof(mk_xgb, [SEED])
P['XGB (5-seed avg)']    = oof(mk_xgb, SEEDS)
P['LightGBM']            = oof(mk_lgbm, [SEED])
P['CatBoost']            = oof(mk_cat, [SEED])
P['Ensemble (equal)']    = (P['XGB (single)'] + P['LightGBM'] + P['CatBoost']) / 3
P['Ensemble (weighted)'] = 0.4*P['XGB (single)'] + 0.3*P['LightGBM'] + 0.3*P['CatBoost']

# Calibrated variants of single XGB (nested calibrator fit on inner OOF)
def inner_oof(Xp, yp):
    skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=SEED+1)
    p = np.zeros(len(yp))
    for tr, vl in skf.split(Xp, yp):
        imp = SimpleImputer(strategy='median')
        Xtr = imp.fit_transform(Xp.iloc[tr]); Xvl = imp.transform(Xp.iloc[vl])
        m = mk_xgb(SEED); m.fit(Xtr, yp[tr]); p[vl] = m.predict_proba(Xvl)[:, 1]
    return p
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=SEED)
iso_oof = np.zeros(len(y)); platt_oof = np.zeros(len(y))
for tr, vl in skf.split(X, y):
    raw = P['XGB (single)'][vl]
    ip = inner_oof(X.iloc[tr], y[tr])
    ir = IsotonicRegression(out_of_bounds='clip'); ir.fit(ip, y[tr])
    lr = LogisticRegression(); lr.fit(ip.reshape(-1, 1), y[tr])
    iso_oof[vl] = ir.predict(raw)
    platt_oof[vl] = lr.predict_proba(raw.reshape(-1, 1))[:, 1]
P['XGB + Isotonic'] = iso_oof
P['XGB + Platt']    = platt_oof

# ---------- metrics
def evaluate(p):
    auc = roc_auc_score(y, p); pr = average_precision_score(y, p)
    brier = brier_score_loss(y, p); ll = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
    ts = np.linspace(0.001, 0.95, 300)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in ts]
    bi = int(np.argmax(f1s)); bf1 = f1s[bi]; bt = ts[bi]
    mcc = matthews_corrcoef(y, (p >= bt).astype(int))
    rec = recall_score(y, (p >= bt).astype(int))
    return dict(AUC=auc, PR_AUC=pr, BestF1=bf1, MCC=mcc, Recall=rec,
                Brier=brier, LogLoss=ll)

rows = {name: evaluate(p) for name, p in P.items()}
df = pd.DataFrame(rows).T

# ---------- composite rank (higher better for first 5, lower for Brier/LogLoss)
asc = {'AUC': False, 'PR_AUC': False, 'BestF1': False, 'MCC': False,
       'Recall': False, 'Brier': True, 'LogLoss': True}
ranks = pd.DataFrame({m: df[m].rank(ascending=asc[m]) for m in df.columns})
df['MeanRank'] = ranks.mean(axis=1)
df = df.sort_values('MeanRank')

pd.set_option('display.width', 140, 'display.float_format', lambda v: f'{v:.4f}')
print("\n" + "="*92)
print("MODEL & ENSEMBLE RANKING  (5-fold OOF, lower MeanRank = better overall)")
print("="*92)
print(df.to_string())

print("\n" + "="*92)
print("BEST BY EACH METRIC")
print("="*92)
for m in ['AUC', 'PR_AUC', 'BestF1', 'MCC', 'Recall', 'Brier', 'LogLoss']:
    best = df[m].idxmin() if asc[m] else df[m].idxmax()
    print(f"  {m:8s}: {best:22s} = {df.loc[best, m]:.4f}")

df.to_csv('model_ranking.csv')
print("\n✓ Saved model_ranking.csv")
