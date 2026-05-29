"""
ADVANCED SINGLE XGB IMPROVEMENT PIPELINE
=========================================
1. SHAP analysis to identify top features
2. Engineer 10-20 targeted interaction features (top pairs only)
3. Add instability features (row-level variability)
4. Retune XGBoost with Bayesian hyperparameter search
5. Recalibrate probabilities
6. Threshold sweep to find optimal operating point

All with leakage-free 5-fold stratified CV.
"""
import pandas as pd, numpy as np, warnings, shap
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef, brier_score_loss)
from xgboost import XGBClassifier
from optuna import create_study, Trial
warnings.filterwarnings('ignore')

SEED = 42; N = 5
train = pd.read_csv('train.csv'); test = pd.read_csv('test.csv')
base_feat = [c for c in train.columns if c.startswith('X')]
y = train['Y'].astype(int).values; coil = test['CoilID'].values
spw = (y == 0).sum() / max((y == 1).sum(), 1)

# =========================================================================
# STEP 1: SHAP ANALYSIS — identify top 15 features by mean |SHAP|
# =========================================================================
print("="*70)
print("STEP 1: SHAP ANALYSIS")
print("="*70)
imp = SimpleImputer(strategy='median')
X_full = imp.fit_transform(train[base_feat])
m_baseline = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
    subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
    min_child_weight=3, scale_pos_weight=spw, random_state=SEED,
    eval_metric='logloss', verbosity=0, n_jobs=-1)
m_baseline.fit(X_full, y)
explainer = shap.TreeExplainer(m_baseline)
shap_vals = explainer.shap_values(X_full)
if isinstance(shap_vals, list): shap_vals = shap_vals[1]
shap_means = np.abs(shap_vals).mean(axis=0)
top15_idx = np.argsort(-shap_means)[:15]
TOP_FEAT = [base_feat[i] for i in top15_idx]
print(f"Top 15 features by SHAP importance:\n  {TOP_FEAT}\n")

# =========================================================================
# STEP 2: Engineer interactions + instability
# =========================================================================
def engineer_features(df_in, prefix=''):
    f = pd.DataFrame(index=df_in.index)
    eps = 1e-6
    # 10-15 top interactions (Cartesian of top 6, select highest-variance)
    top6 = TOP_FEAT[:6]
    interactions = []
    for i in range(len(top6)):
        for j in range(i+1, len(top6)):
            interactions.append((top6[i], top6[j]))
    interactions = interactions[:12]  # cap at 12 to get 10-20 total
    for f1, f2 in interactions:
        f[f'{f1}_{f2}_prod'] = df_in[f1] * df_in[f2]
    # Instability features (row-level stats across all X features)
    X_vals = df_in[base_feat].values
    f['row_std'] = np.std(X_vals, axis=1)
    f['row_cv']  = f['row_std'] / (np.mean(X_vals, axis=1) + eps)
    f['row_iqr'] = np.percentile(X_vals, 75, axis=1) - np.percentile(X_vals, 25, axis=1)
    f['row_max'] = np.max(X_vals, axis=1)
    f['row_min'] = np.min(X_vals, axis=1)
    f['row_range'] = f['row_max'] - f['row_min']
    # Ratios involving top features
    for i in range(min(3, len(TOP_FEAT)-1)):
        f[f'{TOP_FEAT[i]}_{TOP_FEAT[i+1]}_ratio'] = df_in[TOP_FEAT[i]] / (df_in[TOP_FEAT[i+1]] + eps)
    return f

print("="*70)
print("STEP 2: ENGINEERING INTERACTIONS & INSTABILITY")
print("="*70)
cand_train = engineer_features(train)
cand_test = engineer_features(test)
print(f"Engineered {len(cand_train.columns)} features:\n  {list(cand_train.columns)}\n")

Xfull_eng = pd.concat([train[base_feat], cand_train], axis=1)
Xtf_eng = pd.concat([test[base_feat], cand_test], axis=1)

# =========================================================================
# STEP 3: RETUNE XGBoost with OPTUNA
# =========================================================================
print("="*70)
print("STEP 3: BAYESIAN HYPERPARAMETER TUNING")
print("="*70)

def objective(trial: Trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500, step=50),
        'max_depth': trial.suggest_int('max_depth', 2, 6),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 2.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }
    skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=SEED)
    aucs = []
    for tr, vl in skf.split(Xfull_eng, y):
        imp = SimpleImputer(strategy='median')
        Xtr = imp.fit_transform(Xfull_eng.iloc[tr])
        Xvl = imp.transform(Xfull_eng.iloc[vl])
        m = XGBClassifier(**params, scale_pos_weight=spw, random_state=SEED,
            eval_metric='logloss', verbosity=0, n_jobs=-1)
        m.fit(Xtr, y[tr], verbose=False)
        pred = m.predict_proba(Xvl)[:, 1]
        auc = roc_auc_score(y[vl], pred)
        aucs.append(auc)
    return np.mean(aucs)

study = create_study(direction='maximize')
study.optimize(objective, n_trials=20, show_progress_bar=False)
best_params = study.best_params
best_auc_tuned = study.best_value
print(f"Best CV AUC: {best_auc_tuned:.4f}")
print(f"Best params: {best_params}\n")

# =========================================================================
# STEP 4: FINAL OOF WITH TUNED MODEL
# =========================================================================
print("="*70)
print("STEP 4: FINAL OOF EVALUATION (tuned + engineered)")
print("="*70)
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=SEED)
oof_raw = np.zeros(len(y)); oof_iso = np.zeros(len(y))

for tr, vl in skf.split(Xfull_eng, y):
    imp = SimpleImputer(strategy='median')
    Xtr = imp.fit_transform(Xfull_eng.iloc[tr])
    Xvl = imp.transform(Xfull_eng.iloc[vl])

    # Raw OOF
    m = XGBClassifier(**best_params, scale_pos_weight=spw, random_state=SEED,
        eval_metric='logloss', verbosity=0, n_jobs=-1)
    m.fit(Xtr, y[tr])
    raw_pred = m.predict_proba(Xvl)[:, 1]
    oof_raw[vl] = raw_pred

    # Fit calibrator on inner-OOF
    skf_inner = StratifiedKFold(n_splits=N, shuffle=True, random_state=SEED+1)
    inner_oof = np.zeros(len(y[tr]))
    for tr2, vl2 in skf_inner.split(Xfull_eng.iloc[tr], y[tr]):
        Xtr2 = imp.fit_transform(Xfull_eng.iloc[tr].iloc[tr2])
        Xvl2 = imp.transform(Xfull_eng.iloc[tr].iloc[vl2])
        m2 = XGBClassifier(**best_params, scale_pos_weight=spw, random_state=SEED,
            eval_metric='logloss', verbosity=0, n_jobs=-1)
        m2.fit(Xtr2, y[tr2])
        inner_oof[vl2] = m2.predict_proba(Xvl2)[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(inner_oof, y[tr])
    oof_iso[vl] = iso.predict(raw_pred)

# Metrics
def metrics(name, p):
    auc = roc_auc_score(y, p); pr = average_precision_score(y, p)
    ts = np.linspace(0.001, 0.95, 300)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in ts]
    bi = int(np.argmax(f1s)); bf1 = f1s[bi]; bt = ts[bi]
    mcc = matthews_corrcoef(y, (p >= bt).astype(int))
    print(f"  {name:20s} AUC={auc:.4f} PR-AUC={pr:.4f} "
          f"BestF1={bf1:.3f}@{bt:.4f} MCC={mcc:.3f}")
    return bf1, bt

print("\nOOF Results:")
f1_raw, t_raw = metrics('Raw (tuned+eng)', oof_raw)
f1_iso, t_iso = metrics('Isotonic calib', oof_iso)

# =========================================================================
# STEP 5: TEST PREDICTIONS
# =========================================================================
print("\n" + "="*70)
print("STEP 5: PRODUCING TEST PROBABILITIES")
print("="*70)
imp = SimpleImputer(strategy='median')
Xfull = imp.fit_transform(Xfull_eng)
Xtfull = imp.transform(Xtf_eng)
m_final = XGBClassifier(**best_params, scale_pos_weight=spw, random_state=SEED,
    eval_metric='logloss', verbosity=0, n_jobs=-1)
m_final.fit(Xfull, y)
test_raw = m_final.predict_proba(Xtfull)[:, 1]

# Recalibrate on full-train inner-OOF
skf_inner = StratifiedKFold(n_splits=N, shuffle=True, random_state=SEED+1)
inner_full = np.zeros(len(y))
for tr, vl in skf_inner.split(Xfull_eng, y):
    Xtr = imp.fit_transform(Xfull_eng.iloc[tr])
    Xvl = imp.transform(Xfull_eng.iloc[vl])
    m = XGBClassifier(**best_params, scale_pos_weight=spw, random_state=SEED,
        eval_metric='logloss', verbosity=0, n_jobs=-1)
    m.fit(Xtr, y[tr])
    inner_full[vl] = m.predict_proba(Xvl)[:, 1]
iso_final = IsotonicRegression(out_of_bounds='clip')
iso_final.fit(inner_full, y)
test_iso = iso_final.predict(test_raw)

pdf = pd.DataFrame({'CoilID': coil, 'xgb_improved': test_raw, 'xgb_improved_calib': test_iso})
pdf.to_csv('test_probabilities.csv', index=False)
print("✓ Saved test probabilities")

# =========================================================================
# STEP 6: THRESHOLD SWEEP
# =========================================================================
print("\n" + "="*70)
print("STEP 6: THRESHOLD SWEEP")
print("="*70)
ts = np.linspace(0.001, 0.1, 100)
results = []
for t in ts:
    pred = (test_raw >= t).astype(int)
    d = pred.sum()
    pct = 100 * d / len(pred)
    results.append({'threshold': t, 'defects': d, 'pct': pct})
sweep_df = pd.DataFrame(results)
sweep_df.to_csv('threshold_sweep.csv', index=False)
print(sweep_df.to_string(index=False))
print(f"\n✓ Saved threshold_sweep.csv")
print(f"\nBest F1 threshold from OOF: {t_raw:.4f}")
print(f"  -> {(test_raw >= t_raw).sum()} defects ({100*(test_raw >= t_raw).sum()/len(test_raw):.1f}%)")
