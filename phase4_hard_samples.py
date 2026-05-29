"""
PHASE 4 — Hard Sample Analysis
==============================
Study false positives and false negatives to find missing signals.

For each OOF fold, identify:
  - False Positives: predicted 1 but actual 0 (good defects missed)
  - False Negatives: predicted 0 but actual 1 (bad defects caught)
  - Hard Negatives: pred in [0.3, 0.7] but y=0 (borderline non-defects)
  - Hard Positives: pred in [0.3, 0.7] but y=1 (borderline defects)

Then:
  1. Count & analyze
  2. SHAP contribution for each hard sample
  3. Feature value distributions (hard vs correct)
  4. Probability calibration
"""
import pandas as pd, numpy as np, warnings, shap
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

SEED = 42
N_FOLDS = 5

train = pd.read_csv('train.csv')
feat = [c for c in train.columns if c.startswith('X')]
X = train[feat]; y = train['Y'].astype(int).values; coil = train['CoilID']
spw = (y == 0).sum() / max((y == 1).sum(), 1)

def mk_xgb(sd):
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
        min_child_weight=3, scale_pos_weight=spw, random_state=sd,
        eval_metric='logloss', verbosity=0, n_jobs=-1)

# ========== OOF loop: collect hard samples + models for SHAP
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_probs = np.zeros(len(y))
hard_samples = []  # list of (idx, pred, actual, fold)
models_by_fold = []

for fold, (tr, vl) in enumerate(skf.split(X, y)):
    imp = SimpleImputer(strategy='median')
    Xtr = imp.fit_transform(X.iloc[tr])
    Xvl = imp.transform(X.iloc[vl])

    m = mk_xgb(SEED)
    m.fit(Xtr, y[tr])
    pred = m.predict_proba(Xvl)[:, 1]
    oof_probs[vl] = pred
    models_by_fold.append((m, Xtr, y[tr], Xvl, vl, imp))

    # Find hard samples
    for i_local, i_global in enumerate(vl):
        p = pred[i_local]
        a = y[i_global]

        # Hard: predicted wrong or near boundary
        is_hard = False
        label = None
        if a == 0 and p >= 0.3:
            is_hard = True
            label = 'FP' if p > 0.5 else 'hardneg'
        elif a == 1 and p < 0.7:
            is_hard = True
            label = 'FN' if p < 0.5 else 'hardpos'

        if is_hard:
            hard_samples.append({
                'idx': i_global, 'coil': coil.iloc[i_global], 'fold': fold,
                'pred': p, 'actual': a, 'label': label
            })

hard_df = pd.DataFrame(hard_samples)

print("="*70)
print("PHASE 4 — HARD SAMPLE ANALYSIS")
print("="*70)
print(f"\nDataset: {len(y)} samples, {y.sum()} positives")
from sklearn.metrics import roc_auc_score
print(f"\nOOF AUC = {roc_auc_score(y, oof_probs):.4f}")

print(f"\nHARD SAMPLES FOUND: {len(hard_df)}")
print(f"  False Positives (pred >0.5, y=0):  {len(hard_df[hard_df['label']=='FP'])}")
print(f"  False Negatives (pred <0.5, y=1):  {len(hard_df[hard_df['label']=='FN'])}")
print(f"  Hard Negatives (pred 0.3-0.5, y=0): {len(hard_df[hard_df['label']=='hardneg'])}")
print(f"  Hard Positives (pred 0.5-0.7, y=1): {len(hard_df[hard_df['label']=='hardpos'])}")

# ========== Feature analysis
print("\n" + "="*70)
print("FEATURE VALUE DISTRIBUTIONS")
print("="*70)

for label in ['FP', 'FN', 'hardneg', 'hardpos']:
    mask = hard_df['label'] == label
    if not mask.sum():
        continue
    idxs = hard_df[mask]['idx'].values

    print(f"\n{label.upper()} ({len(idxs)} samples):")
    hard_vals = X.iloc[idxs].values
    all_vals = X.values

    # top features by std dev in hard samples
    hard_std = hard_vals.std(axis=0)
    top_idx = np.argsort(-hard_std)[:5]

    for fi in top_idx:
        fname = feat[fi]
        h_mean = hard_vals[:, fi].mean()
        h_std = hard_vals[:, fi].std()
        g_mean = all_vals[:, fi].mean()
        g_std = all_vals[:, fi].std()
        print(f"  {fname:3s}:  hard={h_mean:7.2f}±{h_std:5.2f}  "
              f"global={g_mean:7.2f}±{g_std:5.2f}")

# ========== SHAP for subset of hard samples
print("\n" + "="*70)
print("SHAP ANALYSIS (first hard sample per category)")
print("="*70)

for label in ['FP', 'FN']:
    mask = hard_df['label'] == label
    if not mask.sum():
        continue

    sample = hard_df[mask].iloc[0]
    idx = int(sample['idx'])
    fold = int(sample['fold'])

    # Get the model + data for that fold
    m, Xtr, ytr, Xvl, vl, imp = models_by_fold[fold]

    # Find position of idx in vl
    pos = np.where(vl == idx)[0][0]
    x_sample = Xvl[pos:pos+1]

    # SHAP
    try:
        explainer = shap.TreeExplainer(m)
        shap_vals = explainer.shap_values(x_sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # binary: take class 1 shap
        top_idx = np.argsort(-np.abs(shap_vals[0]))[:5]

        print(f"\n{label} — CoilID {sample['coil']:.0f}, pred={sample['pred']:.4f}, y={sample['actual']}")
        for fi in top_idx:
            fname = feat[fi]
            val = x_sample[0, fi]
            sv = shap_vals[0, fi]
            print(f"  {fname:3s} = {val:7.2f}  [SHAP impact: {sv:+.4f}]")
    except Exception as e:
        print(f"  SHAP error: {e}")

# ========== Probability calibration
print("\n" + "="*70)
print("PROBABILITY CALIBRATION")
print("="*70)

bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for i in range(len(bins) - 1):
    mask = (oof_probs >= bins[i]) & (oof_probs < bins[i+1])
    if mask.sum() == 0:
        continue
    acc = (y[mask] == (oof_probs[mask] >= 0.5).astype(int)).mean()
    actual_pos_rate = y[mask].mean()
    print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}):  "
          f"n={mask.sum():3d}  pos_rate={actual_pos_rate:.3f}  "
          f"acc={acc:.3f}")

# ========== Save report
hard_df.to_csv('hard_samples.csv', index=False)
print("\n✓ Saved hard_samples.csv")
print("\nNext: Inspect hard_samples.csv for patterns.")
print("Look for:")
print("  - Do FN/FP samples cluster at certain probability ranges?")
print("  - Which features differ most between hard and correct?")
print("  - Could missing process-state descriptors be the root cause?")
