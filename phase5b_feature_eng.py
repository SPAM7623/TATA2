"""
PHASE 5B — Targeted Feature Engineering on Hard-Sample Drivers
==============================================================
Phase 4 found that false negatives & false positives are dominated by
X34, X35, X36, X37, X38 (and to a lesser extent X22, X2, X13). These
process-state variables co-vary in misclassified samples.

We engineer a SMALL set of candidate features from exactly these variables
and test each one honestly with leakage-free 5-fold OOF AUC. We ADD a
feature only if it improves OOF AUC over the current best set — greedy
forward selection. This avoids the feature-explosion that hurt earlier.

All transforms are computed per-fold-safe (they are row-wise / use only
training-fold statistics where stats are needed).
"""
import pandas as pd, numpy as np, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

SEED = 42; N_FOLDS = 5
train = pd.read_csv('train.csv'); test = pd.read_csv('test.csv')
base_feat = [c for c in train.columns if c.startswith('X')]
y = train['Y'].astype(int).values
spw = (y == 0).sum() / max((y == 1).sum(), 1)

DRIVERS = ['X34', 'X35', 'X36', 'X37', 'X38', 'X22', 'X2', 'X13']

def mk():
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
        min_child_weight=3, scale_pos_weight=spw, random_state=SEED,
        eval_metric='logloss', verbosity=0, n_jobs=-1)

def add_candidates(df):
    """Row-wise engineered features (no cross-row stats -> leakage-safe)."""
    f = pd.DataFrame(index=df.index)
    eps = 1e-6
    # Ratios between the co-varying process variables
    f['r_34_35'] = df['X34'] / (df['X35'] + eps)
    f['r_36_35'] = df['X36'] / (df['X35'] + eps)
    f['r_37_38'] = df['X37'] / (df['X38'] + eps)
    f['r_34_36'] = df['X34'] / (df['X36'] + eps)
    f['r_22_2']  = df['X22'] / (df['X2'] + eps)
    # Sums / aggregates of the throughput-like group
    f['sum_34_38'] = df[['X34', 'X35', 'X36', 'X37', 'X38']].sum(axis=1)
    f['mean_34_38'] = df[['X34', 'X35', 'X36', 'X37', 'X38']].mean(axis=1)
    f['std_34_38'] = df[['X34', 'X35', 'X36', 'X37', 'X38']].std(axis=1)
    # Log transforms (these vars span many orders of magnitude)
    f['log_x35'] = np.log1p(df['X35'].clip(lower=0))
    f['log_x34'] = np.log1p(df['X34'].clip(lower=0))
    f['log_x36'] = np.log1p(df['X36'].clip(lower=0))
    # Interactions with the boundary drivers X13, X22
    f['x13_x22'] = df['X13'] * df['X22']
    f['x2_x13']  = df['X2'] * df['X13']
    # Missingness flags for the driver group (FN/FP may hide in NaN)
    for c in DRIVERS:
        f[f'na_{c}'] = df[c].isna().astype(int)
    return f

cand = add_candidates(train)
cand_test = add_candidates(test)
CAND_COLS = list(cand.columns)

# Pre-build full augmented frames; we select columns inside the CV loop.
Xfull = pd.concat([train[base_feat], cand], axis=1)

def cv_auc(cols):
    """Leakage-free OOF AUC/PR/F1 for base_feat + given candidate cols."""
    use = base_feat + cols
    Xs = Xfull[use]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, vl in skf.split(Xs, y):
        imp = SimpleImputer(strategy='median')
        Xtr = imp.fit_transform(Xs.iloc[tr]); Xvl = imp.transform(Xs.iloc[vl])
        m = mk(); m.fit(Xtr, y[tr]); oof[vl] = m.predict_proba(Xvl)[:, 1]
    auc = roc_auc_score(y, oof); pr = average_precision_score(y, oof)
    ts = np.linspace(0.001, 0.95, 200)
    bf1 = max(f1_score(y, (oof >= t).astype(int), zero_division=0) for t in ts)
    return auc, pr, bf1

# ============ Baseline
base_auc, base_pr, base_f1 = cv_auc([])
print("="*70)
print("PHASE 5B — GREEDY FORWARD SELECTION OF ENGINEERED FEATURES")
print("="*70)
print(f"\nBaseline (49 raw feats): AUC={base_auc:.4f} PR={base_pr:.4f} F1={base_f1:.3f}")
print(f"Candidate pool: {len(CAND_COLS)} features\n")

# ============ Greedy forward selection
selected = []
best_auc = base_auc
remaining = list(CAND_COLS)
improved = True
round_n = 0
while improved and remaining:
    round_n += 1
    improved = False
    best_gain = 0; best_col = None; best_stats = None
    for c in remaining:
        auc, pr, f1 = cv_auc(selected + [c])
        gain = auc - best_auc
        if gain > best_gain:
            best_gain = gain; best_col = c; best_stats = (auc, pr, f1)
    if best_col is not None and best_gain > 0.0005:   # require real gain
        selected.append(best_col); remaining.remove(best_col)
        best_auc = best_stats[0]; improved = True
        print(f"  Round {round_n}: + {best_col:12s} -> AUC={best_stats[0]:.4f} "
              f"PR={best_stats[1]:.4f} F1={best_stats[2]:.3f}  (+{best_gain:.4f})")
    else:
        print(f"  Round {round_n}: no candidate improves AUC by >0.0005 — stop")

print("\n" + "="*70)
print("RESULT")
print("="*70)
print(f"  Selected features: {selected if selected else 'NONE'}")
print(f"  Baseline AUC : {base_auc:.4f}")
print(f"  Final AUC    : {best_auc:.4f}  ({best_auc-base_auc:+.4f})")

# ============ Produce TEST probabilities if anything was selected
import os
if selected:
    use = base_feat + selected
    Xs = Xfull[use]
    Xts = pd.concat([test[base_feat], cand_test], axis=1)[use]
    imp = SimpleImputer(strategy='median')
    Xtr = imp.fit_transform(Xs); Xte = imp.transform(Xts)
    m = mk(); m.fit(Xtr, y)
    test_fe = m.predict_proba(Xte)[:, 1]
    if os.path.exists('test_probabilities.csv'):
        pdf = pd.read_csv('test_probabilities.csv')
    else:
        pdf = pd.DataFrame({'CoilID': test['CoilID'].values})
    pdf['xgb_fe'] = test_fe
    pdf.to_csv('test_probabilities.csv', index=False)
    print("\n✓ Saved xgb_fe column to test_probabilities.csv")
    print("  Use: python make_submission.py xgb_fe <threshold>")
else:
    print("\nNo features added — engineered features did not beat raw set.")
