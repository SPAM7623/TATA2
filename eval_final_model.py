"""Honest evaluation of the production model via 5-fold OOF predictions."""
import pandas as pd, numpy as np, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    matthews_corrcoef, brier_score_loss)
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

SEED=42; N=5
train=pd.read_csv('train.csv')
X=train.drop(['CoilID','Y'],axis=1); y=train['Y'].astype(int).values
spw=(y==0).sum()/(y==1).sum()

def mk():
    return XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.03,
        subsample=0.7,colsample_bytree=0.7,scale_pos_weight=spw,
        reg_alpha=0.5,reg_lambda=2.0,min_child_weight=3,
        random_state=SEED,eval_metric='logloss',verbosity=0,n_jobs=-1)

# Out-of-fold predictions (no leakage)
skf=StratifiedKFold(n_splits=N,shuffle=True,random_state=SEED)
oof=np.zeros(len(y))
for tr,vl in skf.split(X,y):
    imp=SimpleImputer(strategy='median')
    Xtr=imp.fit_transform(X.iloc[tr]); Xvl=imp.transform(X.iloc[vl])
    m=mk(); m.fit(Xtr,y[tr]); oof[vl]=m.predict_proba(Xvl)[:,1]

auc=roc_auc_score(y,oof); pr=average_precision_score(y,oof)
brier=brier_score_loss(y,oof)

print("="*70)
print("PRODUCTION MODEL - HONEST 5-FOLD OUT-OF-FOLD METRICS")
print("Regularized XGBoost (d3, 300, lr.03, reg) | median imputation")
print("="*70)
print(f"\nThreshold-independent:")
print(f"  ROC-AUC : {auc:.4f}")
print(f"  PR-AUC  : {pr:.4f}")
print(f"  Brier   : {brier:.4f}")

print(f"\nThreshold-dependent metrics:")
print(f"  {'Thresh':>8} {'Prec':>6} {'Recall':>7} {'F1':>6} {'MCC':>6}  {'TP':>3} {'FP':>4} {'FN':>3} {'TN':>4}")
for t in [0.10,0.20,0.28,0.30,0.40,0.50,0.60]:
    pred=(oof>=t).astype(int)
    if pred.sum()==0: continue
    prec=precision_score(y,pred,zero_division=0)
    rec=recall_score(y,pred,zero_division=0)
    f1=f1_score(y,pred,zero_division=0)
    mcc=matthews_corrcoef(y,pred)
    tn,fp,fn,tp=confusion_matrix(y,pred).ravel()
    print(f"  {t:>8.2f} {prec:>6.3f} {rec:>7.3f} {f1:>6.3f} {mcc:>6.3f}  {tp:>3} {fp:>4} {fn:>3} {tn:>4}")

# Best-F1 and best-MCC thresholds
ts=np.linspace(0.01,0.95,95)
f1s=[f1_score(y,(oof>=t).astype(int),zero_division=0) for t in ts]
mccs=[matthews_corrcoef(y,(oof>=t).astype(int)) for t in ts]
bf1=ts[int(np.argmax(f1s))]; bmcc=ts[int(np.argmax(mccs))]
print(f"\nOptimal thresholds:")
print(f"  Best F1  @ {bf1:.2f} -> F1={max(f1s):.3f}")
print(f"  Best MCC @ {bmcc:.2f} -> MCC={max(mccs):.3f}")
print("="*70)
