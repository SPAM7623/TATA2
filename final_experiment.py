"""Confirm best config + multi-seed ensemble, honestly measured."""
import pandas as pd, numpy as np, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

SEED=42; N=5; SEEDS=[42,123,456,789,999]
train=pd.read_csv('train.csv')
X=train.drop(['CoilID','Y'],axis=1); y=train['Y'].astype(int)
spw=(y==0).sum()/(y==1).sum()

def mk(sd):
    return XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.03,
        subsample=0.7,colsample_bytree=0.7,scale_pos_weight=spw,
        reg_alpha=0.5,reg_lambda=2.0,min_child_weight=3,
        random_state=sd,eval_metric='logloss',verbosity=0,n_jobs=-1)

def run(ensemble):
    skf=StratifiedKFold(n_splits=N,shuffle=True,random_state=SEED)
    oof=np.zeros(len(y))
    for tr,vl in skf.split(X,y):
        imp=SimpleImputer(strategy='median')
        Xtr=imp.fit_transform(X.iloc[tr]); Xvl=imp.transform(X.iloc[vl])
        seeds=SEEDS if ensemble else [SEED]
        p=np.zeros(len(vl))
        for sd in seeds:
            m=mk(sd); m.fit(Xtr,y.iloc[tr]); p+=m.predict_proba(Xvl)[:,1]
        oof[vl]=p/len(seeds)
    return roc_auc_score(y,oof), average_precision_score(y,oof)

a1,p1=run(False); print(f"Regularized XGB single-seed   AUC={a1:.4f}  PR-AUC={p1:.4f}")
a2,p2=run(True);  print(f"Regularized XGB 5-seed ensemble AUC={a2:.4f}  PR-AUC={p2:.4f}")
print(f"\nBaseline was 0.8641. Best = {max(a1,a2):.4f} (+{max(a1,a2)-0.8641:+.4f})")
