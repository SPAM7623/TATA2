"""Quick honest experiments: what actually improves CV AUC?"""
import pandas as pd, numpy as np, warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore')

SEED=42; N=5
train=pd.read_csv('train.csv')
X=train.drop(['CoilID','Y'],axis=1); y=train['Y'].astype(int)
spw=(y==0).sum()/(y==1).sum()

def evaluate(make_model, label):
    skf=StratifiedKFold(n_splits=N,shuffle=True,random_state=SEED)
    oof=np.zeros(len(y))
    for tr,vl in skf.split(X,y):
        imp=SimpleImputer(strategy='median')
        Xtr=imp.fit_transform(X.iloc[tr]); Xvl=imp.transform(X.iloc[vl])
        m=make_model(); m.fit(Xtr,y.iloc[tr])
        oof[vl]=m.predict_proba(Xvl)[:,1]
    auc=roc_auc_score(y,oof); pr=average_precision_score(y,oof)
    print(f"{label:48s} AUC={auc:.4f}  PR-AUC={pr:.4f}")
    return auc

print("="*72)
print("HYPERPARAMETER / MODEL EXPERIMENTS (5-fold OOF AUC)")
print("="*72)

evaluate(lambda: XGBClassifier(n_estimators=150,max_depth=6,learning_rate=0.1,
    subsample=0.8,colsample_bytree=0.8,scale_pos_weight=10,
    random_state=SEED,eval_metric='logloss',verbosity=0,n_jobs=-1),
    "XGB orig (d6,150,lr.1,spw10)")

evaluate(lambda: XGBClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,
    subsample=0.8,colsample_bytree=0.8,scale_pos_weight=spw,
    random_state=SEED,eval_metric='logloss',verbosity=0,n_jobs=-1),
    "XGB current (d4,200,lr.05,spw_auto)")

evaluate(lambda: XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.03,
    subsample=0.7,colsample_bytree=0.7,scale_pos_weight=spw,
    reg_alpha=0.5,reg_lambda=2.0,min_child_weight=3,
    random_state=SEED,eval_metric='logloss',verbosity=0,n_jobs=-1),
    "XGB regularized (d3,300,lr.03,reg)")

evaluate(lambda: XGBClassifier(n_estimators=400,max_depth=3,learning_rate=0.02,
    subsample=0.7,colsample_bytree=0.6,scale_pos_weight=spw,
    reg_alpha=1.0,reg_lambda=3.0,min_child_weight=5,gamma=0.1,
    random_state=SEED,eval_metric='logloss',verbosity=0,n_jobs=-1),
    "XGB heavy-reg (d3,400,lr.02)")

evaluate(lambda: LGBMClassifier(n_estimators=300,max_depth=3,learning_rate=0.03,
    subsample=0.7,colsample_bytree=0.7,num_leaves=15,
    class_weight='balanced',reg_alpha=0.5,reg_lambda=2.0,
    random_state=SEED,verbosity=-1,n_jobs=-1),
    "LightGBM (d3,300,lr.03,balanced)")

evaluate(lambda: LGBMClassifier(n_estimators=400,max_depth=4,learning_rate=0.02,
    subsample=0.8,colsample_bytree=0.7,num_leaves=20,
    scale_pos_weight=spw,reg_alpha=1.0,reg_lambda=3.0,min_child_samples=20,
    random_state=SEED,verbosity=-1,n_jobs=-1),
    "LightGBM (d4,400,lr.02,spw)")
print("="*72)
