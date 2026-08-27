import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
ML="/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib"
df=pd.read_parquet(ML+"/accept_reject_clean.parquet")
print("split counts:",df["split"].value_counts().to_dict(),"| base accept=%.3f"%df.judgement.mean())
tr=df["split"]=="train"; ev=df["split"]=="eval"
# some slices may use different split labels; fallback
if tr.sum()<100 or ev.sum()<100:
    vc=df["split"].value_counts(); train_val=vc.idxmax(); 
    tr=df["split"]==train_val; ev=df["split"]!=train_val
print("train=%d eval=%d"%(tr.sum(),ev.sum()))
vec=TfidfVectorizer(min_df=5,max_features=40000,ngram_range=(1,2),sublinear_tf=True)
Xtr=vec.fit_transform(df.loc[tr,"diff"].astype(str)); Xte=vec.transform(df.loc[ev,"diff"].astype(str))
m=LogisticRegression(class_weight="balanced",max_iter=3000,C=1.0).fit(Xtr,df.loc[tr,"judgement"].values)
print("C (TF-IDF) AUC = %.3f"%roc_auc_score(df.loc[ev,"judgement"].values,m.predict_proba(Xte)[:,1]))
names=np.array(vec.get_feature_names_out()); coef=m.coef_[0]
order=np.argsort(coef)
print("\n=== TOP 40 REJECT tokens (coef most negative -> predict reject) ===")
for i in order[:40]:
    print("  %+6.2f  %s"%(coef[i],names[i]))
print("\n=== TOP 40 ACCEPT tokens (coef most positive -> predict accept) ===")
for i in order[-40:][::-1]:
    print("  %+6.2f  %s"%(coef[i],names[i]))
