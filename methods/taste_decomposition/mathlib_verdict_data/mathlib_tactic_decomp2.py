import numpy as np, pandas as pd, re
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
ML="/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib"
df=pd.read_parquet(ML+"/accept_reject_clean.parquet").reset_index(drop=True)
tr=df["split"]=="train"; ev=df["split"]=="eval"; yte=df.loc[ev,"judgement"].values
TACTICS=["grind","aesop","simp","simpa","fun_prop","funprop","cat_disch","catdisch","decide",
         "norm_num","ring","nlinarith","linarith","omega","intro","apply","have","unfold","rw",
         "rewrite","cases","induction","exact","refine","rwa","trans","calc","change","ext",
         "constructor","congr","simps","obtain"]
pat={t:re.compile(r"\b"+re.escape(t)+r"\b") for t in TACTICS}
tc=np.zeros((len(df),len(TACTICS)))
for i,d in enumerate(df["diff"].astype(str)):
    for j,t in enumerate(TACTICS): tc[i,j]=len(pat[t].findall(d))
def fe(X):
    sc=StandardScaler().fit(X[tr.values]); m=LogisticRegression(class_weight="balanced",max_iter=3000).fit(sc.transform(X[tr.values]),df.loc[tr,"judgement"].values)
    return roc_auc_score(yte,m.predict_proba(sc.transform(X[ev.values]))[:,1])
auto={"grind","aesop","simp","simpa","fun_prop","funprop","cat_disch","catdisch","decide","norm_num","ring","nlinarith","linarith"}
ai=[j for j,t in enumerate(TACTICS) if t in auto]; mi=[j for j,t in enumerate(TACTICS) if t not in auto]
ac=tc[:,ai].sum(1); mc=tc[:,mi].sum(1); ratio=(ac/(ac+mc+1e-9)).reshape(-1,1)
V=pd.read_parquet(ML+"/mathlib_diff_v_features.parquet"); vf=[c for c in V.columns if c!="number"]
Vm=df.merge(V[["number"]+vf],on="number",how="left")[vf].apply(pd.to_numeric,errors="coerce").values.astype(float); Vm=np.where(np.isnan(Vm),0,Vm)
print("=== mathlib accept/reject: is the V->C gap tactic-idiom? ===")
print("automation-ratio (1 feat)   = %.3f" % fe(ratio))
print("tactic-counts (32 feats)    = %.3f" % fe(tc))
print("V (det diff feats)          = %.3f" % fe(Vm))
print("V + tactic-counts           = %.3f" % fe(np.hstack([Vm,tc])))
vec=TfidfVectorizer(min_df=5,max_features=40000,ngram_range=(1,2),sublinear_tf=True)
Xtr=vec.fit_transform(df.loc[tr,"diff"].astype(str)); Xte=vec.transform(df.loc[ev,"diff"].astype(str))
m=LogisticRegression(class_weight="balanced",max_iter=3000).fit(Xtr,df.loc[tr,"judgement"].values)
cpred=m.predict_proba(Xte)[:,1]
print("C (TF-IDF)                  = %.3f" % roc_auc_score(yte,cpred))
rg=LinearRegression().fit(tc[ev.values],cpred); resid=cpred-rg.predict(tc[ev.values])
print("C residualized of tactic    = %.3f  (big drop => tactic was C's signal)" % roc_auc_score(yte,resid))
