import numpy as np, pandas as pd, re
from collections import Counter
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
ML="/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib"
df=pd.read_parquet(ML+"/accept_reject_clean.parquet").reset_index(drop=True)
tr=df["split"]=="train"; ev=df["split"]=="eval"; yte=df.loc[ev,"judgement"].values
def strip_author(d):
    return "\n".join(l for l in str(d).split("\n") if not re.search(r"Copyright|Authors:|Released under|Apache 2\.0|described in the file LICENSE|SPDX-License|maintainer",l,re.I))
df["diff_noauth"]=df["diff"].astype(str).map(strip_author)
TACTICS=["grind","aesop","simp","simpa","fun_prop","funprop","cat_disch","catdisch","decide","norm_num","ring","nlinarith","linarith","omega","intro","apply","have","unfold","rw","rewrite","cases","induction","exact","refine","rwa","trans","calc","change","ext","constructor","congr","simps","obtain"]
pat={t:re.compile(r"\b"+t+r"\b") for t in TACTICS}
def tac(dfr,col):
    M=np.zeros((len(dfr),len(TACTICS)))
    for i,d in enumerate(dfr[col].astype(str)):
        for j,t in enumerate(TACTICS): M[i,j]=len(pat[t].findall(d))
    return M
V=pd.read_parquet(ML+"/mathlib_diff_v_features.parquet"); vf=[c for c in V.columns if c!="number"]
def Vmat(dfr): 
    X=dfr.merge(V[["number"]+vf],on="number",how="left")[vf].apply(pd.to_numeric,errors="coerce").values.astype(float); return np.where(np.isnan(X),0,X)
Vm=Vmat(df); Vprime=np.hstack([Vm,tac(df,"diff")])
av=pd.read_json(ML+"/a_metric_verdicts_mathlib.jsonl",lines=True)
anorms=[c for c in av.columns if c.startswith("m") and c[1:3].isdigit()]
av=av[["number"]+anorms].dropna(subset=anorms).drop_duplicates("number")
adf=df.merge(av,on="number",how="inner").reset_index(drop=True)
trA=adf["split"]=="train"; evA=adf["split"]=="eval"; yteA=adf.loc[evA,"judgement"].values
Am=adf[anorms].values.astype(float); Am=np.where(np.isnan(Am),0,Am)
VmA=Vmat(adf); VpA=np.hstack([VmA,tac(adf,"diff")])
def area(d):
    ms=re.findall(r"(?:a|b)/Mathlib/([A-Za-z0-9_]+)/",str(d)); return Counter(ms).most_common(1)[0][0] if ms else "NONE"
df["area"]=df["diff"].astype(str).map(area); adf["area"]=adf["diff"].astype(str).map(area)
top=list(df.area.value_counts().head(25).index)
def Tmat(dfr):
    M=np.zeros((len(dfr),len(top)))
    for i,a in enumerate(dfr.area.values):
        if a in top: M[i,top.index(a)]=1
    return M
T=Tmat(df); TA=Tmat(adf)
def fp(Xtr,ytr,Xte):
    sc=StandardScaler().fit(Xtr); m=LogisticRegression(class_weight="balanced",max_iter=3000).fit(sc.transform(Xtr),ytr); return m.predict_proba(sc.transform(Xte))[:,1]
def resid(p,cov): 
    r=LinearRegression().fit(cov,p); return p-r.predict(cov)
vec=TfidfVectorizer(min_df=5,max_features=40000,ngram_range=(1,2),sublinear_tf=True)
Xtr=vec.fit_transform(df.loc[tr,"diff_noauth"].astype(str)); Xte=vec.transform(df.loc[ev,"diff_noauth"].astype(str))
cm=LogisticRegression(class_weight="balanced",max_iter=3000).fit(Xtr,df.loc[tr,"judgement"].values)
Cpred=cm.predict_proba(Xte)[:,1]
Vpred=fp(Vm[tr.values],df.loc[tr,"judgement"].values,Vm[ev.values])
Vppred=fp(Vprime[tr.values],df.loc[tr,"judgement"].values,Vprime[ev.values])
Apred=fp(Am[trA.values],adf.loc[trA,"judgement"].values,Am[evA.values])
AVppred=fp(np.hstack([Am,VpA])[trA.values],adf.loc[trA,"judgement"].values,np.hstack([Am,VpA])[evA.values])
print("=== DE-CONFOUNDED remeasure: author-stripped + topic residualized ===")
print("n(V,C)=%d  n(A)=%d  base=%.3f" % (len(df),len(adf),df.judgement.mean()))
print("%-22s %8s %10s" % ("model","raw","topic-resid"))
for nm,p,y,Te in [("V (orig)",Vpred,yte,T[ev.values]),("V' (V+tactic)",Vppred,yte,T[ev.values]),
                  ("C (no-auth TFIDF)",Cpred,yte,T[ev.values]),("A (m01-10)",Apred,yteA,TA[evA.values]),
                  ("A + V'",AVppred,yteA,TA[evA.values])]:
    print("%-22s %8.3f %10.3f" % (nm,roc_auc_score(y,p),roc_auc_score(y,resid(p,Te))))
