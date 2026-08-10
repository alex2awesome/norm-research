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
    return "\n".join(l for l in str(d).split("\n") if not re.search(r"Copyright|Authors:|Released under|Apache 2\.0|described in the file LICENSE|SPDX-License-Identifier|maintainer", l, re.I))
df["diff_noauth"]=df["diff"].astype(str).map(strip_author)
def has_auth(d): return int(bool(re.search(r"riou|avigad|carneiro|batteries",str(d),re.I)))
print("=== AUTHOR STRIP ===")
print("author-token present: before=%.3f after=%.3f" % (df["diff"].map(has_auth).mean(), df["diff_noauth"].map(has_auth).mean()))
def C(textcol):
    vec=TfidfVectorizer(min_df=5,max_features=40000,ngram_range=(1,2),sublinear_tf=True)
    Xtr=vec.fit_transform(df.loc[tr,textcol].astype(str)); Xte=vec.transform(df.loc[ev,textcol].astype(str))
    m=LogisticRegression(class_weight="balanced",max_iter=3000).fit(Xtr,df.loc[tr,"judgement"].values)
    return m.predict_proba(Xte)[:,1]
print("C(TF-IDF) before=%s after=%s" % (round(roc_auc_score(yte,C("diff")),3), round(roc_auc_score(yte,C("diff_noauth")),3)))

# ---------- TOPIC = top-level Mathlib area ----------
def area(d):
    ms=re.findall(r"(?:a|b)/Mathlib/([A-Za-z0-9_]+)/",str(d))
    return Counter(ms).most_common(1)[0][0] if ms else "NONE"
df["area"]=df["diff"].astype(str).map(area)
g=df.groupby("area").agg(n=("judgement","size"),acc=("judgement","mean")).sort_values("acc")
print("\n=== ACCEPT RATE BY TOPIC (area; top/bottom, n>=40) ===")
gg=g[g["n"]>=40]
for a,r in pd.concat([gg.head(8),gg.tail(8)]).iterrows():
    print("  %-28s n=%5d accept=%.3f" % (a,int(r["n"]),r["acc"]))
top_areas=list(df.area.value_counts().head(25).index)
T=np.zeros((len(df),len(top_areas)))
for i,a in enumerate(df.area.values):
    if a in top_areas: T[i,top_areas.index(a)]=1
addn=df["additions"].astype(float).values.reshape(-1,1)
V=pd.read_parquet(ML+"/mathlib_diff_v_features.parquet"); vf=[c for c in V.columns if c!="number"]
Vm=df.merge(V[["number"]+vf],on="number",how="left")[vf].apply(pd.to_numeric,errors="coerce").values.astype(float); Vm=np.where(np.isnan(Vm),0,Vm)
def fe(X): 
    sc=StandardScaler().fit(X[tr.values]); m=LogisticRegression(class_weight="balanced",max_iter=3000).fit(sc.transform(X[tr.values]),df.loc[tr,"judgement"].values); return roc_auc_score(yte,m.predict_proba(sc.transform(X[ev.values]))[:,1])
def resid_auc(cpred,cov):
    r=LinearRegression().fit(cov[ev.values],cpred); return roc_auc_score(yte,cpred-r.predict(cov[ev.values]))
print("\n=== TOPIC decomposition ===")
print("TOPIC(area) alone            = %.3f" % fe(T))
print("V (det)                      = %.3f" % fe(Vm))
print("V + TOPIC                    = %.3f  (V adds over topic?)" % fe(np.hstack([Vm,T])))
# does V survive residualizing topic? fit V-model, residualize preds of topic
sc=StandardScaler().fit(Vm[tr.values]); vm=LogisticRegression(class_weight="balanced",max_iter=3000).fit(sc.transform(Vm[tr.values]),df.loc[tr,"judgement"].values)
vpred=vm.predict_proba(sc.transform(Vm[ev.values]))[:,1]
print("V preds resid of TOPIC       = %.3f  (V's topic-independent signal)" % resid_auc(vpred,T))
print("TOPIC resid of size(additions)= %.3f  (is topic just complexity?)" % resid_auc(fe.__wrapped__ if False else LogisticRegression(class_weight="balanced",max_iter=2000).fit(T[tr.values],df.loc[tr,"judgement"].values).predict_proba(T[ev.values])[:,1], addn[ev.values]))
