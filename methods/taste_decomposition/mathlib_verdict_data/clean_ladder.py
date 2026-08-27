import json, numpy as np, pandas as pd, glob
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
ML="/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib"

ar=pd.read_parquet(ML+"/accept_reject_deconfounded.parquet")
ar=ar.rename(columns={"additions":"addn"})
av=pd.read_json(ML+"/a_metric_verdicts_mathlib.jsonl",lines=True)
anorms=[c for c in av.columns if c.startswith("m") and c[1:3].isdigit()]
av=av[["number"]+anorms].dropna(subset=anorms).drop_duplicates("number")
V=pd.read_parquet(ML+"/mathlib_diff_v_features.parquet")
vfeats=[c for c in V.columns if c!="number"]
# de-confounded V feats: drop raw size cols that ARE size (keep style-ish ones? report both)
sizecols_in_V={"added","deleted","net","churn","n_files","n_def","n_lean","add_del_ratio"}
df=ar.merge(av,on="number",how="left").merge(V[["number"]+vfeats],on="number",how="left")
print("merged n=%d  hasA=%d  hasV=%d  base accept=%.3f" % (len(df),df[anorms[0]].notna().sum(),df[vfeats[0]].notna().sum(),df.judgement.mean()))

def cv_auc(d, feat_builder, label="judgement"):
    d=d.dropna(subset=[label]).reset_index(drop=True)
    y=d[label].values.astype(float); pred=np.zeros(len(d)); g=d["number"].values
    skf=StratifiedKFold(5,shuffle=True,random_state=0)
    for tr,te in skf.split(np.zeros(len(d)),y):
        Xtr,Xte=feat_builder(d,tr,te)
        m=LogisticRegression(class_weight="balanced",max_iter=2000).fit(Xtr,y[tr]); pred[te]=m.predict_proba(Xte)[:,1]
    try: return roc_auc_score(y,pred)
    except Exception: return float('nan')
def sz(d,tr,te): return d[["addn"]].iloc[tr].values.astype(float), d[["addn"]].iloc[te].values.astype(float)
def Vb(d,tr,te):
    X=d[vfeats].values.astype(float); X=np.where(np.isnan(X),0,X)
    mu=X[tr].mean(0); sd=np.where(X[tr].std(0)==0,1,X[tr].std(0))
    return (X[tr]-mu)/sd,(X[te]-mu)/sd
def Ab(d,tr,te):
    X=d[anorms].values.astype(float); X=np.where(np.isnan(X),0,X)
    mu=X[tr].mean(0); sd=np.where(X[tr].std(0)==0,1,X[tr].std(0))
    return (X[tr]-mu)/sd,(X[te]-mu)/sd
def Cb(d,tr,te):
    vec=TfidfVectorizer(min_df=3,max_features=20000,ngram_range=(1,2),sublinear_tf=True)
    Xtr=vec.fit_transform(d["diff"].iloc[tr].astype(str)); Xte=vec.transform(d["diff"].iloc[te].astype(str))
    return Xtr,Xte

def report(tag,d):
    d=d.dropna(subset=["diff"]).copy()
    szA=cv_auc(d,sz)
    print("  %-34s n=%5d base=%.3f | size->rej %.3f | V %.3f | A %.3f | C(TFIDF) %.3f" % (
        tag,len(d),d.judgement.mean(),szA,cv_auc(d,Vb),cv_auc(d,Ab),cv_auc(d,Cb)))

# subsets
empty = df["addn"]>0
topcut = df["addn"]<=1000   # cut abnormal mega-PRs (~p99=1021)
reviewed = df["n_reviews"]>0
print("\n=== SIZE->REJECT / V / A / C under cleaning (accept/reject leg, real additions) ===")
report("FULL", df)
report("drop empty (addn==0)", df[empty])
report("drop top mega (addn>1000)", df[empty & topcut])
report("  + control abandon (n_reviews>0)", df[empty & topcut & reviewed])
report("control abandon ONLY (n_reviews>0)", df[reviewed])
print("\n(legend: size->rej = additions predicting judgement; AUC>0.5 means bigger->more rejected)")
print("mega-PRs cut (addn>1000): n=%d, all rejected? accept rate=%.3f, n_reviews==0 share=%.3f" % (
    (df["addn"]>1000).sum(), df[df["addn"]>1000].judgement.mean(), (df[df["addn"]>1000].n_reviews==0).mean()))
