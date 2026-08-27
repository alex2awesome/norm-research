import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
ML="/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib"
ar=pd.read_parquet(ML+"/accept_reject_deconfounded.parquet").rename(columns={"additions":"addn"})
av=pd.read_json(ML+"/a_metric_verdicts_mathlib.jsonl",lines=True)
anorms=[c for c in av.columns if c.startswith("m") and c[1:3].isdigit()]
av=av[["number"]+anorms].dropna(subset=anorms).drop_duplicates("number")
V=pd.read_parquet(ML+"/mathlib_diff_v_features.parquet"); vfeats=[c for c in V.columns if c!="number"]
df=ar.merge(av,on="number",how="left").merge(V[["number"]+vfeats],on="number",how="left")

def cv(d,builder,label="judgement"):
    d=d.dropna(subset=[label]).reset_index(drop=True)
    y=d[label].values.astype(float); pred=np.zeros(len(d))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(len(d)),y):
        Xtr,Xte=builder(d,tr,te)
        m=LogisticRegression(class_weight="balanced",max_iter=2000).fit(Xtr,y[tr]); pred[te]=m.predict_proba(Xte)[:,1]
    try: return roc_auc_score(y,pred)
    except Exception: return float('nan')
def Sz(d,tr,te): return d[["addn"]].iloc[tr].values.astype(float),d[["addn"]].iloc[te].values.astype(float)
def Vb(d,tr,te):
    X=d[vfeats].values.astype(float); X=np.where(np.isnan(X),0,X); mu=X[tr].mean(0); sd=np.where(X[tr].std(0)==0,1,X[tr].std(0)); return (X[tr]-mu)/sd,(X[te]-mu)/sd
def Ab(d,tr,te):
    X=d[anorms].values.astype(float); X=np.where(np.isnan(X),0,X); mu=X[tr].mean(0); sd=np.where(X[tr].std(0)==0,1,X[tr].std(0)); return (X[tr]-mu)/sd,(X[te]-mu)/sd
def Cb(d,tr,te):
    vec=TfidfVectorizer(min_df=3,max_features=20000,ngram_range=(1,2),sublinear_tf=True)
    return vec.fit_transform(d["diff"].iloc[tr].astype(str)),vec.transform(d["diff"].iloc[te].astype(str))
def row(tag,d):
    d=d.dropna(subset=["diff"]).copy()
    print("  %-42s n=%5d base=%.3f | size->rej %.3f | V %.3f | A %.3f | C %.3f" % (
        tag,len(d),d.judgement.mean(),cv(d,Sz),cv(d,Vb),cv(d,Ab),cv(d,Cb)))

# size hygiene
hyg=(df["addn"]>0)&(df["addn"]<=1000)
# Slice A (CANONICAL): drop non-engagement REJECTS only, keep all accepts + size hygiene
sliceA = hyg & ~((df["judgement"]==0)&(df["n_review_threads"]==0))
# Slice B (alt): reviewed-only both classes
sliceB = hyg & (df["n_review_threads"]>0)
print("=== CANONICAL clean slice (Slice A): drop no-review rejects + empty + mega ===")
row("Slice A (canonical)", df[sliceA])
print("\n=== Alt Slice B: reviewed-only (drop no-review both classes) ===")
row("Slice B (reviewed-only)", df[sliceB])
print("\n=== FULL (reference) ===")
row("FULL", df)

# save canonical slice (all original columns)
out=ar[sliceA].copy()
out.to_parquet(ML+"/accept_reject_clean.parquet",index=False)
print("\nsaved canonical slice -> accept_reject_clean.parquet  n=%d base=%.3f" % (len(out),out.judgement.mean()))
print("dropped: empty(add=0)=%d, mega(add>1000)=%d, no-review-rejects=%d" % (
    (df["addn"]==0).sum(),(df["addn"]>1000).sum(),((df["judgement"]==0)&(df["n_review_threads"]==0)).sum()))
