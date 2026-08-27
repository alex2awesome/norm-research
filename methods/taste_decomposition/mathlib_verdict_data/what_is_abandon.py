import numpy as np, pandas as pd, re
from collections import Counter
ML="/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib"
ar=pd.read_parquet(ML+"/accept_reject_deconfounded.parquet")
rej=ar[ar.judgement==0].copy()
ab=rej[rej["n_reviews"]==0].copy()      # "abandoned" by my shorthand
rv=rej[rej["n_reviews"]>0].copy()       # reviewed rejects
print("REJECTS: n=%d  | n_reviews==0 ('abandoned'): n=%d (%.1f%%)  | reviewed: n=%d (%.1f%%)" % (
    len(rej),len(ab),100*len(ab)/len(rej),len(rv),100*len(rv)/len(rej)))

def labfreq(df):
    c=Counter()
    for s in df["labels"].fillna(""):
        for t in str(s).split("|"):
            t=t.strip()
            if t: c[t]+=1
    return c
print("\n=== TOP LABELS among n_reviews==0 rejects (share of %d) ===" % len(ab))
for k,v in labfreq(ab).most_common(12):
    print("  %-28s %4d  (%.1f%%)" % (k,v,100*v/len(ab)))
print("\n=== TOP LABELS among reviewed rejects (share of %d) ===" % len(rv))
for k,v in labfreq(rv).most_common(12):
    print("  %-28s %4d  (%.1f%%)" % (k,v,100*v/len(rv)))

print("\n=== do 'abandoned' rejects have THREAD engagement? (n_review_threads) ===")
for tag,d in [("n_reviews==0 rejects",ab),("reviewed rejects",rv),("accepts",ar[ar.judgement==1])]:
    t=d["n_review_threads"].values
    print("  %-22s threads median=%.0f  ==0 share=%.2f  >=3 share=%.2f" % (tag,np.median(t),(t==0).mean(),(t>=3).mean()))

print("\n=== author_association ===")
for tag,d in [("n_reviews==0 rejects",ab),("reviewed rejects",rv)]:
    print("  ",tag, dict(d.author_association.value_counts()))

print("\n=== title patterns (WIP/Draft/duplicate/stale/close) ===")
pat=re.compile(r"\b(wip|draft|duplicate|stale|close|revert|todo|fixme)\b",re.I)
for tag,d in [("n_reviews==0 rejects",ab),("reviewed rejects",rv)]:
    m=d.title.astype(str).str.contains(pat,case=False).mean()
    print("  %-22s title-match share=%.2f" % (tag,m))

print("\n=== sample TITLES of n_reviews==0 rejects ===")
for t in ab.title.dropna().sample(min(15,len(ab)),random_state=3):
    print("   -",str(t)[:85])
print("\n=== sample TITLES of reviewed rejects ===")
for t in rv.title.dropna().sample(min(8,len(rv)),random_state=3):
    print("   -",str(t)[:85])
