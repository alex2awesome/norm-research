#!/usr/bin/env python3
import pandas as pd, numpy as np, csv, sys, json
csv.field_size_limit(sys.maxsize)
def load(p):
    d=pd.read_csv(p,compression="gzip"); d["text"]=d.text.fillna("")
    def split(t):
        x=t.split("\n\nCONTEXT:",1); return x[0].replace("HEADLINE:","",1).strip(),(x[1].strip() if len(x)>1 else "")
    hc=d.text.apply(lambda t:pd.Series(split(t))); hc.columns=["hl","ctx"]
    return d,hc
orig,_=load("datasets/news-homepages/homepage_newsworthiness_topic_balanced_groupsplit.csv.gz")
v2,hcv=load("datasets/news-homepages/homepage_newsworthiness_clean_v2.csv.gz")
OUTLETS=["nytimes","wsj","latimes","bbc","washingtonpost","cnn","guardian","reuters"]
low=(hcv.hl+" "+hcv.ctx).str.lower()
mat=np.column_stack([low.str.contains(o,regex=False).values for o in OUTLETS])
outlet=np.where(mat.any(axis=1),np.array(OUTLETS)[mat.argmax(axis=1)],"other")
v2s=hcv.sample(min(80,len(hcv)),random_state=2).copy()
v2s["judgement"]=v2.loc[v2s.index,"judgement"].values
v2s["snapshot_id"]=v2.loc[v2s.index,"snapshot_id"].values
v2s["outlet"]=outlet[v2s.index]
v2s[["hl","ctx","judgement","snapshot_id","outlet"]].to_json("datasets/news-homepages/analysis/verify_clean_sample.jsonl",orient="records",lines=True)
df=v2.copy(); df["hl"]=hcv.hl.values; df["outlet"]=outlet
pairs=[]
for snap,sd in df.groupby("snapshot_id"):
    pos=sd[sd.judgement==1]; neg=sd[sd.judgement==0]
    if len(pos)<1 or len(neg)<1: continue
    p=pos.iloc[0]; n=neg.iloc[0]
    pairs.append({"snapshot":snap,"outlet":p.outlet,"hl_top":p.hl,"hl_bot":n.hl})
    if len(pairs)>=80: break
pd.DataFrame(pairs).to_json("datasets/news-homepages/analysis/verify_clean_pairs.jsonl",orient="records",lines=True)
dropped_idx=set(orig.text.str[:60])-set(v2.text.str[:60])
droph=orig[orig.text.str[:60].isin(dropped_idx)].head(40)
with open("datasets/news-homepages/analysis/verify_dropped_sample.jsonl","w") as f:
    for _,r in droph.iterrows():
        x=r.text.split("\n\nCONTEXT:",1); hl=x[0].replace("HEADLINE:","",1).strip()[:120]
        f.write(json.dumps({"hl":hl,"judgement":int(r.judgement)})+"\n")
print(f"wrote verify_clean_sample(80) + verify_clean_pairs({len(pairs)}) + verify_dropped_sample",flush=True)
