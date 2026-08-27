import pandas as pd, numpy as np, csv, sys
csv.field_size_limit(sys.maxsize)
d=pd.read_csv("datasets/news-homepages/homepage_newsworthiness_topic_balanced_groupsplit.csv.gz",compression="gzip")
d["text"]=d.text.fillna("")
def split_hl(t):
    p=t.split("\n\nCONTEXT:",1)
    return p[0].replace("HEADLINE:","",1).strip(), (p[1].strip() if len(p)>1 else "")
hc=d.text.apply(lambda t:pd.Series(split_hl(t))); hc.columns=["hl","ctx"]; d=pd.concat([d,hc],axis=1)
# vectorized outlet detection
low_all=(d.hl+" "+d.ctx).str.lower()
OUTLETS=["nytimes","wsj","latimes","bbc","washingtonpost","cnn","guardian","reuters"]
mat=np.column_stack([low_all.str.contains(o,regex=False).values for o in OUTLETS])
d["outlet"]=np.where(mat.any(axis=1),np.array(OUTLETS)[mat.argmax(axis=1)],"other")
# within-snapshot pairs
pairs=[]
for snap,sd in d.groupby("snapshot_id"):
    pos=sd[sd.judgement==1]; neg=sd[sd.judgement==0]
    if len(pos)<1 or len(neg)<1: continue
    p=pos.iloc[(int(snap,16)%max(1,len(pos)))%len(pos)]; n=neg.iloc[(int(snap,16)%max(1,len(neg)))%len(neg)]
    pairs.append({"snapshot":snap,"outlet":p.outlet,"hl_top":p.hl,"hl_bot":n.hl,"ctx":p.ctx[:1500]})
    if len(pairs)>=220: break
pp=pd.DataFrame(pairs)
print(f"[pairs] {len(pp)} within-snapshot pairs; outlet mix: {pp.outlet.value_counts().to_dict()}",flush=True)
pp.to_json("datasets/news-homepages/analysis/within_snapshot_pairs.jsonl",orient="records",lines=True)
samp=d.sample(60,random_state=1)
samp[["text","judgement","snapshot_id","outlet"]].to_json("datasets/news-homepages/analysis/format_audit_rows.jsonl",orient="records",lines=True)
print("wrote within_snapshot_pairs.jsonl + format_audit_rows.jsonl",flush=True)
