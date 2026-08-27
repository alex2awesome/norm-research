#!/usr/bin/env python3
"""Full-page ranking: give the 70B ALL headlines of a snapshot, ask to RANK by prominence.
Uses full homepage context (vs 1-v-1 pairwise). Metric: per-item AUC of judge-rank predicting
the binary top/bottom label (grouped by snapshot), + within-snap pairwise agreement."""
import os,sys,json,re,csv,warnings
warnings.filterwarnings("ignore"); csv.field_size_limit(sys.maxsize)
sys.path.insert(0,"methods")
import numpy as np,pandas as pd
from metric_implementer.backends import LLMBackend, BACKENDS
from metric_implementer.config import ImplementerConfig
from sklearn.metrics import roc_auc_score
DS="/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages"
open("/tmp/judge_dummy_key.txt","w").write("dummy")
BACKENDS["local"]={"url":"http://127.0.0.1:8005/v1/chat/completions","key":"/tmp/judge_dummy_key.txt","format":"openai"}
cfg=ImplementerConfig(); cfg.backend="local"; cfg.llm_concurrency=24; cfg.request_timeout_s=120
judge=LLMBackend("pr-judge","judge",cfg)
d=pd.read_csv(f"{DS}/homepage_newsworthiness_clean_v8.csv.gz",compression="gzip"); d["text"]=d.text.fillna("")
def split(t):
    x=t.split("\n\nCONTEXT:",1); return x[0].replace("HEADLINE:","",1).strip(),(x[1].strip() if len(x)>1 else "")
hc=d.text.apply(lambda t:pd.Series(split(t))); d["hl"]=hc[0].values
# snapshots with 4-10 articles
rng=np.random.default_rng(0); snaps=list(d.snapshot_id.unique()); rng.shuffle(snaps)
pages=[]
for s in snaps:
    sd=d[d.snapshot_id==s]
    if not (4<=len(sd)<=10): continue
    if sd.judgement.nunique()<2: continue
    pages.append(sd)
    if len(pages)>=500: break
print(f"[rank] {len(pages)} snapshots (full-page ranking)",flush=True)
PROMPT=("You are an expert news editor. Below are {n} headlines that appeared on the SAME news homepage "
 "(same outlet, same day). Rank them by EDITORIAL PROMINENCE - which the editors placed higher on the page. "
 "Judge by newsworthiness: importance, timeliness, prominence of named actors, magnitude, conflict/casualty, "
 "breaking status, institutional stakes, concrete human impact. Hard news (politics/war/disaster/crime) usually "
 "beats soft (lifestyle/entertainment).\n\n{items}\n\nOutput ONLY the IDs in rank order, most-prominent first, "
 "comma-separated (e.g. H3,H1,H4,H2).")
prompts=[]; page_meta=[]
for pi,sd in enumerate(pages):
    rows=[]
    for _,r in sd.iterrows():
        hid=f"H{pi}_{_}"  # unique-ish
        rows.append(f"{hid}: {r['hl'][:160]}")
    items="\n".join(rows)
    # shuffle the item order in the prompt to avoid list-position bias
    rng2=np.random.default_rng(hash(str(sd.snapshot_id.values[0]))%(2**32)); rng2.shuffle(rows)
    items="\n".join(rows)
    prompts.append(PROMPT.format(n=len(sd),items=items))
    page_meta.append(sd)
resps=judge.generate_batch(prompts,max_tokens=200,temperature=0.0)
# parse: rank each headline (1=most prominent). score = rank position (lower=more prominent)
all_y=[]; all_score=[]; pair_right=0; pair_tot=0
for sd,resp in zip(page_meta,resps):
    # parse the ordered IDs
    ids=re.findall(r"H\d+_\d+",resp or "")
    # map row index -> id
    id_by_idx={f"H{pi}_{idx}":idx for pi,idx in enumerate(sd.index) for pi in [pages.index(sd)]}  # messy; simpler below
    rank={}
    for rpos,hid in enumerate(ids):
        try: idx=int(hid.split("_")[1]); rank[idx]=rpos
        except: pass
    # if not all ranked, skip
    if len(rank)<len(sd): continue
    # within-snap pairwise agreement + per-item score
    yu={}; su={}
    for idx in sd.index:
        yu[idx]=int(sd.loc[idx,"judgement"]); su[idx]=rank[idx]
    idxs=list(sd.index)
    for i in range(len(idxs)):
        for j in range(i+1,len(idxs)):
            a,b=idxs[i],idxs[j]
            if yu[a]==yu[b]: continue
            pair_tot+=1
            # top (y=1) should have lower rank (more prominent)
            if (yu[a]>yu[b] and su[a]<su[b]) or (yu[b]>yu[a] and su[b]<su[a]): pair_right+=1
    all_y.extend([yu[i] for i in idxs]); all_score.extend([su[i] for i in idxs])
all_y=np.array(all_y); all_score=np.array(all_score)
# AUC: lower rank = more prominent = predict y=1. So use -score.
auc=roc_auc_score(all_y,-all_score)
pairwise_acc=pair_right/max(pair_tot,1)
print(f"[rank] full-page-ranking: per-item AUC={auc:.4f} (n={len(all_y)}); within-snap pairwise accuracy={pairwise_acc:.4f} (n={pair_tot})",flush=True)
print(f"[rank] refs: simple pairwise=0.569; feature pair-diff=0.69; dense within-snap=0.69; human~0.80",flush=True)
print("RANK_DONE",flush=True)
