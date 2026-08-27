#!/usr/bin/env python3
"""Pairwise/contrastive judge: 70B decides which of two same-homepage headlines was placed MORE
prominently (top vs bottom). A/B order randomized. Measures accuracy — does a holistic contrastive
judge capture what score-based rubrics provably can't (A~0.57)? Human~80%; feature pair-diff AUC~0.69."""
import os,sys,json,re,csv,warnings
warnings.filterwarnings("ignore")
csv.field_size_limit(sys.maxsize)
sys.path.insert(0,"methods")
import numpy as np,pandas as pd
from metric_implementer.backends import LLMBackend, BACKENDS
from metric_implementer.config import ImplementerConfig
DS="/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages"
open("/tmp/judge_dummy_key.txt","w").write("dummy")
BACKENDS["local"]={"url":"http://127.0.0.1:8005/v1/chat/completions","key":"/tmp/judge_dummy_key.txt","format":"openai"}
cfg=ImplementerConfig(); cfg.backend="local"; cfg.llm_concurrency=32; cfg.request_timeout_s=60
judge=LLMBackend("pr-judge","judge",cfg)
d=pd.read_csv(f"{DS}/homepage_newsworthiness_clean_v8.csv.gz",compression="gzip"); d["text"]=d.text.fillna("")
def split(t):
    x=t.split("\n\nCONTEXT:",1); return x[0].replace("HEADLINE:","",1).strip(),(x[1].strip() if len(x)>1 else "")
hc=d.text.apply(lambda t:pd.Series(split(t))); d["hl"]=hc[0].values
# build within-snap pairs
rng=np.random.default_rng(0); pairs=[]
for snap,sd in d.groupby("snapshot_id"):
    pos=sd[sd.judgement==1]; neg=sd[sd.judgement==0]
    if len(pos)<1 or len(neg)<1: continue
    for _,p in pos.iterrows():
        n=neg.sample(1,random_state=int(str(snap),16)%9999).iloc[0]
        # randomize A/B order
        if rng.random()<0.5: a,b,lbl=p["hl"],n["hl"],"A"   # top=A
        else: a,b,lbl=n["hl"],p["hl"],"B"                  # top=B
        pairs.append({"a":a[:300],"b":b[:300],"top":lbl,"snap":snap})
    if len(pairs)>=3000: break
pairs=pairs[:3000]
print(f"[pairwise] {len(pairs)} within-snap pairs (A/B randomized)",flush=True)
PROMPT=("Two news headlines A and B appeared on the SAME news homepage (same outlet, same day). Editors "
 "placed ONE of them MORE prominently (higher on the page, above the fold). Based on editorial newsworthiness "
 "judgment - importance, timeliness, prominence of named actors, magnitude/scale, conflict/casualty, "
 "breaking/developing status, institutional stakes - which headline was placed higher?\n\n"
 "Headline A: {a}\n\nHeadline B: {b}\n\nOutput ONLY a single letter, A or B.")
prompts=[PROMPT.format(a=p["a"],b=p["b"]) for p in pairs]
resps=judge.generate_batch(prompts,max_tokens=4,temperature=0.0)
correct=0; n_parse=0; a_count=0
for p,r in zip(pairs,resps):
    r=(r or "").strip()
    m=re.search(r"\b([AB])\b",r)
    if not m: continue
    n_parse+=1
    pick=m.group(1); a_count+=(pick=="A")
    if pick==p["top"]: correct+=1
acc=correct/max(n_parse,1)
ci=1.96*np.sqrt(acc*(1-acc)/n_parse)
print(f"[pairwise] 70B pairwise-judge accuracy = {acc:.4f}  (+/-{ci:.3f}, n={n_parse})",flush=True)
print(f"[pairwise] position-bias: picks A {a_count/n_parse:.2%} of the time (random=A 50%)",flush=True)
print(f"[pairwise] refs: human~80% pairwise; feature pair-diff AUC~0.69; chance=0.50",flush=True)
print("PAIRWISE_DONE",flush=True)
