#!/usr/bin/env python3
"""Pairwise-v2: CoT + few-shot. Same 3000 within-snap pairs but prompt has (a) 4 worked few-shot
examples with news-value reasoning + correct answer, (b) asks for 1-line reasoning then A/B.
Tests whether the 70B's 0.569 (which underperformed feature pair-diff 0.69) is a prompting failure."""
import os,sys,json,re,csv,warnings
warnings.filterwarnings("ignore"); csv.field_size_limit(sys.maxsize)
sys.path.insert(0,"methods")
import numpy as np,pandas as pd
from metric_implementer.backends import LLMBackend, BACKENDS
from metric_implementer.config import ImplementerConfig
DS="/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages"
open("/tmp/judge_dummy_key.txt","w").write("dummy")
BACKENDS["local"]={"url":"http://127.0.0.1:8005/v1/chat/completions","key":"/tmp/judge_dummy_key.txt","format":"openai"}
cfg=ImplementerConfig(); cfg.backend="local"; cfg.llm_concurrency=32; cfg.request_timeout_s=90
judge=LLMBackend("pr-judge","judge",cfg)
d=pd.read_csv(f"{DS}/homepage_newsworthiness_clean_v8.csv.gz",compression="gzip"); d["text"]=d.text.fillna("")
def split(t):
    x=t.split("\n\nCONTEXT:",1); return x[0].replace("HEADLINE:","",1).strip(),(x[1].strip() if len(x)>1 else "")
hc=d.text.apply(lambda t:pd.Series(split(t))); d["hl"]=hc[0].values
rng=np.random.default_rng(0)
snaps=list(d.snapshot_id.unique()); rng.shuffle(snaps)
fewshot_snaps=set(snaps[:4]); test_snaps=[s for s in snaps[4:] if (d.snapshot_id==s).sum()>=4][:600]
# few-shot examples: one clear pair per fewshot snapshot
fewshot=[]
for s in fewshot_snaps:
    sd=d[d.snapshot_id==s]; pos=sd[sd.judgement==1]; neg=sd[sd.judgement==0]
    if len(pos)<1 or len(neg)<1: continue
    p=pos.iloc[0]; n=neg.iloc[0]
    fewshot.append((p["hl"][:160],n["hl"][:160]))  # (top,bot)
fewshot=fewshot[:4]
# build test pairs (disjoint snapshots)
pairs=[]
for s in test_snaps:
    sd=d[d.snapshot_id==s]; pos=sd[sd.judgement==1]; neg=sd[sd.judgement==0]
    if len(pos)<1 or len(neg)<1: continue
    for _,p in pos.iterrows():
        n=neg.sample(1,random_state=int(str(s),16)%9999).iloc[0]
        if rng.random()<0.5: a,b,lbl=p["hl"],n["hl"],"A"
        else: a,b,lbl=n["hl"],p["hl"],"B"
        pairs.append({"a":a[:200],"b":b[:200],"top":lbl})
    if len(pairs)>=3000: break
pairs=pairs[:3000]
print(f"[pairwise-v2] {len(pairs)} test pairs + {len(fewshot)} few-shot",flush=True)
FS="\n\n".join(f"Example {i+1}:\nA: {t}\nB: {b}\nThe more prominent (placed higher) headline is: {('A')}\nReason: the top story carries stronger newsworthiness (elite actors, conflict, magnitude, or breaking status) than the softer/routine alternative." for i,(t,b) in enumerate(fewshot))
PROMPT=("You are an expert news editor. Two headlines A and B appeared on the SAME news homepage. "
 "Editors placed ONE MORE prominently (higher on the page). Judge by editorial newsworthiness: "
 "importance, timeliness, prominence of named actors, magnitude/scale, conflict/casualty, breaking status, "
 "institutional stakes, concrete human impact. Hard news (politics/war/disaster/crime) usually beats soft "
 "(lifestyle/entertainment/service).\n\n{fs}\n\nNow decide:\nA: {a}\nB: {b}\n\n"
 "Give one short sentence of reasoning, then on a new line 'ANSWER: A' or 'ANSWER: B'.")
prompts=[PROMPT.format(fs=FS,a=p["a"],b=p["b"]) for p in pairs]
resps=judge.generate_batch(prompts,max_tokens=120,temperature=0.0)
correct=0;n_parse=0;a_count=0
for p,r in zip(pairs,resps):
    m=re.search(r"ANSWER:\s*([AB])",(r or "").upper())
    if not m: m=re.search(r"\b([AB])\b\s*$",(r or "").strip())
    if not m: continue
    n_parse+=1; pick=m.group(1); a_count+=(pick=="A")
    if pick==p["top"]: correct+=1
acc=correct/max(n_parse,1); ci=1.96*np.sqrt(acc*(1-acc)/n_parse)
print(f"[pairwise-v2] CoT+few-shot 70B accuracy = {acc:.4f} (+/-{ci:.3f}, n={n_parse})",flush=True)
print(f"[pairwise-v2] position-bias picks-A: {a_count/n_parse:.2%}; baseline (simple pairwise) was 0.569",flush=True)
print("PAIRWISE_V2_DONE",flush=True)
