#!/usr/bin/env python3
"""Codex fix #1: bare-HEADLINE-ONLY judge. Strip summary + context pollution. Tests whether the
70B's 0.57 is input-degradation (vs task-impossible). Runs BOTH the rubric A-layer (14 metrics)
AND the pairwise judge on bare-headline-only, compares to full-text (A=0.569, pairwise=0.569)."""
import os,sys,re,csv,warnings
warnings.filterwarnings("ignore"); csv.field_size_limit(sys.maxsize)
sys.path.insert(0,"methods"); sys.path.insert(0,"datasets/news-homepages/analysis")
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from metrics_tree_infilling.io_metrics import materialize, make_vllm_judge_scorer, MetricSpec
from metrics_tree_infilling.config import InfillConfig
from metric_implementer.backends import LLMBackend, BACKENDS
from metric_implementer.config import ImplementerConfig
from news_newmetrics import NEW_METRICS, EXISTING_TOP4
DS="/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages"
open("/tmp/judge_dummy_key.txt","w").write("dummy")
BACKENDS["local"]={"url":"http://127.0.0.1:8005/v1/chat/completions","key":"/tmp/judge_dummy_key.txt","format":"openai"}
d=pd.read_csv(f"{DS}/homepage_newsworthiness_clean_v8.csv.gz",compression="gzip"); d["text"]=d.text.fillna("")
def bare_hl(t):
    h=t.split("\n\nCONTEXT:",1)[0].replace("HEADLINE:","",1).strip()
    # first sentence/clause only
    m=re.split(r"(?<=[.?!])\s+",h,maxsplit=1)
    h=m[0] if m else h
    return h.strip()[:160]
d["hl_only"]=d.text.apply(bare_hl)
ALL=NEW_METRICS+EXISTING_TOP4
pos=d[d.judgement==1]; neg=d[d.judgement==0]
samp=pd.concat([pos.sample(2200,random_state=0),neg.sample(2200,random_state=0)]).sample(frac=1,random_state=0).reset_index(drop=True)
samp["snapshot_id"]=samp.snapshot_id.astype(str)
samp["text"]="ARTICLE HEADLINE: "+samp.hl_only   # headline-only input
print(f"[hl-only] {len(samp)} items; bare-headline len mean {samp.hl_only.str.len().mean():.0f}",flush=True)
# ---- rubric A-layer (14 metrics) on headline-only ----
cfg=InfillConfig(id_column="snapshot_id",text_column="text",label_column="judgement",
    materialize_backend="openai_compatible",materialize_model="pr-judge",
    openai_base_url="http://127.0.0.1:8005/v1",llm_concurrency=32,max_text_tokens=200,
    output_dir="outputs/news_hlonly",cache_dir="outputs/news_hlonly/judge_cache")
os.makedirs(cfg.cache_dir,exist_ok=True)
mspec=[MetricSpec(metric_id=f"hl_{n}",name=n,description=g,kind="judge",guidance=g) for n,g in ALL]
judge=make_vllm_judge_scorer(cfg)
print("[hl-only] materializing 14 metrics x",len(samp),"via 70B...",flush=True)
sm=materialize(mspec,samp,cfg,judge)
y=samp.judgement.values.astype(int); g=samp.snapshot_id.values
X=np.where(sm.applicable,sm.levels,np.nan)
pipe=make_pipeline(SimpleImputer(strategy="constant",fill_value=0.5),StandardScaler(),LogisticRegression(max_iter=3000,class_weight="balanced"))
ga=cross_val_score(pipe,X,y,cv=StratifiedGroupKFold(5,shuffle=True,random_state=0),groups=g,scoring="roc_auc").mean()
print(f"[hl-only] RUBRIC A (14) headline-only grouped={ga:.4f}  (full-text was 0.569)",flush=True)
print("[hl-only] top per-metric:",flush=True)
res=[]
for j in range(X.shape[1]):
    v=X[:,j]; mk=~np.isnan(v)
    if mk.sum()>200 and np.std(v[mk])>0: res.append((roc_auc_score(y[mk],v[mk]),ALL[j][0]))
for a,nm in sorted(res,key=lambda r:-abs(r[0]-0.5))[:6]: print(f"   {max(a,1-a):.3f}  {nm}",flush=True)
# ---- pairwise judge on headline-only ----
jcfg=ImplementerConfig(); jcfg.backend="local"; jcfg.llm_concurrency=32; jcfg.request_timeout_s=60
pjudge=LLMBackend("pr-judge","judge",jcfg)
rng=np.random.default_rng(0); pairs=[]
for snap,sd in d.groupby("snapshot_id"):
    sp=sd[sd.judgement==1]; sn=sd[sd.judgement==0]
    if len(sp)<1 or len(sn)<1: continue
    for _,p in sp.iterrows():
        n=sn.sample(1,random_state=int(str(snap),16)%9999).iloc[0]
        if rng.random()<0.5: a,b,lbl=p.hl_only,n.hl_only,"A"
        else: a,b,lbl=n.hl_only,p.hl_only,"B"
        pairs.append({"a":a,"b":b,"top":lbl})
    if len(pairs)>=3000: break
pairs=pairs[:3000]
PROMPT=("Two news headlines A and B appeared on the SAME news homepage. Editors placed ONE more "
 "prominently (higher on the page). Based on editorial newsworthiness (importance, timeliness, "
 "prominence of named actors, magnitude, conflict/casualty, breaking status, institutional stakes), "
 "which was placed higher?\nA: {a}\nB: {b}\nOutput ONLY 'A' or 'B'.")
resps=pjudge.generate_batch([PROMPT.format(a=p["a"],b=p["b"]) for p in pairs],max_tokens=4,temperature=0.0)
correct=n=0
for p,r in zip(pairs,resps):
    m=re.search(r"\b([AB])\b",(r or "").strip())
    if not m: continue
    n+=1
    if m.group(1)==p["top"]: correct+=1
print(f"[hl-only] PAIRWISE judge headline-only acc={correct/max(n,1):.4f} (full-text pairwise was 0.569; CoT 0.579)",flush=True)
print("HLONLY_DONE",flush=True)
