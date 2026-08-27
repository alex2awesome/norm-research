#!/usr/bin/env python3
"""Phase B-opt: score the GEPA-OPTIMIZED new-metric prompts (from news_gepa_optimized.jsonl) via 70B
on clean v8. Bank AUC + per-metric + compare to V(0.572)/dense(0.631)/old-A(0.55-0.57)."""
import os,sys,json,warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"methods")
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score
from metrics_tree_infilling.io_metrics import materialize, make_vllm_judge_scorer, MetricSpec
from metrics_tree_infilling.config import InfillConfig
from news_newmetrics import EXISTING_TOP4
DS="/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages"
opt=[json.loads(l) for l in open(f"{DS}/news_gepa_optimized.jsonl")]
# use the BEST prompt per metric (optimized if accepted, else seed)
METRICS=[(r["name"], r["best_prompt"]) for r in opt] + EXISTING_TOP4
print(f"[opt] {len(METRICS)} metrics ({len(opt)} GEPA-opt new + {len(EXISTING_TOP4)} existing)",flush=True)
acc=[r["name"] for r in opt if r.get("accepted")]
print(f"[opt] GEPA-accepted/optimized: {len(acc)}/{len(opt)}: {acc}",flush=True)
d=pd.read_csv(f"{DS}/homepage_newsworthiness_clean_v8.csv.gz",compression="gzip").reset_index(drop=True)
d["text"]=d.text.fillna("")
pos=d[d.judgement==1]; neg=d[d.judgement==0]
samp=pd.concat([pos.sample(2200,random_state=0),neg.sample(2200,random_state=0)]).sample(frac=1,random_state=0).reset_index(drop=True)
samp["snapshot_id"]=samp.snapshot_id.astype(str)
print(f"[opt] eval set {len(samp)} ({samp.judgement.sum()} pos)",flush=True)
cfg=InfillConfig(id_column="snapshot_id",text_column="text",label_column="judgement",
    materialize_backend="openai_compatible",materialize_model="pr-judge",
    openai_base_url="http://127.0.0.1:8005/v1",llm_concurrency=32,max_text_tokens=700,
    output_dir="outputs/news_score_opt",cache_dir="outputs/news_score_opt/judge_cache")
os.makedirs(cfg.cache_dir,exist_ok=True)
mspec=[MetricSpec(metric_id=f"opt_{i}",name=n,description=g,kind="judge",guidance=g) for i,(n,g) in enumerate(METRICS)]
judge=make_vllm_judge_scorer(cfg)
print("[opt] materializing",len(METRICS),"metrics x",len(samp),"items via 70B...",flush=True)
sm=materialize(mspec,samp,cfg,judge)
y=samp.judgement.values.astype(int); g=samp.snapshot_id.values
np.savez(f"{DS}/news_opt_scores.npz",levels=sm.levels,applicable=sm.applicable,y=y,g=g,names=np.array([n for n,_ in METRICS],dtype=object))
X=np.where(sm.applicable,sm.levels,np.nan)
print(f"[opt] NA rate {np.isnan(sm.levels).mean():.3f}",flush=True)
pipe=make_pipeline(SimpleImputer(strategy="constant",fill_value=0.5),StandardScaler(),LogisticRegression(max_iter=3000,class_weight="balanced"))
ga=cross_val_score(pipe,X,y,cv=StratifiedGroupKFold(5,shuffle=True,random_state=0),groups=g,scoring="roc_auc").mean()
fa=[];fb=[]
for snap,sd in samp.groupby("snapshot_id"):
    pi=sd.index[sd.judgement==1].values; ni=sd.index[sd.judgement==0].values
    if len(pi)<1 or len(ni)<1: continue
    for a in pi[:3]:
        for b in ni[:3]: fa.append(a);fb.append(b)
A=np.where(np.isnan(X[fa]),0.5,X[fa]);B=np.where(np.isnan(X[fb]),0.5,X[fb]);D=A-B
D2=np.vstack([D,-D]);y2=np.concatenate([np.ones(len(D)),np.zeros(len(D))])
wj=cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced")),D2,y2,cv=5,scoring="roc_auc").mean()
print(f"[opt] GEPA-OPT BANK grouped={ga:.4f} within-snap={wj:.4f}  (V=0.572 dense=0.631 old-A=0.55-0.57)",flush=True)
for label,cols in [("GEPA-OPT-NEW",list(range(len(opt)))),("EXISTING-4",list(range(len(opt),len(METRICS))))]:
    Xs=X[:,cols]; a=cross_val_score(make_pipeline(SimpleImputer(strategy="constant",fill_value=0.5),StandardScaler(),LogisticRegression(max_iter=3000,class_weight="balanced")),Xs,y,cv=StratifiedGroupKFold(5,shuffle=True,random_state=0),groups=g,scoring="roc_auc").mean()
    print(f"   {label} grouped={a:.4f}",flush=True)
print("[opt] top per-metric:",flush=True)
res=[]
for j in range(X.shape[1]):
    v=X[:,j]; mk=~np.isnan(v)
    if mk.sum()>200 and np.std(v[mk])>0: res.append((roc_auc_score(y[mk],v[mk]),METRICS[j][0]))
for a,nm in sorted(res,key=lambda r:-abs(r[0]-0.5))[:10]: print(f"   {max(a,1-a):.3f}  {nm}",flush=True)
print("NEWS_OPT_DONE",flush=True)
