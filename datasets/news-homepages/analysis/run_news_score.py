#!/usr/bin/env python3
"""Phase B-seed: score the new-metric bank (10 new + 4 existing) via 70B on clean v8.
Bank AUC (grouped snapshot_id) + within-snap joint + per-metric AUC. Compare to V(0.572)/dense(0.631)."""
import os,sys,json,warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"methods")
sys.path.insert(0,"datasets/news-homepages/analysis")
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score
from metrics_tree_infilling.io_metrics import materialize, make_vllm_judge_scorer, MetricSpec
from metrics_tree_infilling.config import InfillConfig
from news_newmetrics import NEW_METRICS, EXISTING_TOP4
DS="/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages"
ALL=NEW_METRICS+EXISTING_TOP4
print(f"[score] {len(ALL)} metrics ({len(NEW_METRICS)} new + {len(EXISTING_TOP4)} existing)",flush=True)
d=pd.read_csv(f"{DS}/homepage_newsworthiness_clean_v8.csv.gz",compression="gzip").reset_index(drop=True)
d["text"]=d.text.fillna("")
pos=d[d.judgement==1]; neg=d[d.judgement==0]
N=2200
samp=pd.concat([pos.sample(min(N,len(pos)),random_state=0),neg.sample(min(N,len(neg)),random_state=0)]).sample(frac=1,random_state=0).reset_index(drop=True)
samp["snapshot_id"]=samp.snapshot_id.astype(str)
print(f"[score] eval set {len(samp)} ({samp.judgement.sum()} pos), {samp.snapshot_id.nunique()} snapshots",flush=True)
cfg=InfillConfig(id_column="snapshot_id",text_column="text",label_column="judgement",
    materialize_backend="openai_compatible",materialize_model="pr-judge",
    openai_base_url="http://127.0.0.1:8005/v1",llm_concurrency=32,max_text_tokens=700,
    output_dir="outputs/news_score",cache_dir="outputs/news_score/judge_cache")
os.makedirs(cfg.cache_dir,exist_ok=True)
mspec=[MetricSpec(metric_id=f"nm_{n}",name=n,description=g,kind="judge",guidance=g) for n,g in ALL]
judge=make_vllm_judge_scorer(cfg)
print("[score] materializing 14 metrics x",len(samp),"items via 70B...",flush=True)
sm=materialize(mspec,samp,cfg,judge)
y=samp.judgement.values.astype(int); g=samp.snapshot_id.values
np.savez(f"{DS}/news_newmetrics_scores.npz",levels=sm.levels,applicable=sm.applicable,y=y,g=g,names=np.array([n for n,_ in ALL],dtype=object))
X=np.where(sm.applicable,sm.levels,np.nan)
print(f"[score] NA rate {np.isnan(sm.levels).mean():.3f}",flush=True)
pipe=make_pipeline(SimpleImputer(strategy="constant",fill_value=0.5),StandardScaler(),LogisticRegression(max_iter=3000,class_weight="balanced"))
ga=cross_val_score(pipe,X,y,cv=StratifiedGroupKFold(5,shuffle=True,random_state=0),groups=g,scoring="roc_auc").mean()
pr=cross_val_predict(pipe,X,y,cv=StratifiedGroupKFold(5,shuffle=True,random_state=0),groups=g,method="predict_proba")[:,1]
# within-snap joint
fa=[];fb=[]
for snap,sd in samp.groupby("snapshot_id"):
    pi=sd.index[sd.judgement==1].values; ni=sd.index[sd.judgement==0].values
    if len(pi)<1 or len(ni)<1: continue
    for a in pi[:3]:
        for b in ni[:3]: fa.append(a);fb.append(b)
if fa:
    A=np.where(np.isnan(X[fa]),0.5,X[fa]);B=np.where(np.isnan(X[fb]),0.5,X[fb]);D=A-B
    D2=np.vstack([D,-D]);y2=np.concatenate([np.ones(len(D)),np.zeros(len(D))])
    wj=cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced")),D2,y2,cv=5,scoring="roc_auc").mean()
else: wj=float("nan")
print(f"[score] BANK (new+existing) grouped={ga:.4f} within-snap={wj:.4f}  (V=0.572 dense=0.631)",flush=True)
print(f"[score] reference: old 14 news-values A-LAYER was ~0.55-0.57",flush=True)
# per-metric + bank for NEW only vs EXISTING only
for label,cols in [("NEW-10",list(range(len(NEW_METRICS)))),("EXISTING-4",list(range(len(NEW_METRICS),len(ALL))))]:
    Xs=X[:,cols]
    a=cross_val_score(make_pipeline(SimpleImputer(strategy="constant",fill_value=0.5),StandardScaler(),LogisticRegression(max_iter=3000,class_weight="balanced")),Xs,y,cv=StratifiedGroupKFold(5,shuffle=True,random_state=0),groups=g,scoring="roc_auc").mean()
    print(f"   {label} grouped={a:.4f}",flush=True)
print("[score] top per-metric univariate AUC:",flush=True)
res=[]
for j in range(X.shape[1]):
    v=X[:,j]; mk=~np.isnan(v)
    if mk.sum()>200 and np.std(v[mk])>0: res.append((roc_auc_score(y[mk],v[mk]),ALL[j][0]))
for a,nm in sorted(res,key=lambda r:-abs(r[0]-0.5))[:10]: print(f"   {max(a,1-a):.3f}  {nm}",flush=True)
print("NEWS_SCORE_DONE",flush=True)
