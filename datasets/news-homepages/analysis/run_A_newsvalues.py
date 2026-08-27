#!/usr/bin/env python3
"""A-layer (hand-built NEWS-VALUES rubrics) on clean_v2: 70B judge materialize -> bank AUC
grouped (snapshot_id) + within-snap joint. Compares to V(0.554)/dense(pending)."""
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
DS="/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages"
RUBRICS=[
("elite_political_actor","Names a head of state, top government official, SCOTUS justice, legislature/party leader, or major US/foreign political figure (e.g. Trump, Biden, Putin, Netanyahu, a president/PM) as a central subject."),
("institutional_action","Reports a concrete government/institutional action: policy, executive order, court ruling, legislation, regulation, treaty, or election result."),
("crisis_disaster","Concerns an active or imminent crisis: war, armed attack, natural disaster, accident, outbreak, or public-safety emergency."),
("conflict_violence","Involves violence, casualties, armed conflict, crime with victims, or direct confrontation."),
("magnitude_scale","Conveys large scale: big numbers of people affected, national/international scope, or major dollar/economic stakes."),
("breaking_developing","Timely/breaking/developing: a just-happened or ongoing event (live updates, developing story, imminent)."),
("legal_accountability","Legal accountability: indictment, arrest, lawsuit, investigation, verdict, sentencing, or scandal."),
("economic_impact","Economic impact on audiences: markets, inflation, jobs, tariffs, prices, major corporate moves affecting wallets."),
("elite_celebrity_org","Names a globally famous celebrity, CEO, or major non-political institution/org as subject."),
("human_interest_drama","Emotional human-interest drama: tragedy, survival, remarkable personal story."),
("surprise_novelty","Unexpected, bizarre, record-breaking, or genuinely novel."),
("proximity_domestic","US/domestic or directly reader-relevant (vs distant foreign with no domestic angle)."),
("hard_vs_soft","HARD NEWS (politics/war/economy/crime/disaster) rather than SOFT (lifestyle, entertainment, sports, service journalism, evergreen)."),
("ongoing_top_story","Part of a top-tier ongoing national/international story (the day's major storyline)."),
]
print(f"[A] {len(RUBRICS)} news-values rubrics",flush=True)
# balanced snapshot-grouped sample from clean_v2
d=pd.read_csv(f"{DS}/homepage_newsworthiness_clean_v2.csv.gz",compression="gzip").reset_index(drop=True)
d["text"]=d.text.fillna("")
pos=d[d.judgement==1]; neg=d[d.judgement==0]
N=2200
samp=pd.concat([pos.sample(min(N,len(pos)),random_state=0),neg.sample(min(N,len(neg)),random_state=0)]).sample(frac=1,random_state=0).reset_index(drop=True)
print(f"[A] eval set {len(samp)} ({samp.judgement.sum()} pos), {samp.snapshot_id.nunique()} snapshots",flush=True)
cfg=InfillConfig(id_column="snapshot_id",text_column="text",label_column="judgement",
    materialize_backend="openai_compatible",materialize_model="pr-judge",
    openai_base_url="http://127.0.0.1:8005/v1",llm_concurrency=32,max_text_tokens=700,
    output_dir="outputs/news_A",cache_dir="outputs/news_A/judge_cache")
os.makedirs(cfg.cache_dir,exist_ok=True)
samp["snapshot_id"]=samp.snapshot_id.astype(str)
mspec=[MetricSpec(metric_id=f"nv_{n}",name=n,description=g,kind="judge",guidance=g) for n,g in RUBRICS]
judge=make_vllm_judge_scorer(cfg)
print("[A] materializing 14 rubrics x",len(samp),"items via 70B...",flush=True)
sm=materialize(mspec,samp,cfg,judge)
y=samp.judgement.values.astype(int); g=samp.snapshot_id.values
np.savez(f"{DS}/news_A_scores.npz",levels=sm.levels,applicable=sm.applicable,y=y,g=g,names=np.array([n for n,_ in RUBRICS],dtype=object))
X=np.where(sm.applicable,sm.levels,np.nan)
print(f"[A] NA rate {np.isnan(sm.levels).mean():.3f}",flush=True)
pipe=make_pipeline(SimpleImputer(strategy="constant",fill_value=0.5),StandardScaler(),LogisticRegression(max_iter=3000,class_weight="balanced"))
ga=cross_val_score(pipe,X,y,cv=StratifiedGroupKFold(5,shuffle=True,random_state=0),groups=g,scoring="roc_auc").mean()
pr=cross_val_predict(pipe,X,y,cv=StratifiedGroupKFold(5,shuffle=True,random_state=0),groups=g,method="predict_proba")[:,1]
# within-snap joint pair-diff
fa=[];fb=[]
for snap,sd in samp.groupby("snapshot_id"):
    pi=sd.index[sd.judgement==1].values; ni=sd.index[sd.judgement==0].values
    if len(pi)<1 or len(ni)<1: continue
    for a in pi[:3]:
        for b in ni[:3]: fa.append(a);fb.append(b)
if fa:
    A=X[fa];B=X[fb]; Ai=np.where(np.isnan(A),0.5,A); Bi=np.where(np.isnan(B),0.5,B); D=Ai-Bi
    D2=np.vstack([D,-D]); y2=np.concatenate([np.ones(len(D)),np.zeros(len(D))])
    wj=cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced")),D2,y2,cv=5,scoring="roc_auc").mean()
else: wj=float("nan")
print(f"[A] NEWS-VALUES A-LAYER: grouped={ga:.4f}  within-snap-joint={wj:.4f}  (V=0.554)",flush=True)
print("[A] top per-rubric univariate AUC:",flush=True)
res=[]
for j in range(X.shape[1]):
    v=X[:,j]; mk=~np.isnan(v)
    if mk.sum()>200 and np.std(v[mk])>0: res.append((roc_auc_score(y[mk],v[mk]),RUBRICS[j][0]))
for a,nm in sorted(res,key=lambda r:-abs(r[0]-0.5))[:8]: print(f"   {max(a,1-a):.3f}  {nm}",flush=True)
print("A_NEWS_DONE",flush=True)
