import numpy as np,pandas as pd,csv,sys,re,warnings
warnings.filterwarnings("ignore"); csv.field_size_limit(sys.maxsize)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
d=pd.read_csv("/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages/homepage_newsworthiness_clean_v9.csv.gz",compression="gzip"); d["text"]=d.text.fillna("")
def split(t):
    x=t.split("\n\nCONTEXT:",1); return x[0].replace("HEADLINE:","",1).strip(),(x[1].strip() if len(x)>1 else "")
hc=d.text.apply(lambda t:pd.Series(split(t)))
ctx=hc[1]
n_seg=ctx.apply(lambda s: float(len([x for x in s.split(";") if x.strip()])))
ctx_words=ctx.str.split().str.len().fillna(0).astype(float)
avg_seg=(ctx_words/ n_seg.clip(lower=1)).fillna(0).astype(float)
y=d.judgement.values.astype(int); g=d.snapshot_id.values
def gauc(v):
    return cross_val_score(make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=1000,class_weight="balanced")),np.array(v).reshape(-1,1).astype(float),y,cv=StratifiedGroupKFold(5,shuffle=True,random_state=0),groups=g,scoring="roc_auc").mean()
print("GLOBAL grouped AUC (is it real signal or within-snap only?):",flush=True)
print(f"  ctx_len (total)     {gauc(ctx_words):.4f}",flush=True)
print(f"  ctx_n_segments      {gauc(n_seg):.4f}",flush=True)
print(f"  ctx_avg_seg_len     {gauc(avg_seg):.4f}",flush=True)
# within-snap pair-diff for each
posidx={};negidx={}
for i,(sn,lab) in enumerate(zip(g,y)): (posidx if lab==1 else negidx).setdefault(sn,[]).append(i)
fa=[];fb=[]
for sn in posidx:
    if sn not in negidx: continue
    for a in posidx[sn][:3]:
        for b in negidx[sn][:3]: fa.append(a);fb.append(b)
fa=np.array(fa);fb=np.array(fb); y2=np.concatenate([np.ones(len(fa)),np.zeros(len(fa))])
def wauc(v):
    D=v[fa]-v[fb]; D2=np.concatenate([D,-D])
    if np.std(D2)==0: return float("nan")
    return cross_val_score(make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=500,class_weight="balanced")),D2.reshape(-1,1),y2,cv=5,scoring="roc_auc").mean()
print("WITHIN-SNAP pair-diff AUC:",flush=True)
print(f"  ctx_len (total)     {wauc(ctx_words.values):.4f}",flush=True)
print(f"  ctx_n_segments      {wauc(n_seg.values):.4f}",flush=True)
print(f"  ctx_avg_seg_len     {wauc(avg_seg.values):.4f}",flush=True)
print(flush=True)
print("If ctx_n_segments is high within-snap but ~0.5 global => position proxy (top articles have more siblings below in context = LEAKAGE).",flush=True)
print("If ctx_avg_seg_len carries it => real sibling-content richness signal.",flush=True)
