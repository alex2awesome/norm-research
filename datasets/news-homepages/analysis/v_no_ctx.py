import numpy as np,pandas as pd,csv,sys,re,warnings
warnings.filterwarnings("ignore"); csv.field_size_limit(sys.maxsize)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
d=pd.read_csv("/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages/homepage_newsworthiness_clean_v9.csv.gz",compression="gzip"); d["text"]=d.text.fillna("")
def split(t):
    x=t.split("\n\nCONTEXT:",1); return x[0].replace("HEADLINE:","",1).strip(),(x[1].strip() if len(x)>1 else "")
hc=d.text.apply(lambda t:pd.Series(split(t)))
NV=r"\b(breaking|urgent|exclusive|live|killed|dead|dies|attack|war|crisis|storm|shooting|blast|crash|fire|flood|riot|protest|bomb|collapse|murder|wounded|hostage|sanction|tariff)\b"
ELITE=r"\b(trump|biden|putin|netanyahu|zelensky|musk|macron|modi|pelosi|schumer|supreme court|senate|house|white house|pentagon|federal reserve|congress)\b"
Vh=pd.DataFrame(index=d.index)  # HEADLINE-ONLY features (no ctx_len)
Vh["hl_len"]=hc[0].str.split().str.len().fillna(0).astype(float)
Vh["hl_numbers"]=hc[0].apply(lambda s:float(len(re.findall(r"\b\d[\d,]*\b",s))))
Vh["hl_elite"]=hc[0].str.lower().str.count(ELITE)
Vh["hl_neg_mag"]=hc[0].str.lower().str.count(NV)
Vh["hl_proper"]=hc[0].apply(lambda s:float(len(re.findall(r"\b[A-Z][a-z]+\b",s))))
Vh["hl_allcaps"]=hc[0].apply(lambda s:float(len(re.findall(r"\b[A-Z]{2,}\b",s))))
ctx_len=hc[1].str.split().str.len().fillna(0).astype(float)
y=d.judgement.values.astype(int); g=d.snapshot_id.values
def gauc(X):
    return cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced")),X,y,cv=StratifiedGroupKFold(5,shuffle=True,random_state=0),groups=g,scoring="roc_auc").mean()
Xh=Vh.fillna(0).values.astype(np.float32)
print("GROUPED AUC (full v9):",flush=True)
print(f"  V headline-only (6 feats, NO ctx_len): {gauc(Xh):.4f}",flush=True)
print(f"  V headline + ctx_len (7 feats)       : {gauc(np.column_stack([Xh,ctx_len.values])):.4f}",flush=True)
print(f"  ctx_len ALONE                        : {gauc(ctx_len.values.reshape(-1,1)):.4f}",flush=True)
print(f"  [ref: A(14 rubrics, 70B) = 0.568; dense = 0.630]",flush=True)
# within-snap
posidx={};negidx={}
for i,(sn,lab) in enumerate(zip(g,y)): (posidx if lab==1 else negidx).setdefault(sn,[]).append(i)
fa=[];fb=[]
for sn in posidx:
    if sn not in negidx: continue
    for a in posidx[sn][:3]:
        for b in negidx[sn][:3]: fa.append(a);fb.append(b)
fa=np.array(fa);fb=np.array(fb); y2=np.concatenate([np.ones(len(fa)),np.zeros(len(fa))])
def wauc(X):
    D=X[fa]-X[fb]; D2=np.concatenate([D,-D])
    return cross_val_score(make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=1000,class_weight="balanced")),D2,y2,cv=5,scoring="roc_auc").mean()
print("WITHIN-SNAP pair-diff AUC:",flush=True)
print(f"  V headline-only (6 feats): {wauc(Xh):.4f}",flush=True)
print(f"  V headline + ctx_len      : {wauc(np.column_stack([Xh,ctx_len.values])):.4f}",flush=True)
print(f"  [ref: A within-snap = 0.600; dense within-snap = 0.692]",flush=True)
