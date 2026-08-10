import numpy as np,pandas as pd,csv,sys,re,warnings
warnings.filterwarnings("ignore"); csv.field_size_limit(sys.maxsize)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
d=pd.read_csv("/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages/homepage_newsworthiness_clean_v9.csv.gz",compression="gzip"); d["text"]=d.text.fillna("")
def split(t):
    x=t.split("\n\nCONTEXT:",1); return x[0].replace("HEADLINE:","",1).strip(),(x[1].strip() if len(x)>1 else "")
hc=d.text.apply(lambda t:pd.Series(split(t)))
FEATS={
 "len_words": hc[0].str.split().str.len().fillna(0).astype(float),
 "n_numbers": hc[0].apply(lambda s:float(len(re.findall(r"\b\d[\d,]*\b",s)))),
 "n_proper_names": hc[0].apply(lambda s:float(len(re.findall(r"\b[A-Z][a-z]{2,}\b",s)))),
 "n_elite_political": hc[0].str.lower().str.count(r"\b(trump|biden|putin|netanyahu|zelensky|musk|senate|house|white house|pentagon|congress|supreme court|pelosi|schumer|mcconnell|harris)\b"),
 "neg_magnitude": hc[0].str.lower().str.count(r"\b(killed|dead|dies|attack|war|crisis|storm|shooting|blast|crash|fire|flood|riot|protest|bomb|collapse|murder|wounded|hostage|sanction)\b"),
 "n_dollars": hc[0].apply(lambda s:float(len(re.findall(r"\$\s?\d",s)))),
 "hardnews_kw": hc[0].str.lower().str.count(r"\b(senate|house|court|bill|law|policy|election|vote|government|official|minister|president|congress)\b"),
 "question": hc[0].str.contains(r"\?",regex=True).astype(float),
 "vivid_verbs": hc[0].str.lower().str.count(r"\b(slam|torch|explode|collapse|seize|blast|erupt|sweep|deadlock|bury|crush|strike|blow)\b"),
}
y=d.judgement.values.astype(int); g=d.snapshot_id.values
posidx={};negidx={}
for i,(sn,lab) in enumerate(zip(g,y)): (posidx if lab==1 else negidx).setdefault(sn,[]).append(i)
fa=[];fb=[]
for sn in posidx:
    if sn not in negidx: continue
    for a in posidx[sn][:3]:
        for b in negidx[sn][:3]: fa.append(a);fb.append(b)
fa=np.array(fa);fb=np.array(fb); y2=np.concatenate([np.ones(len(fa)),np.zeros(len(fa))])
print(f"{len(fa)} within-snap pairs",flush=True)
for name,v in FEATS.items():
    D=v.values[fa]-v.values[fb]; D2=np.concatenate([D,-D])
    if np.std(D2)>0:
        a=cross_val_score(make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=500,class_weight="balanced")),D2.reshape(-1,1),y2,cv=5,scoring="roc_auc").mean()
        print(f"  {name:20} {a:.4f}",flush=True)
J=np.column_stack([FEATS[n].values[fa]-FEATS[n].values[fb] for n in FEATS]); J2=np.concatenate([J,-J])
ja=cross_val_score(make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=1000,class_weight="balanced")),J2,y2,cv=5,scoring="roc_auc").mean()
print(f"  {'JOINT':20} {ja:.4f}",flush=True)
