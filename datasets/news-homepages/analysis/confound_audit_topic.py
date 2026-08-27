import numpy as np,pandas as pd,csv,sys,re,warnings
warnings.filterwarnings("ignore"); csv.field_size_limit(sys.maxsize)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
d=pd.read_csv("/lfs/skampere3/0/alexspan/norm-research/datasets/news-homepages/homepage_newsworthiness_clean_v9.csv.gz",compression="gzip"); d["text"]=d.text.fillna("")
def hl(t):
    return t.split("\n\nCONTEXT:",1)[0].replace("HEADLINE:","",1).strip().lower()
H=d.text.apply(hl)
# ---- keyword topic-proxy ----
TOPICS={
 "politics": r"\b(trump|biden|senate|house|congress|president|election|vote|republican|democrat|governor|mayor|policy|impeach|cabinet|primary)\b",
 "world_conflict": r"\b(war|ukraine|russia|gaza|israel|iran|china|military|strike|nato|united nations|ceasefire|putin|netanyahu|taliban)\b",
 "business": r"\b(market|stock|economy|fed|inflation|jobs|tariff|oil|prices|deal|earnings|revenue|layoff|google|apple|tesla|crypto)\b",
 "crime_justice": r"\b(police|arrest|charged|murder|trial|prison|guilty|sentenced|lawsuit|probe|indict|felony|fbi|defendant)\b",
 "disaster_weather": r"\b(storm|hurricane|fire|earthquake|flood|tornado|wildfire|crash|blast|evacuat|heat|snow|cyclone)\b",
 "science_health": r"\b(study|research|cancer|health|doctor|vaccine|virus|climate|space|scientists|fda|cdc|patient)\b",
 "entertainment": r"\b(movie|star|celebrity|singer|actor|film|album|awards|kanye|taylor|netflix|show|fashion|music)\b",
 "sports": r"\b(nfl|nba|game|team|player|coach|season|cup|league|match|championship|playoff|tournament|goal)\b",
 "lifestyle": r"\b(recipe|food|travel|restaurant|hotel|garden|wedding|wine|dating|wellness|fitness|beauty)\b",
}
def topic(s):
    best,bn="other",0
    for t,p in TOPICS.items():
        n=len(re.findall(p,s))
        if n>bn: bn,best=n,t
    return best if bn>0 else "other"
T=H.apply(topic)
d["topic"]=T.values
print("topic distribution:",flush=True)
print(T.value_counts().to_string(),flush=True)
# ---- V features (headline-only) ----
ELITE=r"\b(trump|biden|putin|netanyahu|zelensky|musk|senate|house|white house|pentagon|congress|supreme court)\b"
NV=r"\b(killed|dead|dies|attack|war|crisis|storm|shooting|blast|crash|fire|flood|riot|protest|bomb|collapse|murder|wounded|hostage|sanction)\b"
H0=d.text.apply(lambda t:t.split("\n\nCONTEXT:",1)[0].replace("HEADLINE:","",1).strip())
FE={
 "hl_len": H0.str.split().str.len().fillna(0).astype(float),
 "hl_elite": H0.str.lower().str.count(ELITE),
 "hl_neg_mag": H0.str.lower().str.count(NV),
 "hl_proper": H0.apply(lambda s:float(len(re.findall(r"\b[A-Z][a-z]+\b",s)))),
 "hl_allcaps": H0.apply(lambda s:float(len(re.findall(r"\b[A-Z]{2,}\b",s)))),
}
y=d.judgement.values.astype(int); g=d.snapshot_id.values; T=T.values
posidx={};negidx={}
for i,(sn,lab) in enumerate(zip(g,y)): (posidx if lab==1 else negidx).setdefault(sn,[]).append(i)
def pairs(same_topic):
    fa=[];fb=[]
    for sn in posidx:
        if sn not in negidx: continue
        for a in posidx[sn][:3]:
            for b in negidx[sn][:3]:
                if same_topic and T[a]!=T[b]: continue
                fa.append(a);fb.append(b)
    return np.array(fa),np.array(fb)
def wauc(fa,fb):
    y2=np.concatenate([np.ones(len(fa)),np.zeros(len(fa))])
    rows=[]
    print(f"\n{'feature':12} {'ALL-pairs':>10} {'SAME-TOPIC-pairs':>16}",flush=True)
    for n,v in FE.items():
        da=v.values[fa]-v.values[fb]
        if np.std(np.concatenate([da,-da]))==0: continue
        a_all=cross_val_score(make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=500,class_weight="balanced")),np.concatenate([da,-da]).reshape(-1,1),y2,cv=5,scoring="roc_auc").mean()
        rows.append((n,v,a_all))
    # joint
    J=np.column_stack([v.values[fa]-v.values[fb] for _,v,_ in rows]);
    jall=cross_val_score(make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=1000,class_weight="balanced")),np.concatenate([J,-J]),y2,cv=5,scoring="roc_auc").mean()
    for n,_,a in rows:
        # recompute same-topic per feature outside; print all for now
        print(f"  {n:12} {a:.4f}",flush=True)
    return rows
fa1,fb1=pairs(False); fa2,fb2=pairs(True)
print(f"\nALL within-snap pairs: {len(fa1)}; SAME-TOPIC within-snap pairs: {len(fa2)}",flush=True)
y2=np.concatenate([np.ones(len(fa1)),np.zeros(len(fa1))])
print(f"\n{'feature':14} {'ALL':>8} {'SAME-TOPIC':>12}  (drop = topic-confound)",flush=True)
for n,v in FE.items():
    da=v.values[fa1]-v.values[fb1]; db=v.values[fa2]-v.values[fb2]
    if np.std(np.concatenate([da,-da]))>0:
        a_all=cross_val_score(make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=500,class_weight="balanced")),np.concatenate([da,-da]).reshape(-1,1),y2,cv=5,scoring="roc_auc").mean()
    else: a_all=float("nan")
    y2b=np.concatenate([np.ones(len(fa2)),np.zeros(len(fa2))])
    if np.std(np.concatenate([db,-db]))>0:
        a_same=cross_val_score(make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=500,class_weight="balanced")),np.concatenate([db,-db]).reshape(-1,1),y2b,cv=5,scoring="roc_auc").mean()
    else: a_same=float("nan")
    print(f"  {n:14} {a_all:.4f}   {a_same:.4f}",flush=True)
# joint all vs same-topic
J1=np.column_stack([v.values[fa1]-v.values[fb1] for v in FE.values()]); J2=np.column_stack([v.values[fa2]-v.values[fb2] for v in FE.values()])
ja=cross_val_score(make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=1000,class_weight="balanced")),np.concatenate([J1,-J1]),np.concatenate([np.ones(len(fa1)),np.zeros(len(fa1))]),cv=5,scoring="roc_auc").mean()
js=cross_val_score(make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=1000,class_weight="balanced")),np.concatenate([J2,-J2]),np.concatenate([np.ones(len(fa2)),np.zeros(len(fa2))]),cv=5,scoring="roc_auc").mean()
print(f"  {'JOINT':14} {ja:.4f}   {js:.4f}",flush=True)
print(f"\nIf SAME-TOPIC >> 0.5: real prominence signal. If SAME-TOPIC ~ 0.5: topic-confound.",flush=True)
