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
# vlayer EXACT feature set + random-3 sampling (to reconcile 0.585 vs 0.682)
NV=r"\b(breaking|urgent|exclusive|live|video|photos?|killed|dead|dies?|attack|war|crisis|storm|earthquake|shooting|blast|crash|fire|flood|riot|protest|strike|siege|assault|bomb|collapse|murder|slain|wounded|injured|hostage|sanction|tariff)\b"
ELITE=r"\b(trump|biden|putin|netanyahu|zelensky|musk|macron|modi|pelosi|schumer|supreme court|senate|house|white house|pentagon|federal reserve|congress)\b"
LIFE=r"\b(recipe|travel|style|fashion|food|wine|garden|horoscope|celebrity|gossip|movie|tv|music|game|sport|nfl|nba|weather|lottery)\b"
V=pd.DataFrame(index=d.index)
V["hl_len"]=hc[0].str.split().str.len().fillna(0).astype(float)
V["hl_numbers"]=hc[0].apply(lambda s:float(len(re.findall(r"\b\d[\d,]*\b",s))))
V["hl_elite"]=hc[0].str.lower().str.count(ELITE)
V["hl_neg_mag"]=hc[0].str.lower().str.count(NV)
V["hl_lifestyle"]=hc[0].str.lower().str.count(LIFE)
V["hl_proper"]=hc[0].apply(lambda s:float(len(re.findall(r"\b[A-Z][a-z]+\b",s))))
V["hl_allcaps"]=hc[0].apply(lambda s:float(len(re.findall(r"\b[A-Z]{2,}\b",s))))
V["ctx_len"]=hc[1].str.split().str.len().fillna(0).astype(float)
y=d.judgement.values.astype(int)
fa={c:( [],[]) for c in V.columns}
for snap,sd in V.assign(judgement=y, snapshot_id=d.snapshot_id.values).groupby("snapshot_id"):
    pos=sd[sd.judgement==1]; neg=sd[sd.judgement==0]
    if len(pos)<1 or len(neg)<1: continue
    ps=pos.sample(min(3,len(pos)),random_state=0); ns=neg.sample(min(3,len(neg)),random_state=0)
    for _,p in ps.iterrows():
        for _,n in ns.iterrows():
            for c in fa: fa[c][0].append(p[c]); fa[c][1].append(n[c])
print(f"{len(fa['hl_len'][0])} pairs (vlayer-style random-3)",flush=True)
for c in V.columns:
    D=np.array(fa[c][0])-np.array(fa[c][1]); D2=np.concatenate([D,-D]); y2=np.concatenate([np.ones(len(D)),np.zeros(len(D))])
    a=cross_val_score(make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=500,class_weight="balanced")),D2.reshape(-1,1),y2,cv=5,scoring="roc_auc").mean()
    print(f"  {c:14} {a:.4f}",flush=True)
J=np.column_stack([np.array(fa[c][0])-np.array(fa[c][1]) for c in V.columns]); J2=np.concatenate([J,-J]); y2=np.concatenate([np.ones(len(J)),np.zeros(len(J))])
ja=cross_val_score(make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=1000,class_weight="balanced")),J2,y2,cv=5,scoring="roc_auc").mean()
print(f"  {'JOINT-8':14} {ja:.4f}   (vlayer_v9 reported 0.6823)",flush=True)
