import pandas as pd,numpy as np,csv,sys,re,warnings
warnings.filterwarnings("ignore"); csv.field_size_limit(sys.maxsize)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold,cross_val_score
d=pd.read_csv("datasets/news-homepages/homepage_newsworthiness_clean_v4.csv.gz",compression="gzip"); d["text"]=d.text.fillna("")
def split(t):
    p=t.split("\n\nCONTEXT:",1); return p[0].replace("HEADLINE:","",1).strip(),(p[1].strip() if len(p)>1 else "")
hc=d.text.apply(lambda t:pd.Series(split(t))); hc.columns=["hl","ctx"]
NV=r"\b(breaking|urgent|exclusive|live|video|photos?|killed|dead|dies?|attack|war|crisis|storm|earthquake|shooting|blast|crash|fire|flood|riot|protest|strike|bomb|collapse|murder|slain|wounded|injured|hostage|sanction|tariff)\b"
ELITE=r"\b(trump|biden|putin|netanyahu|zelensky|musk|macron|modi|pelosi|schumer|supreme court|senate|house|white house|pentagon|federal reserve|congress)\b"
LIFE=r"\b(recipe|travel|style|fashion|food|wine|garden|horoscope|celebrity|gossip|movie|tv|music|sport|nfl|nba|weather|lottery)\b"
V=pd.DataFrame(index=d.index)
V["hl_len"]=hc.hl.str.split().str.len().fillna(0).astype(float)
V["hl_numbers"]=hc.hl.apply(lambda s:float(len(re.findall(r"\b\d[\d,]*\b",s))))
V["hl_elite"]=hc.hl.str.lower().str.count(ELITE); V["hl_neg_mag"]=hc.hl.str.lower().str.count(NV); V["hl_lifestyle"]=hc.hl.str.lower().str.count(LIFE)
V["hl_proper"]=hc.hl.apply(lambda s:float(len(re.findall(r"\b[A-Z][a-z]+\b",s)))); V["hl_allcaps"]=hc.hl.apply(lambda s:float(len(re.findall(r"\b[A-Z]{2,}\b",s))))
V["ctx_len"]=hc.ctx.str.split().str.len().fillna(0).astype(float)
X=V.fillna(0).values.astype(np.float32); y=d.judgement.values; g=d.snapshot_id.values
ga=cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced")),X,y,cv=StratifiedGroupKFold(5,shuffle=True,random_state=0),groups=g,scoring="roc_auc").mean()
d2=d.copy()
for c in V: d2[c]=V[c].values
fa={c:( [],[]) for c in V.columns}
for snap,sd in d2.groupby("snapshot_id"):
    pos=sd[sd.judgement==1]; neg=sd[sd.judgement==0]
    if len(pos)<1 or len(neg)<1: continue
    for _,p in pos.sample(min(3,len(pos)),random_state=0).iterrows():
        for _,n in neg.sample(min(3,len(neg)),random_state=0).iterrows():
            for c in fa: fa[c][0].append(p[c]); fa[c][1].append(n[c])
D=np.column_stack([np.array(fa[c][0])-np.array(fa[c][1]) for c in V.columns])
D2=np.vstack([D,-D]); y2=np.concatenate([np.ones(len(D)),np.zeros(len(D))])
wj=cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced")),D2,y2,cv=5,scoring="roc_auc").mean()
print(f"[v4-V] grouped={ga:.4f} within-snap-joint={wj:.4f}  (v3 0.5630/0.6604; v2 0.5541/0.6593; orig 0.5535/0.6533)",flush=True)
