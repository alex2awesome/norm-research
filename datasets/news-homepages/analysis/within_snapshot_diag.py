"""Within-snapshot conditional diagnostic: per feature, does it rank TOP > BOT within the same homepage?
This is the quantitative analog of the manual pair inspection. ~0.5 = no within-snapshot signal."""
import pandas as pd, numpy as np, csv, sys, re, warnings
warnings.filterwarnings("ignore")
csv.field_size_limit(sys.maxsize)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
d=pd.read_csv("datasets/news-homepages/homepage_newsworthiness_topic_balanced_groupsplit.csv.gz",compression="gzip")
d["text"]=d.text.fillna("")
def split_hl(t):
    p=t.split("\n\nCONTEXT:",1); return p[0].replace("HEADLINE:","",1).strip(),(p[1].strip() if len(p)>1 else "")
hc=d.text.apply(lambda t:pd.Series(split_hl(t))); hc.columns=["hl","ctx"]; d=pd.concat([d,hc],axis=1)
def F(s,p): return float(len(re.findall(p,s)))
NV=r"\b(breaking|urgent|exclusive|live|video|photos?|killed|dead|dies?|attack|war|crisis|storm|earthquake|shooting|blast|crash|fire|flood|riot|protest|strike|siege|assault|bomb|collapse|murder|slain|wounded|injured|hostage|sanction|tariff)\b"
ELITE=r"\b(trump|biden|putin|netanyahu|zelensky|musk|macron|modi|pelosi|schumer|supreme court|senate|house|white house|pentagon|federal reserve|congress)\b"
LIFE=r"\b(recipe|travel|style|fashion|food|wine|garden|horoscope|celebrity|gossip|movie|tv|music|game|sport|nfl|nba|weather|lottery)\b"
feat={}
feat["hl_len"]=d.hl.str.split().str.len().fillna(0).astype(float)
feat["hl_numbers"]=d.hl.apply(lambda s:F(s,r"\b\d[\d,]*\b"))
feat["hl_elite"]=d.hl.str.lower().str.count(ELITE)
feat["hl_neg_mag"]=d.hl.str.lower().str.count(NV)
feat["hl_lifestyle"]=d.hl.str.lower().str.count(LIFE)
feat["hl_proper"]=d.hl.apply(lambda s:F(s,r"\b[A-Z][a-z]+\b"))
feat["hl_allcaps"]=d.hl.apply(lambda s:F(s,r"\b[A-Z]{2,}\b"))
feat["hl_question"]=d.hl.str.endswith("?").astype(float)
feat["ctx_len"]=d.ctx.str.split().str.len().fillna(0).astype(float)
V=pd.DataFrame(feat)
d2=d.copy()
for c in V: d2[c]=V[c].values
# build ALL within-snapshot (top,bot) pairs (cap per snapshot to avoid blowup)
print("[diag] building within-snapshot pairs (cap 3/snapshot)...",flush=True)
tops=[]; bots=[]; ft={} 
from collections import defaultdict
for c in V: ft[c]=([],[])
npair=0
for snap,sd in d2.groupby("snapshot_id"):
    pos=sd[sd.judgement==1]; neg=sd[sd.judgement==0]
    if len(pos)<1 or len(neg)<1: continue
    ps=pos.sample(min(3,len(pos)),random_state=0); ns=neg.sample(min(3,len(neg)),random_state=0)
    for _,p in ps.iterrows():
        for _,n in ns.iterrows():
            for c in V: ft[c][0].append(p[c]); ft[c][1].append(n[c])
            npair+=1
print(f"[diag] {npair} within-snapshot pairs",flush=True)
print(f"\n=== within-snapshot pairwise accuracy (P(feature_TOP > feature_BOT)); 0.5=no signal ===")
for c in V:
    a=np.array(ft[c][0]); b=np.array(ft[c][1])
    acc=((a>b).sum()+0.5*(a==b).sum())/len(a)
    print(f"  {c:14s} {acc:.3f}")
# joint: can a feature SET separate within-snapshot? (logistic on pairwise diff)
print("\n=== joint within-snapshot: logistic on (feature_top - feature_bot) -> predict 1 ===")

D=np.column_stack([np.array(ft[c][0])-np.array(ft[c][1]) for c in V])
y=np.ones(len(D))  # all pairs are (top,bot); test if diff predicts positive mean>0
# proper CV AUC: build symmetric dataset (top,bot)=1 and (bot,top)=0
D2=np.vstack([D,-D]); y2=np.concatenate([np.ones(len(D)),np.zeros(len(D))])
from sklearn.model_selection import cross_val_score
auc=cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced")),
    D2,y2,cv=5,scoring="roc_auc").mean()
print(f"  joint within-snapshot pair-diff AUC = {auc:.4f}  (0.5=no within-snapshot articulable signal)")
print("DIAG_DONE",flush=True)
