"""news-homepages Phase-0 audit + V-layer baseline. CPU. Grouped (snapshot_id) + outlet-leak check."""
import pandas as pd, numpy as np, csv, sys, re, warnings
warnings.filterwarnings("ignore")
csv.field_size_limit(sys.maxsize)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score
d=pd.read_csv("datasets/news-homepages/homepage_newsworthiness_topic_balanced_groupsplit.csv.gz",compression="gzip")
d["text"]=d.text.fillna("")
print(f"[load] {len(d)} rows, pos={int(d.judgement.sum())}, snapshots={d.snapshot_id.nunique()}",flush=True)
# split HEADLINE vs CONTEXT
def split(t):
    parts=t.split("\n\nCONTEXT:",1)
    return parts[0].replace("HEADLINE:","",1).strip(), (parts[1].strip() if len(parts)>1 else "")
hl_ctx=d.text.apply(lambda t:pd.Series(split(t)))
hl_ctx.columns=["hl","ctx"]; d=pd.concat([d,hl_ctx],axis=1)
# ---- Phase 0: outlet-leak check (does outlet identity recoverable from text predict label?) ----
OUTLETS=["nytimes","wsj","latimes","bbc","washingtonpost","cnn","guardian","reuters"]
def outlet_of(row):
    low=(row.hl+" "+row.ctx).lower()
    for o in OUTLETS:
        if o in low: return o
    return "other"
d["outlet"]=d.apply(outlet_of,axis=1)
print("[p0] outlet recovery rate:",round((d.outlet!="other").mean(),3))
oc=d.groupby("outlet").judgement.agg(["mean","count"]).sort_values("count",ascending=False)
print("[p0] outlet base-rates (label mean; balanced should be ~0.5 each):")
print(oc.round(3).to_string())
# outlet one-hot AUC (grouped)
oh=pd.get_dummies(d.outlet).astype(float)
pipe=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced"))
auc_outlet=cross_val_score(pipe,oh.values,d.judgement.values,cv=StratifiedGroupKFold(5,shuffle=True,random_state=0),groups=d.snapshot_id.values,scoring="roc_auc").mean()
print(f"[p0] outlet-identity grouped AUC = {auc_outlet:.4f}  (≈0.5 => balanced/neutralized; >0.55 => residual confound)",flush=True)
# ---- V-layer features ----
def F(s,p): return float(len(re.findall(p,s)))
def hlower(s): return s.lower()
V=pd.DataFrame(index=d.index)
V["hl_words"]=d.hl.str.split().str.len().fillna(0).astype(float)
V["hl_chars"]=d.hl.str.len().astype(float)
V["hl_numbers"]=d.hl.apply(lambda s:F(s,r"\b\d[\d,]*\b"))
V["hl_proper"]=d.hl.apply(lambda s:F(s,r"\b[A-Z][a-z]+\b"))            # magnitude/elite entities
V["hl_allcaps"]=d.hl.apply(lambda s:F(s,r"\b[A-Z]{2,}\b"))
V["hl_question"]=(d.hl.str.endswith("?")).astype(float)
NV=r"\b(breaking|urgent|exclusive|live|video|photos?|killed|dead|dies?|attack|attacks?|war|crisis|storm|earthquake|shooting|blast|crash|fire|flood|riot|protest|strike|siege|assault|bomb|collapse|murder|slain|wounded|injured|hostage|sanction|tariff|election|vote|senate|congress|court|rule|orders?|ban|deal|summit|g7|g20|nato|eu|un security)\b"
V["hl_neg_magnitude"]=d.hl.str.lower().str.count(NV)
ELITE=r"\b(trump|biden|putin|netanyahu|xi jinping|zelensky|musk|macron|putin|modi|putin|pelosi|schumer|mccarthy|bennett|gantz|supreme court|senate|house|white house|pentagon|federal reserve)\b"
V["hl_elite"]=d.hl.str.lower().str.count(ELITE)
V["ctx_count"]=d.ctx.str.count(";").fillna(0).add(1).astype(float) if d.ctx.str.len().sum()>0 else 0
V["ctx_words"]=d.ctx.str.split().str.len().fillna(0).astype(float)
V["ctx_numbers"]=d.ctx.apply(lambda s:F(s,r"\b\d[\d,]*\b"))
V["ctx_neg_mag"]=d.ctx.str.lower().str.count(NV)
V["ctx_elite"]=d.ctx.str.lower().str.count(ELITE)
X=V.fillna(0).values.astype(np.float32); y=d.judgement.values; g=d.snapshot_id.values
print(f"\n[v] {X.shape[1]} V-features; grouped LR (snapshot_id)...",flush=True)
auc_v=cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced")),
    X,y,cv=StratifiedGroupKFold(5,shuffle=True,random_state=0),groups=g,scoring="roc_auc").mean()
print(f"[v] V-layer grouped AUC = {auc_v:.4f}  (deconfounded dense ref ~0.753)",flush=True)
# per-feature univariate AUC
print("[v] top univariate V-features:")
res=[]
for j,c in enumerate(V.columns):
    a=roc_auc_score(y,X[:,j]); res.append((max(a,1-a),V.columns[j]))
for a,n in sorted(res,reverse=True)[:10]: print(f"   {a:.3f}  {n}")
print("V_DONE",flush=True)
