#!/usr/bin/env python3
"""Stage 1: regex-clean canonical news-homepages CSV -> _v2, then re-measure V-layer (grouped)
+ within-snapshot joint pair-diff AUC on original vs _v2. Reports drop stats by reason."""
import pandas as pd, numpy as np, csv, sys, re, warnings
warnings.filterwarnings("ignore")
csv.field_size_limit(sys.maxsize)
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

SRC="datasets/news-homepages/homepage_newsworthiness_topic_balanced_groupsplit.csv.gz"
OUT="datasets/news-homepages/homepage_newsworthiness_clean_v2.csv.gz"

JS = re.compile(r"function\s+\w+|const\s+\w+|var\s+\w+|fallbackImage|imageLoadError|/media/sites/|=>|\{[^}]{0,40}\}|window\.|document\.",re.I)
HTML = re.compile(r"<[^>]+>")
READTIME = re.compile(r"\b\d+\s*(min|hr|hour|minute)s?\s*read\b",re.I)
VIDDUR = re.compile(r"\b\d{1,2}:\d{2}\b")
UIBLOB = re.compile(r"[•·]\s*(Video|Gallery|Live|Photos?|Subscribers?|Watch|Show all|Analysis|Opinion|Breaking)?\s*\d*:?\d*",re.I)
SECTLEAD = re.compile(r"^\s*(Analysis|Opinion|Live\s*:?\s*updates?|Video|Photos?|Gallery|For Subscribers|Sign up[^;]{0,40}|Show all|Breaking News|The Great Read|Explainer|Watch|Read|Summary|Report)\s*[\-:–|–]\s*",re.I)
OUTLETTAG = re.compile(r"\s*[|\-–]\s*(CNN|BBC|New York Times|Washington Post|Wall Street Journal|WSJ|Guardian|Reuters|AP|NPR|Latimes|L\.A\. Times)\s*$",re.I)
IMGCREDIT = re.compile(r"(/?(Getty|AFP)\s*Images?|Animation by|/iStock|iStockphoto|Photo(?:graph)? by|via Reuters|/AP$|^[A-Z][a-z]+\s+[A-Z][a-z]+/(Getty|AFP))",re.I)
PROMO = re.compile(r"^\s*(Sign up|For Subscribers|Subscribe|Show all|See more|Read more|Watch|Listen|Follow|Newsletter|The Recap|The Morning|The Evening)\b",re.I)
URLISH = re.compile(r"^(https?://|www\.|/\w+/[\w-]+/?)")
PT = re.compile(r"\b(não|nas|nos|das|dos|para|que|uma|um|com|mais|está|estão|ser|por|como|já|sempre|segunda|terça|quarta|quinta|sexta|feira|presidente|segundo|após|contra|segundo|disse|segundo|também|ainda|entre|sobre|sem|sua|seu|nação|país|governo|ministro)\b",re.I)
WS = re.compile(r"\s+")

def clean_seg(s):
    if not s: return ""
    s = JS.sub(" ", s); s = HTML.sub(" ", s); s = READTIME.sub(" ", s); s = VIDDUR.sub(" ", s)
    s = UIBLOB.sub(" ", s); s = OUTLETTAG.sub("", s)
    for _ in range(3): s = SECTLEAD.sub("", s)
    s = WS.sub(" ", s).strip()
    return s

def drop_reason(hl_clean, hl_raw):
    if not hl_clean or len(hl_clean) < 15: return "short"
    if JS.search(hl_raw) or "imageLoadError" in hl_raw or "fallbackImage" in hl_raw: return "js"
    if IMGCREDIT.search(hl_raw) and len(hl_clean) < 45: return "imgcredit"
    if PROMO.match(hl_clean): return "promo"
    if URLISH.match(hl_raw.strip()): return "url"
    # non-english (portuguese): PT-word density on the raw headline
    words=re.findall(r"\w+",hl_raw.lower())
    if words and sum(1 for w in words if PT.fullmatch(w or PT.match(w) and w))/len(words)>0.18: return "nonenglish"
    return None

# load + split
d=pd.read_csv(SRC,compression="gzip"); d["text"]=d.text.fillna("")
print(f"[load] {len(d)} rows",flush=True)
def split_hl(t):
    p=t.split("\n\nCONTEXT:",1)
    return p[0].replace("HEADLINE:","",1).strip(),(p[1].strip() if len(p)>1 else "")
hc=d.text.apply(lambda t:pd.Series(split_hl(t))); hc.columns=["hl","ctx"]
keep=[]; reasons=Counter(); new_text=[]
for i in range(len(d)):
    hl=hc.hl.iloc[i]; ctx=hc.ctx.iloc[i]
    hlc=clean_seg(hl)
    r=drop_reason(hlc,hl)
    if r: reasons[r]+=1; continue
    # clean context segments
    segs=[clean_seg(s) for s in ctx.split(";")]
    segs=[s for s in segs if s and len(s)>=8 and not JS.search(s) and not (IMGCREDIT.search(s) and len(s)<45)]
    newctx="; ".join(segs)
    new_text.append("HEADLINE: %s\n\nCONTEXT: %s"%(hlc,newctx)); keep.append(i)
print(f"[clean] dropped {len(d)-len(keep)} rows; reasons: {dict(reasons)}",flush=True)
v2=d.iloc[keep].copy(); v2["text"]=new_text
print(f"[clean] _v2 rows={len(v2)} pos={int(v2.judgement.sum())} snapshots={v2.snapshot_id.nunique()}",flush=True)
v2[["text","judgement","snapshot_id"]].to_csv(OUT,compression="gzip",index=False)
print(f"[clean] wrote {OUT}",flush=True)

# ---- V-layer + within-snap joint on ORIGINAL vs _v2 ----
NV=r"\b(breaking|urgent|exclusive|live|video|photos?|killed|dead|dies?|attack|war|crisis|storm|earthquake|shooting|blast|crash|fire|flood|riot|protest|strike|siege|assault|bomb|collapse|murder|slain|wounded|injured|hostage|sanction|tariff)\b"
ELITE=r"\b(trump|biden|putin|netanyahu|zelensky|musk|macron|modi|pelosi|schumer|supreme court|senate|house|white house|pentagon|federal reserve|congress)\b"
LIFE=r"\b(recipe|travel|style|fashion|food|wine|garden|horoscope|celebrity|gossip|movie|tv|music|game|sport|nfl|nba|weather|lottery)\b"
def feats(df):
    hc=df.text.apply(lambda t:pd.Series(split_hl(t))); hc.columns=["hl","ctx"]
    V=pd.DataFrame(index=df.index)
    V["hl_len"]=hc.hl.str.split().str.len().fillna(0).astype(float)
    V["hl_numbers"]=hc.hl.apply(lambda s:float(len(re.findall(r"\b\d[\d,]*\b",s))))
    V["hl_elite"]=hc.hl.str.lower().str.count(ELITE)
    V["hl_neg_mag"]=hc.hl.str.lower().str.count(NV)
    V["hl_lifestyle"]=hc.hl.str.lower().str.count(LIFE)
    V["hl_proper"]=hc.hl.apply(lambda s:float(len(re.findall(r"\b[A-Z][a-z]+\b",s))))
    V["hl_allcaps"]=hc.hl.apply(lambda s:float(len(re.findall(r"\b[A-Z]{2,}\b",s))))
    V["ctx_len"]=hc.ctx.str.split().str.len().fillna(0).astype(float)
    return V.fillna(0).values.astype(np.float32),hc
def grouped_auc(X,y,g):
    return cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced")),
        X,y,cv=StratifiedGroupKFold(5,shuffle=True,random_state=0),groups=g,scoring="roc_auc").mean()
def within_snap_joint(df,V):
    d2=df.copy();
    for j,c in enumerate(["hl_len","hl_numbers","hl_elite","hl_neg_mag","hl_lifestyle","hl_proper","hl_allcaps","ctx_len"]): d2[c]=V[:,j]
    fa={c:( [],[]) for c in ["hl_len","hl_numbers","hl_elite","hl_neg_mag","hl_lifestyle","hl_proper","hl_allcaps","ctx_len"]}
    for snap,sd in d2.groupby("snapshot_id"):
        pos=sd[sd.judgement==1]; neg=sd[sd.judgement==0]
        if len(pos)<1 or len(neg)<1: continue
        ps=pos.sample(min(3,len(pos)),random_state=0); ns=neg.sample(min(3,len(neg)),random_state=0)
        for _,p in ps.iterrows():
            for _,n in ns.iterrows():
                for c in fa: fa[c][0].append(p[c]); fa[c][1].append(n[c])
    cols=list(fa.keys())
    D=np.column_stack([np.array(fa[c][0])-np.array(fa[c][1]) for c in cols])
    D2=np.vstack([D,-D]); y2=np.concatenate([np.ones(len(D)),np.zeros(len(D))])
    return cross_val_score(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced")),
        D2,y2,cv=5,scoring="roc_auc").mean(),len(D)

for name,df in [("ORIGINAL",d),("CLEAN_v2",v2)]:
    V,hc=feats(df); y=df.judgement.values; g=df.snapshot_id.values
    ga=grouped_auc(V,y,g); wj,npair=within_snap_joint(df,V)
    print(f"[meas] {name:9s} V-grouped={ga:.4f}  within-snap-joint={wj:.4f} (n={npair} pairs)",flush=True)
print("STAGE1_DONE",flush=True)
