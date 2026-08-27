#!/usr/bin/env python3
"""Confounder audit on clean_v2 (vs original). Test whether anything beyond articulable news-value
signal predicts the label: outlet identity, byline/author, date/recency, snapshot base-rate,
position-in-context. All AUCs grouped by snapshot_id."""
import pandas as pd, numpy as np, csv, sys, re, warnings
warnings.filterwarnings("ignore")
csv.field_size_limit(sys.maxsize)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

FILES={"ORIGINAL":"datasets/news-homepages/homepage_newsworthiness_topic_balanced_groupsplit.csv.gz",
       "CLEAN_v2":"datasets/news-homepages/homepage_newsworthiness_clean_v2.csv.gz"}
OUTLETS=["nytimes","wsj","latimes","bbc","washingtonpost","cnn","guardian","reuters"]
# known columnist/opinion-writer byline names (NYT + others) that the a11y enrichment may prepend
COLUMNISTS={"jessica grose","david wallace-wells","nicholas kristof","frank bruni","ross douthat",
 "maureen dowd","thomas friedman","bret stephens","ezra klein","paul krugman","gail collins",
 "charles blow","michelle goldberg","jamelle bouie","lydia polgreen","jen gunter","farhad manjoo",
 "heather long","megan mcardle","max read","ezra klein","spencer ackerman"}
BYLINE_SIG=re.compile(r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+){0,2})([A-Z][a-z]+)")  # CamelCase join "FirstName LastNameHeadline"
DATE=re.compile(r"\b(20\d{2}|january|february|march|april|may|june|july|august|september|october|november|december|\d{1,2}/\d{1,2}/\d{2,4}|\d+\s*(?:hours?|hrs?|days?|minutes?|mins?)\s*ago)\b",re.I)

def load(name):
    d=pd.read_csv(FILES[name],compression="gzip"); d["text"]=d.text.fillna("")
    def split(t):
        p=t.split("\n\nCONTEXT:",1); return p[0].replace("HEADLINE:","",1).strip(),(p[1].strip() if len(p)>1 else "")
    hc=d.text.apply(lambda t:pd.Series(split(t))); hc.columns=["hl","ctx"]
    return d,hc

def auc(X,y,g):
    if X.ndim==1: X=X.reshape(-1,1)
    if np.unique(X).size<2: return float("nan")
    return cross_val_score(make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=1000,class_weight="balanced",C=1.0)),
        X,y,cv=StratifiedGroupKFold(5,shuffle=True,random_state=0),groups=g,scoring="roc_auc").mean()

for name in ["ORIGINAL","CLEAN_v2"]:
    d,hc=load(name); y=d.judgement.values; g=d.snapshot_id.values; low=(hc.hl+" "+hc.ctx).str.lower()
    print(f"\n===== {name} ({len(d)} rows) =====",flush=True)
    # 1. outlet identity (one-hot)
    oh=np.column_stack([low.str.contains(o,regex=False).astype(int).values for o in OUTLETS])
    print(f"  outlet-identity AUC = {auc(oh,y,g):.4f}",flush=True)
    # 2. byline: CamelCase-join signature + known columnist presence
    byline_sig=hc.hl.apply(lambda s:1.0 if BYLINE_SIG.match(s or "") else 0.0).values
    columnist=hc.hl.str.lower().apply(lambda s:1.0 if any(c in (s or "") for c in COLUMNISTS) else 0.0).values
    byline=np.column_stack([byline_sig,columnist])
    print(f"  byline-signature rate={byline_sig.mean():.3f};  byline AUC = {auc(byline,y,g):.4f}",flush=True)
    # 3. date/recency
    hasdate=hc.hl.apply(lambda s:1.0 if DATE.search(s or "") else 0.0).values
    ctxdate=hc.ctx.apply(lambda s:float(len(DATE.findall(s or "")))).values
    print(f"  date-in-headline rate={hasdate.mean():.3f};  date AUC = {auc(np.column_stack([hasdate,ctxdate]),y,g):.4f}",flush=True)
    # 4. snapshot base-rate distribution (should be ~0.5 each after rebalance)
    br=d.groupby("snapshot_id").judgement.mean()
    print(f"  snapshot base-rate: mean={br.mean():.3f} std={br.std():.3f} min={br.min():.2f} max={br.max():.2f} | snapshots all-0/1-both: {((br==0).sum(),(br==1).sum(),((br>0)&(br<1)).sum())}",flush=True)
    # 5. position-in-context (target's char-position in full text — should be ~constant after HEADLINE prefix)
    pos=hc.hl.str.len().values  # offset of CONTEXT start = headline len; if label leaks via position, ctx_len would track
    print(f"  headline-len AUC = {auc(pos.astype(float),y,g):.4f}  ctx-len AUC = {auc(hc.ctx.str.split().str.len().fillna(0).astype(float).values,y,g):.4f}",flush=True)
    # 6. snapshot-group leakage check: rows-per-snapshot (if huge, within-snap dominance)
    rps=d.groupby("snapshot_id").size()
    print(f"  rows/snapshot: mean={rps.mean():.1f} max={rps.max()}",flush=True)
print("\nAUDIT_DONE",flush=True)
