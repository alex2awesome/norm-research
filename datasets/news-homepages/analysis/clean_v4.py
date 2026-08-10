#!/usr/bin/env python3
"""Stage 1.6: v4. Targets v3-verify residuals: photo-credit headlines (Name/Outlet patterns),
stronger non-English (PT/ES), UI artifacts (Trending/Loading/Show all), date appendages, context
JS fragments. Keeps the conservative byline-strip."""
import pandas as pd, numpy as np, csv, sys, re, warnings
warnings.filterwarnings("ignore"); csv.field_size_limit(sys.maxsize)
from collections import Counter
SRC="datasets/news-homepages/homepage_newsworthiness_clean_v3.csv.gz"  # build on v3 (already byline-stripped)
# actually rebuild from ORIGINAL to keep one clean pipeline:
SRC="datasets/news-homepages/homepage_newsworthiness_topic_balanced_groupsplit.csv.gz"
OUT="datasets/news-homepages/homepage_newsworthiness_clean_v4.csv.gz"
JS=re.compile(r"function\s+\w+|const\s+\w+|var\s+\w+|fallbackImage|imageLoadError|/media/|=>|\{[^}]{0,60}\}|\(img\)\s*\{|cnn-f|window\.|document\.|getelementby|addeventlistener",re.I)
HTML=re.compile(r"<[^>]+>"); READTIME=re.compile(r"\b\d+\s*(min|hr|hour|minute)s?\s*read\b",re.I); VIDDUR=re.compile(r"\b\d{1,2}:\d{2}\b")
UIBLOB=re.compile(r"[•·]\s*(Video|Gallery|Live|Photos?|Subscribers?|Watch|Show all|Analysis|Opinion|Breaking|Trending|Loading|More)?\s*\d*:?\d*",re.I)
SECTLEAD=re.compile(r"^\s*(Analysis|Opinion|Live\s*:?\s*updates?|Video|Photos?|Gallery|For Subscribers|Sign up[^;]{0,40}|Show all|Breaking News|The Great Read|Explainer|Watch|Read|Summary|Report|First on CNN|Trending|Loading|More from|See more|News Breaking News|Live)\s*[\-:–|–]?\s*",re.I)
OUTLETTAG=re.compile(r"\s*[|\-–]\s*(CNN|BBC|New York Times|Washington Post|Wall Street Journal|WSJ|Guardian|Reuters|AP|NPR|Latimes|L\.A\. Times)\s*$",re.I)
PROMO=re.compile(r"^\s*(Sign up|For Subscribers|Subscribe|Show all|See more|Read more|Watch the latest|Watch|Listen|Follow|Newsletter|The Recap|The Morning|The Evening|More from|Trending|Loading|Expert[- ]backed guides)\b",re.I)
UIANY=re.compile(r"\b(Show all|Trending|Loading|See more|More from|For Subscribers)\b",re.I)
URLISH=re.compile(r"^(https?://|www\.|/\w+/[\w-]+/?)")
DATE=re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b",re.I)
BYLINE=re.compile(r"^([A-Z][a-z]+ [A-Z][a-z]+)(?=[A-Z][a-z]+)")
# photo-credit detection (drop): explicit phrases OR Name/Outlet[/Format] pattern OR known agencies
CREDITPHRASE=re.compile(r"\b(styling by|photo(?:graph)? by|picture by|via (?:Reuters|AP|AFP|Getty)|animation by)\b",re.I)
AGENCYSLASH=re.compile(r"/(Getty|AFP|AP|Reuters|Sipa|CBS|CNN|NBC|ABC|Wirecutter|NYT|POOL|Bloomberg|AP|Fox|Shutterstock|Adobe|iStock|USA|File)\b",re.I)
NAMEOUTLET=re.compile(r"^[A-Z][a-z'-.]+(?: [A-Z][a-z'-.]+){0,2}/[A-Z][\w&' .-]+(?:/[A-Z][\w&' .-]+)?\s*$")  # "First Last / Outlet / File"
BARENAME=re.compile(r"^[A-Z][a-z]+(?:[ -][A-Z][a-z]+){1,4}$")
PT=re.compile(r"\b(n[ãa]o|nas|nos|das|dos|para|que|uma|um|com|mais|est[áa]|est[ãa]o|ser|por|como|j[áa]|sempre|segunda|ter[çc]a|quarta|quinta|sexta|feira|presidente|segundo|ap[óo]s|contra|disse|tamb[ée]m|ainda|entre|sobre|sem|sua|seu|na[çc][ãa]o|pa[íi]s|governo|ministro|depois|antes|mulher|homem|diz|vai|foi|foram|tem|tinha|desde|durante|segundo|segundo|apoia|destravaria|superou|crise|pesquisa)\b",re.I)
WS=re.compile(r"\s+")
def clean_seg(s):
    if not s: return ""
    s=JS.sub(" ",s); s=HTML.sub(" ",s); s=READTIME.sub(" ",s); s=VIDDUR.sub(" ",s); s=UIBLOB.sub(" ",s); s=OUTLETTAG.sub("",s); s=UIANY.sub("",s); s=DATE.sub("",s)
    for _ in range(3): s=SECTLEAD.sub("",s)
    s=WS.sub(" ",s).strip(); return s
def is_credit(hl):
    if CREDITPHRASE.search(hl): return True
    if AGENCYSLASH.search(hl) and len(hl)<60: return True
    if NAMEOUTLET.match(hl): return True
    return False
def drop_reason(hlc,hl):
    if not hlc or len(hlc)<15: return "short"
    if JS.search(hl): return "js"
    if is_credit(hl): return "credit"
    if PROMO.match(hlc): return "promo"
    if URLISH.match(hl.strip()): return "url"
    w=re.findall(r"\w+",hl.lower())
    if w and sum(1 for x in w if PT.match(x))/len(w)>0.14: return "nonenglish"
    return None
d=pd.read_csv(SRC,compression="gzip"); d["text"]=d.text.fillna("")
def split_hl(t):
    p=t.split("\n\nCONTEXT:",1); return p[0].replace("HEADLINE:","",1).strip(),(p[1].strip() if len(p)>1 else "")
hc=d.text.apply(lambda t:pd.Series(split_hl(t))); hc.columns=["hl","ctx"]
keep=[]; reasons=Counter(); new_text=[]; nbs=0
for i in range(len(d)):
    hl=hc.hl.iloc[i]; ctx=hc.ctx.iloc[i]; h0=hl
    m=BYLINE.match(hl or "")
    if m:
        rest=hl[m.end():]
        if len(rest)>=15 and re.match(r"[A-Z][a-z]+",rest): hl=rest; nbs+=1
    hlc=clean_seg(hl); r=drop_reason(hlc,h0)
    if r: reasons[r]+=1; continue
    segs=[]
    for s in ctx.split(";"):
        s=clean_seg(s)
        if not s or len(s)<8: continue
        if JS.search(s) or is_credit(s) or PROMO.match(s) or URLISH.match(s.strip()) or BARENAME.match(s): continue
        segs.append(s)
    new_text.append("HEADLINE: %s\n\nCONTEXT: %s"%(hlc,"; ".join(segs))); keep.append(i)
print(f"[v4] dropped {len(d)-len(keep)}; reasons {dict(reasons)}; byline-stripped {nbs}",flush=True)
v4=d.iloc[keep].copy(); v4["text"]=new_text
print(f"[v4] rows={len(v4)} pos={int(v4.judgement.sum())} snapshots={v4.snapshot_id.nunique()}",flush=True)
v4[["text","judgement","snapshot_id"]].to_csv(OUT,compression="gzip",index=False)
print(f"[v4] wrote {OUT}",flush=True)
