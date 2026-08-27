#!/usr/bin/env python3
"""Stage 1.5: stricter clean -> v3. Adds (a) conservative byline-strip on headlines (no-space
CamelCase-join artifact only), (b) tighter context-segment cleaning (drop image-credit chains,
promo segments, bare byline names). Re-measures V + reports drop/delta vs v2."""
import pandas as pd, numpy as np, csv, sys, re, warnings
warnings.filterwarnings("ignore")
csv.field_size_limit(sys.maxsize)
from collections import Counter
SRC="datasets/news-homepages/homepage_newsworthiness_topic_balanced_groupsplit.csv.gz"
OUT="datasets/news-homepages/homepage_newsworthiness_clean_v3.csv.gz"
JS=re.compile(r"function\s+\w+|const\s+\w+|var\s+\w+|fallbackImage|imageLoadError|/media/sites/|=>|\{[^}]{0,40}\}|window\.|document\.",re.I)
HTML=re.compile(r"<[^>]+>"); READTIME=re.compile(r"\b\d+\s*(min|hr|hour|minute)s?\s*read\b",re.I); VIDDUR=re.compile(r"\b\d{1,2}:\d{2}\b")
UIBLOB=re.compile(r"[•·]\s*(Video|Gallery|Live|Photos?|Subscribers?|Watch|Show all|Analysis|Opinion|Breaking)?\s*\d*:?\d*",re.I)
SECTLEAD=re.compile(r"^\s*(Analysis|Opinion|Live\s*:?\s*updates?|Video|Photos?|Gallery|For Subscribers|Sign up[^;]{0,40}|Show all|Breaking News|The Great Read|Explainer|Watch|Read|Summary|Report|First on CNN)\s*[\-:–|–]?\s*",re.I)
OUTLETTAG=re.compile(r"\s*[|\-–]\s*(CNN|BBC|New York Times|Washington Post|Wall Street Journal|WSJ|Guardian|Reuters|AP|NPR|Latimes|L\.A\. Times)\s*$",re.I)
PROMO=re.compile(r"^\s*(Sign up|For Subscribers|Subscribe|Show all|See more|Read more|Watch the latest|Watch|Listen|Follow|Newsletter|The Recap|The Morning|The Evening|More from)\b",re.I)
URLISH=re.compile(r"^(https?://|www\.|/\w+/[\w-]+/?)")
SHOWALL=re.compile(r"\bShow all\b",re.I)
# conservative byline strip: 2-word capitalized name immediately followed (NO space) by a Capitalized word
BYLINE=re.compile(r"^([A-Z][a-z]+ [A-Z][a-z]+)(?=[A-Z][a-z]+)")
# credit-chain / bare-name context-segment drop
CREDIT=re.compile(r"\b(getty|afp|reuters|animation by|istock|adobe stock|shutterstock|bauer[- ]?griffin|gc images|media.?punch|underscored|photo(?:graph)? by|via |ap$)\b",re.I)
BARENAME=re.compile(r"^[A-Z][a-z]+(?:[ -][A-Z][a-z]+){1,3}$")  # a bare 2-4 token name (byline) alone
PT=re.compile(r"\b(não|nas|nos|das|dos|para|que|uma|com|mais|está|estão|ser|por|como|já|sempre|segunda|terça|quarta|quinta|sexta|feira|presidente|segundo|após|contra|disse|também|ainda|entre|sobre|sem|sua|seu|nação|país|governo|ministro)\b",re.I)
WS=re.compile(r"\s+")
def clean_seg(s):
    if not s: return ""
    s=JS.sub(" ",s); s=HTML.sub(" ",s); s=READTIME.sub(" ",s); s=VIDDUR.sub(" ",s); s=UIBLOB.sub(" ",s); s=OUTLETTAG.sub("",s); s=SHOWALL.sub("",s)
    for _ in range(3): s=SECTLEAD.sub("",s)
    s=WS.sub(" ",s).strip(); return s
def drop_reason(hlc,hl):
    if not hlc or len(hlc)<15: return "short"
    if JS.search(hl) or "imageLoadError" in hl or "fallbackImage" in hl: return "js"
    if CREDIT.search(hl) and len(hlc)<45: return "imgcredit"
    if PROMO.match(hlc): return "promo"
    if URLISH.match(hl.strip()): return "url"
    w=re.findall(r"\w+",hl.lower());
    if w and sum(1 for x in w if PT.match(x))/len(w)>0.18: return "nonenglish"
    return None
d=pd.read_csv(SRC,compression="gzip"); d["text"]=d.text.fillna("")
def split_hl(t):
    p=t.split("\n\nCONTEXT:",1); return p[0].replace("HEADLINE:","",1).strip(),(p[1].strip() if len(p)>1 else "")
hc=d.text.apply(lambda t:pd.Series(split_hl(t))); hc.columns=["hl","ctx"]
keep=[]; reasons=Counter(); new_text=[]; n_byline_stripped=0
for i in range(len(d)):
    hl=hc.hl.iloc[i]; ctx=hc.ctx.iloc[i]
    # conservative byline strip on headline
    h0=hl
    m=BYLINE.match(hl or "")
    if m:
        rest=hl[m.end():]
        if len(rest)>=15 and re.match(r"[A-Z][a-z]+",rest): hl=rest; n_byline_stripped+=1
    hlc=clean_seg(hl)
    r=drop_reason(hlc,h0)
    if r: reasons[r]+=1; continue
    segs=[clean_seg(s) for s in ctx.split(";")]
    segs2=[]
    for s in segs:
        if not s or len(s)<8: continue
        if JS.search(s) or CREDIT.search(s) or PROMO.match(s) or URLISH.match(s.strip()): continue
        if BARENAME.match(s): continue  # bare byline name
        segs2.append(s)
    new_text.append("HEADLINE: %s\n\nCONTEXT: %s"%(hlc,"; ".join(segs2))); keep.append(i)
print(f"[v3] dropped {len(d)-len(keep)}; reasons {dict(reasons)}; byline-stripped {n_byline_stripped} headlines",flush=True)
v3=d.iloc[keep].copy(); v3["text"]=new_text
print(f"[v3] rows={len(v3)} pos={int(v3.judgement.sum())} snapshots={v3.snapshot_id.nunique()}",flush=True)
v3[["text","judgement","snapshot_id"]].to_csv(OUT,compression="gzip",index=False)
print(f"[v3] wrote {OUT}",flush=True)
