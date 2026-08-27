#!/usr/bin/env python3
"""Stage 1.8: v6. Fast stopword-based language detection (no slow langdetect) + non-length-gated
credit detection + outlet-prefix-mash strip. Builds from ORIGINAL."""
import pandas as pd,numpy as np,csv,sys,re,warnings
warnings.filterwarnings("ignore"); csv.field_size_limit(sys.maxsize)
from collections import Counter
SRC="datasets/news-homepages/homepage_newsworthiness_topic_balanced_groupsplit.csv.gz"
OUT="datasets/news-homepages/homepage_newsworthiness_clean_v7.csv.gz"
JS=re.compile(r"function\s+\w+|const\s+\w+|var\s+\w+|fallbackImage|imageLoadError|/media/|=>|\{[^}]{0,60}\}|\(img\)\s*\{|cnn-f|window\.|document\.|getelementby|addeventlistener",re.I)
HTML=re.compile(r"<[^>]+>|&lt;img[^&]*&gt;|&amp;"); READTIME=re.compile(r"\b\d+\s*(min|hr|hour|minute)s?\s*read\b",re.I); VIDDUR=re.compile(r"\b\d{1,2}:\d{2}\b")
UIBLOB=re.compile(r"[•·]\s*(Video|Gallery|Live|Photos?|Subscribers?|Watch|Show all|Analysis|Opinion|Breaking|Trending|Loading|More|Open modal)?\s*\d*:?\d*",re.I)
SECTLEAD=re.compile(r"^\s*(Analysis|Opinion|Live\s*:?\s*updates?|Video|Photos?|Gallery|For Subscribers|Sign up[^;]{0,40}|Show all|Breaking News|The Great Read|Explainer|Watch|Read|Summary|Report|First on CNN|Trending|Loading|More from|See more|News Breaking News|Live)\s*[\-:–|–]?\s*",re.I)
OUTLETTAG=re.compile(r"\s*[|\-–]\s*(CNN|BBC|New York Times|Washington Post|Wall Street Journal|WSJ|Guardian|Reuters|AP|NPR|Latimes|L\.A\. Times)\s*$",re.I)
# outlet-prefix mash: "The Wall Street JournalHeadline" or "New York TimesHeadline"
OUTLETPREFIX=re.compile(r"^(The\s+)?(Wall Street Journal|New York Times|Washington Post|CNN|BBC|Guardian|Reuters|Latimes|L\.A\. Times|Associated Press)(?=[A-Z])")
PROMO=re.compile(r"^\s*(Sign up|For Subscribers|Subscribe|Show all|See more|Read more|Watch the latest|Watch|Listen|Follow|Newsletter|The Recap|The Morning|The Evening|More from|Trending|Loading|Expert[- ]backed guides|Open modal)\b",re.I)
UIANY=re.compile(r"\b(Show all|Trending|Loading|See more|More from|For Subscribers|Open modal at item)\b",re.I)
URLISH=re.compile(r"^(https?://|www\.|/\w+/[\w-]+/?)")
DATE=re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b",re.I)
AGOAPPEND=re.compile(r"\b\d+\s*(hours?|hrs?|minutes?|mins?|days?|seconds?|moments?)\s*ago\b.*$",re.I|re.DOTALL)
SECTTAG=re.compile(r"(US\s*(?:&|and)\s*Canada|UK|Europe|Asia|Middle East|Africa|Americas|Australia|World|Business|Technology|Science|Health|Entertainment[s]?|Sports?|Politics|Climate|Environment|England|Scotland|Wales|Northern Ireland|Latin America|India|China|Russia|Ukraine)\s*$",re.I)
ILLUST=re.compile(r"\b(Illustration|Photo(?:graph)?|Animation|Picture|Graphic) by [A-Z][\w .'-]{2,40}\s*$",re.I)
HANDLE=re.compile(r"\s*@[A-Za-z0-9_]+/(Instagram|Twitter|X|Facebook|TikTok)\s*$",re.I)
BYLINE=re.compile(r"^([A-Z][a-z]+ [A-Z][a-z]+)(?=[A-Z][a-z]+)")
# credit detection: NOT length-gated. agency-slash anywhere, or credit phrase, or starts Name/Outlet
AGENCYSLASH=re.compile(r"/(Getty|AFP|AP|Reuters|Sipa|CBS|CNN|NBC|ABC|Wirecutter|NYT|POOL|Bloomberg|Fox|Shutterstock|Adobe|iStock|USA|File|Paris Match|Corbis|EPA)\b",re.I)
CREDITPHRASE=re.compile(r"\b(styling by|photo(?:graph)? by|picture by|via (?:Reuters|AP|AFP|Getty)|animation by|illustration by)\b",re.I)
NAMEOUTLETSTART=re.compile(r"^[A-Z][a-z'-.]+(?: [A-Z][a-z'-.]+){0,2}/[A-Z]")
BARENAME=re.compile(r"^[A-Z][a-z]+(?:[ -][A-Z][a-z]+){1,4}$")
WS=re.compile(r"\s+")
# stopword-based language detection (fast, no langdetect)
PTES=r"\b(da|do|das|dos|que|uma|com|para|por|na|nas|nos|sua|seu|suas|seus|mais|como|j[áa]|são|est[áa]|estão|foi|ser|ainda|mesmo|isso|este|esse|aquele|porque|então|também|depois|sobre|sem|muito|pouco|sempre|ele|ela|eles|elas|deles|delas|nação|país|governo|presidente|ministro|após|contra|durante|segundo|crise|pesquis|superou|destrav|cessar|morre|fundador|golpista|chegam|primeira|encerra|strei|atinge|interior|usina|pede|novela|representando|matéria|queda|patente|aprovou|vendida|espaço|saquê|objetivo|regime|aiatolás|irã|israel|podem?)\b"
EN=r"\b(the|of|to|in|and|a|is|for|on|with|as|by|at|it|that|this|from|be|was|are|or|an|were|has|have|had|not|but|his|her|they|their|its|which|who|when|where|will|can|said|says|after|before|over|under|about|into|out|up|down|new|more|most|some|any|all|each|other|one|two|three|trump|biden|senate|house|court|state|states|official|police|president|government|bill|law|attack|war|killed|says|told|report)\b"
def is_nonenglish(s):
    s=(s or "").lower()
    if len(s)<12: return False
    pes=len(re.findall(PTES,s)); en=len(re.findall(EN,s))
    return pes>en and pes>=2
def clean_seg(s):
    if not s: return ""
    s=AGOAPPEND.sub("",s); s=JS.sub(" ",s); s=HTML.sub(" ",s); s=READTIME.sub(" ",s); s=VIDDUR.sub(" ",s); s=UIBLOB.sub(" ",s); s=OUTLETTAG.sub("",s); s=OUTLETPREFIX.sub("",s); s=UIANY.sub("",s); s=DATE.sub("",s); s=ILLUST.sub("",s); s=HANDLE.sub("",s); s=SECTTAG.sub("",s)
    for _ in range(3): s=SECTLEAD.sub("",s)
    s=WS.sub(" ",s).strip(); return s
def is_credit(hl):
    if CREDITPHRASE.search(hl): return True
    if AGENCYSLASH.search(hl): return True   # any agency slash anywhere (not length-gated)
    if NAMEOUTLETSTART.match(hl): return True
    return False
def drop_reason(hlc,hl):
    if not hlc or len(hlc)<15: return "short"
    if JS.search(hl): return "js"
    if is_credit(hl): return "credit"
    if PROMO.match(hlc): return "promo"
    if URLISH.match(hl.strip()): return "url"
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
    if is_nonenglish(hl): reasons["nonenglish"]+=1; continue
    hlc=clean_seg(hl); r=drop_reason(hlc,h0)
    if r: reasons[r]+=1; continue
    segs=[]
    for s in ctx.split(";"):
        s=clean_seg(s)
        if not s or len(s)<8: continue
        if JS.search(s) or is_credit(s) or PROMO.match(s) or URLISH.match(s.strip()) or BARENAME.match(s) or is_nonenglish(s): continue
        segs.append(s)
    new_text.append("HEADLINE: %s\n\nCONTEXT: %s"%(hlc,"; ".join(segs))); keep.append(i)
print(f"[v7] dropped {len(d)-len(keep)}; reasons {dict(reasons)}; byline-stripped {nbs}",flush=True)
v6=d.iloc[keep].copy(); v6["text"]=new_text
print(f"[v7] rows={len(v6)} pos={int(v6.judgement.sum())} snapshots={v6.snapshot_id.nunique()}",flush=True)
v6[["text","judgement","snapshot_id"]].to_csv(OUT,compression="gzip",index=False)
print(f"[v7] wrote {OUT}",flush=True)
