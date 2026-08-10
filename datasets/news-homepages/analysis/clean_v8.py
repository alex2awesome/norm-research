#!/usr/bin/env python3
"""Stage 1.9 (FINAL clean): v8. langdetect gated on a BROAD pre-filter (accented PT char OR any
common PT/ES stopword) -> accurate non-English drop on ~35% of rows. Broader credits ('Courtesy X',
'X via Y'). Keeps byline-strip, appendage strip, outlet-prefix strip. Builds from ORIGINAL."""
import pandas as pd,numpy as np,csv,sys,re,warnings
warnings.filterwarnings("ignore"); csv.field_size_limit(sys.maxsize)
from collections import Counter
from langdetect import detect, DetectorFactory
DetectorFactory.seed=0
SRC="datasets/news-homepages/homepage_newsworthiness_topic_balanced_groupsplit.csv.gz"
OUT="datasets/news-homepages/homepage_newsworthiness_clean_v8.csv.gz"
JS=re.compile(r"function\s+\w+|const\s+\w+|var\s+\w+|fallbackImage|imageLoadError|/media/|=>|\{[^}]{0,60}\}|\(img\)\s*\{|cnn-f|window\.|document\.|getelementby|addeventlistener",re.I)
HTML=re.compile(r"<[^>]+>|&lt;img[^&]*&gt;|&amp;"); READTIME=re.compile(r"\b\d+\s*(min|hr|hour|minute)s?\s*read\b",re.I); VIDDUR=re.compile(r"\b\d{1,2}:\d{2}\b")
UIBLOB=re.compile(r"[•·]\s*(Video|Gallery|Live|Photos?|Subscribers?|Watch|Show all|Analysis|Opinion|Breaking|Trending|Loading|More|Open modal)?\s*\d*:?\d*",re.I)
SECTLEAD=re.compile(r"^\s*(Analysis|Opinion|Live\s*:?\s*updates?|Video|Photos?|Gallery|For Subscribers|Sign up[^;]{0,40}|Show all|Breaking News|The Great Read|Explainer|Watch|Read|Summary|Report|First on CNN|Trending|Loading|More from|See more|News Breaking News|Live)\s*[\-:–|–]?\s*",re.I)
OUTLETTAG=re.compile(r"\s*[|\-–]\s*(CNN|BBC|New York Times|Washington Post|Wall Street Journal|WSJ|Guardian|Reuters|AP|NPR|Latimes|L\.A\. Times)\s*$",re.I)
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
AGENCYSLASH=re.compile(r"/(Getty|AFP|AP|Reuters|Sipa|CBS|CNN|NBC|ABC|Wirecutter|NYT|POOL|Bloomberg|Fox|Shutterstock|Adobe|iStock|USA|File|Paris Match|Corbis|EPA)\b",re.I)
CREDITPHRASE=re.compile(r"\b(styling by|photo(?:graph)? by|picture by|via (?:Reuters|AP|AFP|Getty)|animation by|illustration by|courtesy of|courtesy)\b",re.I)
CREDITPHRASE2=re.compile(r"\b(Photography via|via [A-Z][a-z]+|Courtesy [A-Z][a-z]+)\b")
NAMEOUTLETSTART=re.compile(r"^[A-Z][a-z'-.]+(?: [A-Z][a-z'-.]+){0,2}/[A-Z]")
BARENAME=re.compile(r"^[A-Z][a-z]+(?:[ -][A-Z][a-z]+){1,4}$")
WS=re.compile(r"\s+")
ACCENT=re.compile(r"[ãõçáéíóúêôâàÃÕÇÁÉÍÓÚÊÔÂÀ]")
PTSTOP=re.compile(r"\b(de|da|do|das|dos|que|com|para|por|uma|em|na|nas|nos|no|sua|seu|mais|como|ser|sem|apo|s[ãa]o|est[áa]|sempre|tamb[ée]m|ainda|depois|durante|segundo|contra|ap[óo]s|presidente|ministro|governo|pa[íi]s|crise|pesquis|superou|destrav|cessar|morre|fundador|golpista|primeira|encerra|strei|atinge|interior|usina|pede|novela|representando|mat[ée]ria|queda|patente|aprovou|vendida|espa[çc]o|saqu[êe]|objetivo|regime|aiatol[áa]s|ir[ãa]|podem?|vira|provoca|indica|contrata|lucram|empolga|lealdade|vence|dobra|teia|vorcaro|dele[çc][ãa]o)\b",re.I)
def clean_seg(s):
    if not s: return ""
    s=AGOAPPEND.sub("",s); s=JS.sub(" ",s); s=HTML.sub(" ",s); s=READTIME.sub(" ",s); s=VIDDUR.sub(" ",s); s=UIBLOB.sub(" ",s); s=OUTLETTAG.sub("",s); s=OUTLETPREFIX.sub("",s); s=UIANY.sub("",s); s=DATE.sub("",s); s=ILLUST.sub("",s); s=HANDLE.sub("",s); s=SECTTAG.sub("",s)
    for _ in range(3): s=SECTLEAD.sub("",s)
    s=WS.sub(" ",s).strip(); return s
def is_english(s):
    s=(s or "").strip()
    if len(s)<8: return False
    try: return detect(s)=="en"
    except: return False
def needs_langcheck(s):
    # broad pre-filter: any accented PT char OR any PT stopword -> langdetect
    return bool(ACCENT.search(s or "")) or bool(PTSTOP.search(s or ""))
def is_credit(hl):
    if CREDITPHRASE.search(hl) or CREDITPHRASE2.search(hl): return True
    if AGENCYSLASH.search(hl): return True
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
keep=[]; reasons=Counter(); new_text=[]; nbs=0; nlc=0
for i in range(len(d)):
    hl=hc.hl.iloc[i]; ctx=hc.ctx.iloc[i]; h0=hl
    m=BYLINE.match(hl or "")
    if m:
        rest=hl[m.end():]
        if len(rest)>=15 and re.match(r"[A-Z][a-z]+",rest): hl=rest; nbs+=1
    if needs_langcheck(hl):
        nlc+=1
        if not is_english(hl): reasons["nonenglish"]+=1; continue
    hlc=clean_seg(hl); r=drop_reason(hlc,h0)
    if r: reasons[r]+=1; continue
    segs=[]
    for s in ctx.split(";"):
        s=clean_seg(s)
        if not s or len(s)<8: continue
        if JS.search(s) or is_credit(s) or PROMO.match(s) or URLISH.match(s.strip()) or BARENAME.match(s): continue
        if needs_langcheck(s) and not is_english(s): continue
        segs.append(s)
    new_text.append("HEADLINE: %s\n\nCONTEXT: %s"%(hlc,"; ".join(segs))); keep.append(i)
print(f"[v8] dropped {len(d)-len(keep)}; reasons {dict(reasons)}; byline {nbs}; langchecked {nlc}",flush=True)
v8=d.iloc[keep].copy(); v8["text"]=new_text
print(f"[v8] rows={len(v8)} pos={int(v8.judgement.sum())} snapshots={v8.snapshot_id.nunique()}",flush=True)
v8[["text","judgement","snapshot_id"]].to_csv(OUT,compression="gzip",index=False)
print(f"[v8] wrote {OUT}",flush=True)
