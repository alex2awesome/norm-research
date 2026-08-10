#!/usr/bin/env python3
"""v9 FINAL: strip codex's residual read-times/bylines/credits. KEY FIX: read-times are MASHED
('Years3 min read') so no leading word-boundary. Also drop 'X min' at segment ends."""
import pandas as pd,csv,sys,re,warnings
warnings.filterwarnings("ignore"); csv.field_size_limit(sys.maxsize)
SRC="datasets/news-homepages/homepage_newsworthiness_clean_v8.csv.gz"
OUT="datasets/news-homepages/homepage_newsworthiness_clean_v9.csv.gz"
# NO leading \b (digit mashed to word). catches "Years3 min read", "5 min read", etc.
READTIME=re.compile(r"\d+\s*(?:min(?:ute)?|hr|hour)s?\s*read",re.I)
READTIME_END=re.compile(r"\d+\s*(?:min(?:ute)?|hr|hour)s?(?=\s*(?:;|$))",re.I)  # "4 min" at seg end
BYANYWHERE=re.compile(r"\bBy\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")
AGENCY=re.compile(r"/(?:Getty|AFP|AP|Reuters|Sipa|CBS|CNN|NBC|ABC|Wirecutter|NYT|POOL|Bloomberg|Fox|Shutterstock|Adobe|iStock|USA|File|Paris Match|Corbis|EPA)\b",re.I)
CREDITPHR=re.compile(r"\b(?:styling by|photo(?:graph)? by|picture by|animation by|illustration by|courtesy of|courtesy|via (?:Reuters|AP|AFP|Getty))\b",re.I)
NAMEOUTLET=re.compile(r"^[A-Z][a-z'-.]+(?: [A-Z][a-z'-.]+){0,2}/[A-Z]")
BARENAME=re.compile(r"^[A-Z][a-z]+(?:[ -][A-Z][a-z]+){1,4}$")
def scrub(s):
    if not s: return ""
    s=READTIME.sub(" ",s); s=READTIME_END.sub(" ",s); s=BYANYWHERE.sub(" ",s); s=AGENCY.sub(" ",s); s=CREDITPHR.sub(" ",s)
    s=re.sub(r"\s+"," ",s).strip(); return s
d=pd.read_csv(SRC,compression="gzip"); d["text"]=d.text.fillna("")
def split(t):
    p=t.split("\n\nCONTEXT:",1); return p[0].replace("HEADLINE:","",1).strip(),(p[1].strip() if len(p)>1 else "")
hc=d.text.apply(lambda t:pd.Series(split(t))); hc.columns=["hl","ctx"]
keep=[]; new_text=[]; ndrop=0
for i in range(len(d)):
    hl=scrub(hc.hl.iloc[i])
    segs=[scrub(s) for s in hc.ctx.iloc[i].split(";")]
    segs=[s for s in segs if s and len(s)>=8 and not NAMEOUTLET.match(s) and not BARENAME.match(s)]
    if len(hl)<15: ndrop+=1; continue
    new_text.append("HEADLINE: %s\n\nCONTEXT: %s"%(hl,"; ".join(segs))); keep.append(i)
v9=d.iloc[keep].copy(); v9["text"]=new_text
print(f"[v9] dropped {len(d)-len(keep)} ({ndrop} short); kept {len(v9)} (pos {int(v9.judgement.sum())}, {v9.snapshot_id.nunique()} snapshots)",flush=True)
v9[["text","judgement","snapshot_id"]].to_csv(OUT,compression="gzip",index=False)
# CORRECT verify (compiled regexes, single backslash)
hc9=v9.text.apply(lambda t:pd.Series(split(t))); full=(hc9[0]+" "+hc9[1])
RE_RT=re.compile(r"\d+\s*(?:min|minute|hr|hour)s?\s*read",re.I)
RE_BY=re.compile(r"\bBy\s+[A-Z][a-z]+\s+[A-Z][a-z]+")
RE_AG=re.compile(r"/(?:Getty|AFP|AP|Reuters|CBS|Sipa)\b",re.I)
print(f"[verify] readtime={sum(bool(RE_RT.search(t)) for t in full)/len(full):.4f}  byline={sum(bool(RE_BY.search(t)) for t in full)/len(full):.4f}  agency={sum(bool(RE_AG.search(t)) for t in full)/len(full):.4f}  short_hl={(hc9[0].str.len()<15).mean():.4f}",flush=True)
print("[v9] wrote "+OUT,flush=True)
