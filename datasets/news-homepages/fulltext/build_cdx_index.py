#!/usr/bin/env python3
"""Bulk-CDX index: resolve availability + snapshot timestamp for ALL corpus URLs with a few
hundred prefix queries instead of 662k per-URL lookups.
Strategy: group corpus URLs by (domain, path-prefix = /YYYY/MM or outlet-specific), issue
cdx?url=domain/prefix/*&collapse=urlkey (paginated via resumeKey), intersect with corpus.
Runs through rotating proxies. Output: cdx_index.jsonl {url, ts} for every archived corpus URL.
Run: python3 build_cdx_index.py [--outlets wapo,reuters,...]"""
import json, os, re, sys, time, random, argparse, urllib.request, urllib.parse
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
URLS = os.path.join(BASE, "..", "twitter_engagement", "urls_to_scrape.jsonl")
OUT = os.path.join(BASE, "cdx_index.jsonl")
PROXY_FILE = os.path.expanduser("~/.proxies_list.txt")   # user:pass@host:port per line

def load_proxies():
    ps = [l.strip() for l in open(PROXY_FILE) if l.strip()]
    random.shuffle(ps)
    return ps

UAS = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
       "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
       "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0"]

def get(url, proxies, timeout=90, tries=5):
    for t in range(tries):
        proxy = random.choice(proxies)
        try:
            h = urllib.request.ProxyHandler({"http": f"http://{proxy}", "https": f"http://{proxy}"})
            op = urllib.request.build_opener(h)
            req = urllib.request.Request(url, headers={"User-Agent": random.choice(UAS)})
            with op.open(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception:
            time.sleep(2 * (t + 1))
    return None

def norm_url(u):
    u = re.sub(r"^https?://(www\.)?", "", (u or "").lower()).rstrip("/")
    return u.split("?")[0]

def prefix_of(url):
    """Group key: domain + first path segments incl. /YYYY/MM(/DD) when present."""
    m = re.match(r"^https?://(?:www\.)?([^/]+)(/.*)?$", url)
    if not m: return None
    dom, path = m.group(1), (m.group(2) or "/")
    pm = re.match(r"^(.*?/20\d\d/(?:[a-z]{3}|\d\d))/", path)
    if pm: return dom + pm.group(1)
    segs = [s for s in path.split("/") if s]
    return dom + "/" + "/".join(segs[:2]) if len(segs) >= 2 else dom + "/" + (segs[0] if segs else "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_group", type=int, default=8,
                    help="prefixes with fewer corpus URLs than this get per-URL lookups later")
    args = ap.parse_args()
    proxies = load_proxies()
    print(f"[cdx] {len(proxies)} proxies", flush=True)
    done_urls = set()
    if os.path.exists(OUT):
        for ln in open(OUT):
            try: done_urls.add(json.loads(ln)["url"])
            except Exception: pass
    corpus = {}
    for ln in open(URLS):
        try: r = json.loads(ln)
        except Exception: continue
        u = r.get("url")
        if u and u not in done_urls: corpus.setdefault(norm_url(u), u)
    groups = defaultdict(list)
    for nu, u in corpus.items():
        p = prefix_of(u)
        if p: groups[p].append(nu)
    big = {p: us for p, us in groups.items() if len(us) >= args.min_group}
    print(f"[cdx] {len(corpus)} URLs -> {len(groups)} prefixes ({len(big)} big >= {args.min_group})", flush=True)
    fout = open(OUT, "a")
    hits = 0
    for gi, (prefix, members) in enumerate(sorted(big.items(), key=lambda kv: -len(kv[1]))):
        member_set = set(members)
        resume = ""
        page_hits = 0
        while True:
            q = ("http://web.archive.org/cdx/search/cdx?url=" + urllib.parse.quote(prefix, safe="")
                 + "/*&output=json&filter=statuscode:200&collapse=urlkey&fl=original,timestamp"
                 + "&limit=5000&showResumeKey=true" + (f"&resumeKey={resume}" if resume else ""))
            body = get(q, proxies)
            if body is None: break
            try: rows = json.loads(body)
            except Exception: break
            resume_next = ""
            if rows and rows[-2:] and rows[-2] == [] :
                resume_next = rows[-1][0] if rows[-1] else ""
                rows = rows[:-2]
            for r in rows[1:]:
                if len(r) < 2: continue
                nu = norm_url(r[0])
                if nu in member_set:
                    fout.write(json.dumps({"url": corpus[nu], "ts": r[1]}) + "\n")
                    hits += 1; page_hits += 1
            fout.flush()
            if not resume_next: break
            resume = urllib.parse.quote(resume_next, safe="")
        if gi % 20 == 0:
            print(f"[cdx] {gi}/{len(big)} prefixes, {hits} archived-URL hits", flush=True)
        time.sleep(0.3)
    print(f"[cdx] DONE: {hits} archived corpus URLs indexed", flush=True)
    print("CDX_INDEX_DONE", flush=True)

if __name__ == "__main__":
    main()
