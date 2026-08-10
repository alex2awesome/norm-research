#!/usr/bin/env python3
"""v2 sharded, proxy-rotating full-text fetcher.
Inputs: cdx_index.jsonl (url,ts from build_cdx_index) for wayback-direct fetches; guardian URLs
go via Guardian API; URLs without a CDX entry fall back to live fetch.
Sharding: --shard K --nshards N (deterministic hash on url) -> run one shard per host (sk1/2/3).
Proxies: rotating webshare list (~/.proxies_list.txt), random UA per request, per-proxy cooldown
on 429/403; direct (no-proxy) fallback for guardian API. BS4 .get_text(' ').
Run: python3 fetch_fulltext_v2.py --shard 0 --nshards 3 [--workers 30] [--priority twitter|all]"""
import json, os, re, sys, time, gzip, random, argparse, hashlib, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from bs4 import BeautifulSoup

BASE = os.path.dirname(os.path.abspath(__file__))
URLS = os.path.join(BASE, "..", "twitter_engagement", "urls_to_scrape.jsonl")
TW_DONE = os.path.join(BASE, "..", "twitter_engagement", "tweet_engagement.jsonl")
CDX = os.path.join(BASE, "cdx_index.jsonl")
PROXY_FILE = os.path.expanduser("~/.proxies_list.txt")
GUARDIAN_KEY = os.environ.get("GUARDIAN_KEY", "test")

UAS = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
       "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
       "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
       "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0"]

class ProxyPool:
    def __init__(self, path):
        self.proxies = [l.strip() for l in open(path) if l.strip()]
        self.cooldown = {}
        self.lock = Lock()
    def pick(self):
        with self.lock:
            now = time.time()
            ok = [p for p in self.proxies if self.cooldown.get(p, 0) < now]
            return random.choice(ok if ok else self.proxies)
    def penalize(self, p, secs=120):
        with self.lock:
            self.cooldown[p] = time.time() + secs

def get(url, pool, timeout=45, tries=4, use_proxy=True):
    last = None
    for t in range(tries):
        try:
            if use_proxy:
                p = pool.pick()
                h = urllib.request.ProxyHandler({"http": f"http://{p}", "https": f"http://{p}"})
                op = urllib.request.build_opener(h)
            else:
                p = None; op = urllib.request.build_opener()
            req = urllib.request.Request(url, headers={"User-Agent": random.choice(UAS)})
            with op.open(req, timeout=timeout) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip": raw = gzip.decompress(raw)
                return r.status, raw.decode("utf-8", "ignore")
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 403, 503) and use_proxy and p: pool.penalize(p)
            if e.code == 404: raise
        except Exception as e:
            last = e
            if use_proxy and p: pool.penalize(p, 30)
        time.sleep(1 + t)
    raise last if last else RuntimeError("fetch failed")

def soup_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "iframe", "svg"]): t.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, default=3)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--priority", default="twitter", choices=["twitter", "all"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or os.path.join(BASE, f"fulltext_v2_shard{args.shard}.jsonl")
    pool = ProxyPool(PROXY_FILE)
    print(f"[v2 s{args.shard}] {len(pool.proxies)} proxies", flush=True)
    cdx = {}
    if os.path.exists(CDX):
        for ln in open(CDX):
            try: r = json.loads(ln); cdx[r["url"]] = r["ts"]
            except Exception: pass
    done = set()
    if os.path.exists(out_path):
        for ln in open(out_path):
            try: done.add(json.loads(ln)["url"])
            except Exception: pass
    tw = set()
    if os.path.exists(TW_DONE):
        for ln in open(TW_DONE):
            try: tw.add(json.loads(ln)["url"])
            except Exception: pass
    todo, seen = [], set()
    for ln in open(URLS):
        try: r = json.loads(ln)
        except Exception: continue
        u = r.get("url")
        if not u or u in seen or u in done: continue
        seen.add(u)
        if int(hashlib.sha1(u.encode()).hexdigest(), 16) % args.nshards != args.shard: continue
        todo.append(r)
    todo.sort(key=lambda r: 0 if r["url"] in tw else 1)
    if args.priority == "twitter":
        todo = [r for r in todo if r["url"] in tw]
    print(f"[v2 s{args.shard}] todo={len(todo)} (cdx-index={len(cdx)}, done={len(done)})", flush=True)
    lock = Lock(); fout = open(out_path, "a"); n = [0, 0]
    def work(rec):
        url, outlet = rec["url"], rec.get("outlet", "?")
        row = {"url": url, "outlet": outlet, "route": "FAIL", "text_len": 0, "ts_fetched": int(time.time())}
        try:
            if outlet == "guardian":
                path = urllib.parse.urlparse(url).path.strip("/")
                st, body = get(f"https://content.guardianapis.com/{path}?api-key={GUARDIAN_KEY}&show-fields=bodyText,headline",
                               pool, use_proxy=False)
                f = json.loads(body).get("response", {}).get("content", {}).get("fields", {})
                txt = (f.get("headline", "") + "\n\n" + f.get("bodyText", "")).strip()
                if len(txt) > 400:
                    row.update(route="guardian_api", text_len=len(txt), text=txt[:40000])
            if row["route"] == "FAIL" and url in cdx:
                st, html = get(f"https://web.archive.org/web/{cdx[url]}id_/{url}", pool)
                txt = soup_text(html)
                if len(txt) > 400:
                    row.update(route="wayback", text_len=len(txt), text=txt[:40000])
            if row["route"] == "FAIL" and url not in cdx:
                st, html = get(url, pool, tries=2)
                txt = soup_text(html)
                if len(txt) > 400:
                    row.update(route="live", text_len=len(txt), text=txt[:40000])
        except Exception as e:
            row["err"] = str(e)[:60]
        with lock:
            fout.write(json.dumps(row) + "\n"); fout.flush()
            n[0] += 1; n[1] += (row["route"] != "FAIL")
            if n[0] % 200 == 0:
                print(f"[v2 s{args.shard}] {n[0]}/{len(todo)} ok={n[1]} ({n[1]/max(n[0],1):.0%})", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    print(f"[v2 s{args.shard}] DONE {n[0]} ok={n[1]}", flush=True)

if __name__ == "__main__":
    main()
