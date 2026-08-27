#!/usr/bin/env python3
"""Stage-1 full-text backfill for homepage article URLs.
Route per URL: guardian -> Guardian Open Platform API (clean bodyText, skips HTML);
else Wayback CDX->snapshot (id_ raw); else live fetch. HTML -> BeautifulSoup .get_text(' ').
Resumable (done-set on url), JSONL append, polite rate limits.
Run on sk3: python3 fetch_fulltext.py [--limit N] [--priority twitter|all]
Output rows: {url, outlet, route, http, html_len, text_len, text(<=40k), ts_fetched}
"""
import json, os, re, sys, time, gzip, random, argparse, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
try:
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("pip install beautifulsoup4")

BASE = os.path.dirname(os.path.abspath(__file__))
URLS = os.path.join(BASE, "..", "twitter_engagement", "urls_to_scrape.jsonl")
TW_DONE = os.path.join(BASE, "..", "twitter_engagement", "tweet_engagement.jsonl")
OUT = os.path.join(BASE, "fulltext.jsonl")
LOG = os.path.join(BASE, "fetch.log")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
GUARDIAN_KEY = os.environ.get("GUARDIAN_KEY", "test")

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

import threading
_backoff = {"wayback": 0.0, "guardian": 0.0}   # epoch until which the host is paused
_bo_lock = threading.Lock()

def _host_key(url):
    if "web.archive.org" in url: return "wayback"
    if "guardianapis.com" in url: return "guardian"
    return None

def http_get(url, timeout=30, max_tries=4):
    """GET with graceful backoff. 429/503 (and Wayback connection-limit resets) trigger a
    shared per-host pause with exponential growth (30s -> 60 -> 120 -> 240), so ALL workers
    stop hammering the host, then retry. Other errors raise after max_tries."""
    key = _host_key(url)
    delay = 30.0
    for attempt in range(max_tries):
        if key:
            with _bo_lock: until = _backoff[key]
            wait = until - time.time()
            if wait > 0: time.sleep(wait + random.uniform(0, 2))
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return r.status, raw.decode("utf-8", "ignore")
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and key and attempt < max_tries - 1:
                retry_after = e.headers.get("Retry-After")
                pause = float(retry_after) if (retry_after or "").isdigit() else delay
                with _bo_lock:
                    _backoff[key] = max(_backoff[key], time.time() + pause)
                log(f"BACKOFF {key} {e.code}: pausing {pause:.0f}s (attempt {attempt+1})")
                delay *= 2
                continue
            raise
        except (ConnectionResetError, urllib.error.URLError) as e:
            # Wayback drops connections when rate-limited; treat as soft backoff
            if key and attempt < max_tries - 1:
                with _bo_lock:
                    _backoff[key] = max(_backoff[key], time.time() + delay)
                log(f"BACKOFF {key} conn-error: pausing {delay:.0f}s ({str(e)[:40]})")
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")

def soup_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "iframe", "svg"]):
        t.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()

def fetch_guardian(url):
    path = urllib.parse.urlparse(url).path.strip("/")
    q = f"https://content.guardianapis.com/{path}?api-key={GUARDIAN_KEY}&show-fields=bodyText,headline"
    st, body = http_get(q)
    d = json.loads(body)
    c = d.get("response", {}).get("content", {})
    f = c.get("fields", {})
    txt = (f.get("headline", "") + "\n\n" + f.get("bodyText", "")).strip()
    return ("guardian_api", st, len(body), txt) if len(txt) > 200 else (None, st, 0, "")

def fetch_wayback(url):
    q = ("https://web.archive.org/cdx/search/cdx?url=" + urllib.parse.quote(url, safe="")
         + "&output=json&limit=2&filter=statuscode:200&collapse=digest")
    st, body = http_get(q)
    rows = json.loads(body)
    if len(rows) < 2: return (None, st, 0, "")
    ts = rows[1][1]
    st2, html = http_get(f"https://web.archive.org/web/{ts}id_/{url}")
    return ("wayback", st2, len(html), soup_text(html))

def fetch_live(url):
    st, html = http_get(url)
    return ("live", st, len(html), soup_text(html))

def fetch_one(rec):
    url, outlet = rec["url"], rec.get("outlet", "?")
    routes = ([fetch_guardian] if outlet == "guardian" else []) + [fetch_wayback, fetch_live]
    last_err = ""
    for fn in routes:
        try:
            route, st, hlen, txt = fn(url)
            if route and len(txt) > 400:
                return {"url": url, "outlet": outlet, "route": route, "http": st,
                        "html_len": hlen, "text_len": len(txt), "text": txt[:40000],
                        "ts_fetched": int(time.time())}
        except Exception as e:
            last_err = f"{fn.__name__}:{str(e)[:60]}"
        time.sleep(0.4)
    return {"url": url, "outlet": outlet, "route": "FAIL", "err": last_err,
            "text_len": 0, "ts_fetched": int(time.time())}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--priority", default="twitter", choices=["twitter", "all"])
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    done = set()
    if os.path.exists(OUT):
        for ln in open(OUT):
            try: done.add(json.loads(ln)["url"])
            except Exception: pass
    tw_urls = set()
    if os.path.exists(TW_DONE):
        for ln in open(TW_DONE):
            try: tw_urls.add(json.loads(ln)["url"])
            except Exception: pass
    todo, seen = [], set()
    for ln in open(URLS):
        try: r = json.loads(ln)
        except Exception: continue
        u = r.get("url")
        if not u or u in done or u in seen: continue
        seen.add(u)
        if args.priority == "twitter" and u not in tw_urls: continue
        todo.append(r)
    # twitter-priority ordering also within 'all'
    if args.priority == "all":
        todo.sort(key=lambda r: 0 if r["url"] in tw_urls else 1)
    if args.limit: todo = todo[:args.limit]
    log(f"todo={len(todo)} (done={len(done)}, twitter_pool={len(tw_urls)}, priority={args.priority})")
    lock = Lock(); n = [0, 0]
    fout = open(OUT, "a")
    def work(rec):
        row = fetch_one(rec)
        with lock:
            fout.write(json.dumps(row) + "\n"); fout.flush()
            n[0] += 1; n[1] += (row["route"] != "FAIL")
            if n[0] % 50 == 0:
                log(f"progress {n[0]}/{len(todo)} ok={n[1]} ({n[1]/max(n[0],1):.0%})")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    log(f"DONE {n[0]} fetched, ok={n[1]}")

if __name__ == "__main__":
    main()
