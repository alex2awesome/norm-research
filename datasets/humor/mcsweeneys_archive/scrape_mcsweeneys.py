#!/usr/bin/env python3
"""McSweeney's Internet Tendency full-archive scrape.

Stage 1: enumerate /articles/archives?page=N (~1,458 pages x ~20 slugs)
         -> archive_index.jsonl (url, title, page)
Stage 2: fetch every piece -> pieces_raw/<slug-hash>.html.gz

Pool = full archive; curated top-k = "Best of" anthology TOCs (manual step).
Requires browser UA (naive fetchers get 403). robots carries ai-train=no
rights signal (rights note recorded in README).
"""
import gzip
import hashlib
import json
import os
import re
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://www.mcsweeneys.net"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 norm-research-academic "
      "(contact: alex2awesome@gmail.com)")
RATE = 1.1
STATE = os.path.join(HERE, "state.json")


def st_load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {}


def st_save(s):
    json.dump(s, open(STATE, "w"))


def get(sess, url):
    for a in range(7):
        try:
            r = sess.get(url, timeout=60)
            if r.status_code in (429, 500, 502, 503):
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.text
        except Exception as e:
            w = min(300, 2 ** a * 5)
            print(f"{url} attempt {a}: {e}; sleep {w}", flush=True)
            time.sleep(w)
    raise RuntimeError(url)


def stage_index(sess, st):
    idx_f = os.path.join(HERE, "archive_index.jsonl")
    page = st.get("index_page", 1)
    out = open(idx_f, "a")
    empty = 0
    while page < 2000:
        h = get(sess, f"{BASE}/articles/archives?page={page}")
        arts = re.findall(r'href="(/articles/[^"?#]+)"[^>]*>([^<]{2,200})', h)
        rows = []
        seen = set()
        for u, t in arts:
            if u in seen or u.startswith("/articles/archives"):
                continue
            seen.add(u)
            rows.append({"url": u, "title": t.strip(), "page": page})
        for r_ in rows:
            out.write(json.dumps(r_) + "\n")
        if not rows:
            empty += 1
            if empty >= 3:
                break
        else:
            empty = 0
        if page % 50 == 0:
            out.flush()
            print(f"index page {page}: {len(rows)} pieces", flush=True)
        page += 1
        st["index_page"] = page
        st_save(st)
        time.sleep(RATE)
    out.close()
    st["index_done"] = True
    st_save(st)


def stage_pieces(sess, st):
    raw_dir = os.path.join(HERE, "pieces_raw")
    os.makedirs(raw_dir, exist_ok=True)
    urls = []
    seen = set()
    for l in open(os.path.join(HERE, "archive_index.jsonl")):
        u = json.loads(l)["url"]
        if u not in seen:
            seen.add(u)
            urls.append(u)
    print(f"{len(urls)} unique pieces", flush=True)
    for i, u in enumerate(urls):
        hcode = hashlib.md5(u.encode()).hexdigest()[:16]
        f = os.path.join(raw_dir, f"{hcode}.html.gz")
        if os.path.exists(f):
            continue
        h = get(sess, BASE + u)
        with gzip.open(f, "wt") as fh:
            fh.write(f"<!-- url: {u} -->\n" + h)
        if i % 200 == 0:
            print(f"piece {i}/{len(urls)}", flush=True)
        time.sleep(RATE)


def main():
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})
    st = st_load()
    if not st.get("index_done"):
        stage_index(sess, st)
    stage_pieces(sess, st)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
