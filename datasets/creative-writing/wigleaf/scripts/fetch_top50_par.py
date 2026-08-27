#!/usr/bin/env python3
"""Parallel Top50 fetch (live -> Wayback). Per-domain throttle, thread pool."""
import json
import os
import re
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, quote

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_engine import cached_fetch
from wig_textproc import full_pipeline, looks_like_junk_page, normalize_magazine

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/wigleaf"
CACHE = os.path.join(BASE, "text_cache", "top50")
MIN_TEXT = 300
WORKERS = 16

PARKED_DOMAINS = {"hugedomains.com", "sedo.com", "sedoparking.com", "dan.com",
                  "afternic.com", "godaddy.com", "namecheap.com", "porkbun.com"}
PARKED_TITLE = re.compile(r"domain (?:is )?for sale|buy this domain|parked free|"
                          r"this domain may be for sale|domain expired|seized|account suspended", re.I)


def is_parked(html_str, final_url):
    dom = urlparse(final_url).netloc.lower().removeprefix("www.")
    if dom in PARKED_DOMAINS:
        return True
    m = re.search(r"<title[^>]*>(.*?)</title>", html_str[:20000], re.S | re.I)
    return bool(m and PARKED_TITLE.search(m.group(1)))


def homepage_redirect(o, f):
    op, fp = urlparse(o), urlparse(f)
    return len(op.path.strip("/")) > 1 and len(fp.path.strip("/")) <= 1 and not fp.query


def wb_avail(url, year):
    api = ("http://archive.org/wayback/available?url=" + quote(url, safe="") + f"&timestamp={year}0601")
    res = cached_fetch(api, "wbavail", CACHE)
    if not res.get("html"):
        return None
    try:
        snap = json.loads(res["html"])["archived_snapshots"].get("closest")
    except Exception:
        return None
    if snap and snap.get("available"):
        return snap["url"].replace("http://web.archive.org", "https://web.archive.org")
    return None


def wb_cdx(url, year):
    api = ("https://web.archive.org/cdx/search/cdx?url=" + quote(url, safe="") +
           "&output=json&limit=8&filter=statuscode:200&collapse=digest")
    res = cached_fetch(api, "wbcdx", CACHE)
    if not res.get("html"):
        return None
    try:
        rows = json.loads(res["html"])
    except Exception:
        return None
    if len(rows) < 2:
        return None
    target = int(f"{year}0601000000")
    best = min(rows[1:], key=lambda r: abs(int(r[1]) - target))
    return f"https://web.archive.org/web/{best[1]}/{best[2]}"


def fetch_one(url, year):
    res = cached_fetch(url, "live", CACHE)
    ok = (res.get("status") == 200 and res.get("html")
          and not homepage_redirect(url, res["final_url"])
          and not is_parked(res["html"], res["final_url"]))
    if ok:
        raw, clean = full_pipeline(res["html"])
        if len(clean) >= MIN_TEXT and not looks_like_junk_page(clean):
            return "live", raw, clean
    for wb in filter(None, [wb_avail(url, year), wb_cdx(url, year)]):
        wr = cached_fetch(wb, "wayback", CACHE)
        if wr.get("status") == 200 and wr.get("html"):
            raw, clean = full_pipeline(wr["html"])
            if len(clean) >= MIN_TEXT and not looks_like_junk_page(clean):
                return "wayback", raw, clean
    return "dead", "", ""


def main():
    os.makedirs(CACHE, exist_ok=True)
    df = pd.read_csv(os.path.join(BASE, "wigleaf_labels_fixed.csv"))
    pool = df[(df.tier == "top50") & df.story_url.notna()].to_dict("records")
    out_path = os.path.join(BASE, "top50_texts.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                done.add(json.loads(l)["story_url"])
            except Exception:
                pass
    pool = [r for r in pool if r["story_url"] not in done]
    print(f"Top50 to fetch (remaining): {len(pool)}", flush=True)
    lock = threading.Lock()
    counts = Counter()
    fout = open(out_path, "a")
    n = [0]

    def work(r):
        st, raw, clean = fetch_one(r["story_url"], int(r["year"]))
        rec = {"year": int(r["year"]), "tier": "top50", "title": r["title"],
               "author": r["author"], "magazine": normalize_magazine(r["magazine"]),
               "story_url": r["story_url"], "fetch_status": st, "fetch_source": st,
               "domain": urlparse(r["story_url"]).netloc.lower().removeprefix("www."),
               "raw_text": raw, "text": clean}
        with lock:
            counts[st] += 1
            n[0] += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if n[0] % 50 == 0:
                print(f"[{n[0]}/{len(pool)}] {dict(counts)}", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(as_completed([ex.submit(work, r) for r in pool]))
    fout.close()
    print("DONE top50:", dict(counts), flush=True)


if __name__ == "__main__":
    main()
