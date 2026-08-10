#!/usr/bin/env python3
"""Parallel longlist recovery via Wayback CDX + slug match. Per-domain throttle.

Groups longlist entries BY DOMAIN so each domain's CDX index is fetched once,
then matches all its stories. Domains run in parallel (thread pool); archive.org
is throttled by the shared engine.
"""
import hashlib
import json
import os
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, quote, unquote

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_engine import cached_fetch
from wig_textproc import full_pipeline, looks_like_junk_page, normalize_magazine
from recover_longlist import (build_mag2dom, domain_for, slugify, title_tokens,
                              match_in_cdx, content_ok, CURATED)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/wigleaf"
CACHE = os.path.join(BASE, "text_cache", "longlist")
CDX_CACHE = os.path.join(BASE, "text_cache", "cdx")
MIN_TEXT = 300
WORKERS = 6   # gentle on archive.org (it refused under 12)

import time as _time
from fetch_engine import polite_get


def cdx_urls(domain):
    """All archived 200/text-html URLs on a domain. Caches ONLY non-empty,
    non-error results (empty = likely a transient archive.org failure)."""
    key = hashlib.sha1(("cdx2" + domain).encode()).hexdigest()[:16]
    cpath = os.path.join(CDX_CACHE, key + ".json")
    if os.path.exists(cpath):
        try:
            d = json.load(open(cpath))
            if isinstance(d, list):
                return d
        except Exception:
            pass
    api = ("https://web.archive.org/cdx/search/cdx?url=" + quote(domain, safe="") +
           "*&output=json&collapse=urlkey&fl=timestamp,original,statuscode,mimetype"
           "&filter=statuscode:200&filter=mimetype:text/html&limit=20000")
    rows, ok = [], False
    for attempt in range(4):
        r, err = polite_get(api, timeout=90)
        if r is not None and r.status_code == 200:
            try:
                data = json.loads(r.text)
                rows = data[1:] if len(data) > 1 else []
                ok = True
                break
            except Exception:
                pass
        _time.sleep(2.0 * (attempt + 1))
    # cache ONLY a confirmed result (even if genuinely empty for that domain);
    # do NOT cache a transient failure
    if ok:
        tmp = cpath + f".tmp{threading.get_ident()}"
        json.dump(rows, open(tmp, "w"))
        os.replace(tmp, cpath)
    return rows


def _fetch_snapshot(wb_url):
    """Fetch a Wayback snapshot, retrying transient archive.org failures and
    caching only confirmed 200 bodies (so a transient refusal isn't cached)."""
    key = hashlib.sha1(("ll_wb" + wb_url).encode()).hexdigest()[:16]
    cpath = os.path.join(CACHE, key + ".json")
    if os.path.exists(cpath):
        try:
            d = json.load(open(cpath))
            if d.get("status") == 200 and d.get("html"):
                return d
        except Exception:
            pass
    for attempt in range(3):
        r, err = polite_get(wb_url, timeout=60)
        if r is not None and r.status_code == 200 and r.text:
            try:
                r.encoding = r.apparent_encoding or "utf-8"
            except Exception:
                pass
            out = {"url": wb_url, "status": 200, "html": r.text}
            tmp = cpath + f".tmp{threading.get_ident()}"
            json.dump(out, open(tmp, "w"))
            os.replace(tmp, cpath)
            return out
        if r is not None and r.status_code in (404, 403):
            return {"status": r.status_code, "html": None}
        _time.sleep(2.0 * (attempt + 1))
    return {"status": None, "html": None}


def recover_one(title, author, magazine, year, domain):
    if not domain:
        return "not_found", "", "", ""
    rows = cdx_urls(domain)
    m = match_in_cdx(rows, title, year)
    if m:
        wb_url, score = m
        res = _fetch_snapshot(wb_url)
        if res.get("status") == 200 and res.get("html"):
            raw, clean = full_pipeline(res["html"])
            if (len(clean) >= MIN_TEXT and not looks_like_junk_page(clean)
                    and content_ok(clean, title, author)):
                return "wayback", wb_url, raw, clean
    return "not_found", "", "", ""


def main():
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(CDX_CACHE, exist_ok=True)
    mag2dom, alias_map, alias_fn = build_mag2dom()
    df = pd.read_csv(os.path.join(BASE, "wigleaf_labels_fixed.csv"))
    ll = df[df.tier == "longlist"].copy()
    out_path = os.path.join(BASE, "longlist_texts.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                d = json.loads(l)
                done.add((d["title"], d["author"], int(d["year"])))
            except Exception:
                pass
    records = []
    for _, r in ll.iterrows():
        key = (r["title"], r["author"], int(r["year"]))
        if key in done:
            continue
        dom = domain_for(r["magazine"], mag2dom, alias_map, alias_fn)
        records.append((r, dom))
    # group by domain so each domain's CDX is built once (serially within group),
    # but DIFFERENT domains run in parallel
    by_dom = defaultdict(list)
    for r, dom in records:
        by_dom[dom or "__nodomain__"].append(r)
    print(f"Longlist to recover (remaining): {len(records)} across {len(by_dom)} domains", flush=True)

    lock = threading.Lock()
    counts = Counter()
    n = [0]
    fout = open(out_path, "a")

    def do_domain(dom, rws):
        local = []
        real_dom = None if dom == "__nodomain__" else dom
        for r in rws:
            st, url, raw, clean = recover_one(
                str(r["title"]), str(r["author"]), str(r["magazine"]),
                int(r["year"]), real_dom)
            rec = {"year": int(r["year"]), "tier": "longlist", "title": r["title"],
                   "author": r["author"], "magazine": normalize_magazine(r["magazine"]),
                   "story_url": url, "fetch_status": st, "fetch_source": st,
                   "domain": real_dom or "", "raw_text": raw, "text": clean}
            local.append((st, rec))
        with lock:
            for st, rec in local:
                counts[st] += 1
                n[0] += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{n[0]}/{len(records)}] dom={dom[:30]} +{len(rws)} {dict(counts)}", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(do_domain, dom, rws) for dom, rws in
                sorted(by_dom.items(), key=lambda kv: -len(kv[1]))]
        list(as_completed(futs))
    fout.close()
    print("DONE longlist:", dict(counts), flush=True)


if __name__ == "__main__":
    main()
