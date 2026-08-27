#!/usr/bin/env python3
"""Fetch ALL 905 Wigleaf Top50 story URLs (live -> Wayback fallback), extract
story prose, strip bio tails + CMS boilerplate, normalize presentation.

Reuses the validated pilot fetch logic (parked-domain detection, homepage-redirect
detection, Wayback availability + CDX fallback) and the SHARED wig_textproc
pipeline so Top50 and longlist get identical presentation normalization.

Politeness: <=1 req/sec per domain, academic UA, backoff on 429/503, disk cache.
Output: top50_texts.jsonl  (one row per Top50 entry, fetch_status in
{live,wayback,dead}).
"""
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from urllib.parse import urlparse, quote

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wig_textproc import full_pipeline, looks_like_junk_page, normalize_magazine

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/wigleaf"
CACHE = os.path.join(BASE, "text_cache", "top50")
UA = ("norm-research-academic-crawler/0.2 "
      "(alex2awesome@gmail.com; Stanford research; respects robots, low-rate)")
SLEEP_S = 1.05
MIN_TEXT = 300

PARKED_DOMAINS = {"hugedomains.com", "sedo.com", "sedoparking.com", "dan.com",
                  "afternic.com", "godaddy.com", "namecheap.com", "porkbun.com"}
import re
PARKED_TITLE_PAT = re.compile(
    r"domain (?:is )?for sale|buy this domain|parked free|"
    r"this domain may be for sale|domain expired|seized|account suspended", re.I)

_last_fetch = {}


def is_parked(html_str, final_url):
    dom = urlparse(final_url).netloc.lower().removeprefix("www.")
    if dom in PARKED_DOMAINS:
        return True
    m = re.search(r"<title[^>]*>(.*?)</title>", html_str[:20000], re.S | re.I)
    return bool(m and PARKED_TITLE_PAT.search(m.group(1)))


def looks_like_homepage_redirect(orig_url, final_url):
    op, fp = urlparse(orig_url), urlparse(final_url)
    if len(op.path.strip("/")) > 1 and len(fp.path.strip("/")) <= 1 and not fp.query:
        return True
    return False


def polite_get(session, url, timeout=30):
    domain = urlparse(url).netloc
    el = time.monotonic() - _last_fetch.get(domain, 0)
    if el < SLEEP_S:
        time.sleep(SLEEP_S - el)
    for attempt in range(3):
        try:
            r = session.get(url, headers={"User-Agent": UA}, timeout=timeout,
                            allow_redirects=True)
            _last_fetch[domain] = time.monotonic()
            if r.status_code in (429, 503):
                time.sleep(2.0 * (attempt + 1))
                continue
            return r, None
        except requests.RequestException as e:
            _last_fetch[domain] = time.monotonic()
            if attempt == 2:
                return None, type(e).__name__
            time.sleep(1.0 * (attempt + 1))
    return None, "retries_exhausted"


def cached_fetch(session, url, tag):
    key = hashlib.sha1((tag + url).encode()).hexdigest()[:16]
    cpath = os.path.join(CACHE, key + ".json")
    if os.path.exists(cpath):
        try:
            with open(cpath) as f:
                return json.load(f)
        except Exception:
            pass
    r, err = polite_get(session, url)
    out = {"url": url}
    if r is None:
        out.update(status=None, final_url=None, html=None, error=err)
    else:
        enc = r.apparent_encoding if r.apparent_encoding else "utf-8"
        try:
            r.encoding = enc
        except Exception:
            pass
        out.update(status=r.status_code, final_url=r.url,
                   html=r.text if r.status_code == 200 else None, error=None)
    with open(cpath, "w") as f:
        json.dump(out, f)
    return out


def wayback_lookup(session, url, year):
    api = ("http://archive.org/wayback/available?url=" + quote(url, safe="") +
           f"&timestamp={year}0601")
    res = cached_fetch(session, api, "wbavail")
    if not res.get("html"):
        return None
    try:
        snap = json.loads(res["html"])["archived_snapshots"].get("closest")
    except (json.JSONDecodeError, KeyError, AttributeError):
        return None
    if snap and snap.get("available"):
        return snap["url"].replace("http://web.archive.org", "https://web.archive.org")
    return None


def wayback_cdx_lookup(session, url, year):
    api = ("https://web.archive.org/cdx/search/cdx?url=" + quote(url, safe="") +
           "&output=json&limit=8&filter=statuscode:200&collapse=digest")
    res = cached_fetch(session, api, "wbcdx")
    if not res.get("html"):
        return None
    try:
        rows = json.loads(res["html"])
    except json.JSONDecodeError:
        return None
    if len(rows) < 2:
        return None
    target = int(f"{year}0601000000")
    best = min(rows[1:], key=lambda r: abs(int(r[1]) - target))
    return f"https://web.archive.org/web/{best[1]}/{best[2]}"


def fetch_one(session, url, year):
    """-> (status, raw_text, clean_text). status in {live,wayback,dead}."""
    res = cached_fetch(session, url, "live")
    live_ok = (res.get("status") == 200 and res.get("html")
               and not looks_like_homepage_redirect(url, res["final_url"])
               and not is_parked(res["html"], res["final_url"]))
    if live_ok:
        raw, clean = full_pipeline(res["html"])
        if len(clean) >= MIN_TEXT and not looks_like_junk_page(clean):
            return "live", raw, clean
    for wb in filter(None, [wayback_lookup(session, url, year),
                            wayback_cdx_lookup(session, url, year)]):
        wres = cached_fetch(session, wb, "wayback")
        if wres.get("status") == 200 and wres.get("html"):
            raw, clean = full_pipeline(wres["html"])
            if len(clean) >= MIN_TEXT and not looks_like_junk_page(clean):
                return "wayback", raw, clean
    return "dead", "", ""


def main():
    os.makedirs(CACHE, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(BASE, "wigleaf_labels.csv"))))
    pool = [r for r in rows if r["tier"] == "top50" and r["story_url"]]
    print(f"Top50 to fetch: {len(pool)}", flush=True)
    session = requests.Session()
    out_path = os.path.join(BASE, "top50_texts.jsonl")
    # resume: skip rows already written
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                done.add(json.loads(l)["story_url"])
            except Exception:
                pass
    fout = open(out_path, "a")
    counts = Counter()
    for i, r in enumerate(pool):
        url = r["story_url"]
        if url in done:
            continue
        domain = urlparse(url).netloc.lower().removeprefix("www.")
        status, raw, clean = fetch_one(session, url, int(r["year"]))
        counts[status] += 1
        rec = {"year": int(r["year"]), "tier": "top50", "title": r["title"],
               "author": r["author"],
               "magazine": normalize_magazine(r["magazine"]),
               "story_url": url, "fetch_status": status, "fetch_source": status,
               "domain": domain, "raw_text": raw, "text": clean}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{len(pool)}] {dict(counts)}", flush=True)
    fout.close()
    print("DONE top50:", dict(counts), flush=True)


if __name__ == "__main__":
    main()
