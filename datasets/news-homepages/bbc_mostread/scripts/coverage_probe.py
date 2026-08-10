#!/usr/bin/env python3
"""
Map Wayback coverage of the BBC dedicated Most-Read page across years.

The dedicated page lives at (over time):
  bbc.com/news/popular/read         (HTML page, ~2014-2019)
  bbc.co.uk/news/popular/read       (HTML page)
  bbc.co.uk/news/popular/read.fragment  (AJAX fragment, sparse)
  bbc.com/news/popular/read         later may 404/redirect

Print per-year capture counts (collapsed to daily) for each URL form so we can
estimate the achievable scale (n distinct capture-days) before committing to a
full crawl.
"""
import sys
import time
import urllib.parse
from collections import Counter
import requests

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 "
      "(research crawler; contact alex2awesome@gmail.com)")
S = requests.Session(); S.headers.update({"User-Agent": UA})
_last = [0.0]
def throttle():
    dt = time.time() - _last[0]
    if dt < 1.1: time.sleep(1.1 - dt)
    _last[0] = time.time()
def get(url, mr=6, to=60):
    for a in range(mr):
        throttle()
        try: r = S.get(url, timeout=to)
        except requests.RequestException: time.sleep(2**a); continue
        if r.status_code in (429,500,502,503,504): time.sleep(min(60,2**a*3)); continue
        return r
    return None

def cdx(url, collapse="timestamp:8"):
    base = "http://web.archive.org/cdx/search/cdx"
    p = {"url": url, "output": "json", "from": "20130101", "to": "20251231",
         "collapse": collapse, "fl": "timestamp,original,statuscode,mimetype"}
    r = get(base + "?" + urllib.parse.urlencode(p))
    if not r or r.status_code != 200:
        print(f"  CDX FAIL {url}: {r.status_code if r else 'none'}", file=sys.stderr)
        return []
    try: d = r.json()
    except Exception: return []
    if not d: return []
    return [dict(zip(d[0], row)) for row in d[1:]]

forms = [
    "bbc.com/news/popular/read",
    "www.bbc.com/news/popular/read",
    "bbc.co.uk/news/popular/read",
    "www.bbc.co.uk/news/popular/read",
    "bbc.co.uk/news/popular/read.fragment",
    "www.bbc.co.uk/news/popular/read.fragment",
]
grand_days = set()
for f in forms:
    rows = cdx(f, collapse="timestamp:8")  # daily collapse
    ok = [r for r in rows if r.get("statuscode") == "200"]
    by_year = Counter(r["timestamp"][:4] for r in ok)
    days = set(r["timestamp"][:8] for r in ok)
    grand_days |= days
    print(f"\n{f}")
    print(f"  total 200 daily-captures: {len(ok)}   distinct days: {len(days)}")
    print(f"  by year: {dict(sorted(by_year.items()))}")
print(f"\n=== UNION distinct capture-days across all forms: {len(grand_days)} ===")
yrs = Counter(d[:4] for d in grand_days)
print(f"union by year: {dict(sorted(yrs.items()))}")
