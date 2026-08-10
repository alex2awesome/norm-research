#!/usr/bin/env python3
"""
Deeper diagnostic: find the REAL Most Read module per era.

Findings from feasibility.py:
  - 2013-2015 bbc.co.uk/news: my parser caught a "1: ... 2: ..." carousel (NOT most read)
  - 2016-2017: Most Read is an AJAX fragment (/news/popular/read.fragment) -> not inline
  - 2018-2023: my parser caught "Video ..." carousel captions (NOT most read)
  - 2024-2025: caught nav "Innovation"

So: (A) check whether the /news/popular/read.fragment is itself archived (the real
list for the 2014-2019 era lived there), and (B) dump the raw HTML around every
"most read"/"most popular" marker per capture so we can see the true container and
write correct selectors.
"""
import re
import sys
import time
import urllib.parse
import requests
from bs4 import BeautifulSoup

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 "
      "(research crawler; contact alex2awesome@gmail.com)")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})
MIN_INTERVAL = 1.1
_last = [0.0]


def throttle():
    dt = time.time() - _last[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _last[0] = time.time()


def get(url, max_retries=5, timeout=60):
    for a in range(max_retries):
        throttle()
        try:
            r = SESSION.get(url, timeout=timeout)
        except requests.RequestException as e:
            time.sleep(2 ** a); continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(60, 2 ** a * 3)); continue
        return r
    return None


def cdx(url, frm, to, collapse="timestamp:8", limit=20):
    base = "http://web.archive.org/cdx/search/cdx"
    params = {"url": url, "output": "json", "from": frm, "to": to,
              "collapse": collapse, "limit": str(limit),
              "fl": "timestamp,original,statuscode,mimetype"}
    r = get(base + "?" + urllib.parse.urlencode(params))
    if not r or r.status_code != 200:
        return []
    try:
        d = r.json()
    except Exception:
        return []
    if not d:
        return []
    return [dict(zip(d[0], row)) for row in d[1:]]


def fetch(ts, orig):
    r = get(f"http://web.archive.org/web/{ts}id_/{orig}")
    return r.text if r else None


# ---------------------------------------------------------------------------
print("### PART A: is /news/popular/read.fragment archived? ###")
for pat in ["bbc.co.uk/news/popular/read.fragment",
            "bbc.com/news/popular/read",
            "bbc.co.uk/news/popular/read",
            "bbc.co.uk/news/0/most_popular_content/read"]:
    rows = cdx(pat, "20130101", "20251231", collapse="timestamp:6", limit=40)
    rows = [r for r in rows if r.get("statuscode") == "200"]
    print(f"\n  {pat}: {len(rows)} captures (200)")
    for r in rows[:6]:
        print(f"    {r['timestamp']}  {r['original']}")
    if rows:
        # fetch one to see the markup
        r0 = rows[len(rows)//2]
        html = fetch(r0["timestamp"], r0["original"])
        if html:
            print(f"    --- sample fragment {r0['timestamp']} ({len(html)} bytes) ---")
            soup = BeautifulSoup(html, "html.parser")
            # show first 8 li/a texts
            for i, a in enumerate(soup.find_all("a")[:10]):
                t = re.sub(r"\s+", " ", a.get_text(" ", strip=True))
                if t:
                    print(f"      a[{i}]: {t[:80]}")
            print(f"      RAW HEAD: {html[:400]}")


# ---------------------------------------------------------------------------
print("\n\n### PART B: raw HTML around 'most read' markers, per era ###")
# pick representative homepage captures (use same eras as before)
homepages = [
    ("20130101002650", "http://www.bbc.co.uk/news/"),
    ("20160101024908", "http://www.bbc.co.uk/news"),
    ("20170101013439", "http://www.bbc.com/news"),
    ("20180101000714", "http://www.bbc.com/news"),
    ("20200101000238", "https://www.bbc.com/news"),
    ("20230101001524", "https://www.bbc.co.uk/news"),
    ("20250101015548", "https://www.bbc.com/news"),
]
for ts, orig in homepages:
    print("\n" + "=" * 76)
    print(f"CAPTURE {ts}  {orig}")
    html = fetch(ts, orig)
    if not html:
        print("  FETCH FAIL"); continue
    low = html.lower()
    # find all marker positions
    for marker in ["most read", "most-read", "mostread", "mostRead".lower()]:
        start = 0
        hits = 0
        while True:
            idx = low.find(marker, start)
            if idx < 0 or hits >= 3:
                break
            hits += 1
            snippet = html[max(0, idx-200):idx+400]
            snippet = re.sub(r"\s+", " ", snippet)
            print(f"  [{marker}] ...{snippet}...")
            start = idx + len(marker)
    # also: count data-component values present
    soup = BeautifulSoup(html, "html.parser")
    dcs = {}
    for el in soup.find_all(attrs={"data-component": True}):
        dcs[el["data-component"]] = dcs.get(el["data-component"], 0) + 1
    if dcs:
        print(f"  data-component values: {dcs}")
