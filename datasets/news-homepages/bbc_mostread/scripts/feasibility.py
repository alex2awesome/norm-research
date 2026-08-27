#!/usr/bin/env python3
"""
BBC Most Read feasibility test.

Step 1 of the build. Queries the Wayback CDX API for BBC News homepage captures,
fetches ~10 captures spread across 2013-2025, and tries to parse the "Most Read"
module with per-era selectors. Prints extracted ranked headline lists so we can
confirm we are getting real ranked most-read headlines, not nav junk.

Politeness: <=1 req/sec, retries + backoff on 429/503, real UA with contact email.
"""
import json
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

MIN_INTERVAL = 1.1  # seconds between requests (politeness)
_last_req = [0.0]


def throttle():
    dt = time.time() - _last_req[0]
    if dt < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - dt)
    _last_req[0] = time.time()


def get(url, max_retries=5, timeout=60):
    for attempt in range(max_retries):
        throttle()
        try:
            r = SESSION.get(url, timeout=timeout)
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  [retry {attempt}] {type(e).__name__}: {e}; sleep {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        if r.status_code in (429, 503, 502, 500, 504):
            wait = min(60, 2 ** attempt * 3)
            print(f"  [retry {attempt}] HTTP {r.status_code}; sleep {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        return r
    return None


def cdx_query(url_pattern, frm, to, collapse="timestamp:8", limit=None,
              filters=None, match_type=None):
    base = "http://web.archive.org/cdx/search/cdx"
    params = {
        "url": url_pattern,
        "output": "json",
        "from": frm,
        "to": to,
        "collapse": collapse,
        "fl": "timestamp,original,statuscode,digest,mimetype",
    }
    if limit:
        params["limit"] = str(limit)
    if match_type:
        params["matchType"] = match_type
    qs = urllib.parse.urlencode(params)
    if filters:
        for f in filters:
            qs += "&filter=" + urllib.parse.quote(f)
    r = get(base + "?" + qs)
    if r is None or r.status_code != 200:
        print(f"CDX query failed: {r.status_code if r else 'no response'}", file=sys.stderr)
        return []
    try:
        data = r.json()
    except Exception as e:
        print(f"CDX json parse failed: {e}; body head: {r.text[:200]}", file=sys.stderr)
        return []
    if not data:
        return []
    header = data[0]
    rows = [dict(zip(header, row)) for row in data[1:]]
    return rows


def fetch_capture(timestamp, original):
    # id_ suffix gets the raw original capture without Wayback's rewriting toolbar
    wb = f"http://web.archive.org/web/{timestamp}id_/{original}"
    r = get(wb)
    if r is None:
        return None
    return r.text


# ---------------------------------------------------------------------------
# Per-era Most Read parsers. Each returns a list of (rank, headline) or [].
# ---------------------------------------------------------------------------

def clean_text(s):
    if s is None:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    # strip leading rank numbers like "1." or "1 " that some markups inline
    return s


def parse_modern_datacomponent(soup):
    """2019+? : <div data-component="mostRead"> ... or section/nav with mostRead."""
    out = []
    nodes = soup.select('[data-component="mostRead"]')
    for node in nodes:
        items = node.find_all("li")
        rank = 0
        for li in items:
            a = li.find("a")
            txt = None
            if a:
                txt = a.get_text(" ", strip=True)
            else:
                txt = li.get_text(" ", strip=True)
            txt = clean_text(txt)
            # strip a leading standalone rank digit if present
            txt = re.sub(r"^\d{1,2}\s+", "", txt) if re.match(r"^\d{1,2}\s+\D", txt) else txt
            if txt and len(txt) > 8:
                rank += 1
                out.append((rank, txt))
        if out:
            return out
    return out


def parse_modern_heading(soup):
    """Find a heading 'Most read'/'Most Read'/'Most popular' then the following list."""
    out = []
    headings = soup.find_all(re.compile(r"^h[1-6]$"))
    for h in headings:
        ht = h.get_text(" ", strip=True).lower()
        if "most read" in ht or "most popular" in ht or "most watched" in ht:
            # find nearest following ol/ul, or parent container's list
            container = h.find_parent()
            lst = None
            sib = h.find_next(["ol", "ul"])
            if sib:
                lst = sib
            if lst is None and container:
                lst = container.find(["ol", "ul"])
            if lst:
                rank = 0
                for li in lst.find_all("li"):
                    a = li.find("a")
                    txt = a.get_text(" ", strip=True) if a else li.get_text(" ", strip=True)
                    txt = clean_text(txt)
                    txt = re.sub(r"^\d{1,2}\s+", "", txt) if re.match(r"^\d{1,2}\s+\D", txt) else txt
                    if txt and len(txt) > 8:
                        rank += 1
                        out.append((rank, txt))
                if out:
                    return out
    return out


def parse_old_most_popular(soup):
    """2010-2016 era: div#most-popular, ol.most-popular-list, .mostPopular, etc."""
    out = []
    selectors = [
        "div#most-popular", "#mostPopular", ".most-popular", ".mostPopular",
        "ol.most-popular-list", ".most-popular-list",
        "#blq-most-popular", ".story-most-popular",
        "div.most-popular-read", ".mostread", "#mostread",
    ]
    for sel in selectors:
        for node in soup.select(sel):
            # most-popular module often has tabs (read / shared / watched);
            # grab the "read"/"shared" list. Try ol first.
            lists = node.find_all(["ol", "ul"])
            for lst in lists:
                items = lst.find_all("li")
                if len(items) < 3:
                    continue
                rank = 0
                tmp = []
                for li in items:
                    a = li.find("a")
                    txt = a.get_text(" ", strip=True) if a else li.get_text(" ", strip=True)
                    txt = clean_text(txt)
                    txt = re.sub(r"^\d{1,2}\.?\s+", "", txt) if re.match(r"^\d{1,2}\.?\s+\D", txt) else txt
                    if txt and len(txt) > 8:
                        rank += 1
                        tmp.append((rank, txt))
                if len(tmp) >= 3:
                    return tmp  # first plausible list (usually "most read" tab)
    return out


PARSERS = [
    ("modern_datacomponent", parse_modern_datacomponent),
    ("modern_heading", parse_modern_heading),
    ("old_most_popular", parse_old_most_popular),
]


def parse_all(html):
    soup = BeautifulSoup(html, "html.parser")
    results = {}
    for name, fn in PARSERS:
        try:
            res = fn(soup)
        except Exception as e:
            res = []
            print(f"    parser {name} error: {e}", file=sys.stderr)
        results[name] = res
    return results, soup


def main():
    # Gather homepage captures across eras. BBC news homepage URLs over time:
    #   bbc.co.uk/news  (primary)
    #   news.bbc.co.uk  (older, pre-2011 mostly)
    targets = ["bbc.co.uk/news", "www.bbc.co.uk/news", "bbc.com/news", "www.bbc.com/news"]

    # one CDX call per target, full range, collapse to ~daily
    all_caps = []
    for t in targets:
        rows = cdx_query(t, "20130101", "20251231", collapse="timestamp:6")  # ~monthly
        for r in rows:
            r["_target"] = t
        all_caps.extend(rows)
        print(f"CDX {t}: {len(rows)} captures (monthly-collapsed)", file=sys.stderr)

    # keep only 200 status, text/html
    all_caps = [c for c in all_caps
                if c.get("statuscode") == "200"
                and "html" in c.get("mimetype", "")]
    all_caps.sort(key=lambda c: c["timestamp"])
    print(f"\nTotal 200/html captures: {len(all_caps)}")
    if all_caps:
        print(f"Earliest: {all_caps[0]['timestamp']}  Latest: {all_caps[-1]['timestamp']}")

    # pick ~12 spread across years
    years_seen = {}
    test_caps = []
    target_years = list(range(2013, 2026))
    for c in all_caps:
        yr = int(c["timestamp"][:4])
        if yr in target_years and years_seen.get(yr, 0) < 1:
            years_seen[yr] = years_seen.get(yr, 0) + 1
            test_caps.append(c)
    print(f"\nSelected {len(test_caps)} test captures across years: "
          f"{sorted(years_seen.keys())}\n")

    summary = []
    for c in test_caps:
        ts = c["timestamp"]
        orig = c["original"]
        print("=" * 78)
        print(f"CAPTURE {ts}  ({c['_target']})  {orig}")
        html = fetch_capture(ts, orig)
        if not html:
            print("  FETCH FAILED")
            summary.append((ts, c["_target"], "FETCH_FAIL", 0, None))
            continue
        print(f"  html bytes: {len(html)}")
        results, soup = parse_all(html)
        best = None
        best_name = None
        for name, res in results.items():
            if res and (best is None or len(res) > len(best)):
                best, best_name = res, name
        if best:
            print(f"  >>> parser={best_name}  n={len(best)}")
            for rank, txt in best[:12]:
                print(f"      {rank:2d}. {txt[:90]}")
            summary.append((ts, c["_target"], best_name, len(best), best[0][1][:60] if best else None))
        else:
            print("  >>> NO MOST-READ MODULE FOUND by any parser")
            # diagnostic: look for the words 'most read' / 'most popular' in raw html
            low = html.lower()
            for marker in ["most read", "most popular", "mostread", "most-popular", "most-read"]:
                idx = low.find(marker)
                if idx >= 0:
                    snippet = html[max(0, idx-80):idx+120].replace("\n", " ")
                    print(f"      marker '{marker}' present near: ...{snippet}...")
                    break
            summary.append((ts, c["_target"], "NONE", 0, None))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for ts, tgt, parser, n, first in summary:
        print(f"  {ts}  {tgt:20s}  parser={parser:22s}  n={n:2d}  first={first}")


if __name__ == "__main__":
    main()
