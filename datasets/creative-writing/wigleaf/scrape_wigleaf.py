#!/usr/bin/env python3
"""Scrape wigleaf.com Top 50 (+ longlist) pages for 2008-2025, with polite
rate-limiting and a local HTML cache so re-runs never re-fetch.

Outputs (staging dir):
  html_cache/<page>.htm        -- raw fetched HTML
  manifest.json                -- per-year page map {year: {main, top50_pages, longlist_pages}}
"""
import json
import os
import re
import sys
import time
from urllib.parse import urljoin, urlparse

import requests

STAGING = "/Users/spangher/Projects/stanford-research/norm-research/.staging/norm-datasets/creative-writing/wigleaf"
CACHE_DIR = os.path.join(STAGING, "html_cache")
BASE = "https://wigleaf.com/"
UA = "norm-research-academic-crawler/0.1 (alex2awesome@gmail.com; Stanford research)"
SLEEP_S = 2.0

_last_fetch = {}  # domain -> monotonic time of last request


def cache_path(url):
    name = os.path.basename(urlparse(url).path) or "homepage.html"
    return os.path.join(CACHE_DIR, name)


def fetch(url, session):
    """Fetch with cache. Returns (html_text, status). Cached pages never re-fetch."""
    p = cache_path(url)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            return f.read(), "cached"
    domain = urlparse(url).netloc
    elapsed = time.monotonic() - _last_fetch.get(domain, 0)
    if elapsed < SLEEP_S:
        time.sleep(SLEEP_S - elapsed)
    r = session.get(url, headers={"User-Agent": UA}, timeout=30)
    _last_fetch[domain] = time.monotonic()
    if r.status_code != 200:
        print(f"  WARN {r.status_code} for {url}", file=sys.stderr)
        return None, r.status_code
    r.encoding = r.apparent_encoding or "utf-8"
    with open(p, "w", encoding="utf-8") as f:
        f.write(r.text)
    return r.text, "fetched"


def links_in(html):
    return re.findall(r'href="([^"]+)"', html, flags=re.IGNORECASE)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    session = requests.Session()

    archive_html, _ = fetch(urljoin(BASE, "archive.htm"), session)
    if archive_html is None:
        sys.exit("Could not fetch archive.htm")

    main_pages = sorted(set(re.findall(r'(\d{2}top50main\.htm)', archive_html)))
    print(f"Found {len(main_pages)} top50 main pages: {main_pages}")

    manifest = {}
    for mp in main_pages:
        yy = int(mp[:2])
        year = 2000 + yy
        html, _ = fetch(urljoin(BASE, mp), session)
        if html is None:
            manifest[str(year)] = {"main": mp, "error": "main page fetch failed"}
            continue
        hrefs = links_in(html)
        top50_first = sorted({h for h in hrefs if re.fullmatch(rf'{year}top50\d\.htm', h)})
        longlist_first = sorted({h for h in hrefs
                                 if re.search(rf'{year}.*long', h, re.I)
                                 or re.search(rf'long.*{year}', h, re.I)})
        # Follow top50 pagination transitively (PAGE 1 2 3 links).
        top50_pages, queue = [], list(top50_first)
        seen = set()
        while queue:
            page = queue.pop(0)
            if page in seen:
                continue
            seen.add(page)
            phtml, _ = fetch(urljoin(BASE, page), session)
            if phtml is None:
                continue
            top50_pages.append(page)
            for h in links_in(phtml):
                if re.fullmatch(rf'{year}top50\d\.htm', h) and h not in seen:
                    queue.append(h)
        top50_pages.sort()
        # Longlist pages (+ any pagination among them).
        longlist_pages, queue, seen = [], list(longlist_first), set()
        while queue:
            page = queue.pop(0)
            if page in seen:
                continue
            seen.add(page)
            phtml, _ = fetch(urljoin(BASE, page), session)
            if phtml is None:
                continue
            longlist_pages.append(page)
            for h in links_in(phtml):
                if re.search(rf'{year}.*long', h, re.I) and h.endswith(".htm") and h not in seen:
                    queue.append(h)
        longlist_pages.sort()
        manifest[str(year)] = {"main": mp, "top50_pages": top50_pages,
                               "longlist_pages": longlist_pages}
        print(f"{year}: top50 pages={top50_pages} longlist pages={longlist_pages}")
        if not top50_pages:
            print(f"  WARN {year}: no top50 list pages found", file=sys.stderr)
        if not longlist_pages:
            print(f"  WARN {year}: no longlist page found", file=sys.stderr)

    with open(os.path.join(STAGING, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {os.path.join(STAGING, 'manifest.json')}")


if __name__ == "__main__":
    main()
