#!/usr/bin/env python3
"""Validate the PRODUCTION parsers (imported from scrape_bbc_mostread.py) against
one representative Wayback capture per era. Prints the ranked most-read list +
a few control headlines so we can confirm real most-read text, not carousel/nav."""
import sys
sys.path.insert(0, ".")
from scrape_bbc_mostread import (fetch_capture, parse_most_read,
                                 harvest_other_headlines, cdx)

# (label, url-form-for-cdx, from, to) -- pick one good capture per era
ERAS = [
    ("2014 popular/read page", "bbc.com/news/popular/read", "20140601", "20141231"),
    ("2016 popular/read page", "bbc.co.uk/news/popular/read", "20160101", "20160601"),
    ("2016 .fragment",         "bbc.co.uk/news/popular/read.fragment", "20151201", "20160201"),
    ("2018 homepage Morph",    "bbc.com/news", "20180101", "20180115"),
    ("2020 homepage Morph",    "bbc.com/news", "20200101", "20200110"),
    ("2022 homepage Morph",    "bbc.co.uk/news", "20220101", "20220110"),
    ("2023 homepage Morph",    "bbc.co.uk/news", "20230101", "20230110"),
    ("2024 homepage React",    "bbc.com/news", "20240101", "20240110"),
    ("2025 homepage React",    "bbc.com/news", "20250101", "20250110"),
]

for label, form, frm, to in ERAS:
    print("=" * 78)
    print(f"ERA: {label}   ({form}  {frm}..{to})")
    rows = cdx(form, frm, to, "timestamp:8", limit=10)
    rows = [r for r in rows if r.get("statuscode") == "200" and "html" in r.get("mimetype", "")]
    if not rows:
        print("  NO CAPTURE FOUND in window")
        continue
    r = rows[0]
    html = fetch_capture(r["timestamp"], r["original"])
    if not html:
        print("  FETCH FAIL")
        continue
    parser, items, soup = parse_most_read(html)
    print(f"  capture {r['timestamp']}  bytes={len(html)}  parser={parser}  n={len(items)}")
    for it in items[:10]:
        print(f"    {str(it.get('rank')):>3}. {it['headline'][:88]}   [{it.get('href','')[:40]}]")
    if items:
        others = harvest_other_headlines(soup, [it["href"] for it in items])
        print(f"  -- {len(others)} control (non-most-read) headlines on this page; sample:")
        for o in others[:5]:
            print(f"       o: {o['headline'][:80]}   [{o['href'][:40]}]")
