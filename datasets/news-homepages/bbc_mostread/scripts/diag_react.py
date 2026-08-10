#!/usr/bin/env python3
"""Inspect React-era (2024-2025) card markup to find a headline-only selector."""
import sys, re
sys.path.insert(0, ".")
from scrape_bbc_mostread import fetch_capture, cdx
from bs4 import BeautifulSoup

for form, frm, to in [("bbc.com/news", "20250101", "20250110"),
                      ("bbc.com/news", "20240101", "20240110")]:
    rows = cdx(form, frm, to, "timestamp:8", limit=10)
    rows = [r for r in rows if r.get("statuscode") == "200" and "html" in r.get("mimetype","")]
    if not rows:
        print(f"no capture {form} {frm}"); continue
    r = rows[0]
    html = fetch_capture(r["timestamp"], r["original"])
    soup = BeautifulSoup(html, "html.parser")
    print("="*70)
    print(f"{form} {r['timestamp']}")
    sec = soup.find("section", attrs={"data-analytics_group_name": re.compile("most read", re.I)})
    if not sec:
        print("  no most-read section"); continue
    cards = sec.find_all(attrs={"data-testid": "cambridge-card"})
    print(f"  {len(cards)} cambridge-card nodes; dumping testids of first card's descendants:")
    if cards:
        c = cards[0]
        # show structure: every element with a data-testid, plus its text
        for el in c.find_all(attrs={"data-testid": True}):
            t = re.sub(r"\s+"," ", el.get_text(" ", strip=True))[:60]
            print(f"    testid={el['data-testid']:30s} <{el.name}> : {t}")
        # also headings
        for h in c.find_all(["h1","h2","h3","span"]):
            cls = " ".join(h.get("class") or [])
            t = re.sub(r"\s+"," ", h.get_text(" ", strip=True))[:60]
            if t:
                print(f"    <{h.name} class='{cls[:40]}'> : {t}")
