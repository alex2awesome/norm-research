#!/usr/bin/env python3
"""Inspect the 2025 capture: (a) the rank-7 'years ago' most-read card, and
(b) where the dirty controls ('... 2 hrs ago Cul') come from."""
import sys, re
sys.path.insert(0, ".")
from scrape_bbc_mostread import fetch_capture, cdx, _react_card_headline, _norm_href, ARTICLE_HREF_RE, _txt
from bs4 import BeautifulSoup

rows = cdx("bbc.com/news", "20250101", "20250110", "timestamp:8", limit=5)
rows = [r for r in rows if r.get("statuscode")=="200" and "html" in r.get("mimetype","")]
r = rows[0]
html = fetch_capture(r["timestamp"], r["original"])
soup = BeautifulSoup(html, "html.parser")
print(f"capture {r['timestamp']}")

# (a) the most-read section rank-7 card
sec = soup.find("section", attrs={"data-analytics_group_name": re.compile("most read", re.I)})
cards = sec.find_all(attrs={"data-testid":"cambridge-card"})
print(f"\nmost-read cambridge-cards: {len(cards)}")
for i,c in enumerate(cards):
    h = c.find(attrs={"data-testid":"card-headline"})
    ht = _txt(h) if h else "(no card-headline)"
    print(f"  card[{i}] card-headline raw: {ht[:90]}")

# (b) which links became controls: replicate harvester anchor-fallback and show dirty ones
print("\n--- control anchors with metadata-suffix (generic fallback) ---")
react_keys = set(_norm_href(it["href"]).split("?")[0].rstrip("/") for it in
                 [ _react_card_headline(c) for c in cards] if it)
shown=0
for a in soup.find_all("a", href=True):
    href=_norm_href(a["href"]); key=href.split("?")[0].rstrip("/")
    if not ARTICLE_HREF_RE.search(href) or key in react_keys: continue
    txt=_txt(a)
    if len(txt)<12 or len(txt.split())<3: continue
    # is this anchor inside a cambridge-card with a card-headline?
    card = a.find_parent(attrs={"data-testid":"cambridge-card"})
    has_ch = bool(card and card.find(attrs={"data-testid":"card-headline"}))
    if re.search(r"\b\d+ ?(hrs?|hours?|days?|mins?) ago\b", txt) or "More than" in txt:
        print(f"  DIRTY anchor (in_card={has_ch}): {txt[:85]}")
        if card:
            ch=card.find(attrs={"data-testid":"card-headline"})
            print(f"      -> parent-card card-headline: {(_txt(ch) if ch else None)}")
        shown+=1
    if shown>=8: break
