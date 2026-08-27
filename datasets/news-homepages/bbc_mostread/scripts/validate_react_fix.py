#!/usr/bin/env python3
"""Final check of the React fixes on 2024 + 2025 captures."""
import sys, re
sys.path.insert(0, ".")
from scrape_bbc_mostread import fetch_capture, parse_most_read, harvest_other_headlines, cdx

# also import the builder's strip_prefix to confirm end-to-end cleanliness
import importlib.util
spec = importlib.util.spec_from_file_location("bld", "scripts/build_bbc_mostread.py")
bld = importlib.util.module_from_spec(spec); spec.loader.exec_module(bld)

for frm, to in [("20240101","20240110"), ("20250101","20250110")]:
    rows = cdx("bbc.com/news", frm, to, "timestamp:8", limit=5)
    rows = [r for r in rows if r.get("statuscode")=="200" and "html" in r.get("mimetype","")]
    r = rows[0]; html = fetch_capture(r["timestamp"], r["original"])
    parser, items, soup = parse_most_read(html)
    print("="*70); print(f"{r['timestamp']}  parser={parser}  n={len(items)}")
    for it in items:
        clean = bld.strip_prefix(bld.normalize(it["headline"]))
        print(f"  {it['rank']:>2}. {clean[:88]}")
    others = harvest_other_headlines(soup, [it["href"] for it in items])
    print(f"  -- {len(others)} controls; cleaned sample:")
    for o in others[:8]:
        clean = bld.strip_prefix(bld.normalize(o["headline"]))
        print(f"     o: {clean[:80]}")
