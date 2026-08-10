#!/usr/bin/env python3
import json, sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wig_textproc import (strip_bio_tail, strip_cms_boilerplate,
                          strip_trailing_junk, strip_inline_bio, normalize_text)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = [json.loads(l) for l in open(os.path.join(BASE, "wigleaf_pilot_texts.jsonl"))]
ok = [r for r in rows if r.get("text") and len(r["text"]) > 300]

def pipe(raw):
    c = strip_cms_boilerplate(raw)
    c = strip_trailing_junk(c)
    c = strip_bio_tail(c)
    c = strip_inline_bio(c)
    c = strip_trailing_junk(c)
    return normalize_text(c)

print("=== last 180 chars after FULL strip ===\n")
for r in ok:
    raw = r["raw_text"]
    c = pipe(raw)
    removed = len(raw) - len(c)
    print("### %-34s removed=%d" % (r["title"][:34], removed))
    print("    TAIL:", repr(c[-180:]))
    print()
