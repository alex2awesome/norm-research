#!/usr/bin/env python3
"""Re-apply the (updated) presentation pipeline to already-fetched texts.

Reprocesses the `text` field from `raw_text` in top50_texts.jsonl and
longlist_texts.jsonl using the current wig_textproc functions (stronger nav/CMS
stripping, bio/junk tails). IDENTICAL pipeline for both classes. Does not refetch.
Overwrites the `text` field in place; keeps `raw_text` as the audit trail.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wig_textproc import (strip_cms_boilerplate, strip_trailing_junk,
                          strip_bio_tail, strip_inline_bio, normalize_text,
                          looks_like_junk_page)

BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/creative-writing/wigleaf"


def reproc(raw):
    c = strip_cms_boilerplate(raw)
    c = strip_trailing_junk(c)
    c = strip_bio_tail(c)
    c = strip_inline_bio(c)
    c = strip_trailing_junk(c)
    return normalize_text(c)


for fn in ["top50_texts.jsonl", "longlist_texts.jsonl"]:
    p = os.path.join(BASE, fn)
    if not os.path.exists(p):
        continue
    rows = [json.loads(l) for l in open(p)]
    n_changed = 0
    for r in rows:
        raw = r.get("raw_text") or ""
        if not raw:
            continue
        new = reproc(raw)
        if looks_like_junk_page(new):
            new = ""  # drop junk-page extractions consistently
        if new != r.get("text"):
            n_changed += 1
        r["text"] = new
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with_text = sum(1 for r in rows if r.get("text") and len(r["text"]) >= 300)
    print(f"{fn}: {len(rows)} rows, changed {n_changed}, with text>=300 now {with_text}")
