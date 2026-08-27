#!/usr/bin/env python3
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from wig_textproc import strip_bio_tail, normalize_text, strip_cms_boilerplate

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = [json.loads(l) for l in open(os.path.join(BASE, "wigleaf_pilot_texts.jsonl"))]
ok = [r for r in rows if r.get("text")]
print("=== bio-tail + cms strip validation on pilot ===")
nstripped = 0
for r in ok[:16]:
    t = r["raw_text"]
    t2 = strip_cms_boilerplate(t)
    t3 = strip_bio_tail(t2)
    removed = len(t2) - len(t3)
    cms_removed = len(t) - len(t2)
    if removed > 10:
        nstripped += 1
    title = r["title"][:32]
    dom = r["domain"][:20]
    print("--- %-32s %-20s raw=%d clean=%d biostrip=%d cmsstrip=%d" % (
        title, dom, len(t), len(t3), removed, cms_removed))
    if removed > 10:
        print("    TAIL:", repr(t2[len(t3):][:150]))
    if cms_removed > 10:
        print("    HEAD:", repr(t[:cms_removed][:150]))
print()
print("bio stripped in %d/16 shown" % nstripped)
