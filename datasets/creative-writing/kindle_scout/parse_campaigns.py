#!/usr/bin/env python3
"""Parse raw Kindle Scout campaign HTML -> analysis-ready campaigns_parsed.jsonl.

Fields: campaign_id, title, author, genres, verdict (from action-bar),
action_msg, excerpt_text (from expand-excerpt capture when present, else the
main page's project-detail-excerpt block), capture ts, sel_carousel flag.
"""
import glob
import gzip
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

VERDICT = {"published": "accept", "selected": "accept",
           "notselected": "reject",
           "voting": "live_at_capture", "reviewvotes": "in_review_at_capture"}


def clean(x):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", x)).strip()


def excerpt_from(h):
    m = re.search(r'project-detail-excerpt.*?</h3>(.*?)</div>\s*</div>', h, re.S)
    if not m:
        return None
    paras = re.findall(r"<p[^>]*>(.*?)</p>", m.group(1), re.S)
    if not paras:
        t = clean(m.group(1))
        return t or None
    return "\n\n".join(clean(p) for p in paras if clean(p))


def main():
    meta = {json.loads(l)["campaign_id"]: json.loads(l)
            for l in open(os.path.join(HERE, "campaigns.jsonl"))}
    out = open(os.path.join(HERE, "campaigns_parsed.jsonl"), "w")
    n = with_excerpt = 0
    for f in sorted(glob.glob(os.path.join(HERE, "raw", "*.html.gz"))):
        base = os.path.basename(f)
        if base.endswith(".excerpt.html.gz"):
            continue
        cid = base[:-8]
        h = gzip.open(f, "rt").read()
        m = re.search(r"<h2[^>]*>\s*([^<]{2,150}?)\s*</h2>", h)
        title = m.group(1).strip() if m else None
        m = re.search(r"[Bb]y\s+([A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,3})\s*<", h)
        author = m.group(1).strip() if m else None
        rec = meta.get(cid, {"campaign_id": cid})
        exc = None
        ef = os.path.join(HERE, "raw", f"{cid}.excerpt.html.gz")
        if os.path.exists(ef):
            exc = excerpt_from(gzip.open(ef, "rt").read())
        if not exc:
            exc = excerpt_from(h)
        row = {"campaign_id": cid, "title": title, "author": author,
               "genres": rec.get("genres", []),
               "verdict": VERDICT.get(rec.get("action_bar"), rec.get("action_bar")),
               "action_msg": rec.get("action_msg"),
               "sel_carousel": rec.get("sel_carousel"),
               "capture": rec.get("capture"),
               "excerpt_text": exc,
               "excerpt_chars": len(exc) if exc else 0}
        out.write(json.dumps(row) + "\n")
        n += 1
        if exc:
            with_excerpt += 1
    out.close()
    print(f"parsed {n} campaigns, {with_excerpt} with excerpt text")


if __name__ == "__main__":
    main()
