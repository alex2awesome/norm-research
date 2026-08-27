#!/usr/bin/env python3
"""Parse the RoyalRoad deep fiction pages (fiction_pages_raw/<id>.html.gz,
fetched by scrape_deep_metrics.py) into one JSONL row per fiction.

Extracted per page (ld+json Book blob + stats block):
  fiction_id, title, description_text (HTML stripped, whitespace collapsed),
  date_created, date_modified, genres (from genre= URL params),
  rating_value, rating_count, pages, total_views, average_views,
  followers_deep, favorites, is_404.

Output: rr_deep_parsed.jsonl.gz (append-safe rebuild: writes to a temp file
then renames; never edits raw HTML).
"""
import glob
import gzip
import html
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "fiction_pages_raw")
OUT = os.path.join(HERE, "rr_deep_parsed.jsonl.gz")

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
LDJSON_RE = re.compile(r'application/ld\+json">(.*?)</script>', re.S)
STAT_RES = {
    "total_views": re.compile(r"Total Views\s*:?\s*</[^>]+>\s*<[^>]+>\s*([\d,]+)", re.S),
    "average_views": re.compile(r"Average Views\s*:?\s*</[^>]+>\s*<[^>]+>\s*([\d,]+)", re.S),
    "followers_deep": re.compile(r"Followers\s*:?\s*</[^>]+>\s*<[^>]+>\s*([\d,]+)", re.S),
    "favorites": re.compile(r"Favorites\s*:?\s*</[^>]+>\s*<[^>]+>\s*([\d,]+)", re.S),
    "ratings_n": re.compile(r"Ratings\s*:?\s*</[^>]+>\s*<[^>]+>\s*([\d,]+)", re.S),
}


def strip_html(s):
    return WS_RE.sub(" ", TAG_RE.sub(" ", html.unescape(s or ""))).strip()


def parse_one(fid, text):
    row = {"fiction_id": fid, "is_404": False}
    if "<!-- 404 -->" in text[:200]:
        row["is_404"] = True
        return row
    m = LDJSON_RE.search(text)
    if m:
        try:
            d = json.loads(m.group(1))
        except json.JSONDecodeError:
            d = {}
        row["title"] = d.get("name")
        row["description_text"] = strip_html(d.get("description", ""))
        row["date_created"] = d.get("dateCreated")
        row["date_modified"] = d.get("dateModified")
        genres = []
        for g in d.get("genre") or []:
            mm = re.search(r"genre=([\w%\-]+)", g)
            if mm:
                genres.append(mm.group(1))
        row["genres"] = genres
        ar = d.get("aggregateRating") or {}
        row["rating_value"] = ar.get("ratingValue")
        row["rating_count"] = ar.get("ratingCount")
        row["pages"] = d.get("numberOfPages")
        ic = d.get("interactionStatistic") or {}
        row["total_views_ld"] = ic.get("userInteractionCount")
    for k, rx in STAT_RES.items():
        mm = rx.search(text)
        row[k] = int(mm.group(1).replace(",", "")) if mm else None
    return row


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.html.gz")))
    print(f"{len(files)} raw pages")
    tmp = OUT + ".tmp"
    n_ok = n_404 = n_nodesc = 0
    with gzip.open(tmp, "wt") as out:
        for i, f in enumerate(files):
            fid = int(os.path.basename(f).split(".")[0])
            try:
                text = gzip.open(f, "rt", errors="replace").read()
            except Exception as e:
                print(f"fid {fid}: read error {e}")
                continue
            row = parse_one(fid, text)
            if row["is_404"]:
                n_404 += 1
            elif not row.get("description_text"):
                n_nodesc += 1
            else:
                n_ok += 1
            out.write(json.dumps(row) + "\n")
            if i % 2000 == 0:
                print(f"{i}/{len(files)}", flush=True)
    os.replace(tmp, OUT)
    print(f"DONE ok={n_ok} 404={n_404} nodesc={n_nodesc} -> {OUT}")


if __name__ == "__main__":
    main()
