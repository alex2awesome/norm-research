#!/usr/bin/env python3
"""Assemble the S2 reuse cache for the full acad-C re-fetch.

Merges the two existing S2 caches (NeurIPS-2023 build cache + best_papers
cache) and matches the 29,228 DBLP membership rows against them by normalized
title (accepted records only). Writes:
  - s2_reuse_cache.jsonl : one accepted S2 record per matched DBLP key,
    re-keyed to the DBLP key and re-stamped with the DBLP target_venue/year,
    so the downstream build + the fetch-resume both see it as already-cached.
  - s2_todo.csv : the genuinely-missing DBLP rows (venue,year,title,dblp_key)
    to feed to the fetcher.

Reports cached-vs-to-fetch counts overall and per venue/year.
"""
import json
import re
import unicodedata

import pandas as pd

HERE = "/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/openalex_citations"
BP = "/lfs/skampere3/0/alexspan/norm-research/datasets/peer-review/best_papers/v2_build/s2_cache.jsonl"
DBLP = f"{HERE}/dblp_papers.csv"
N23_CACHE = f"{HERE}/s2_cache.jsonl"
REUSE_OUT = f"{HERE}/s2_reuse_cache.jsonl"
TODO_OUT = f"{HERE}/s2_todo.csv"


def norm_text(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = (s.replace("‘", "'").replace("’", "'")
          .replace("“", '"').replace("”", '"')
          .replace("–", "-").replace("—", "-"))
    return s


def nt(s):
    s = norm_text(s).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_accepted(path):
    """Return {ntitle: record} for accepted records (first wins)."""
    d = {}
    try:
        f = open(path)
    except FileNotFoundError:
        return d
    with f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not r.get("accepted"):
                continue
            t = r.get("s2_title") or r.get("title")
            key = nt(t)
            if key and key not in d:
                d[key] = r
    return d


def main():
    dblp = pd.read_csv(DBLP)
    dblp["ntitle"] = dblp.title.map(nt)
    # within-venue-year dup titles: keep first (matches join_and_build)
    dblp = dblp.drop_duplicates(["venue", "year", "ntitle"]).copy()
    print(f"DBLP membership (deduped): {len(dblp)} rows; "
          f"unique ntitle {dblp.ntitle.nunique()}")

    c_n23 = load_accepted(N23_CACHE)
    c_bp = load_accepted(BP)
    print(f"NeurIPS-2023 cache accepted (unique ntitle): {len(c_n23)}")
    print(f"best_papers cache accepted (unique ntitle): {len(c_bp)}")
    merged = dict(c_bp)
    merged.update(c_n23)  # prefer the acad-C-native cache on collision
    print(f"merged accepted (unique ntitle): {len(merged)}")

    reuse_rows = []
    todo_rows = []
    for r in dblp.itertuples():
        rec = merged.get(r.ntitle)
        if rec is not None:
            # re-stamp onto this DBLP key/target so build + resume see it
            out = dict(rec)
            out["key"] = str(r.dblp_key)
            out["title"] = str(r.title)
            out["target_year"] = int(r.year)
            out["target_venue"] = str(r.venue)
            reuse_rows.append(out)
        else:
            todo_rows.append({
                "venue": r.venue, "year": int(r.year),
                "title": str(r.title), "dblp_key": str(r.dblp_key),
            })

    with open(REUSE_OUT, "w") as f:
        for r in reuse_rows:
            f.write(json.dumps(r) + "\n")
    pd.DataFrame(todo_rows).to_csv(TODO_OUT, index=False)

    print(f"\n=== REUSE SUMMARY ===")
    print(f"already cached (reused): {len(reuse_rows)} / {len(dblp)} "
          f"({len(reuse_rows)/len(dblp)*100:.1f}%)")
    print(f"to fetch (missing):     {len(todo_rows)} / {len(dblp)}")
    print(f"wrote reuse cache: {REUSE_OUT}")
    print(f"wrote todo:        {TODO_OUT}")

    rdf = pd.DataFrame(reuse_rows)
    tdf = pd.DataFrame(todo_rows)
    print("\nreused per venue/year:")
    if len(rdf):
        print(rdf.groupby(["target_venue", "target_year"]).size().to_string())
    print("\nto-fetch per venue/year:")
    print(tdf.groupby(["venue", "year"]).size().to_string())


if __name__ == "__main__":
    main()
