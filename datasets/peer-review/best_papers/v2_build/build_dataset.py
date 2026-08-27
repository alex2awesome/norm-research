#!/usr/bin/env python3
"""Build best_papers_v2 dataset from DBLP pool + S2 citations/abstracts.

y=1 = best-paper award papers; y=0 = same venue-year non-award papers (DBLP
pool minus awards). Require an abstract (from S2). X = TITLE + ABSTRACT,
presentation-normalized (NFKC, curly->straight, dashes, ellipsis, collapse
ws); raw kept too. Metadata: cited_by_count (S2 citationCount), venue, year.

Stable hash split by md5(dblp_key)%10: 0-7=train, 8=eval, 9=test.
"""
import argparse
import hashlib
import json
import re
import unicodedata

import pandas as pd


def norm_pres(s):
    """Presentation normalization: NFKC, curly->straight, dashes, ellipsis,
    collapse whitespace. Returns cleaned text."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = (s.replace("‘", "'").replace("’", "'")
          .replace("“", '"').replace("”", '"')
          .replace("–", "-").replace("—", "-").replace("−", "-")
          .replace("…", "..."))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_of(key):
    b = int(hashlib.md5(str(key).encode()).hexdigest(), 16) % 10
    return "train" if b <= 7 else ("eval" if b == 8 else "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="s2_cache.jsonl")
    ap.add_argument("--input", default="s2_input.csv")
    ap.add_argument("--out", default="best_papers_v2_full.csv.gz")
    args = ap.parse_args()

    # S2 cache: key -> record (last write wins)
    s2 = {}
    with open(args.cache) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            s2[r["key"]] = r

    inp = pd.read_csv(args.input)
    rows = []
    n_no_s2 = n_rej = n_no_abs = 0
    for r in inp.itertuples():
        key = str(r.dblp_key)
        rec = s2.get(key)
        if rec is None:
            n_no_s2 += 1
            continue
        if not rec.get("accepted"):
            n_rej += 1
            continue
        abstract = rec.get("abstract") or ""
        if not abstract.strip():
            n_no_abs += 1
            continue
        title_raw = str(r.title)
        title = norm_pres(title_raw)
        abs_n = norm_pres(abstract)
        text = (title + "\n\n" + abs_n).strip()
        rows.append({
            "dblp_key": key,
            "venue": r.venue, "year": int(r.year),
            "label": int(r.is_award),
            "title_raw": title_raw, "abstract_raw": abstract,
            "title": title, "abstract": abs_n,
            "text": text,
            "cited_by_count": rec.get("citationCount"),
            "s2_paper_id": rec.get("s2_paper_id"),
            "s2_year": rec.get("s2_year"),
            "split": split_of(key),
        })
    df = pd.DataFrame(rows)
    # drop rows with null citation (rare) -- keep but fill 0 for metadata use
    df["cited_by_count"] = pd.to_numeric(df["cited_by_count"], errors="coerce")
    df.to_csv(args.out, index=False, compression="gzip")

    print(f"input papers: {len(inp)}")
    print(f"  no S2 record (not fetched yet): {n_no_s2}")
    print(f"  S2 rejected: {n_rej}")
    print(f"  no abstract: {n_no_abs}")
    print(f"  KEPT: {len(df)}  (y=1 {int(df.label.sum())}, y=0 {int((df.label==0).sum())})")
    print(f"  venue-years with >=1 award AND >=1 non-award (usable for within-vy):")
    vy = df.groupby(["venue", "year"]).agg(pos=("label", "sum"),
                                           n=("label", "size")).reset_index()
    usable = vy[(vy.pos >= 1) & (vy.n - vy.pos >= 1)]
    print(f"    {len(usable)} venue-years; awards in them = {int(usable.pos.sum())}")
    print(f"  split counts:")
    print(df.groupby("split").label.agg(["size", "sum"]).to_string())
    df.to_csv(args.out, index=False, compression="gzip")


if __name__ == "__main__":
    main()
