#!/usr/bin/env python3
"""Recover the OUTLET label for the frozen homepage-curation population without
touching the population itself.

Why this is needed: `outlet` was stripped at the very first save of the homepage
dataset (it leaks hard -- BBC 34% top-half vs Washington Post 73%), so every
shipped CSV from `homepage_newsworthiness.csv.gz` down to
`homepage_newsworthiness_clean_v9.csv.gz` carries only `text, judgement,
snapshot_id`. The registry's journalism-curation row is nevertheless specified
OUTLET-HELD-OUT, so an outlet column has to come back from somewhere.

Method (no reprocessing of the labelling pipeline, so no risk of perturbing the
frozen population): the raw Internet Archive dumps are still on disk, one
directory per outlet, and each `*.hyperlinks.json` lists the link texts rendered
on that homepage capture. Build, per outlet, the SET of normalised link texts it
ever showed; then assign each `snapshot_id` the outlet whose set covers the most
of that snapshot's headlines. A snapshot is one outlet's capture, so the
assignment is a plurality vote over its own rows -- individual wire headlines
shared between outlets cannot flip it.

Reported per snapshot: winning outlet, its coverage share, and the runner-up's
share. Snapshots whose margin is below `--min-margin` are labelled `ambiguous`
and must be dropped from any outlet-held-out arm.

Usage (CPU only, run on sk3):
  python3 recover_outlet_map.py \
     --raw datasets/news-homepages/raw_data \
     --pop datasets/news-homepages/homepage_newsworthiness_clean_v9.csv.gz \
     --out datasets/news-homepages/outlet_map_v9.csv.gz
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

NORM_RE = re.compile(r"[^a-z0-9 ]+")
WS_RE = re.compile(r"\s+")
KEY_CHARS = 48


def norm(s: str) -> str:
    s = NORM_RE.sub(" ", (s or "").lower())
    return WS_RE.sub(" ", s).strip()[:KEY_CHARS]


def outlet_key_sets(raw: Path):
    sets = {}
    for d in sorted(p for p in raw.iterdir() if p.is_dir()):
        keys = set()
        n_files = 0
        for f in d.glob("*.hyperlinks.json"):
            n_files += 1
            try:
                links = json.load(open(f))
            except Exception:
                continue
            for l in links:
                k = norm(l.get("text", ""))
                if len(k) >= 12:
                    keys.add(k)
        sets[d.name] = keys
        print(f"[{d.name}] {n_files} snapshots -> {len(keys):,} distinct headline keys",
              flush=True)
    return sets


def headline_of(text: str) -> str:
    t = str(text)
    head = t.split("\n\nCONTEXT: ", 1)[0]
    return head.removeprefix("HEADLINE: ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--pop", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-margin", type=float, default=0.15)
    a = ap.parse_args()

    sets = outlet_key_sets(Path(a.raw))
    df = pd.read_csv(a.pop)
    df["_key"] = [norm(headline_of(t)) for t in df["text"]]
    print(f"population n={len(df):,} snapshots={df['snapshot_id'].nunique():,}")

    votes = defaultdict(Counter)
    seen = defaultdict(int)
    for sid, key in zip(df["snapshot_id"], df["_key"]):
        seen[sid] += 1
        if len(key) < 12:
            continue
        for o, ks in sets.items():
            if key in ks:
                votes[sid][o] += 1

    rows = []
    for sid, n in seen.items():
        c = votes.get(sid, Counter())
        ranked = c.most_common(2)
        top, top_n = (ranked[0] if ranked else ("ambiguous", 0))
        second_n = ranked[1][1] if len(ranked) > 1 else 0
        share = top_n / n
        margin = (top_n - second_n) / n
        outlet = top if (top_n > 0 and margin >= a.min_margin) else "ambiguous"
        rows.append({"snapshot_id": sid, "outlet": outlet, "n_rows": n,
                     "top_share": share, "margin": margin,
                     "runner_up": ranked[1][0] if len(ranked) > 1 else ""})
    m = pd.DataFrame(rows)
    m.to_csv(a.out, index=False)
    res = m.merge(df.groupby("snapshot_id").size().rename("rows").reset_index(),
                  on="snapshot_id")
    by = res.groupby("outlet")["rows"].agg(["size", "sum"]).rename(
        columns={"size": "snapshots", "sum": "rows"})
    print(by.to_string())
    amb = int(res.loc[res.outlet == "ambiguous", "rows"].sum())
    print(f"ambiguous rows: {amb:,} / {len(df):,} ({amb/len(df):.3%})")
    print(f"median top_share among resolved: "
          f"{m.loc[m.outlet!='ambiguous','top_share'].median():.3f}")
    print("OUTLET_MAP_DONE")


if __name__ == "__main__":
    main()
