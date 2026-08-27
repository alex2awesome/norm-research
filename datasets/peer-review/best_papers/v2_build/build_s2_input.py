#!/usr/bin/env python3
"""Build the S2 fetch input: all award papers + a capped, stable non-award
sample per venue-year. Output schema matches what s2_title_join.py expects:
columns venue,year,title,dblp_key (used as the cache key).

Why cap: the full DBLP pool is ~198K papers; at ~1.1s/S2 request that is
~60h. We need citations for (a) every award (y=1) and (b) enough same-vy
non-awards to (i) form a y=0 set and (ii) support citation-decile matching.
A per-vy cap of CAP non-awards keeps the fetch tractable (~28K papers, ~9h)
while leaving plenty for decile matching (awards are 1-3 per vy).

Sampling is by stable md5(dblp_key) rank so it is reproducible and resumable.
"""
import csv
import hashlib

import pandas as pd

CAP = 40  # non-awards per venue-year


def h(s):
    return int(hashlib.md5(str(s).encode()).hexdigest(), 16)


def main():
    pool = pd.read_csv("dblp_pool.csv")
    awk = pd.read_csv("award_dblp_keys.csv")
    award_keys = set(awk.dblp_key.tolist())
    award_vy = set(map(tuple, awk[["venue", "year"]].drop_duplicates().values.tolist()))

    pool["is_award"] = pool.dblp_key.isin(award_keys)
    # only keep venue-years that actually contain a matched award
    pool["vy"] = list(zip(pool.venue, pool.year))
    pool = pool[pool.vy.isin(award_vy)].copy()

    rows = []
    for (v, y), g in pool.groupby(["venue", "year"]):
        aw = g[g.is_award]
        non = g[~g.is_award].copy()
        non["hr"] = non.dblp_key.map(h)
        non = non.sort_values("hr").head(CAP)
        for r in pd.concat([aw, non]).itertuples():
            rows.append({"venue": r.venue, "year": r.year, "title": r.title,
                         "dblp_key": r.dblp_key,
                         "is_award": int(r.is_award)})
    out = pd.DataFrame(rows)
    out.to_csv("s2_input.csv", index=False)
    print(f"s2_input: {len(out)} papers "
          f"({int(out.is_award.sum())} awards + {int((~out.is_award.astype(bool)).sum())} non-awards) "
          f"across {out[['venue','year']].drop_duplicates().shape[0]} venue-years")


if __name__ == "__main__":
    main()
