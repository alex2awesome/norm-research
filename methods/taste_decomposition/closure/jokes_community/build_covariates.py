#!/usr/bin/env python3
"""OBSERVED COVARIATE BUILD for the reddit-jokes community closure cell.

FREEZE ADDENDUM 4 (position-in-container) asks every campaign to carry an ordinal /
position covariate for its corpus.  An r/Jokes post has no container of siblings the way
a math.SE answer has its question or a patent claim has its claim-set, so the honest
analogue on this corpus is POSITION IN THE SUBREDDIT'S OWN TIMELINE: when the post was
made, i.e. which era of the subreddit's conventions, meme stock and repost cycle it comes
from.  That is a genuine ordinal in a container (the subreddit's posting stream), and it
is exactly the family that has produced the programme's strongest spurious findings
elsewhere (patents claim ordinal, code repo-recency).

The A/V population (`datasets/humor/reddit_jokes/va/population.csv.gz`) carries only
row_id / group / topic / text / judgement, so the timestamp has to be recovered from the
raw scrape `datasets/humor/reddit_jokes_1m.csv.gz` (despite the name, a PLAIN CSV).

JOIN.  `row_id = sha1(text)[:20]` and the modelling text is `title + " " + selftext`
(or the bare `title` when selftext is empty).  Both forms are hashed and matched against
the population's row_ids; the join is therefore exact, not fuzzy, and unmatched rows are
reported rather than imputed.

WHAT IS AND IS NOT CARRIED.  `created_utc` only.  The scrape also carries `score`, which
is the quantity y is DEFINED from (top vs bottom quartile within a
length-bin x format x topic stratum); pulling it into the campaign would put the label
one join away from every readout, so it is deliberately NOT written to disk here.

CPU only.  Usage: python3 build_covariates.py
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
POP = REPO / "datasets" / "humor" / "reddit_jokes" / "va" / "population.csv.gz"
RAW = REPO / "datasets" / "humor" / "reddit_jokes_1m.csv.gz"   # plain CSV despite the name
OUT = HERE / "jokes_community_covariates.csv"

csv.field_size_limit(10 ** 9)


def sha(t: str) -> str:
    return hashlib.sha1(t.encode()).hexdigest()[:20]


def main():
    want = {}
    with gzip.open(POP, "rt") as fh:
        for r in csv.DictReader(fh):
            want[r["row_id"]] = None
    print(f"population rows: {len(want)}")

    n_scanned, n_hit, dupes = 0, 0, 0
    with open(RAW) as fh:
        for r in csv.DictReader(fh):
            n_scanned += 1
            title, self_ = r.get("title") or "", r.get("selftext") or ""
            for cand in (title + " " + self_, title):
                h = sha(cand)
                if h in want:
                    if want[h] is None:
                        want[h] = r.get("created_utc")
                        n_hit += 1
                    else:
                        dupes += 1
                    break
    matched = {k: v for k, v in want.items() if v is not None}
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row_id", "created_utc"])
        for k, v in want.items():
            w.writerow([k, v if v is not None else ""])
    rep = {
        "raw_rows_scanned": n_scanned,
        "population_rows": len(want),
        "matched": len(matched),
        "match_rate": len(matched) / len(want),
        "later_duplicate_hits": dupes,
        "columns_carried": ["created_utc"],
        "columns_deliberately_omitted": ["score (defines y)"],
        "join": 'row_id = sha1(title + " " + selftext)[:20], falling back to sha1(title)[:20]',
    }
    (HERE / "jokes_community_covariates.report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
