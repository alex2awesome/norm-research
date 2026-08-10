#!/usr/bin/env python3
"""Job 3 (task D7): recover created_utc + within-thread comment rank for the
7,008-row cw_community evaluation-valid population by EXACT-TEXT-JOINING back
to the already-downloaded raw archive datasets/creative-writing/
writingprompts_comments.jsonl.gz (Arctic Shift API mirror, NOT the live Reddit
API -- already on disk, no new fetching).

Pass 1: stream the raw comments file once, build sha1(body) -> (created_utc,
        link_id, score) [list, in case of duplicate story text across
        different threads -- keep all, resolve later]
Pass 2: for each row in cw_honest_population.csv (n=7008, id/prompt_id/story),
        look up sha1(story) in the index.
Pass 3: for ALL raw comments (not just matched ones) sharing a recovered
        link_id, rank by created_utc to get thread_rank (1-indexed, ties
        broken by id order) and thread_size -- this is the pool of
        substantive (>=200 char, non-bot, top-level) submissions per prompt,
        i.e. exactly the same filtering the original download already applied,
        NOT the full raw Reddit thread (short/removed/bot replies are not in
        the pool at all -- documented caveat).

Outputs (this dir):
  cw_match_report.json            coverage stats
  cw_position_covariates.csv      id, created_utc, thread_rank, thread_size,
                                   link_id, thread_rank_frac (rank/size)
"""
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict

csv.field_size_limit(2**31 - 1)

BASE = "/lfs/skampere3/0/alexspan/norm-research"
COMMENTS = BASE + "/datasets/creative-writing/writingprompts_comments.jsonl.gz"
POP = BASE + "/methods/taste_decomposition/closure/cw_community/cw_honest_population.csv"
OUT_COV = BASE + "/methods/taste_decomposition/closure/cw_community/cw_position_covariates.csv"
OUT_REPORT = BASE + "/methods/taste_decomposition/closure/cw_community/cw_match_report.json"


def sha1(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def main():
    print("PASS 1: indexing raw comments by sha1(body) ...", flush=True)
    index = defaultdict(list)  # sha1(body) -> [(created_utc, link_id, score), ...]
    by_link = defaultdict(list)  # link_id -> [(created_utc, comment_sha1), ...] (ALL pool comments, for thread rank)
    n = 0
    with gzip.open(COMMENTS, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            n += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            body = rec.get("body", "")
            cu = rec.get("created_utc")
            lid = rec.get("link_id", "")
            sc = rec.get("score")
            h = sha1(body)
            index[h].append((cu, lid, sc))
            by_link[lid].append((cu, h))
            if n % 300000 == 0:
                print(f"  scanned {n:,} raw comments...", flush=True)
    print(f"PASS 1 done: {n:,} raw comments, {len(index):,} distinct bodies, {len(by_link):,} distinct link_ids",
          flush=True)

    print("PASS 2: computing thread rank within each link_id (pool-relative) ...", flush=True)
    rank_of = {}  # (link_id, sha1) -> (rank, size)  -- rank 1 = earliest created_utc
    for lid, items in by_link.items():
        items_sorted = sorted(items, key=lambda t: (t[0] if t[0] is not None else 0, t[1]))
        size = len(items_sorted)
        for i, (cu, h) in enumerate(items_sorted):
            rank_of[(lid, h)] = (i + 1, size)
    print(f"PASS 2 done: ranked {len(rank_of):,} (link_id, body) slots", flush=True)

    print("PASS 3: joining population rows ...", flush=True)
    rows = []
    n_pop = 0
    n_matched = 0
    n_ambiguous = 0
    with open(POP, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            n_pop += 1
            story = row["story"]
            h = sha1(story)
            cands = index.get(h, [])
            if not cands:
                continue
            if len(cands) > 1:
                n_ambiguous += 1
            cu, lid, sc = cands[0]
            rk, sz = rank_of.get((lid, h), (None, None))
            n_matched += 1
            rows.append({
                "id": row["id"], "prompt_id": row["prompt_id"], "judgement": row["judgement"],
                "created_utc": cu, "link_id": lid, "raw_score": sc,
                "thread_rank": rk, "thread_size": sz,
                "thread_rank_frac": (rk / sz) if (rk and sz) else None,
                "n_candidates": len(cands),
            })

    print(f"PASS 3 done: population n={n_pop}, matched={n_matched} ({n_matched/n_pop:.4f}), "
          f"ambiguous(>1 raw match)={n_ambiguous}", flush=True)

    with open(OUT_COV, "w", newline="") as f:
        cols = ["id", "prompt_id", "judgement", "created_utc", "link_id", "raw_score",
                "thread_rank", "thread_size", "thread_rank_frac", "n_candidates"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    report = {
        "n_raw_comments_scanned": n, "n_distinct_bodies": len(index), "n_distinct_link_ids": len(by_link),
        "n_population": n_pop, "n_matched": n_matched, "match_rate": n_matched / n_pop,
        "n_ambiguous_multi_match": n_ambiguous,
        "output_csv": OUT_COV,
    }
    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print("BUILD_DONE")


if __name__ == "__main__":
    main()
