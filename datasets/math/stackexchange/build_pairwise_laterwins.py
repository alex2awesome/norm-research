#!/usr/bin/env python3
"""Build the within-question pairwise "later answer wins" companion dataset.

Input : preference_pairs.jsonl.gz (411,604 rows) with columns
        question_id, question_tags, question_text, chosen_id, chosen_score,
        chosen_text, rejected_id, rejected_score, rejected_text, score_diff,
        n_answers.

Stack Exchange post IDs are globally sequential, so chosen_id > rejected_id
means the WINNING answer was posted LATER. The known confound is that the
earlier answer wins ~67% of pairs (first-mover advantage); the clean subset is
pairs where the later answer won despite that disadvantage.

Filters (applied in order, counts logged in the manifest):
  1. chosen_id > rejected_id     (later answer won)
  2. score_diff >= 3             (meaningful preference margin)
  3. len(chosen_text) >= 50 and len(rejected_text) >= 50

Output columns: question_id, question_tags, chosen_text, rejected_text,
chosen_score, rejected_score, score_diff, split (80/10/10 hashed by
question_id via md5 % 100, same convention as the other math.SE builds).

Outputs:
  math_se_pairwise_laterwins.csv.gz
  math_se_pairwise_laterwins_manifest.json
  (if final count >> 100K) math_se_pairwise_laterwins_100k.csv.gz
      — 100K subsample retaining ALL rows whose tags contain a priority tag.
"""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

BASE = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/math/stackexchange")
SRC = BASE / "preference_pairs.jsonl.gz"
OUT = BASE / "math_se_pairwise_laterwins.csv.gz"
OUT_100K = BASE / "math_se_pairwise_laterwins_100k.csv.gz"
MANIFEST = BASE / "math_se_pairwise_laterwins_manifest.json"

PRIORITY_TAGS = {
    "proof-writing", "proof-verification", "proof-explanation",
    "alternative-proof", "solution-verification", "intuition", "soft-question",
}
SEED = 0
TARGET_100K = 100_000


def split_of(qid) -> str:
    h = int(hashlib.md5(str(qid).encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("eval" if h < 90 else "test")


def parse_tags(t):
    """Normalize question_tags into a list of tag strings."""
    if isinstance(t, list):
        return [str(x) for x in t]
    if t is None:
        return []
    s = str(t)
    # formats like "<tag1><tag2>" or "tag1|tag2" or "tag1,tag2"
    if "<" in s:
        return [x.strip("<>") for x in s.split("><") if x.strip("<>")]
    for sep in ("|", ","):
        if sep in s:
            return [x.strip() for x in s.split(sep) if x.strip()]
    return [s.strip()] if s.strip() else []


print(f"reading {SRC} ...", flush=True)
rows = []
with gzip.open(SRC, "rt") as f:
    for line in f:
        rows.append(json.loads(line))
df = pd.DataFrame(rows)
del rows
n0 = len(df)
print(f"loaded {n0} rows", flush=True)

manifest = {"source": str(SRC), "n_input": int(n0), "filters": []}

# 1. later answer won
df = df[df["chosen_id"] > df["rejected_id"]]
manifest["filters"].append({"filter": "chosen_id > rejected_id (later wins)",
                            "n_after": int(len(df))})
print(f"after later-wins filter: {len(df)} "
      f"({len(df)/n0:.1%} of input; earlier-wins share = {1-len(df)/n0:.1%})",
      flush=True)

# 2. score margin
df = df[df["score_diff"] >= 3]
manifest["filters"].append({"filter": "score_diff >= 3", "n_after": int(len(df))})
print(f"after score_diff>=3: {len(df)}", flush=True)

# 3. minimum text length
df = df[(df["chosen_text"].astype(str).str.len() >= 50)
        & (df["rejected_text"].astype(str).str.len() >= 50)]
manifest["filters"].append({"filter": "both texts >= 50 chars",
                            "n_after": int(len(df))})
print(f"after min-length filter: {len(df)}", flush=True)

# split + final columns
df["split"] = df["question_id"].apply(split_of)
df["tag_list"] = df["question_tags"].apply(parse_tags)
df["question_tags"] = df["tag_list"].apply(lambda ts: "|".join(ts))

COLS = ["question_id", "question_tags", "chosen_text", "rejected_text",
        "chosen_score", "rejected_score", "score_diff", "split"]
final = df[COLS + ["tag_list"]].copy()
n_final = len(final)

tag_counts = Counter(t for ts in final["tag_list"] for t in ts)
manifest["n_final"] = int(n_final)
manifest["split_counts"] = final["split"].value_counts().to_dict()
manifest["tag_distribution_top20"] = dict(tag_counts.most_common(20))

final[COLS].to_csv(OUT, index=False, compression="gzip")
print(f"wrote {OUT} ({n_final} rows)", flush=True)
print("splits:", manifest["split_counts"], flush=True)

# optional 100K subsample with priority-tag retention
if n_final > 1.2 * TARGET_100K:
    is_priority = final["tag_list"].apply(
        lambda ts: any(t in PRIORITY_TAGS for t in ts))
    keep = final[is_priority]
    rest = final[~is_priority]
    n_needed = TARGET_100K - len(keep)
    if n_needed > 0:
        sampled = rest.sample(n=min(n_needed, len(rest)), random_state=SEED)
        sub = pd.concat([keep, sampled]).sort_index()
    else:
        sub = keep
    sub[COLS].to_csv(OUT_100K, index=False, compression="gzip")
    manifest["subsample_100k"] = {
        "path": str(OUT_100K),
        "n_rows": int(len(sub)),
        "n_priority_tag_rows_kept_all": int(len(keep)),
        "n_random_fill": int(len(sub) - len(keep)),
        "split_counts": sub["split"].value_counts().to_dict(),
        "priority_tags": sorted(PRIORITY_TAGS),
    }
    print(f"wrote {OUT_100K} ({len(sub)} rows; "
          f"{len(keep)} priority-tag rows kept in full)", flush=True)
else:
    manifest["subsample_100k"] = None
    print("final count close to/below 100K; no subsample emitted", flush=True)

with open(MANIFEST, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"wrote {MANIFEST}", flush=True)
print("done")
