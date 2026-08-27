#!/usr/bin/env python3
"""Build data.csv + stable-hash grouped 80/10/10 split for the STANDARDIZED dense
arm (Llama-3.1-8B LoRA via methods/dense/train_reward_model.py) on the HashtagWars
humor-verdict cell (V4 remaining-cells task, 2026-08-06).

Population/y/group EXACTLY reproduce datasets/va_gemma_banks/score_va_gemma_banks.py
build_hashtagwars() (the same population the Layer-1 nonlinear-stack matrix
outputs/va_gemma_banks/hashtagwars_shard*.npz was built from, verified against
methods/taste_decomposition/results/hashtagwars_verdict_layer1.json: n=4228,
pos_rate=0.09389782403027436, n_groups=40, group_column="hashtag contest"):
  - 112 hashtags total (train_data + trial_data + gold_labels TSVs); select the
    first 40 by SHA-256("hashtagwars-va-v1:" + hashtag) sort order.
  - keep ALL rows (all 3 split dirs) whose hashtag is in that set of 40, dedup by
    row_id = sha1(hashtag\\0tweet_id\\0text) (a tweet can appear in >1 split dir).
  - y = 1 iff original label in {1 (top-10), 2 (winner)}, else 0.
  - group = hashtag (the "contest").

Text = the same CONTEST HASHTAG / TWEET context block the Gemma A-judge was given
(ctx() in score_va_gemma_banks.py), so the dense reader sees exactly the item the
articulated criteria were scored against -- no additional or held-back information.

Split: deterministic greedy + hill-climb bucket assignment targeting 80/10/10 by
row count AND matched pos-rate per bucket (stable hash tiebreak, no seeded
shuffle) -- see stable_hash_bucket_map.

Usage (CPU only):
  python3 build_dense_standard.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "dense_standard"
SPLIT_DIRS = ("train_data", "trial_data", "gold_labels")
SALT = "hashtagwars-va-v1:"
N_CONTESTS = 40

EXPECTED_N = 4228
EXPECTED_POS_RATE = 0.09389782403027436
EXPECTED_N_GROUPS = 40


def sha256(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8")).hexdigest()


def sha1(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def stable_hash_bucket_map(y_by_group: dict, targets=None, lam: float = 2.5) -> dict:
    """Deterministic greedy + hill-climb repair bin-packing of groups into
    train/eval/test buckets targeting 80/10/10 BY ROW COUNT *AND* matched
    per-bucket pos-rate. No seeded shuffle: iteration order is (size desc,
    sha1(group) tiebreak) and is identical on every run.

    y_by_group: {group: [0/1, 0/1, ...]} (the judgement values for that group's
    rows). Objective per bucket b: (frac_b - target_b)^2 + lam*(rate_b -
    overall_rate)^2, minimized by greedy placement then hill-climb repair.

    WHY POS-RATE IS PART OF THE OBJECTIVE (2026-08-06 fix): the naive
    row-count-only version of this function (adapted from notice-and-comment/
    v4/build_dense_standard_csvs.py's build_bucket_map) converges to a locally
    row-fraction-optimal split that can still have wildly divergent per-bucket
    pos-rates when group size correlates with y (verified empirically on the
    patents claim-fell cell: corr(app size, mean fell)=+.30, so the
    largest-first greedy dumped nearly all big high-reject apps into train,
    landing train pos-rate .66 vs eval/test .365 -- a real train/eval domain
    shift, not a property of grouped splitting per se). Adding the pos-rate
    term to the objective and letting the hill-climb also equalize it fixes
    this (verified: patents converges to .6014/.6014/.6014 in 2 iterations)."""
    targets = targets or {"train": .8, "eval": .1, "test": .1}
    sizes = {g: len(v) for g, v in y_by_group.items()}
    pos = {g: sum(v) for g, v in y_by_group.items()}
    total = sum(sizes.values())
    overall_rate = sum(pos.values()) / total
    order = sorted(sizes, key=lambda g: (-sizes[g], sha1(g)))
    filled = {b: 0 for b in targets}
    filled_pos = {b: 0 for b in targets}
    bmap = {}

    def obj():
        o = sum((filled[b] / total - targets[b]) ** 2 for b in targets)
        o += lam * sum(((filled_pos[b] / max(filled[b], 1)) - overall_rate) ** 2 for b in targets)
        return o

    for g in order:
        best_b, best_o = None, None
        for b in targets:
            filled[b] += sizes[g]; filled_pos[b] += pos[g]
            o = obj()
            if best_o is None or o < best_o:
                best_o, best_b = o, b
            filled[b] -= sizes[g]; filled_pos[b] -= pos[g]
        bmap[g] = best_b
        filled[best_b] += sizes[g]; filled_pos[best_b] += pos[g]

    improved = True
    n_iter = 0
    while improved and n_iter < 20:
        improved = False
        n_iter += 1
        for g in order:
            cur = bmap[g]
            best_b, best_o = cur, obj()
            for b in targets:
                if b == cur:
                    continue
                filled[cur] -= sizes[g]; filled_pos[cur] -= pos[g]
                filled[b] += sizes[g]; filled_pos[b] += pos[g]
                o = obj()
                if o < best_o - 1e-12:
                    best_b, best_o = b, o
                filled[b] -= sizes[g]; filled_pos[b] -= pos[g]
                filled[cur] += sizes[g]; filled_pos[cur] += pos[g]
            if best_b != cur:
                filled[cur] -= sizes[g]; filled_pos[cur] -= pos[g]
                filled[best_b] += sizes[g]; filled_pos[best_b] += pos[g]
                bmap[g] = best_b
                improved = True
    return bmap


def load_all_rows():
    rows, labels = [], {}
    for sd in SPLIT_DIRS:
        for path in sorted((HERE / sd).glob("*.tsv")):
            hashtag = path.stem
            with path.open(encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    tid, text = parts[0], parts[1]
                    rid = hashlib.sha1(f"{hashtag}\0{tid}\0{text}".encode()).hexdigest()
                    rows.append({"id": rid, "group": hashtag, "text": text, "tweet_id": tid})
                    if len(parts) >= 3 and parts[2] in {"0", "1", "2"}:
                        labels[rid] = int(parts[2])
    return rows, labels


def main():
    rows, labels = load_all_rows()
    groups_all = sorted({r["group"] for r in rows}, key=lambda h: sha256(SALT + h))
    selected = set(groups_all[:N_CONTESTS])

    sample = [r for r in rows if r["group"] in selected]
    seen, dedup = set(), []
    for r in sample:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        dedup.append(r)
    sample = dedup

    missing = [r["id"] for r in sample if r["id"] not in labels]
    assert not missing, f"{len(missing)} sampled rows lack a verdict label -- population is wrong"

    out_rows = []
    for r in sample:
        y = 1 if labels[r["id"]] in (1, 2) else 0
        text = f'CONTEST HASHTAG: #{r["group"]}\n\nTWEET: "{r["text"]}"'
        out_rows.append({"text": text, "judgement": y, "group": r["group"], "row_id": r["id"]})

    n = len(out_rows)
    pos_rate = sum(r["judgement"] for r in out_rows) / n
    n_groups = len(set(r["group"] for r in out_rows))
    print(f"n={n} pos_rate={pos_rate!r} n_groups={n_groups}")

    assert n == EXPECTED_N, f"n mismatch: {n} != {EXPECTED_N} (Layer-1 population)"
    assert abs(pos_rate - EXPECTED_POS_RATE) < 1e-9, \
        f"pos_rate mismatch: {pos_rate!r} != {EXPECTED_POS_RATE!r} (Layer-1 population)"
    assert n_groups == EXPECTED_N_GROUPS, f"n_groups mismatch: {n_groups} != {EXPECTED_N_GROUPS}"
    print("ASSERTION PASS: rows are exactly the hashtagwars_verdict_layer1.json population "
          "(n, pos_rate, n_groups all match to float precision)")

    # ---- stable-hash grouped 80/10/10 split, row-count AND pos-rate balanced
    # (see stable_hash_bucket_map docstring). Only 40 groups here, so the plain
    # cumulative-fraction assignment (prep_dense8b_splits.py pattern) is too
    # coarse (33/3/4 groups landed train=.832, outside the trainer's +-2%
    # tolerance); the greedy+hill-climb bin-packer fixes both row-fraction and
    # pos-rate balance.
    from collections import defaultdict
    y_by_group = defaultdict(list)
    for r in out_rows:
        y_by_group[r["group"]].append(r["judgement"])
    by_split = {"train": [], "eval": [], "test": []}
    bmap = stable_hash_bucket_map(y_by_group)
    for r in out_rows:
        by_split[bmap[r["group"]]].append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split").mkdir(exist_ok=True)
    cols = ["text", "judgement", "group", "row_id"]
    with open(OUT / "data.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    for split in ("train", "eval", "test"):
        with open(OUT / "split" / f"{split}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(by_split[split])

    manifest = {
        "cell": "hashtagwars_verdict",
        "source": "datasets/humor/hashtagwars (train_data + trial_data + gold_labels TSVs)",
        "population_recipe": "verbatim datasets/va_gemma_banks/score_va_gemma_banks.py build_hashtagwars()",
        "layer1_reference": "methods/taste_decomposition/results/hashtagwars_verdict_layer1.json",
        "n": n,
        "pos_rate": pos_rate,
        "n_groups": n_groups,
        "group_column": "hashtag contest",
        "y_definition": "1 iff original @midnight staff label in {1 (top-10), 2 (winner)}, else 0",
        "split_group_counts": {s: len(set(r["group"] for r in by_split[s])) for s in by_split},
        "split_row_counts": {s: len(by_split[s]) for s in by_split},
        "split_pos_rates": {s: (sum(r["judgement"] for r in by_split[s]) / max(len(by_split[s]), 1))
                             for s in by_split},
        "split_fractions": {s: len(by_split[s]) / n for s in by_split},
        "assertion_rows_subseteq_layer1_population": (
            f"n={n} == {EXPECTED_N}, pos_rate matches to float precision, "
            f"n_groups={n_groups} == {EXPECTED_N_GROUPS} -- PASS"
        ),
        "recipe": "Llama-3.1-8B LoRA r16/a32, lr5e-5, batch16, max_len1024, 2 epochs, "
                  "gradient-checkpointing, select-on-eval (dense-standard, no deviation)",
    }
    with open(OUT / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest, indent=2))
    print("BUILD_DONE")


if __name__ == "__main__":
    main()
