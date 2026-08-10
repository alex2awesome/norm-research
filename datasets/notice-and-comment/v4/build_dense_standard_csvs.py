#!/usr/bin/env python3
"""Build data.csv + grouped 80/10/10 split CSVs for the STANDARDIZED dense arm
(Llama-3.1-8B LoRA via methods/dense/train_reward_model.py) — decision 2026-07-27:
one recipe everywhere, no TF-IDF, single full-data run per (task, y).

Emits 6 bundles:
  peer vat_3y  : verdict / curation / revealed — reuses the rung files' existing
                 stable-hash title-grouped train/eval/test split verbatim.
  N&C v4       : outcome-majority / agree-vs-disagree / responded-or-not — same item
                 universes as aggregate_nc_multiy.py (scored ids only), docket-grouped
                 80/10/10 by stable hash (one docket->bucket map shared by all 3 y's).
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

NC = Path("/Users/spangher/Projects/stanford-research/norm-research/datasets/notice-and-comment/v4")
PEER = Path("/Users/spangher/Projects/stanford-research/norm-research/datasets/peer-review/vat_3y")
sys.path.insert(0, str(NC))
from aggregate_nc_multiy import (MATCHED_SHARDS, UNMATCHED_NPZ, SAMPLE_JSONL,  # noqa: E402
                                 UNMATCHED_JSONL, LABELS_FULL, load_shard_scores,
                                 load_jsonl_texts, load_label_ys)


def write_bundle(outdir, rows_by_split, extra_cols):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "split").mkdir(exist_ok=True)
    cols = ["text", "judgement"] + extra_cols
    allrows = [r for rs in rows_by_split.values() for r in rs]
    with open(outdir / "data.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(allrows)
    for split, fname in [("train", "train.csv"), ("eval", "eval.csv"), ("test", "test.csv")]:
        with open(outdir / "split" / fname, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows_by_split[split])
    n = len(allrows)
    fr = {s: len(rows_by_split[s]) / n for s in rows_by_split}
    pos = {s: (sum(r["judgement"] for r in rows_by_split[s]) / max(len(rows_by_split[s]), 1))
           for s in rows_by_split}
    print(f"{outdir.name}: n={n} fractions " +
          " ".join(f"{s}={fr[s]:.3f}" for s in ("train", "eval", "test")) +
          "  pos " + " ".join(f"{s}={pos[s]:.3f}" for s in ("train", "eval", "test")))
    assert abs(fr["train"] - .8) <= .02 and abs(fr["eval"] - .1) <= .02 and abs(fr["test"] - .1) <= .02, \
        f"{outdir}: split fractions outside trainer tolerance"


# ------------------------------------------------------------------ peer ---
for rung in ("verdict", "curation", "revealed"):
    rows_by_split = {"train": [], "eval": [], "test": []}
    for line in open(PEER / f"{rung}.jsonl"):
        r = json.loads(line)
        try:
            y = int(float(r["judgement"]))
        except (TypeError, ValueError):
            continue
        if y not in (0, 1):
            continue
        rows_by_split[r["split"]].append(
            {"text": r["text"], "judgement": y, "group": r["ntitle"]})
    write_bundle(PEER / "dense_llama" / rung, rows_by_split, ["group"])

# ------------------------------------------------------------------- N&C ---
X_m, docket_m, _, _ = load_shard_scores(MATCHED_SHARDS)
X_u, docket_u, _, _ = load_shard_scores([str(UNMATCHED_NPZ)])
overlap = set(X_m) & set(X_u)
for did in overlap:
    del X_u[did]
    del docket_u[did]
text_m = load_jsonl_texts(SAMPLE_JSONL)
text_u = load_jsonl_texts(UNMATCHED_JSONL)
label_ys = load_label_ys(LABELS_FULL)


# Deterministic greedy bin-pack over dockets (largest first, md5 tiebreak) on the
# union universe, so all three y-subsets land inside the trainer's 80/10/10 +-2%
# tolerance despite docket-size skew. One shared map -> same comment never crosses
# buckets between y's.
def build_bucket_map(docket_by, universes):
    """universes: {yname: set(ids)}. Each docket carries a per-y count vector; greedily
    assign (largest total first, md5 tiebreak) to the bucket with the largest summed
    normalized deficit, so EVERY y-subset lands near 80/10/10 simultaneously."""
    from collections import defaultdict
    vec = defaultdict(lambda: defaultdict(int))
    totals = {y: max(len(ids), 1) for y, ids in universes.items()}
    for y, ids in universes.items():
        for d in ids:
            vec[docket_by[d]][y] += 1
    targets = {"train": .8, "eval": .1, "test": .1}
    filled = {b: defaultdict(int) for b in targets}
    bmap = {}
    order = sorted(vec.items(),
                   key=lambda kv: (-sum(kv[1].values()),
                                   hashlib.md5(kv[0].encode()).hexdigest()))
    for dk, v in order:
        def score(b):
            return sum(targets[b] - filled[b][y] / totals[y] for y in universes)
        b = max(targets, key=score)
        bmap[dk] = b
        for y in v:
            filled[b][y] += v[y]

    # deterministic hill-climb repair: minimize sum of squared (fraction - target)
    # deviations over all (y, bucket) cells
    def obj():
        return sum((filled[b][y] / totals[y] - targets[b]) ** 2
                   for b in targets for y in universes)

    improved = True
    while improved:
        improved = False
        for dk, v in order:
            cur = bmap[dk]
            best_b, best_o = cur, obj()
            for b in targets:
                if b == cur:
                    continue
                for y in v:
                    filled[cur][y] -= v[y]
                    filled[b][y] += v[y]
                o = obj()
                if o < best_o - 1e-12:
                    best_b, best_o = b, o
                for y in v:
                    filled[b][y] -= v[y]
                    filled[cur][y] += v[y]
            if best_b != cur:
                for y in v:
                    filled[cur][y] -= v[y]
                    filled[best_b][y] += v[y]
                bmap[dk] = best_b
                improved = True
    return bmap


def nc_bundle(name, id_y_pairs, docket_by, bmap):
    rows_by_split = {"train": [], "eval": [], "test": []}
    for did, y in id_y_pairs:
        dk = docket_by[did]
        txt = text_m.get(did) or text_u.get(did) or ""
        if not txt:
            continue
        rows_by_split[bmap[dk]].append(
            {"text": txt, "judgement": y, "group": dk})
    write_bundle(NC / "dense_llama" / name, rows_by_split, ["group"])


y_out = [(d, label_ys[d]["outcome_majority"]) for d in X_m
         if d in label_ys and label_ys[d]["outcome_majority"] in (0, 1)]
y_agr = [(d, label_ys[d]["agree_vs_disagree"]) for d in X_m
         if d in label_ys and label_ys[d]["agree_vs_disagree"] in (0, 1)]
docket_all = dict(docket_m)
docket_all.update(docket_u)
y_resp = [(d, 1) for d in X_m] + [(d, 0) for d in X_u]

bmap = build_bucket_map(docket_all, {
    "out": {d for d, _ in y_out},
    "agr": {d for d, _ in y_agr},
    "resp": {d for d, _ in y_resp},
})
nc_bundle("outcome", y_out, docket_m, bmap)
nc_bundle("agree", y_agr, docket_m, bmap)
nc_bundle("responded", y_resp, docket_all, bmap)
print("done")
