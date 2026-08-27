#!/usr/bin/env python3
"""Pull the math.SE vote-score dense-standard per-seed probabilities off sk3 and
write `mathse_vote_dense_preds.csv` (row_id, dense_split, group, judgement, p42,
p1, p2 -- whichever seeds exist).

`rm_out_seed*/preds_{eval,test}.csv` carry NO row key; they are row-aligned with
`split/{eval,test}.csv`, which do.  The join is positional and is asserted on BOTH
the `judgement` and `group` columns for every row before anything is written, so a
silent misalignment cannot survive.

Usage (from the campaign dir):  python3 fetch_dense.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "mathse_vote_dense_preds.csv"

REMOTE = r'''
import csv, json, os, sys
BASE = "/lfs/skampere3/0/alexspan/norm-research/datasets/math-stackexchange/v2_va"
D = os.path.join(BASE, "dense_standard_mathse_vote_score")
csv.field_size_limit(10**9)
seeds = [42, 1, 2]
have = [s for s in seeds
        if os.path.exists(os.path.join(D, f"rm_out_seed{s}", "preds_eval.csv"))
        and os.path.exists(os.path.join(D, f"rm_out_seed{s}", "preds_test.csv"))]
rows, order = {}, []
for split in ("eval", "test"):
    with open(os.path.join(D, "split", f"{split}.csv")) as fh:
        sp = list(csv.DictReader(fh))
    for r in sp:
        rid = r["row_id"]
        rows[rid] = {"row_id": rid, "dense_split": split, "group": r["group"],
                     "judgement": r["judgement"]}
        order.append(rid)
    for s in have:
        with open(os.path.join(D, f"rm_out_seed{s}", f"preds_{split}.csv")) as fh:
            pr = list(csv.DictReader(fh))
        assert len(pr) == len(sp), (split, s, len(pr), len(sp))
        for r, p in zip(sp, pr):
            assert str(r["judgement"]) == str(p["judgement"]), "judgement misalign"
            assert str(r["group"]) == str(p["group"]), "group misalign"
            rows[r["row_id"]][f"p{s}"] = p["prob"]
cols = ["row_id", "dense_split", "group", "judgement"] + [f"p{s}" for s in have]
w = csv.DictWriter(sys.stdout, fieldnames=cols); w.writeheader()
for rid in order:
    w.writerow({k: rows[rid].get(k, "") for k in cols})
sys.stderr.write(json.dumps({"seeds_found": have, "n_rows": len(order)}) + "\n")
'''


def main():
    proc = subprocess.run(
        ["ssh", "sk3", "export HOME=/lfs/skampere3/0/alexspan; python3 -"],
        input=REMOTE, capture_output=True, text=True)
    if proc.returncode != 0 or not proc.stdout.strip():
        sys.exit(f"fetch failed rc={proc.returncode}\n{proc.stderr[-2000:]}")
    OUT.write_text(proc.stdout)
    print(proc.stderr.strip())
    print(f"-> {OUT} ({len(proc.stdout.splitlines()) - 1} rows)")


if __name__ == "__main__":
    main()
