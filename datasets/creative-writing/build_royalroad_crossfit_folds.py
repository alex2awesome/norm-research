#!/usr/bin/env python3
"""5-FOLD CROSS-FITTED dense splits for cw_royalroad_verdict.

WHY: the 2026-08-10 dense arm put T at chance (.4994, seeds .4822/.485/.531) on a
141-row eval split. That was a POWER CAP, not a demonstration of signal absence --
141 rows / 60 positives cannot resolve an AUC near .55, and the same-split simple
baselines were themselves unstable (TF-IDF .435 eval vs .615 test). This build
converts the unusable 141-row eval into a ~600-row SELECTION-FREE honest set.

DESIGN (SO-campaign cross-fit template, fiction_id-hash buckets):
  bucket t = md5("split::" + fiction_id) % 1000 // 100          -> t in 0..9
  fold k (k = 0..4):  test tenth = 2k
                      select (eval) tenth = 2k + 1
                      train = the remaining 8 tenths
  honest set = union of the 5 test tenths {0,2,4,6,8} = half the rows.

Every honest-set row is predicted by a model that never trained on it AND never
selected a checkpoint on it, so the honest set is selection-free -- which the
single 141-row eval split was NOT (eval was the checkpoint-selection split).

The bucket rule is BYTE-DERIVED from the existing canonical stable hash
(scripts/datasets/build_topic_stratified.py splitof: b<800 train / <900 eval /
test), so tenths 0-7/8/9 reproduce the shipped split exactly. This is the split-
provenance landmine recorded in notes/2026-08-08__cw_royalroad_wigleaf_rebuild.md
S1: deconfound_v2.py's md5(fiction_id)%10 belongs to the SMALLER n=564 build and
reproduces this population's split at only .656 (chance). Do not use it.

  python datasets/creative-writing/build_royalroad_crossfit_folds.py
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

REPO = Path(os.environ.get("NR_REPO", "/lfs/skampere3/0/alexspan/norm-research"))
CELL = REPO / "datasets/creative-writing/royalroad_stubs"
POP = CELL / "va/population.csv.gz"
OUT = CELL / "dense_crossfit"
N_FOLDS = 5
COLS = ["text", "judgement", "group", "row_id"]


def bucket(fiction_id: str) -> int:
    """Tenth of the canonical stable hash. Byte-identical to splitof()'s input."""
    return int(hashlib.md5(("split::" + str(fiction_id)).encode()).hexdigest(), 16) % 1000 // 100


def main():
    df = pd.read_csv(POP)
    df["row_id"] = df["row_id"].astype(str)
    df["bucket"] = df["row_id"].map(bucket)

    # sanity: tenths 0-7/8/9 must reproduce the shipped canonical split exactly
    recon = df["bucket"].map(lambda t: "train" if t <= 7 else ("eval" if t == 8 else "test"))
    agree = float((recon == df["split"]).mean())
    print(f"[check] tenths reproduce the canonical split: {agree:.4f}")
    assert agree == 1.0, "bucket rule does not reproduce the shipped split"

    sizes = df["bucket"].value_counts().sort_index().to_dict()
    print(f"[check] tenth sizes: {sizes}")

    manifest = {
        "cell": "cw_royalroad_verdict",
        "design": "5-fold cross-fit; fold k: test=tenth 2k, select=tenth 2k+1, "
                  "train=remaining 8 tenths",
        "bucket_rule": 'md5("split::"+fiction_id)%1000//100 (byte-derived from '
                       "build_topic_stratified.splitof; canonical-split agreement 1.0)",
        "why": "the single 141-row eval split was a power cap AND was the "
               "checkpoint-selection split; the union of test tenths is a ~600-row "
               "SELECTION-FREE honest set",
        "n_total": int(len(df)),
        "tenth_sizes": {str(k): int(v) for k, v in sizes.items()},
        "folds": {},
    }

    honest_ids, honest_pos = [], 0
    for k in range(N_FOLDS):
        te_t, ev_t = 2 * k, 2 * k + 1
        te = df[df.bucket == te_t]
        ev = df[df.bucket == ev_t]
        tr = df[~df.bucket.isin([te_t, ev_t])]
        d = OUT / f"fold{k}"
        (d / "split").mkdir(parents=True, exist_ok=True)
        df[COLS].to_csv(d / "data.csv", index=False)
        tr[COLS].to_csv(d / "split/train.csv", index=False)
        ev[COLS].to_csv(d / "split/eval.csv", index=False)
        te[COLS].to_csv(d / "split/test.csv", index=False)
        honest_ids += te["row_id"].tolist()
        honest_pos += int(te["judgement"].sum())
        manifest["folds"][f"fold{k}"] = {
            "test_tenth": te_t, "select_tenth": ev_t,
            "n_train": int(len(tr)), "n_eval": int(len(ev)), "n_test": int(len(te)),
            "pos_train": int(tr.judgement.sum()), "pos_eval": int(ev.judgement.sum()),
            "pos_test": int(te.judgement.sum()),
            "frac_train": round(len(tr) / len(df), 4),
            "frac_eval": round(len(ev) / len(df), 4),
            "frac_test": round(len(te) / len(df), 4)}
        print(f"[fold{k}] train {len(tr)} (pos {int(tr.judgement.sum())}) | "
              f"select tenth{ev_t} {len(ev)} (pos {int(ev.judgement.sum())}) | "
              f"TEST tenth{te_t} {len(te)} (pos {int(te.judgement.sum())})")

    assert len(honest_ids) == len(set(honest_ids)), "honest set has duplicate ids"
    manifest["honest_set"] = {
        "n": len(honest_ids), "n_pos": honest_pos, "n_neg": len(honest_ids) - honest_pos,
        "pos_rate": round(honest_pos / len(honest_ids), 4),
        "tenths": [0, 2, 4, 6, 8],
        "selection_free": True,
        "note": "each row predicted by a model that neither trained nor selected on it",
        "ids": honest_ids}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[honest set] n={len(honest_ids)} pos={honest_pos} "
          f"({honest_pos/len(honest_ids):.4f}) vs the old 141-row eval "
          f"-> {len(honest_ids)/141:.1f}x the rows")
    print(f"wrote {OUT}/manifest.json")
    print("BUILD_CROSSFIT_FOLDS_DONE")


if __name__ == "__main__":
    main()
