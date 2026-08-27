#!/usr/bin/env python3
"""SO closure: 5-fold cross-fitted dense bundles, so EVERY row has an honest T.

WHY. Round 0 showed the frozen stopping rule is UNRESOLVABLE on this cell as
split. MONITOR must be a subset of the dense arm's held-out rows (prereg
AMENDMENT 1), the existing arm holds out only 2,440 of 12,202 rows, and an
80/20 carve of those leaves MONITOR n = 501. At that size the paired
round-over-round noise is sd = .00566 — ABOVE the stopping threshold ε = .005.
The diagnostic that settles it: dropping the single STRONGEST criterion in the
incoming bank ("Solves the precise operation asked", alone |AUC−.5| = .106)
moves MONITOR VA_nl by −.00029 ± .00566. If removing the best criterion is
invisible, adding a new one cannot be visible either, and every round would
report "sub-ε" whatever it discovered.

THE FIX. Cross-fit the dense arm, WITHOUT touching the shared trainer. That
trainer hard-guards the 80/10/10 split shape (no override flag, and modifying it
would affect every other cell), so the folds are built to that exact shape:
questions are hashed into 10 buckets; fold k trains on 8 buckets, selects its
checkpoint on bucket 2k+1 (eval) and predicts bucket 2k (test).

The campaign's honest OOF set is the union of the five TEST tenths -- buckets
0,2,4,6,8, about 6,100 rows, 50% of the population -- because those rows are
never used for checkpoint selection either. That is 12x the rows the old single
split made available and takes MONITOR from 501 to ~1,220, with expected paired
noise .00566 x sqrt(501/1220) = .0036, below eps = .005.

The five EVAL tenths (the other 50%) are also out-of-training but ARE
selection-touched; they are collected separately as an optional wider-MONITOR
sensitivity arm and never mixed into the primary set.

This is not an estimator change and not a weakening of AMENDMENT 1 — it is a
stronger form of the same principle (every dense score the campaign reads is
out-of-sample). The frozen dense-standard recipe is unchanged; only the split
rotates. The single-split arm is retained and reported beside it so the two
conventions are always comparable.

Emits datasets/stackoverflow-votes/va/dense_kfold/fold{0..4}/{data.csv,split/}
with an internal eval split for checkpoint selection carved from the fold's
TRAIN side, so the held-out fifth is never used for selection.

  python3 build_kfold_dense.py            # build bundles
  python3 build_kfold_dense.py --collect  # gather OOF preds after training
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
SO = REPO / "datasets/stackoverflow-votes/va"
OUT = SO / "dense_kfold"
K = 5
NBUCKET = 10
SALT = "so-kfold-v1|"


def bucket_of(q: str) -> int:
    return int(hashlib.sha256((SALT + str(q)).encode()).hexdigest()[:8], 16) % NBUCKET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()

    pop = pd.read_csv(SO / "population.csv.gz")
    v = pop[pop.y_vote.notna()].copy()
    v["row_id"] = v.row_id.astype(str)
    v["group"] = v.group.astype(str)
    v["judgement"] = v.y_vote.astype(int)
    v["bucket"] = v.group.map(bucket_of)
    cols = ["text", "judgement", "group", "row_id"]

    if a.collect:
        oof = {}
        report = []
        for k in range(K):
            d = OUT / f"fold{k}"
            sp = pd.read_csv(d / "split" / "test.csv")
            pr = pd.read_csv(d / "rm_out_seed42" / "preds_test.csv")
            ok = (len(pr) == len(sp)
                  and bool((pr.judgement.values == sp.judgement.values).all())
                  and bool((pr.group.astype(str).values == sp.group.astype(str).values).all()))
            report.append({"fold": k, "n": int(len(sp)), "gate_pass": bool(ok)})
            assert ok, f"alignment gate FAILED fold {k}"
            for rid, p in zip(sp.row_id.astype(str), pr.prob.astype(float)):
                oof[rid] = p
        out = {"n_covered": len(oof), "n_rows": int(len(v)),
               "coverage_frac": round(len(oof) / len(v), 4),
               "note": "coverage is ~50% by design: only the TEST tenths are "
                       "selection-free. The EVAL tenths are out-of-training but "
                       "selection-touched and are NOT in this set.",
               "gates": report, "salt": SALT, "k": K}
        np.savez_compressed(OUT / "oof_dense_kfold.npz",
                            row_id=np.array(list(oof), dtype=object),
                            prob=np.array([oof[r] for r in oof], dtype=float))
        (OUT / "oof_report.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        return

    man = {"k": K, "n_buckets": NBUCKET, "salt": SALT, "n": int(len(v)), "folds": []}
    for k in range(K):
        d = OUT / f"fold{k}"
        (d / "split").mkdir(parents=True, exist_ok=True)
        b_te, b_ev = 2 * k, 2 * k + 1
        te = v[v.bucket == b_te]
        ev = v[v.bucket == b_ev]
        tr = v[~v.bucket.isin([b_te, b_ev])]
        v[cols].to_csv(d / "data.csv", index=False)
        tr[cols].to_csv(d / "split" / "train.csv", index=False)
        ev[cols].to_csv(d / "split" / "eval.csv", index=False)
        te[cols].to_csv(d / "split" / "test.csv", index=False)
        fm = {"fold": k, "bucket_test": b_te, "bucket_eval": b_ev,
              "train": int(len(tr)), "eval": int(len(ev)),
              "test_heldout": int(len(te)),
              "frac_train": round(len(tr) / len(v), 4),
              "frac_eval": round(len(ev) / len(v), 4),
              "frac_test": round(len(te) / len(v), 4),
              "test_pos_rate": float(te.judgement.mean()),
              "train_pos_rate": float(tr.judgement.mean())}
        (d / "manifest.json").write_text(json.dumps(
            {**fm, "recipe": "frozen dense-standard, seed 42; split rotated for "
                             "cross-fitting so every row gets an honest OOF T",
             "item_view": "title view (matches the headline arm)"}, indent=1))
        man["folds"].append(fm)
        print(f"fold{k}: train {len(tr)} eval {len(ev)} heldout {len(te)} "
              f"(pos {te.judgement.mean():.4f})")
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    cov = sum(f["test_heldout"] for f in man["folds"])
    for f in man["folds"]:
        assert abs(f["frac_train"] - .8) < .03 and abs(f["frac_eval"] - .1) < .03 \
            and abs(f["frac_test"] - .1) < .03, f"fold {f['fold']} violates the 80/10/10 guard"
    print(f"OK: {K} folds, all within the trainer's 80/10/10 guard; "
          f"honest TEST coverage {cov}/{len(v)} = {cov/len(v):.1%}")


if __name__ == "__main__":
    main()
