#!/usr/bin/env python3
"""Stage 1: splits + population export, N&C RESPONDED closure campaign.

Prereg: notes/2026-08-05__layer3-closure-prereg.md, FREEZE DECLARATION 2026-08-06
  "Splits: FIT+MINE/MONITOR stable-hash on group key; MONITOR subset-of dense-held-out rows."

Implementation decision recorded HERE, BEFORE any round runs (the pilot's practice
of recording amendments explicitly rather than silently):

  * hash unit  = docket (the cell's frozen Layer-1 grouping unit).
  * threshold  = .80 (prereg Splits section; the freeze does not revise it).
  * MONITOR_FULL = {hash(docket) >= .80}                       -> the literal 80/20 split.
  * MONITOR      = MONITOR_FULL and dense_split in {eval,test}  -> the freeze's
    "MONITOR subset-of dense-held-out"; T is honest here, so Delta_r = T - VA_nl is
    computable on it.
  * FIT+MINE     = {hash(docket) < .80}.  The rows in MONITOR_FULL that are dense-TRAIN
    are used by NOTHING: not fit on (they are monitor-side dockets) and not read as
    MONITOR (T is contaminated there).  This is strictly more conservative than
    folding them back into FIT+MINE and it keeps the 80/20 hash split literal.
  * Mining slice M = FIT+MINE and dense-held-out (dense scores honest on M).

  Because this cell's dense chain held out only 20% of rows, MONITOR (the T-honest
  set) is n=377 -- too thin to carry the frozen saturation statistic (the stopping
  rule turns on the VA_nl gain, which needs VA honesty, NOT T honesty).  So the
  readout is split, pre-declared:
     - VA_nl gain (SATURATION STATISTIC) -> MONITOR_FULL, n=1,892, VA-honest.
       This is the exact analogue of the pilot's MONITOR (n=1,192) on which the
       stopping rule was actually applied.
     - Delta_r = T - VA_nl (LEVEL)       -> MONITOR, n=377, both honest.
     - Delta honest level (better powered, mildly mining-contaminated hence
       conservative) -> all dense-held-out rows, n=1,904.
  Both MONITOR readouts are reported every round.

CPU only.  Writes:
  nc_responded_splits.json      split map + summary counts
  nc_responded_population.csv   doc_id/docket/y/split/text (uploaded to sk3 for scoring)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import nc_closure_lib as L

HERE = Path(__file__).resolve().parent
DENSE_SLIM = HERE.parent / "samerows_preds" / "nc_responded_dense_preds_slim.csv"
TRUNC = 4000  # matches datasets/notice-and-comment/v4/score_va_gemma_nc.py


def main():
    pop = L.load_population()
    doc_id, y, docket, texts = pop["doc_id"], pop["y"], pop["docket"], pop["texts"]
    n = len(y)

    # ALIGNMENT: the sk3 same-rows rescore wrote its own row order, which is a
    # PERMUTATION of the local loader's (both iterate the same dicts, but the shard
    # read order on sk3 differed).  doc_id is unique in both (verified below), so the
    # join is exact; positional alignment is never assumed.
    dense = pd.read_csv(DENSE_SLIM)
    assert len(dense) == n, (len(dense), n)
    d_ids = dense["doc_id"].astype(str).values
    assert len(set(d_ids)) == n and len(set(doc_id)) == n, "doc_id not unique"
    assert set(d_ids) == set(doc_id), "dense preds population mismatch"
    dense = dense.set_index("doc_id").loc[doc_id].reset_index()
    assert (dense["judgement"].values == y).all(), "dense preds label mismatch"
    assert (dense["docket"].astype(str).values == docket).all(), "dense preds docket mismatch"
    dense.to_csv(HERE / "nc_responded_dense_preds_aligned.csv", index=False)

    dsplit = dense["dense_split"].astype(str).values
    dense_heldout = np.isin(dsplit, ["eval", "test"])

    hv = np.array([L.hash_unit(k) for k in docket])
    monitor_full = hv >= L.THRESH
    fit_mine = ~monitor_full
    monitor = monitor_full & dense_heldout
    unused = monitor_full & ~dense_heldout
    mining = fit_mine & dense_heldout

    split = np.where(monitor, "monitor", np.where(fit_mine, "fit_mine", "unused_monitor_side"))

    recs = [
        {
            "i": int(i),
            "doc_id": str(doc_id[i]),
            "docket": str(docket[i]),
            "y": int(y[i]),
            "split": str(split[i]),
            "monitor_full": bool(monitor_full[i]),
            "dense_split": str(dsplit[i]),
            "in_mining_slice": bool(mining[i]),
        }
        for i in range(n)
    ]

    summary = {
        "cell": "nc_responded",
        "population_n": n,
        "n_dockets": int(len(set(docket))),
        "pos_rate": float(y.mean()),
        "hash": "sha256(docket)/2**256 < 0.80 -> fit_mine",
        "counts": {
            "fit_mine": int(fit_mine.sum()),
            "monitor_full": int(monitor_full.sum()),
            "monitor": int(monitor.sum()),
            "unused_monitor_side": int(unused.sum()),
        },
        "dockets": {
            "fit_mine": int(len(set(docket[fit_mine]))),
            "monitor_full": int(len(set(docket[monitor_full]))),
            "monitor": int(len(set(docket[monitor]))),
        },
        "pos_rate_by_split": {
            "fit_mine": float(y[fit_mine].mean()),
            "monitor_full": float(y[monitor_full].mean()),
            "monitor": float(y[monitor].mean()),
        },
        "dense_split_counts": {s: int((dsplit == s).sum()) for s in ("train", "eval", "test")},
        "dense_heldout_n": int(dense_heldout.sum()),
        "mining_slice_n": int(mining.sum()),
        "docket_disjoint_fit_vs_monitor": bool(
            len(set(docket[fit_mine]) & set(docket[monitor_full])) == 0
        ),
    }

    (HERE / "nc_responded_splits.json").write_text(json.dumps({"summary": summary, "rows": recs}, indent=1))

    pd.DataFrame(
        {
            "i": np.arange(n),
            "doc_id": doc_id,
            "docket": docket,
            "y": y,
            "split": split,
            "dense_split": dsplit,
            "text": [t[:TRUNC] for t in texts],
        }
    ).to_csv(HERE / "nc_responded_population.csv", index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
