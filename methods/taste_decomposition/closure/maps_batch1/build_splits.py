#!/usr/bin/env python3
"""Stage 1 of the map-focused batch: FIT+MINE / MONITOR splits + population export.

FROZEN prereg (notes/2026-08-05__layer3-closure-prereg.md, FREEZE DECLARATION):
  * FIT+MINE (80%) / MONITOR (20%) by stable sha256 of the GROUP key (never a
    seeded shuffle);
  * **"MONITOR must be defined INSIDE the dense-held-out rows"** (AMENDMENT from
    pilot round 1, restated in the FREEZE DECLARATION as "MONITOR subset of
    dense-held-out rows").  Implemented literally: the .80 hash is applied
    WITHIN the dense-held-out rows, so MONITOR = 20% of the held-out groups and
    FIT+MINE = the other 80% of held-out groups PLUS every dense-train group.
    (Hashing the whole population and then intersecting would leave MONITOR at
    ~4% of the population -- 228-310 rows on these cells -- and was rejected for
    that reason; the amendment's own wording is the inside-H reading.)
    Verified precondition: every cell's dense split is GROUP-PURE (no ntitle /
    contest / docket straddles two dense splits), so hashing groups inside H
    keeps FIT+MINE and MONITOR group-disjoint.
  * Mining slice M = FIT+MINE rows that are ALSO dense-held-out.
  * Because every MONITOR row is dense-held-out by construction, T on MONITOR is
    honest for all five cells (the pilot's 943-contaminated-row problem cannot
    recur).

Writes  <cell>_splits.json  and  <cell>_population.csv  (the latter is also the
input for the peer-curation / peer-revealed same-rows dense rescore on sk3).

CPU only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import cells as C

HERE = Path(__file__).resolve().parent
# Threshold applied INSIDE the dense-held-out groups (see the module docstring).
# OPERATIONAL ADAPTATION, recorded not silent: on these five cells only ~20% of the
# population is dense-held-out, so a .80 threshold inside H lands MONITOR at 92-310
# rows (peer_revealed: 92 rows / 54 positives).  The held-out rows are split 50/50
# instead: MONITOR = held-out groups with hash >= .50 (~10% of the population,
# 239-1,095 rows), mining slice M = the other half of H (~equal size, always >= 239
# rows, ample for a top-60 disagreement slice).  The frozen DECISION rules (eps,
# saturation, routing, scoring) are untouched; only the FIT+MINE/MONITOR proportion
# inside H moves, and every readout is additionally reported on the full honest
# dense-held-out population (MONITOR union M), which is the pilot's own primary
# population (`stratified_dense_heldout_1244_q10` in stage4_readout.py).
THRESH = 0.50


def hash_unit(key: str) -> float:
    return int(hashlib.sha256(str(key).encode("utf-8")).hexdigest(), 16) / float(1 << 256)


def build(cell):
    d = C.load(cell)
    ids, groups, y = d["ids"], d["groups"], d["y"]
    dsplit = d["dense_split"]
    heldout = np.isin(dsplit, ["eval", "test"])

    # group-purity precondition for hashing inside the held-out rows
    g2s = {}
    for g, s in zip(groups, dsplit):
        g2s.setdefault(g, set()).add(s)
    n_mixed = sum(1 for v in g2s.values() if len(v) > 1)
    assert n_mixed == 0, f"{cell}: {n_mixed} groups straddle two dense splits"

    hv = np.array([hash_unit(g) for g in groups])
    split = np.where(heldout & (hv >= THRESH), "monitor", "fit_mine").astype(object)
    mining = (split == "fit_mine") & heldout

    summary = {
        "cell": cell,
        "population_n": int(len(y)),
        "n_groups": int(len(set(groups))),
        "group_column": d["meta"]["group_column"],
        "pos_rate": float(y.mean()),
        "hash": "sha256(group)/2**256 >= .80 AND dense_split in {eval,test} -> monitor",
        "counts": {s: int((split == s).sum()) for s in ("fit_mine", "monitor")},
        "pos_rate_by_split": {s: (float(y[split == s].mean()) if (split == s).sum() else None)
                              for s in ("fit_mine", "monitor")},
        "dense_split_counts": {s: int((dsplit == s).sum())
                               for s in ("train", "eval", "test", "unmapped")},
        "dense_heldout_n": int(heldout.sum()),
        "mining_slice_n": int(mining.sum()),
        "monitor_n": int((split == "monitor").sum()),
        "n_pos_monitor": int(y[split == "monitor"].sum()),
        "n_groups_monitor": int(len(set(groups[split == "monitor"]))),
        "n_groups_fit_mine": int(len(set(groups[split == "fit_mine"]))),
        "group_overlap_fitmine_monitor": int(len(
            set(groups[split == "fit_mine"]) & set(groups[split == "monitor"]))),
        "dense_available": bool(d["dense"] is not None and np.isfinite(d["dense"]).all()),
    }

    recs = [{"i": int(i), "id": str(ids[i]), "group": str(groups[i]),
             "split": str(split[i]), "dense_split": str(dsplit[i]),
             "in_mining_slice": bool(mining[i])} for i in range(len(y))]
    (HERE / f"{cell}_splits.json").write_text(json.dumps({"summary": summary, "rows": recs}))

    pd.DataFrame({"i": np.arange(len(y)), "id": ids, "text": d["texts"],
                  "judgement": y, "group": groups, "split": split,
                  "dense_split": dsplit}).to_csv(HERE / f"{cell}_population.csv", index=False)
    return summary


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or C.CELLS
    out = {}
    for c in todo:
        out[c] = build(c)
        print(json.dumps(out[c], indent=1))
    (HERE / "splits_summary.json").write_text(json.dumps(out, indent=1))
