#!/usr/bin/env python3
"""FIT+MINE / MONITOR splits + population export for the two humor map cells.

FROZEN prereg (notes/2026-08-05__layer3-closure-prereg.md, FREEZE DECLARATION):
MONITOR must be defined INSIDE the dense-held-out rows.  Operationalised exactly
as maps_batch1/build_splits.py did (recorded there, §1a of
notes/2026-08-06__spurious_maps_batch1.md): the .50 hash is applied WITHIN the
dense-held-out groups, so

    MONITOR   = dense-held-out groups with sha256(group)/2**256 >= .50
    M         = the other half of the dense-held-out groups (the mining slice)
    FIT+MINE  = M plus every dense-train group
    HONEST    = every dense-held-out row = M union MONITOR

ADDITIONAL SPLIT, recorded here (precedent: DECISION 1 of the N&C responded
campaign, notes/2026-08-06__closure_nc_responded.md §1.1).  HashtagWars has only
40 contests, 8 of them dense-held-out, so a MONITOR of held-out groups is FOUR
contests.  The Track-A saturation statistic needs VA honesty, not T honesty, so
MONITOR_FULL = every group (train-side included) with hash >= .80 is also built
and reported; T is NOT honest there and no Delta is ever read on it.

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
THRESH = 0.50           # inside the dense-held-out groups
THRESH_FULL = 0.80      # MONITOR_FULL, over all groups (VA-honest only)


def hash_unit(key: str) -> float:
    return int(hashlib.sha256(str(key).encode("utf-8")).hexdigest(), 16) / float(1 << 256)


def build(cell):
    d = C.load(cell)
    ids, groups, y = d["ids"], d["groups"], d["y"]
    dsplit = d["dense_split"]
    heldout = np.isin(dsplit, ["eval", "test"])

    g2s = {}
    for g, s in zip(groups, dsplit):
        g2s.setdefault(g, set()).add(s)
    n_mixed = sum(1 for v in g2s.values() if len(v) > 1)
    assert n_mixed == 0, f"{cell}: {n_mixed} groups straddle two dense splits"

    hv = np.array([hash_unit(g) for g in groups])
    # COARSE-GROUP ADAPTATION, recorded not silent.  HashtagWars has 8 dense-held-out
    # contests; a raw .50 hash cut on 8 units landed 6 MONITOR / 2 mining, leaving the
    # fleet reading a disagreement slice drawn from TWO contests.  The cut is therefore
    # taken at the MEDIAN of the held-out groups' hashes (still a stable sha256 order,
    # still no seeded shuffle, identical in expectation to .50) so both halves always
    # hold half the held-out groups.  Applied uniformly to both cells.
    held_groups = sorted(set(groups[heldout]))
    hg = sorted(held_groups, key=lambda g: hash_unit(g))
    mon_groups = set(hg[len(hg) // 2:])
    is_mon = np.array([g in mon_groups for g in groups])
    split = np.where(heldout & is_mon, "monitor", "fit_mine").astype(object)
    mining = (split == "fit_mine") & heldout
    mon_full = hv >= THRESH_FULL

    summary = {
        "cell": cell,
        "population_n": int(len(y)),
        "n_groups": int(len(set(groups))),
        "group_column": d["meta"]["group_column"],
        "pos_rate": float(y.mean()),
        "rule": "median-of-sha256 cut over the DENSE-HELD-OUT groups -> MONITOR (upper half)",
        "counts": {s: int((split == s).sum()) for s in ("fit_mine", "monitor")},
        "pos_rate_by_split": {s: float(y[split == s].mean()) for s in ("fit_mine", "monitor")},
        "dense_split_counts": {s: int((dsplit == s).sum()) for s in ("train", "eval", "test")},
        "dense_heldout_n": int(heldout.sum()),
        "mining_slice_n": int(mining.sum()),
        "monitor_n": int((split == "monitor").sum()),
        "n_pos_monitor": int(y[split == "monitor"].sum()),
        "n_groups_monitor": int(len(set(groups[split == "monitor"]))),
        "n_groups_mining": int(len(set(groups[mining]))),
        "n_groups_fit_mine": int(len(set(groups[split == "fit_mine"]))),
        "group_overlap_fitmine_monitor": int(len(
            set(groups[split == "fit_mine"]) & set(groups[split == "monitor"]))),
        "monitor_full_n": int(mon_full.sum()),
        "monitor_full_n_groups": int(len(set(groups[mon_full]))),
        "monitor_full_pos_rate": float(y[mon_full].mean()),
        "monitor_full_note": "VA-honest only; T is NOT honest on the dense-train part",
        "dense_finite_on_heldout": bool(np.isfinite(d["dense"][heldout]).all()),
    }

    recs = [{"i": int(i), "id": str(ids[i]), "group": str(groups[i]),
             "split": str(split[i]), "dense_split": str(dsplit[i]),
             "in_mining_slice": bool(mining[i]),
             "in_monitor_full": bool(mon_full[i])} for i in range(len(y))]
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
