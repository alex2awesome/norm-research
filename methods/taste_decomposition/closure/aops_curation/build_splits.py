#!/usr/bin/env python3
"""FIT+MINE / MONITOR splits + population export for the AoPS CURATION cell.

FROZEN prereg (notes/2026-08-05__layer3-closure-prereg.md): "FIT+MINE (80%) /
MONITOR (20%): sha256 hash of group key, threshold at .80", plus the FREEZE
DECLARATION's binding refinement "MONITOR must live INSIDE the dense-held-out
rows".

WHY THIS CELL USES THE BASE 80/20 RULE AND NOT THE .50-WITHIN-HELD-OUT RULE.
The math.SE / press cells had only ~20% of their A/V population dense-held-out,
so the amendment forced a cut *inside* that fifth (MONITOR = held-out questions
with hash >= .50) and FIT+MINE picked up the whole dense-train remainder.  On
this cell the A/V population WAS DEFINED AS the dense arm's held-out set
(build_va_population.py, reuse-first directive 2026-08-07): all 5,202 rows are
dense-held-out.  The amendment's constraint is therefore satisfied by every
possible cut, and applying the .50 rule anyway would throw away half the fitting
data for no honesty gain.  The base 80/20 rule is used, and this paragraph is the
record of that decision being made before any split existed.

    MONITOR   = problems with salted hash >= .80          (all dense-held-out)
    FIT+MINE  = the other problems                        (all dense-held-out)
    M         = FIT+MINE in full -- dense scores are honest on every row, so the
                mining slice and the fitting set coincide on this cell
    HONEST    = the FULL population = the master ledger's E rows

SALT, recorded here and not silent.  The dense arm's own 80/10/10 was NOT a plain
sha256 cut -- it was the greedy size+prevalence-balancing bucket map in
build_va_population.py::stable_hash_bucket_map, which orders problems by
sha1(problem).  A plain sha256 cut on the same key is a different function, but
the two could still align by accident, so this build hashes
`sha256("aops-curation-closure|" + problem)` and `salt_collision_check` reports
what the unsalted cut would have produced and how the salted MONITOR distributes
over the dense eval/test halves.

NO seeded shuffle anywhere (standing rule: stable-hash splits only).

The exported `aops_curation_population.csv` carries the BANK'S OWN ITEM VIEW in
its `text` column (cells.item_view: PROBLEM statement truncated to 1,500 chars +
FORUM SOLUTION body under the deterministic HEAD-3000/TAIL-2000 middle omission).
score_gemma_maps.py consumes that column unchanged -- it must NOT re-truncate,
because a whole-view HEAD/TAIL cut would show the mined criteria a different
document from the one the incoming A bank was scored on.

CPU only.  Usage: python3 build_splits.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import cells as C

HERE = Path(__file__).resolve().parent
SALT = "aops-curation-closure|"
THRESH = 0.80            # prereg base rule: MONITOR = top 20% of the hash


def hash_unit(key: str, salt: str = "") -> float:
    return int(hashlib.sha256((salt + str(key)).encode("utf-8")).hexdigest(), 16) / float(1 << 256)


def build(cell="aops_curation"):
    d = C.load(cell)
    ids, groups, y = np.array(d["ids"], dtype=object), d["groups"], d["y"]
    dsplit = d["dense_split"]
    heldout = np.ones(len(y), dtype=bool)          # asserted in cells.load()

    g2s = {}
    for g, s in zip(groups, dsplit):
        g2s.setdefault(g, set()).add(s)
    n_mixed = sum(1 for v in g2s.values() if len(v) > 1)
    assert n_mixed == 0, f"{cell}: {n_mixed} problems straddle the dense eval/test halves"

    hv = np.array([hash_unit(g, SALT) for g in groups])
    hv_unsalted = np.array([hash_unit(g) for g in groups])

    is_mon = hv >= THRESH
    split = np.where(is_mon, "monitor", "fit_mine").astype(object)
    mining = split == "fit_mine"

    all_groups = sorted({str(g) for g in groups})
    unsalted_mon_groups = sum(1 for g in all_groups if hash_unit(g) >= THRESH)
    salted_mon_groups = {str(g) for g in groups[is_mon]}
    unsalted_mon_set = {g for g in all_groups if hash_unit(g) >= THRESH}

    summary = {
        "cell": cell,
        "sklearn": C.sklearn_guard(),
        "population_n": int(len(y)),
        "n_groups": int(len(all_groups)),
        "group_column": d["meta"]["group_column"],
        "pos_rate": float(y.mean()),
        "salt": SALT,
        "rule": f"sha256(SALT + problem)/2**256 >= {THRESH} -> MONITOR; the rest is "
                "FIT+MINE. Every row is dense-held-out on this cell, so the FREEZE's "
                "'MONITOR inside the dense-held-out rows' constraint holds by "
                "construction and the prereg's base 80/20 rule applies.",
        "counts": {s: int((split == s).sum()) for s in ("fit_mine", "monitor")},
        "pos_rate_by_split": {s: float(y[split == s].mean()) for s in ("fit_mine", "monitor")},
        "dense_split_counts": {s: int((dsplit == s).sum()) for s in ("eval", "test")},
        "dense_heldout_n": int(heldout.sum()),
        "dense_heldout_n_groups": len(all_groups),
        "mining_slice_n": int(mining.sum()),
        "monitor_n": int((split == "monitor").sum()),
        "n_pos_monitor": int(y[split == "monitor"].sum()),
        "n_groups_monitor": int(len(salted_mon_groups)),
        "n_groups_fit_mine": int(len({str(g) for g in groups[mining]})),
        "group_overlap_fitmine_monitor": int(len(
            {str(g) for g in groups[mining]} & salted_mon_groups)),
        "monitor_eval_test_mix": {
            s: int(((split == "monitor") & (dsplit == s)).sum()) for s in ("eval", "test")},
        "mining_eval_test_mix": {
            s: int((mining & (dsplit == s)).sum()) for s in ("eval", "test")},
        "HONEST_note": "HONEST = the FULL population = the master ledger's E rows "
                       "(vat_fullgrid_aops_curation.json n_E 5202 / n_groups_E 606 / "
                       "pos_rate_E .6733948481353326); M (mining slice) = FIT+MINE in "
                       "full, because dense scores are honest on every row",
        "salt_collision_check": {
            "problems": len(all_groups),
            "would_be_MONITOR_unsalted": unsalted_mon_groups,
            "actual_MONITOR_salted": int(len(salted_mon_groups)),
            "overlap_salted_vs_unsalted": int(len(salted_mon_groups & unsalted_mon_set)),
            "jaccard_salted_vs_unsalted": float(
                len(salted_mon_groups & unsalted_mon_set)
                / max(1, len(salted_mon_groups | unsalted_mon_set))),
            "unsalted_hash_range": [float(hv_unsalted.min()), float(hv_unsalted.max())],
            "dense_split_is_not_a_hash_cut": "the dense arm's own 80/10/10 came from "
                                             "build_va_population.py::stable_hash_bucket_map, "
                                             "a greedy size+prevalence balancer ordered by "
                                             "sha1(problem) -- a different function from "
                                             "either sha256 cut, so the relevant collision "
                                             "check is the eval/test balance of MONITOR "
                                             "reported above",
        },
        "dense_finite": bool(np.isfinite(d["dense"]).all()),
        "dense_seeds_present": list(d["dense_seed_ids"]),
        "alignment_gate": d["alignment_gate"]["GATE_PASS"],
        "group_size_note": "problem groups are very unequal (1 to 106 solutions, median 4), "
                           "so a 20% cut on GROUPS does not give a 20% cut on ROWS; the "
                           "realised row counts above are the ones that matter",
    }

    recs = [{"i": int(i), "id": str(ids[i]), "group": str(groups[i]),
             "split": str(split[i]), "dense_split": str(dsplit[i]),
             "in_mining_slice": bool(mining[i]),
             "in_monitor_full": bool(is_mon[i])} for i in range(len(y))]
    (HERE / f"{cell}_splits.json").write_text(json.dumps({"summary": summary, "rows": recs}))

    cov = d["cov"] or {}
    frame = {"i": np.arange(len(y)), "id": ids, "text": d["texts"],
             "judgement": y, "group": groups, "split": split,
             "dense_split": dsplit}
    for c in ("post_number", "sol_rank", "n_sols_group", "position_pct",
              "thread_age_days", "years_after_contest", "post_year", "contest_year"):
        if c in cov:
            frame[c] = cov[c]
    pd.DataFrame(frame).to_csv(HERE / f"{cell}_population.csv", index=False)
    return summary


if __name__ == "__main__":
    s = build()
    print(json.dumps(s, indent=1))
