#!/usr/bin/env python3
"""FIT+MINE / MONITOR splits + population export for the reddit-jokes community cell.

FROZEN prereg (notes/2026-08-05__layer3-closure-prereg.md, FREEZE DECLARATION): MONITOR
must be defined INSIDE the dense-held-out rows.  Operationalised exactly as
mathse_vote/build_splits.py and press_verdict/build_splits.py did:

    MONITOR   = dense-held-out TOPICS whose hash unit >= .50
    M         = the other half of the dense-held-out topics (the mining slice)
    FIT+MINE  = M plus every dense-train topic
    HONEST    = every dense-held-out row = M union MONITOR

SALT, recorded here and not silent.  The dense arm's own 80/10/10 is a stable
row-count/pos-rate bin-pack over the SAME key (LDA topic).  An unsalted second cut on
the same key is not guaranteed independent of the first, so this build hashes
`sha256("jokes-community-closure|" + topic)`; `salt_collision_check` reports what the
unsalted cut would have produced, so the choice is measured.

GRANULARITY, recorded because this cell is the coarsest-grouped in the programme.  There
are only 50 topics and only TEN of them are dense-held-out, so a topic-level MONITOR
holds ~5 topics.  A topic-level cut is nonetheless the right one: it is the only cut
under which MONITOR shares no grouping unit with the rows the bank is fitted on, and it
matches the Layer-1 ledger's grouped-OOF protocol.  The consequence -- a group bootstrap
over ~5 clusters is coarse -- is handled by reporting the item-level bootstrap ALONGSIDE
the group one and quoting the group one, never by silently switching to rows.

NO seeded shuffle anywhere (standing rule: stable-hash splits only).

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
SALT = "jokes-community-closure|"
THRESH = 0.50            # inside the dense-held-out topics
THRESH_FULL = 0.80       # MONITOR_FULL over all topics (VA-honest only)


def hash_unit(key: str, salt: str = "") -> float:
    return int(hashlib.sha256((salt + str(key)).encode("utf-8")).hexdigest(), 16) / float(1 << 256)


def build(cell="jokes_community"):
    d = C.load(cell)
    ids, groups, y = np.array(d["ids"], dtype=object), d["groups"], d["y"]
    dsplit = d["dense_split"]
    heldout = np.isin(dsplit, ["eval", "test"])

    g2s = {}
    for g, s in zip(groups, dsplit):
        g2s.setdefault(g, set()).add(s)
    n_mixed = sum(1 for v in g2s.values() if len(v) > 1)
    assert n_mixed == 0, f"{cell}: {n_mixed} topics straddle two dense splits"

    hv = np.array([hash_unit(g, SALT) for g in groups])
    hv_unsalted = np.array([hash_unit(g) for g in groups])

    is_mon = hv >= THRESH
    split = np.where(heldout & is_mon, "monitor", "fit_mine").astype(object)
    mining = (split == "fit_mine") & heldout
    mon_full = hv >= THRESH_FULL

    held_groups = sorted({str(g) for g in groups[heldout]})
    unsalted_mon_groups = sum(1 for g in held_groups if hash_unit(g) >= THRESH)

    summary = {
        "cell": cell,
        "sklearn": C.sklearn_guard(),
        "population_n": int(len(y)),
        "n_groups": int(len({str(g) for g in groups})),
        "group_column": d["meta"]["group_column"],
        "pos_rate": float(y.mean()),
        "salt": SALT,
        "rule": f"sha256(SALT + topic)/2**256 >= {THRESH} WITHIN the dense-held-out topics "
                "-> MONITOR; the rest of the held-out topics are the mining slice M; "
                "FIT+MINE = M + all dense-train topics",
        "counts": {s: int((split == s).sum()) for s in ("fit_mine", "monitor")},
        "pos_rate_by_split": {s: float(y[split == s].mean()) for s in ("fit_mine", "monitor")},
        "dense_split_counts": {s: int((dsplit == s).sum()) for s in ("train", "eval", "test")},
        "dense_heldout_n": int(heldout.sum()),
        "dense_heldout_n_groups": len(held_groups),
        "mining_slice_n": int(mining.sum()),
        "monitor_n": int((split == "monitor").sum()),
        "n_pos_monitor": int(y[split == "monitor"].sum()),
        "n_groups_monitor": int(len({str(g) for g in groups[split == "monitor"]})),
        "monitor_topics": sorted({str(g) for g in groups[split == "monitor"]}),
        "n_groups_mining": int(len({str(g) for g in groups[mining]})),
        "mining_topics": sorted({str(g) for g in groups[mining]}),
        "n_groups_fit_mine": int(len({str(g) for g in groups[split == "fit_mine"]})),
        "group_overlap_fitmine_monitor": int(len(
            {str(g) for g in groups[split == "fit_mine"]}
            & {str(g) for g in groups[split == "monitor"]})),
        "monitor_eval_test_mix": {
            s: int(((split == "monitor") & (dsplit == s)).sum()) for s in ("eval", "test")},
        "mining_eval_test_mix": {
            s: int((mining & (dsplit == s)).sum()) for s in ("eval", "test")},
        "monitor_full_n": int(mon_full.sum()),
        "monitor_full_n_groups": int(len({str(g) for g in groups[mon_full]})),
        "monitor_full_pos_rate": float(y[mon_full].mean()),
        "monitor_full_note": "VA-honest only; T is NOT honest on the dense-train part; "
                             "no Delta is ever read on it",
        "salt_collision_check": {
            "held_out_topics": len(held_groups),
            "would_be_MONITOR_unsalted": unsalted_mon_groups,
            "actual_MONITOR_salted": int(len({str(g) for g in groups[split == "monitor"]})),
            "unsalted_hash_range_over_heldout": [
                float(hv_unsalted[heldout].min()), float(hv_unsalted[heldout].max())],
            "note": "if the unsalted count were 0 or all, the dense arm's split salt "
                    "collides with the plain one and an unsalted cut would be degenerate",
        },
        "granularity_note": "only 10 topics are dense-held-out, so MONITOR holds ~5 "
                            "grouping units; the group bootstrap over MONITOR is coarse "
                            "by construction and every readout reports the item-level "
                            "band beside it (group band is the quoted one).",
        "dense_finite_on_heldout": bool(np.isfinite(d["dense"][heldout]).all()),
        "dense_seeds_present": list(d["dense_seed_ids"]),
        "alignment_gate": d["alignment_gate"]["GATE_PASS"],
    }

    recs = [{"i": int(i), "id": str(ids[i]), "group": str(groups[i]),
             "split": str(split[i]), "dense_split": str(dsplit[i]),
             "in_mining_slice": bool(mining[i]),
             "in_monitor_full": bool(mon_full[i])} for i in range(len(y))]
    (HERE / f"{cell}_splits.json").write_text(json.dumps({"summary": summary, "rows": recs}))

    pd.DataFrame({"i": np.arange(len(y)), "id": ids, "text": d["texts"],
                  "judgement": y, "group": groups, "split": split,
                  "dense_split": dsplit,
                  "topic": d["topic"], "created_utc": d["created_utc"],
                  }).to_csv(HERE / f"{cell}_population.csv", index=False)
    return summary


if __name__ == "__main__":
    s = build()
    print(json.dumps(s, indent=1))
