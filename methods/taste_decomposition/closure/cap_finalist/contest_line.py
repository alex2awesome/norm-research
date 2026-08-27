#!/usr/bin/env python3
"""FREEZE ADDENDUM 4 observed-covariate line for CAP_FINALIST, plus the within-contest
(TIER 2) AUC helper the rest of the campaign imports.

WHAT THE CONTAINER IS.  A caption's container is its CONTEST.  Two ordinals exist and
both are recovered here as REAL observed values -- never as text fingerprints (the
programme's standing ruling from jokes/math.SE: LLM proposers can name a
position-in-container channel and cannot see it in the text):

  1. CONTEST NUMBER (530..895) -- the contest's position in the New Yorker's own contest
     series, i.e. which era of the feature's editorial taste the entry comes from.  This
     is the direct analogue of the jokes cell's `created_utc`.

  2. POSITION WITHIN THE CONTEST'S ENTRY LIST -- **DOES NOT EXIST IN THIS CORPUS.**  The
     raw scrape (datasets/humor/newyorker_caption_ratings.csv.gz) carries a per-contest
     `rank`, and its row index is exactly `rank` within every contest (verified), so the
     file order is the CROWD RANKING, not an arrival order.  There is no submission
     timestamp and no entry-stream index anywhere in the source.  Recorded as ABSENT
     rather than searched-for-and-null.

WHY THE ADDENDUM-4 FAMILY IS STRUCTURALLY NULL HERE.  The hard-negative pool takes
exactly 3 finalists and ~20 hard negatives per contest, so the label rate is a constant
.1304 in 225 of 227 contests.  A container-level covariate therefore CANNOT predict y by
construction.  This is a design fact about the pool, not a measurement, and it is the
reason this cell's covariate line is short.

THE COVARIATE THAT DOES CARRY: WITHIN-CONTEST CROWD RANK.  Every caption has a
crowd rating from the NEXTML rating experiment (98.3% coverage).  It is an observed,
non-text measure of the SAME item by a DIFFERENT judge population, and it is the
sharpest thing to say about this cell -- see the report this script writes.

CPU only.  Usage: python contest_line.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import cells as C
import closure_core as L

HERE = Path(__file__).resolve().parent


def within_group_auc(y, p, groups, min_n=8):
    """n-weighted mean of within-container AUCs (TIER 2 on this cell).

    y here is a WITHIN-CONTEST selection (3 finalists of ~23 entries), so the
    within-contest readout is the one that matches the y-definition.  Returned with its
    own coverage info; never substituted for the pooled TIER-1 number."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    num, tot, used, dropped = 0.0, 0, 0, 0
    for g in np.unique(groups):
        m = groups == g
        n = int(m.sum())
        if n < min_n or len(set(y[m].tolist())) < 2:
            dropped += n
            continue
        num += n * roc_auc_score(y[m], p[m])
        tot += n
        used += 1
    if tot == 0:
        return float("nan"), {"n_groups_used": 0, "n_rows_used": 0, "n_rows_dropped": dropped}
    return float(num / tot), {"n_groups_used": used, "n_rows_used": int(tot),
                              "n_rows_dropped": int(dropped)}


def main():
    d = C.load("cap_finalist")
    sp = json.loads((HERE / "cap_finalist_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    y, groups, dense = d["y"], d["groups"], d["dense"]
    held = np.isin(d["dense_split"], ["eval", "test"])
    monm = split == "monitor"

    covs = {
        "contest_number (position of the contest in the series)": d["contest_no"],
        "crowd_mean (raw)": d["crowd_mean"],
        "crowd_votes": d["crowd_votes"],
        "crowd percentile rank WITHIN its contest": d["crowd_pct_in_contest"],
    }
    rep = {"cell": "cap_finalist", "n": int(len(y)),
           "container": "contest",
           "arrival_order_ordinal": "ABSENT FROM THIS CORPUS (raw scrape row index == "
                                    "crowd rank within contest; no submission timestamp)",
           "pos_rate_is_constant_per_contest": True,
           "covariates": {}}

    for name, v in covs.items():
        m = np.isfinite(v)
        row = {"coverage": float(m.mean())}
        for pop, mask in (("full", np.ones(len(y), bool)), ("HONEST", held),
                          ("MONITOR", monm)):
            mm = mask & m
            if len(set(y[mm].tolist())) < 2:
                continue
            row[f"alone_AUC_{pop}"] = float(roc_auc_score(y[mm], v[mm]))
        mm = held & m
        row["within_contest_AUC_HONEST"] = within_group_auc(y[mm], v[mm], groups[mm])[0]
        dm = mm & np.isfinite(dense)
        row["spearman_with_dense_HONEST"] = float(
            np.corrcoef(np.argsort(np.argsort(v[dm])), np.argsort(np.argsort(dense[dm])))[0, 1])
        rep["covariates"][name] = row

    # ---- how much of the residual does the crowd covariate account for? --------
    # Stratification-free stacked increment on HONEST rows with crowd coverage.
    from readout import stack_oof
    import stage1_slice as S1
    blocks, tags = S1.current_blocks(d, "1")     # bank state 0 = V + A
    fitm = split == "fit_mine"
    r0 = L.fit_block(blocks, fitm, monm, y, groups)
    va = np.full(len(y), np.nan)
    va[fitm] = r0["oof_nl_fitmine"]
    va[monm] = r0["nl_mon"]

    cm = d["crowd_pct_in_contest"]
    m = held & np.isfinite(cm) & np.isfinite(va) & np.isfinite(dense)
    yy, gg = y[m], groups[m]
    s_cd = stack_oof([cm[m], dense[m]], yy, gg)
    s_cv = stack_oof([cm[m], va[m]], yy, gg)
    rep["stacked_increment_over_crowd_rank_HONEST"] = {
        "n": int(m.sum()),
        "AUC_crowd_rank_alone": float(roc_auc_score(yy, cm[m])),
        "AUC_dense": float(roc_auc_score(yy, dense[m])),
        "AUC_bank0": float(roc_auc_score(yy, va[m])),
        "AUC_crowd_plus_dense": float(roc_auc_score(yy, s_cd)),
        "AUC_crowd_plus_bank0": float(roc_auc_score(yy, s_cv)),
        "dense_increment_over_crowd": float(roc_auc_score(yy, s_cd) - roc_auc_score(yy, cm[m])),
        "bank_increment_over_crowd": float(roc_auc_score(yy, s_cv) - roc_auc_score(yy, cm[m])),
    }
    (HERE / "cap_finalist_contest_line.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
