#!/usr/bin/env python3
"""GEPA Stage 4 (N&C): swap ACCEPTED winners' full-population scores into the
terminal (round-5) 67-mined-criterion bank and recompute Delta on both the
honest population (n=1,904) and the eval-only, selection-free half (n=952),
old vs new.  CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import nc_closure_lib as L
import readout as RO
from readout import fit_state, load_dense

HERE = Path(__file__).resolve().parent


def load_round_scores_gepa(rounds, winner_cols):
    """Same as readout.load_round_scores, but splices in GEPA-winner columns for
    any (round, cid) present in winner_cols (row-order aligned to `doc_id`)."""
    mats, names = [], []
    for r in rounds:
        p = HERE / f"round{r}_scores.npz"
        z = np.load(p, allow_pickle=True)
        routed = json.loads((HERE / f"round{r}_routing_final.json").read_text())
        keep_ids = {c["id"] for c in routed["A"]}
        gate = json.loads((HERE / f"round{r}_score_report.json").read_text())
        cids = [str(s) for s in z["crit_ids"]]
        cnames = [str(s) for s in z["crit_names"]]
        cols, nms = [], []
        for k, cid in enumerate(cids):
            if cid not in keep_ids:
                continue
            if gate["per_criterion"][cid]["collapsed"]:
                continue
            tag = f"r{r}:{cid}"
            col = winner_cols.get(tag, z["X"][:, k])
            cols.append(col)
            nms.append(f"r{r}:{cid}:{cnames[k]}")
        if cols:
            mats.append(np.column_stack(cols))
            names += nms
    return np.column_stack(mats), names


def main():
    pop = L.load_population()
    y = pop["y"]

    winners = json.loads((HERE / "gepa_winners_nc.json").read_text())
    accepted = [w for w in winners if w["ACCEPTED"]]
    wz = np.load(HERE / "gepa_winners_scores_nc.npz", allow_pickle=True)
    w_doc = [str(s) for s in wz["doc_id"]]
    assert w_doc == list(pop["doc_id"]), (
        "winner rescore row order != load_population() row order")
    wrep = json.loads((HERE / "gepa_winners_scores_nc.report.json").read_text())
    w_collapsed = {c["parent_tag"] for c in wrep["collapse"] if c["COLLAPSED"]}

    winner_cols, swapped, skipped = {}, [], []
    for j, tag in enumerate(wz["parent_tags"]):
        tag = str(tag)
        if tag in w_collapsed:
            skipped.append(tag)
            continue
        winner_cols[tag] = wz["X"][:, j]
        swapped.append(tag)
    print(f"Swapping {len(swapped)} GEPA-accepted, non-collapsed winners: {swapped}")
    if skipped:
        print(f"SKIPPED (collapsed on full-population rescore): {skipped}")

    dense = load_dense()

    rounds = (1, 2, 3, 4, 5)

    # PRE-GEPA: reproduce the campaign's own terminal (round-5) state exactly
    res_pre, arrs_pre = fit_state(rounds, dense_prob=dense, tag="pre_gepa")

    # POST-GEPA: same rounds, winner columns spliced in
    orig_load = RO.load_round_scores
    RO.load_round_scores = lambda rr: load_round_scores_gepa(rr, winner_cols)
    try:
        res_post, arrs_post = fit_state(rounds, dense_prob=dense, tag="post_gepa")
    finally:
        RO.load_round_scores = orig_load

    def extract(res):
        return {
            "honest": {"n": res["honest"]["n"], "T": res["honest"].get("T"),
                      "VA_nl_on_T_rows": res["honest"].get("VA_nl_on_T_rows"),
                      "Delta": res["honest"].get("Delta")},
            "monitor_full": {"n": res["monitor_full"]["n"],
                             "VA_nl": res["monitor_full"]["VA_nl"]},
        }

    out = {
        "n_targeted": len(winners), "n_accepted": len(accepted),
        "swapped_tags": swapped, "skipped_collapsed_tags": skipped,
        "pre_gepa": extract(res_pre), "post_gepa": extract(res_post),
    }
    out["Delta_movement_honest_1904"] = (out["post_gepa"]["honest"]["Delta"]
                                         - out["pre_gepa"]["honest"]["Delta"])

    # eval-only, selection-free half (T not selected on this half) -- the
    # campaign's own decisive caveat; recompute it for both states too
    import pandas as pd
    _, split, dsplit, mining, monitor_full = L.load_splits()
    heldout = np.isin(dsplit, ["eval", "test"])
    evalonly = heldout & (dsplit == "eval")

    def evalonly_delta(arrs, res_full):
        # reconstruct the full nl_mean vector already computed inside fit_state
        # (arrs['nl_mean'] over the WHOLE population, aligned to pop row order)
        nl_mean = arrs["nl_mean"]
        m = evalonly
        va = L.auc(y[m], nl_mean[m])
        t = L.auc(y[m], dense[m])
        return {"n": int(m.sum()), "T": t, "VA_nl": va, "Delta": t - va}

    out["pre_gepa"]["eval_only"] = evalonly_delta(arrs_pre, res_pre)
    out["post_gepa"]["eval_only"] = evalonly_delta(arrs_post, res_post)
    out["Delta_movement_eval_only"] = (out["post_gepa"]["eval_only"]["Delta"]
                                       - out["pre_gepa"]["eval_only"]["Delta"])

    (HERE / "gepa_finalize_nc_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
