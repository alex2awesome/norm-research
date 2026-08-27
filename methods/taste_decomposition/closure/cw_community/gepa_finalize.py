#!/usr/bin/env python3
"""GEPA Stage 4 (CW): swap ACCEPTED winners' full-population scores into the
terminal (round-7) bank state and recompute the frozen readout (fit_block),
old vs new, on MONITOR and the honest population.  CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import closure_lib_cw as C

HERE = Path(__file__).resolve().parent


def reconstruct_bank_order():
    """Reproduce round7_state.npz's bank_names column order WITH cid provenance
    (bank_names alone has 2 duplicate names and cannot be used to look up a
    column unambiguously)."""
    z0 = np.load(HERE / "round0_state.npz", allow_pickle=True)
    order = [(None, str(s)) for s in z0["bank_names"]]
    for r in (1, 2, 3, 4, 5, 6):
        sc = np.load(HERE / f"round{r}_scores.npz", allow_pickle=True)
        cids = [str(s) for s in sc["cids"]]
        names = [str(s) for s in sc["names"]]
        rt = json.loads((HERE / f"round{r}_routing.json").read_text())
        final = {d["cid"]: d["final_track"] for d in rt["decisions"]}
        rep = json.loads((HERE / f"round{r}_scores.report.json").read_text())
        collapsed = {c["cid"] for c in rep["collapse"] if c["COLLAPSED"]}
        keepA = [j for j, c in enumerate(cids)
                 if final.get(c) == "A" and c not in collapsed]
        order += [(cids[j], names[j]) for j in keepA]
    r7crit = {c["cid"]: c["name"] for c in
              json.loads((HERE / "round7_criteria.json").read_text())}
    r7rt = json.loads((HERE / "round7_routing.json").read_text())
    final7 = {d["cid"]: d["final_track"] for d in r7rt["decisions"]}
    sc7 = np.load(HERE / "round7_scores.npz", allow_pickle=True)
    cids7 = [str(s) for s in sc7["cids"]]
    keepA7 = [c for c in cids7 if final7.get(c) == "A"]
    order += [(c, r7crit[c]) for c in keepA7]
    return order


def main():
    z = np.load(HERE / "round7_state.npz", allow_pickle=True)
    VA_old = z["VA"]
    y, groups, split, ids = z["y"], z["groups"], z["split"].astype(str), z["ids"].astype(str)
    bank_names_old = [str(s) for s in z["bank_names"]]

    order = reconstruct_bank_order()
    assert [n for _, n in order] == bank_names_old, "column-order reconstruction mismatch"
    cid_at_col = [c for c, _ in order]

    winners = json.loads((HERE / "gepa_winners.json").read_text())
    accepted = [w for w in winners if w["ACCEPTED"]]
    wz = np.load(HERE / "gepa_winners_scores.npz", allow_pickle=True)
    w_ids = [str(s) for s in wz["ids"]]
    assert w_ids == list(ids), "winner rescore row order != bank state row order"
    wrep = json.loads((HERE / "gepa_winners_scores.report.json").read_text())
    w_collapsed = {c["parent_cid"] for c in wrep["collapse"] if c["COLLAPSED"]}

    VA_new = VA_old.copy()
    swapped, skipped = [], []
    for j, cid in enumerate(wz["parent_cids"]):
        cid = str(cid)
        if cid in w_collapsed:
            skipped.append(cid)
            continue
        col = cid_at_col.index(cid)
        VA_new[:, col] = wz["X"][:, j]
        swapped.append(cid)
    print(f"Swapped {len(swapped)} GEPA-accepted, non-collapsed winners into the "
          f"terminal bank: {swapped}")
    if skipped:
        print(f"SKIPPED (collapsed on full-population rescore): {skipped}")

    res_old = C.fit_block(VA_old, y, groups, split)
    res_new = C.fit_block(VA_new, y, groups, split)

    def auc_pop(res):
        return float(roc_auc_score(y, res["_pop_nl"]))

    T = z["T"]
    mon = split == "monitor"
    tst = split == "test"
    out = {
        "swapped_cids": swapped, "skipped_collapsed_cids": skipped,
        "n_targeted": 10, "n_accepted": len(accepted),
        "pre_gepa": {
            "VA_nl_MONITOR": res_old["va_nl_monitor"],
            "VA_lin_MONITOR": res_old["va_lin_monitor"],
            "VA_nl_population": auc_pop(res_old),
            "T_MONITOR": float(roc_auc_score(y[mon], T[mon])),
            "T_population": float(roc_auc_score(y, T)),
            "Delta_beyond_MONITOR": float(roc_auc_score(y[mon], T[mon])) - res_old["va_nl_monitor"],
            "Delta_beyond_population": float(roc_auc_score(y, T)) - auc_pop(res_old),
        },
        "post_gepa": {
            "VA_nl_MONITOR": res_new["va_nl_monitor"],
            "VA_lin_MONITOR": res_new["va_lin_monitor"],
            "VA_nl_population": auc_pop(res_new),
            "T_MONITOR": float(roc_auc_score(y[mon], T[mon])),
            "T_population": float(roc_auc_score(y, T)),
            "Delta_beyond_MONITOR": float(roc_auc_score(y[mon], T[mon])) - res_new["va_nl_monitor"],
            "Delta_beyond_population": float(roc_auc_score(y, T)) - auc_pop(res_new),
        },
    }
    if tst.sum():
        out["pre_gepa"]["VA_nl_TEST_DO_NOT_REQUOTE"] = float(
            roc_auc_score(y[tst], res_old["_pop_nl"][tst]))
        out["post_gepa"]["VA_nl_TEST_DO_NOT_REQUOTE"] = float(
            roc_auc_score(y[tst], res_new["_pop_nl"][tst]))
    out["Delta_MONITOR_movement"] = (out["post_gepa"]["Delta_beyond_MONITOR"]
                                     - out["pre_gepa"]["Delta_beyond_MONITOR"])
    out["Delta_population_movement"] = (out["post_gepa"]["Delta_beyond_population"]
                                        - out["pre_gepa"]["Delta_beyond_population"])
    (HERE / "gepa_finalize_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
