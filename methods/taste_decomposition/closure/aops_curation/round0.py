#!/usr/bin/env python3
"""ROUND 0 for the AoPS CURATION Layer-3 articulation-closure campaign.

Produces, in one pass, everything the frozen prereg requires before round 1 runs:

  * the OOF ALIGNMENT GATE (registry 2026-08-10) -- refused to proceed otherwise;
  * READOUT TIERS declared in advance (below), so the governing statistic is not
    chosen after seeing a gain;
  * T on every population.  ONE reused dense arm, ONE seed: T and the "seed
    ensemble" figure are the same number on this cell, so no cross-convention
    caveat attaches to any Delta here (contrast the math.SE cells);
  * the round-0 bank state (V + A) fitted under the frozen closure spec:
    grouped-OOF inside FIT+MINE, refit-and-predict on MONITOR;
  * the closure-protocol round-0 residual on each tier, next to the Layer-1
    number it is NOT comparable to (prereg AMENDMENT 1), and the same-rows
    reconciliation against the master ledger's E-row full-grid arm;
  * the swap baseline (C+, C-);
  * a leave-one-problem-out jackknife of the residual over MONITOR problems.

WHAT MAKES THIS CELL'S ROUND 0 DECISIVE.  The dispatched Layer-1 residual is
Delta_beyond = T .7806 - VA_nl .7705 = +.0101 -- the smallest starting residual
in the sweep.  The brief's instruction is explicit: if Delta_0 on MONITOR is
already sub-epsilon, ONE SEALED PROPOSING ROUND IS STILL REQUIRED before any
terminal language, because a sub-epsilon round 0 is not a round (2026-08-08
addendum; prereg AMENDMENT 2's warning that the round-1 null would have been
declared a taste bound had the rule allowed stopping at one sub-epsilon round).

READOUT TIERS (declared BEFORE any round; the code asserts nothing about which
one "wins"):
  TIER 1, GOVERNING -- pooled AUC on MONITOR.  The tier the Layer-1 gate quantity
    and the master ledger's E frame both live on, so the only tier on which the
    dispatch number and the curve are commensurable.
  TIER 2, SECONDARY -- n-weighted within-PROBLEM AUC.  y asks whether THIS
    solution matches the problem's editorial approach, and every problem carries
    many solutions with a mix of labels, so the within-problem readout removes
    the between-problem component (how canonical a problem's approach is) and
    asks the discrimination question inside a fixed problem.  Reported every
    round, never substituted.
    NOTE, unlike the math.SE vote cell: y here is NOT a within-group median
    split, so group-level covariates are NOT structurally neutralised and the
    pooled tier is a legitimate reading rather than an arithmetic artifact.
  TIER 3, DIAGNOSTIC -- eval-only / test-only, and HONEST.
    HONEST = the FULL 5,202-row population = the master ledger's E rows.  Its VA
    is grouped-OOF on the FIT+MINE side and held-out on the MONITOR side, so
    every prediction is out-of-sample; it is quoted as the same-rows level, not
    as a round-over-round statistic.

CPU only.  Usage: OMP_NUM_THREADS=6 python3 round0.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import cells as C
import closure_core as L
from position_line import within_group_auc
from readout import swap_pair

HERE = Path(__file__).resolve().parent


def delta_per_seed(d, mask, va):
    y = d["y"][mask]
    per = [float(roc_auc_score(y, d["dense_seeds"][mask, j]))
           for j in range(d["dense_seeds"].shape[1])]
    t = float(np.mean(per))
    v = float(roc_auc_score(y, va[mask]))
    return t - v, t, v, per


def delta_within_group(d, mask, va):
    y, g = d["y"][mask], d["groups"][mask]
    per = [within_group_auc(y, d["dense_seeds"][mask, j], g)[0]
           for j in range(d["dense_seeds"].shape[1])]
    t = float(np.mean(per))
    v, info = within_group_auc(y, va[mask], g)
    return t - v, t, float(v), info


def jackknife_groups(d, mask, va, max_report=20):
    y, g = d["y"], np.array([str(x) for x in d["groups"]])
    qs = sorted({q for q in g[mask]})
    rows = []
    for q in qs:
        m = mask & (g != q)
        if len(set(y[m].tolist())) < 2:
            continue
        dl, t, v, _ = delta_per_seed(d, m, va)
        rows.append({"dropped_problem": q, "n_remaining": int(m.sum()), "Delta": dl})
    dv = np.array([r["Delta"] for r in rows])
    n = len(dv)
    se = float(np.sqrt((n - 1) / n * ((dv - dv.mean()) ** 2).sum())) if n > 1 else None
    pooled = delta_per_seed(d, mask, va)[0]
    worst = max(rows, key=lambda r: abs(r["Delta"] - pooled))
    return {"pooled_Delta": pooled, "n_problems": n,
            "jackknife_mean_Delta": float(dv.mean()), "jackknife_SE": se,
            "jackknife_range": [float(dv.min()), float(dv.max())],
            "jackknife_pseudo_CI95": [float(dv.mean() - 1.96 * se),
                                      float(dv.mean() + 1.96 * se)],
            "most_influential_problem": worst["dropped_problem"],
            "Delta_without_it": worst["Delta"],
            "extremes": sorted(rows, key=lambda r: r["Delta"])[:max_report]
            + sorted(rows, key=lambda r: -r["Delta"])[:max_report]}


def masterledger_reconciliation(d, va0, out):
    """The same-rows question every closure cell must answer before mining: does
    the master ledger's E-frame residual survive a closure-protocol VA refit on
    the identical rows?  On this cell E = the FULL population, so the comparison
    is exact rather than approximate."""
    fg = json.loads((C.RESULTS / "vat_fullgrid_aops_curation.json").read_text())
    y = d["y"]
    held = np.ones(len(y), bool)
    rec = {
        "E_rows": fg["n_E"], "E_groups": fg["n_groups_E"], "E_pos_rate": fg["pos_rate_E"],
        "identity_check": {
            "closure_HONEST_n": int(held.sum()),
            "closure_HONEST_n_groups": int(len({str(x) for x in d["groups"]})),
            "closure_HONEST_pos_rate": float(y.mean()),
            "SAME_ROWS": bool(int(held.sum()) == fg["n_E"]
                              and len({str(x) for x in d["groups"]}) == fg["n_groups_E"])},
        "T_master": fg["T"],
        "VA_nl_master_OOF_arm": fg["VA_nl"],
        "VA_nl_master_fullfit_at_E": fg["VA_nl_fullfit_at_E"],
        "VA_nl_closure_refit_on_E": float(roc_auc_score(y, va0)),
        "Delta_master_OOF_arm": fg["T"] - fg["VA_nl"],
        "Delta_master_fullfit_reference": fg["T"] - fg["VA_nl_fullfit_at_E"],
        "Delta_closure_refit": fg["T"] - float(roc_auc_score(y, va0)),
        "note": "three VA fits on ONE row set. The master ledger's OOF arm pools "
                "GroupKFold over all 5,202 rows; the closure protocol fits on FIT+MINE "
                "(4,267 rows) and predicts MONITOR while keeping grouped-OOF inside "
                "FIT+MINE. Levels are protocol-specific (prereg AMENDMENT 1) and are "
                "differenced here BY NAME, never silently.",
    }
    out["master_ledger_reconciliation"] = rec
    return rec


def main():
    sk = C.sklearn_guard()
    d = C.load()
    sp = json.loads((HERE / "aops_curation_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.ones(len(d["y"]), bool)                # every row is dense-held-out
    ev, te = d["dense_split"] == "eval", d["dense_split"] == "test"
    y, g = d["y"], d["groups"]

    out = {"cell": "aops_curation", "sklearn": sk, "splits": sp["summary"],
           "alignment_gate": d["alignment_gate"],
           "readout_tiers": {
               "TIER1_GOVERNING": "pooled AUC on MONITOR (the tier the Layer-1 gate "
                                  "quantity and the master ledger E frame live on)",
               "TIER2_SECONDARY": "n-weighted within-PROBLEM AUC (removes the "
                                  "between-problem 'how canonical is this problem's "
                                  "approach' component)",
               "TIER3_DIAGNOSTIC": "eval-only / test-only / HONEST (= the full "
                                   "population = the master ledger's E rows)"},
           "cell_structure": {
               "every_row_dense_heldout": True,
               "HONEST_equals_full_population": True,
               "mining_slice_equals_FITMINE": True,
               "n_dense_seeds": 1,
               "convention_note": "one reused dense arm means T (mean of per-seed AUCs) "
                                  "and the seed-ensemble AUC are the SAME number, so "
                                  "every Delta on this cell -- curve, discount, matched "
                                  "-- is on one convention and may be differenced freely, "
                                  "unlike the math.SE cells"},
           "layer1_reference": d["layer1"]["ledger"],
           "stopping_rule_note": "Layer-1 Delta_beyond is +.0101, the smallest starting "
                                 "residual in the sweep. Per the brief and the 2026-08-08 "
                                 "addendum, a sub-epsilon round 0 is NOT a round: at least "
                                 "one full sealed PROPOSING round runs before any terminal "
                                 "language, whatever Delta_0 reads."}

    # ------------------------------------------------------------------- T ----
    out["T"] = {"HONEST_full_population": C.T_by_seed(d, held),
                "MONITOR": C.T_by_seed(d, monm),
                "mining_slice_M_eq_FITMINE": C.T_by_seed(d, fitm),
                "eval_only": C.T_by_seed(d, ev), "test_only": C.T_by_seed(d, te),
                "convention": "one dense seed; T = its AUC. No mean-over-seeds vs "
                              "ensemble distinction exists on this cell.",
                "provenance": d["layer1"]["T_info"]["provenance"]}

    # -------------------------------------------------- bank state 0 ----------
    r0 = L.fit_block([d["V"], d["A"]], fitm, monm, y, g)
    va0 = np.full(len(y), np.nan)
    va0[fitm] = r0["oof_nl_fitmine"]
    va0[monm] = r0["nl_mon"]
    lin0 = np.full(len(y), np.nan)
    lin0[fitm] = r0["oof_lin_fitmine"]
    lin0[monm] = r0["lin_mon"]

    out["state0_bank"] = {
        "n_features": r0["n_features"],
        "n_features_V": int(d["V"].shape[1]), "n_features_A": int(d["A"].shape[1]),
        "A_na_rate": float(np.isnan(d["A"]).mean()),
        "VA_lin_MONITOR": L.auc(y[monm], r0["lin_mon"]),
        "VA_nl_MONITOR": L.auc(y[monm], r0["nl_mon"]),
        "VA_nl_MONITOR_per_seed": [L.auc(y[monm], p) for p in r0["nl_mon_seeds"]],
        "VA_nl_OOF_fitmine": L.auc(y[fitm], r0["oof_nl_fitmine"]),
        "VA_lin_OOF_fitmine": L.auc(y[fitm], r0["oof_lin_fitmine"]),
        "VA_nl_MONITOR_within_problem": within_group_auc(y[monm], r0["nl_mon"], g[monm])[0],
        "grid_picks": r0["picks"],
        "impute_note": "closure_core.clean_fit median-imputes the A block inside FIT+MINE "
                       "and screens near-constant columns there; the Layer-1 build used "
                       "the same median convention, so there is no const-0.5 fork on this "
                       "cell.",
    }
    out["state0_bank"]["VA_nl_MONITOR_seed_spread"] = float(
        max(out["state0_bank"]["VA_nl_MONITOR_per_seed"])
        - min(out["state0_bank"]["VA_nl_MONITOR_per_seed"]))

    # ------------------------------------------------------------- Deltas -----
    out["round0_delta_TIER1_pooled"] = {}
    for name, m in (("MONITOR", monm), ("HONEST", held), ("mining_slice_M", fitm),
                    ("eval_only", ev), ("test_only", te)):
        dl, t, v, per = delta_per_seed(d, m, va0)
        out["round0_delta_TIER1_pooled"][name] = {
            "n": int(m.sum()), "n_problems": int(len({str(x) for x in g[m]})),
            "T": t, "T_per_seed": per, "VA_nl": v, "Delta_beyond": dl}
    out["round0_delta_TIER2_within_problem"] = {}
    for name, m in (("MONITOR", monm), ("HONEST", held)):
        dl, t, v, info = delta_within_group(d, m, va0)
        out["round0_delta_TIER2_within_problem"][name] = {
            "T": t, "VA_nl": v, "Delta_beyond": dl, **info}

    out["round0_delta_bootstrap_MONITOR"] = L.group_boot_ci(
        y[monm], d["dense"][monm], va0[monm], np.array([str(x) for x in g[monm]]))
    out["round0_delta_bootstrap_HONEST"] = L.group_boot_ci(
        y, d["dense"], va0, np.array([str(x) for x in g]))
    out["round0_delta_bootstrap_note"] = (
        "problem-cluster paired bootstrap of AUC(dense) - AUC(VA_nl); read for WIDTH. "
        "With one dense seed the bootstrap and the point estimate use the SAME score "
        "vector, so unlike the math.SE cells there is no centring offset to warn about.")

    out["jackknife_MONITOR"] = jackknife_groups(d, monm, va0)

    # ------------------------------------------------------------ swap --------
    out["swap_baseline_MONITOR"] = swap_pair(y[monm], d["dense"][monm], va0[monm])
    out["swap_baseline_HONEST"] = swap_pair(y, d["dense"], va0)

    masterledger_reconciliation(d, va0, out)

    # ------------------------------------------ label composition (context) ---
    gs = np.array([str(x) for x in g])
    sizes = np.array([int((gs == q).sum()) for q in sorted(set(gs))])
    rates = np.array([float(y[gs == q].mean()) for q in sorted(set(gs))])
    out["label_composition"] = {
        "n_problems": int(len(sizes)),
        "solutions_per_problem": {"min": int(sizes.min()), "median": float(np.median(sizes)),
                                  "mean": float(sizes.mean()), "max": int(sizes.max())},
        "problems_all_positive": int((rates == 1).sum()),
        "problems_all_negative": int((rates == 0).sum()),
        "problems_mixed": int(((rates > 0) & (rates < 1)).sum()),
        "rows_in_mixed_problems": int(sum(s for s, r in zip(sizes, rates) if 0 < r < 1)),
        "note": "a problem whose solutions are all-positive or all-negative contributes "
                "NOTHING to the within-problem tier; the fraction of rows that survive "
                "into TIER 2 is reported here so that tier's n is never a surprise.",
    }

    np.savez_compressed(HERE / "aops_curation_r0_preds.npz",
                        va_nl=va0, va_lin=lin0, dense=d["dense"],
                        dense_seeds=d["dense_seeds"], y=y,
                        groups=np.array([str(x) for x in g], dtype=object),
                        split=split, held=held, monitor_full=monm)
    (HERE / "aops_curation_r0_context.json").write_text(json.dumps(out, indent=1, default=float))
    slim = {k: v for k, v in out.items() if k != "jackknife_MONITOR"}
    slim["jackknife_MONITOR"] = {k: v for k, v in out["jackknife_MONITOR"].items()
                                 if k != "extremes"}
    print(json.dumps(slim, indent=1, default=float))


if __name__ == "__main__":
    main()
