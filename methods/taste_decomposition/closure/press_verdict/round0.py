#!/usr/bin/env python3
"""ROUND 0 for the press/journalism verdict closure campaign.

Produces, in one pass, everything the frozen prereg requires before round 1 runs:

  * the split-width diagnostic that DECISION 1 of the N&C campaign turns on
    (VA_nl seed spread on MONITOR vs MONITOR_FULL vs HONEST) -- recorded BEFORE any
    round, so the choice of saturation statistic is not made after seeing a gain;
  * the round-0 bank state (V + A_base) fitted under the frozen closure spec:
    grouped-OOF inside FIT+MINE, refit-and-predict on MONITOR;
  * T on every population, as the MEAN OVER DENSE SEEDS OF THE AUC (the same
    convention VA_nl uses), with the seed ensemble reported but never quoted;
  * the same-rows Delta anchor UNDER THE CLOSURE PROTOCOL, next to the Layer-1
    number it is NOT comparable to;
  * the eval-only / test-only split of the residual (this cell's dense chain
    selected on EVAL, so TEST is the selection-free half -- the mirror of N&C,
    whose chain selected on test);
  * the swap baseline (C+, C-);
  * the leave-one-company-out jackknife on Delta over the dense-held-out companies
    (the HashtagWars readout the brief asks for);
  * an A-block imputation sensitivity check (closure-standard median impute inside
    FIT+MINE vs Layer 1's constant-0.5 fill).

CPU only.  Usage: python round0.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import cells as C
import closure_core as L
from readout import swap_pair

HERE = Path(__file__).resolve().parent


def T_on(d, mask):
    return C.T_by_seed(d, mask)


def delta_per_seed(d, mask, va):
    """Delta = mean_s AUC(y, p_s) - AUC(y, va): T averaged over seeds, one bank."""
    y = d["y"][mask]
    t = float(np.mean([roc_auc_score(y, d["dense_seeds"][mask, j])
                       for j in range(d["dense_seeds"].shape[1])]))
    v = float(roc_auc_score(y, va[mask]))
    return t - v, t, v


def jackknife_companies(d, mask, va):
    """Leave-one-company-out over the dense-held-out companies."""
    y, g = d["y"], d["groups"]
    idx = np.where(mask)[0]
    comps = sorted(set(g[idx].tolist()))
    rows = []
    for c in comps:
        m = mask & (g != c)
        if len(set(y[m].tolist())) < 2:
            continue
        dl, t, v = delta_per_seed(d, m, va)
        rows.append({"dropped_company": c, "n_remaining": int(m.sum()),
                     "T": t, "VA_nl": v, "Delta": dl})
    dv = np.array([r["Delta"] for r in rows])
    n = len(dv)
    se = float(np.sqrt((n - 1) / n * ((dv - dv.mean()) ** 2).sum())) if n > 1 else None
    pooled, _, _ = delta_per_seed(d, mask, va)
    worst = max(rows, key=lambda r: abs(r["Delta"] - pooled))
    return {"pooled_Delta": pooled, "n_companies": n,
            "jackknife_mean_Delta": float(dv.mean()), "jackknife_SE": se,
            "jackknife_range": [float(dv.min()), float(dv.max())],
            "jackknife_pseudo_CI95": [float(dv.mean() - 1.96 * se), float(dv.mean() + 1.96 * se)],
            "most_influential_company": worst["dropped_company"],
            "Delta_without_it": worst["Delta"],
            "leave_one_out": sorted(rows, key=lambda r: r["Delta"])}


def fit_state(d, blocks, fitm, monm):
    return L.fit_block(blocks, fitm, monm, d["y"], d["groups"])


def va_full(d, r, fitm, monm):
    v = np.full(len(d["y"]), np.nan)
    v[fitm] = r["oof_nl_fitmine"]
    v[monm] = r["nl_mon"]
    return v


def main():
    sk = C.sklearn_guard()
    d = C.load()
    sp = json.loads((HERE / "press_verdict_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    monfull = np.array([r["in_monitor_full"] for r in sp["rows"]])
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])
    ev, te = d["dense_split"] == "eval", d["dense_split"] == "test"
    y, g = d["y"], d["groups"]

    out = {"cell": "press_verdict", "sklearn": sk, "splits": sp["summary"],
           "layer1_reference": {k: d["layer1"]["ledger"][k] for k in
                                ("V_lin", "V_nl_mean", "A_lin", "A_nl_mean",
                                 "VA_lin", "VA_nl_mean", "Delta_interact")}}

    # ---------------------------------------------------------------- T -------
    out["T"] = {"HONEST_eval_plus_test": T_on(d, held),
                "MONITOR": T_on(d, monm),
                "mining_slice_M": T_on(d, fitm & held),
                "eval_only": T_on(d, ev), "test_only": T_on(d, te),
                "convention": "T = mean over dense seeds {42,1,2} of AUC (mirrors VA_nl's "
                              "mean-over-seeds convention). The seed ENSEMBLE AUC is higher "
                              "because averaging denoises and is never quoted as T.",
                "selection_note": "the dense chain selected on EVAL (dense-standard, no "
                                  "deviation), so TEST is the selection-free half here -- "
                                  "the mirror of the N&C cell, whose chain selected on test."}

    # -------------------------------------------------- bank state 0 ---------
    states, vas = {}, {}
    for label, Ablk in (("PRIMARY_layer1_const05", d["A"]),
                        ("SENSITIVITY_closure_median_impute", d["A_median_impute"])):
        r = fit_state(d, [d["V"], Ablk], fitm, monm)
        v = va_full(d, r, fitm, monm)
        rec = {"n_features": r["n_features"],
               "VA_lin_MONITOR": L.auc(y[monm], r["lin_mon"]),
               "VA_nl_MONITOR": L.auc(y[monm], r["nl_mon"]),
               "VA_nl_MONITOR_per_seed": [L.auc(y[monm], p) for p in r["nl_mon_seeds"]],
               "VA_nl_HONEST": L.auc(y[held], v[held]),
               "VA_nl_OOF_fitmine": L.auc(y[fitm], r["oof_nl_fitmine"]),
               "VA_lin_OOF_fitmine": L.auc(y[fitm], r["oof_lin_fitmine"]),
               "grid_picks": r["picks"]}
        rec["VA_nl_MONITOR_seed_spread"] = float(max(rec["VA_nl_MONITOR_per_seed"])
                                                 - min(rec["VA_nl_MONITOR_per_seed"]))
        states[label] = rec
        vas[label] = v
        if label == "PRIMARY_layer1_const05":
            r0, va0 = r, v
    out["state0_bank"] = states
    out["A_impute_DECISION"] = (
        "DECISION 1, recorded BEFORE round 1. Layer 1's own production pipeline and the "
        "A-layer's published gate BOTH fill an inapplicable rubric cell with the constant "
        "0.5, so that matrix is this cell's actual scorecard and it is the PRIMARY bank "
        "here. The closure-standard median-impute variant is a SENSITIVITY. The gap is "
        "large and one-directional -- median-imputing erases the applicability pattern, "
        "costs the bank .029 of held-out AUC and would inflate Delta_beyond by the same "
        "amount -- so running on the primary is also the conservative choice. Mined "
        "criteria added in rounds 1-5 are NaN-coded and median-imputed by clean_fit, "
        "matching how each block was built; recorded, not silently mixed.")

    # ---------- seed-spread width diagnostic on the three candidate monitors --
    width = {}
    for name, m in (("MONITOR_dense_heldout", monm), ("MONITOR_FULL_hash80", monfull),
                    ("HONEST_all_dense_heldout", held)):
        per = [L.auc(y[m], (r0["nl_mon"] if name != "HONEST_all_dense_heldout"
                            else va0[held]))] if False else None
        width[name] = {"n": int(m.sum()), "n_companies": int(len(set(g[m].tolist()))),
                       "pos_rate": float(y[m].mean())}
    # per-seed VA_nl on each candidate population (needs seed-wise full vectors)
    seedwise = {}
    for j, s in enumerate(L.SEEDS):
        vj = np.full(len(y), np.nan)
        vj[fitm] = r0["oof_nl_fitmine"]          # OOF is already the seed mean; see note
        vj[monm] = r0["nl_mon_seeds"][j]
        seedwise[str(s)] = {"MONITOR": L.auc(y[monm], vj[monm])}
    width["MONITOR_dense_heldout"]["VA_nl_per_seed"] = [seedwise[str(s)]["MONITOR"] for s in L.SEEDS]
    width["MONITOR_dense_heldout"]["VA_nl_seed_spread"] = float(
        max(width["MONITOR_dense_heldout"]["VA_nl_per_seed"])
        - min(width["MONITOR_dense_heldout"]["VA_nl_per_seed"]))
    # MONITOR_FULL needs its own refit (it overlaps FIT+MINE, so VA is NOT honest there
    # unless refit); build it explicitly
    rF = fit_state(d, [d["V"], d["A"]], ~monfull, monfull)
    width["MONITOR_FULL_hash80"]["VA_nl"] = L.auc(y[monfull], rF["nl_mon"])
    width["MONITOR_FULL_hash80"]["VA_nl_per_seed"] = [L.auc(y[monfull], p) for p in rF["nl_mon_seeds"]]
    width["MONITOR_FULL_hash80"]["VA_nl_seed_spread"] = float(
        max(width["MONITOR_FULL_hash80"]["VA_nl_per_seed"])
        - min(width["MONITOR_FULL_hash80"]["VA_nl_per_seed"]))
    width["MONITOR_FULL_hash80"]["note"] = (
        "separate 80/20 hash split over ALL companies; VA-honest, T NOT honest. Reported "
        "only as a width comparison -- on this cell it is barely larger than MONITOR "
        "(companies are very unevenly sized), so unlike N&C it buys no power and the "
        "frozen MONITOR-inside-dense-held-out rule is kept.")
    out["monitor_width_diagnostic"] = width

    # ------------------------------------------------------------- Deltas ----
    out["round0_delta"] = {}
    for name, m in (("HONEST", held), ("MONITOR", monm), ("mining_slice_M", fitm & held),
                    ("eval_only", ev), ("test_only", te)):
        dl, t, v = delta_per_seed(d, m, va0)
        out["round0_delta"][name] = {"n": int(m.sum()), "T": t, "VA_nl": v, "Delta_beyond": dl}
    out["round0_delta"]["layer1_matched_reference"] = {
        "T_eval_mean": 0.7496865354938271, "VA_nl_layer1_pooled": d["layer1"]["ledger"]["VA_nl_mean"],
        "Delta_beyond_as_dispatched": 0.7496865354938271 - d["layer1"]["ledger"]["VA_nl_mean"],
        "note": "the dispatched +.0486 compares the dense EVAL-mean AUC against the Layer-1 "
                "VA_nl fitted by pooled GroupKFold OOF over all 2,956 rows. The closure "
                "protocol fits VA on FIT+MINE only and reads it on the dense-held-out rows, "
                "so its levels are protocol-specific and NOT comparable (the prereg's "
                "AMENDMENT-1 clause); only round-over-round changes and the same-rows "
                "honest level are quotable."}

    # group bootstrap on the round-0 Delta (company clusters), using the seed-mean
    # dense score as the T vector (the AUC-level T is the mean of seed AUCs; the
    # bootstrap needs one vector, so this is reported as the bootstrap's own basis)
    out["round0_delta_bootstrap_HONEST"] = L.group_boot_ci(
        y[held], d["dense"][held], va0[held], g[held])
    out["round0_delta_bootstrap_note"] = (
        "company-cluster paired bootstrap of AUC(dense seed-mean) - AUC(VA_nl); the seed-mean "
        "score vector is used because a bootstrap needs one vector, and its pooled AUC "
        "(.7744) sits above the quoted T (.7508 = mean of seed AUCs), so this CI is centred "
        "high relative to the quoted Delta and is read for WIDTH, not level.")

    # ------------------------------------------------------------ jackknife --
    out["jackknife_HONEST"] = jackknife_companies(d, held, va0)
    out["jackknife_HONEST_SENSITIVITY_median_impute"] = jackknife_companies(
        d, held, vas["SENSITIVITY_closure_median_impute"])
    out["round0_delta_SENSITIVITY_median_impute"] = {}
    for name, m in (("HONEST", held), ("MONITOR", monm), ("eval_only", ev), ("test_only", te)):
        dl, t, v = delta_per_seed(d, m, vas["SENSITIVITY_closure_median_impute"])
        out["round0_delta_SENSITIVITY_median_impute"][name] = {
            "n": int(m.sum()), "T": t, "VA_nl": v, "Delta_beyond": dl}

    # ----------------------------------------------------------------- swap --
    out["swap_baseline_HONEST"] = swap_pair(y[held], d["dense"][held], va0[held])
    out["swap_baseline_MONITOR"] = swap_pair(y[monm], d["dense"][monm], va0[monm])

    np.savez_compressed(HERE / "press_verdict_r0_preds.npz",
                        va_nl=va0, va_nl_median_impute=vas["SENSITIVITY_closure_median_impute"], dense=d["dense"], dense_seeds=d["dense_seeds"],
                        y=y, groups=np.array(g, dtype=object),
                        split=split, held=held, monitor_full=monfull)
    (HERE / "press_verdict_r0_context.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "jackknife_HONEST"},
                     indent=1, default=float))
    j = out["jackknife_HONEST"]
    print(f"\nJACKKNIFE over {j['n_companies']} held-out companies: pooled Delta "
          f"{j['pooled_Delta']:+.4f}  mean {j['jackknife_mean_Delta']:+.4f}  SE {j['jackknife_SE']:.4f}  "
          f"range [{j['jackknife_range'][0]:+.4f}, {j['jackknife_range'][1]:+.4f}]  "
          f"most influential: {j['most_influential_company']} -> {j['Delta_without_it']:+.4f}")


if __name__ == "__main__":
    main()
