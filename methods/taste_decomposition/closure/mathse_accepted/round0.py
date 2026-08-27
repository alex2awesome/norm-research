#!/usr/bin/env python3
"""ROUND 0 for the math.SE ACCEPTED-VERDICT Layer-3 articulation-closure campaign.

Produces, in one pass, everything the frozen prereg requires before round 1 runs:

  * the OOF ALIGNMENT GATE (registry 2026-08-10) -- refused to proceed otherwise;
  * READOUT TIERS declared in advance (below), so the governing statistic is not
    chosen after seeing a gain;
  * T on every population as the MEAN OVER DENSE SEEDS OF THE AUC (the same
    convention VA_nl uses), with the seed ensemble reported and never quoted;
  * the round-0 bank state (V + A) fitted under the frozen closure spec:
    grouped-OOF inside FIT+MINE, refit-and-predict on MONITOR;
  * the closure-protocol round-0 residual on each tier, next to the Layer-1
    number it is NOT comparable to (prereg AMENDMENT 1);
  * the eval-only / test-only split of the residual (this cell's dense chain
    selected on EVAL, so TEST is the selection-free half) and the GATE-UNCERTAINTY
    characterisation of the seed-42 quantity while seeds 1-2 are still running;
  * the swap baseline (C+, C-);
  * a leave-one-question-out jackknife of the residual over held-out questions.

READOUT TIERS (declared BEFORE any round; the code asserts nothing about which
one "wins"):
  TIER 1, GOVERNING -- pooled AUC on MONITOR. This is the tier the Layer-1 gate
    quantity (Delta_beyond = T .6319 - VA_nl .6320 = -.0001) lives on, so it is
    the only tier on which the dispatch number and the curve are commensurable.
  TIER 2, SECONDARY -- n-weighted within-QUESTION AUC. y is "the ASKER picked
    THIS answer among the answers to their own question", i.e. a within-question
    choice, so the within-question readout is the one that matches the
    y-definition; it is reported every round but never substituted for tier 1.
  TIER 3, DIAGNOSTIC -- eval-only vs test-only, and HONEST (= M u MONITOR).
    HONEST is VA-honest only where VA came from the OOF side; it is quoted as the
    same-rows level, not as a round-over-round statistic.

CPU only.  Usage: OMP_NUM_THREADS=6 python3 round0.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import cells as C
import closure_core as L
from position_line import within_question_auc
from readout import swap_pair

HERE = Path(__file__).resolve().parent


def delta_per_seed(d, mask, va):
    y = d["y"][mask]
    per = [float(roc_auc_score(y, d["dense_seeds"][mask, j]))
           for j in range(d["dense_seeds"].shape[1])]
    t = float(np.mean(per))
    v = float(roc_auc_score(y, va[mask]))
    return t - v, t, v, per


def delta_within_question(d, mask, va):
    y, g = d["y"][mask], d["groups"][mask]
    per = [within_question_auc(y, d["dense_seeds"][mask, j], g)[0]
           for j in range(d["dense_seeds"].shape[1])]
    t = float(np.mean(per))
    v, info = within_question_auc(y, va[mask], g)
    return t - v, t, float(v), info


def jackknife_questions(d, mask, va, max_report=25):
    y, g = d["y"], np.array([str(x) for x in d["groups"]])
    qs = sorted({q for q in g[mask]})
    rows = []
    for q in qs:
        m = mask & (g != q)
        if len(set(y[m].tolist())) < 2:
            continue
        dl, t, v, _ = delta_per_seed(d, m, va)
        rows.append({"dropped_question": q, "n_remaining": int(m.sum()), "Delta": dl})
    dv = np.array([r["Delta"] for r in rows])
    n = len(dv)
    se = float(np.sqrt((n - 1) / n * ((dv - dv.mean()) ** 2).sum())) if n > 1 else None
    pooled = delta_per_seed(d, mask, va)[0]
    worst = max(rows, key=lambda r: abs(r["Delta"] - pooled))
    return {"pooled_Delta": pooled, "n_questions": n,
            "jackknife_mean_Delta": float(dv.mean()), "jackknife_SE": se,
            "jackknife_range": [float(dv.min()), float(dv.max())],
            "jackknife_pseudo_CI95": [float(dv.mean() - 1.96 * se),
                                      float(dv.mean() + 1.96 * se)],
            "most_influential_question": worst["dropped_question"],
            "Delta_without_it": worst["Delta"],
            "extremes": sorted(rows, key=lambda r: r["Delta"])[:max_report]
            + sorted(rows, key=lambda r: -r["Delta"])[:max_report]}


def gate_uncertainty(d, out):
    """What the gate quantity looks like at the seeds ON DISK, with its width
    honestly characterised -- so the 3-seed verdict is read against a stated
    prior rather than a point."""
    y = d["y"]
    ev, te = d["dense_split"] == "eval", d["dense_split"] == "test"
    led = d["layer1"]
    rec = {
        "layer1_gate_quantity": {
            "T_eval_seed42": led["T_dense"],
            "VA_nl_mean3_pooled": led["ledger"]["VA_nl_mean"],
            "Delta_beyond": led["ledger"]["Delta_beyond"],
            "note": "the dispatched gate quantity: dense EVAL AUC at seed 42 vs the "
                    "Layer-1 VA_nl (pooled GroupKFold OOF over all 13,001 rows, mean of "
                    "seeds 0/1/2). Protocol-specific; the closure curve is read on "
                    "MONITOR and its LEVEL is not comparable (AMENDMENT 1)."},
        "same_seed_split_half_width": {
            "T_eval_seed42": led["T_info"]["raw"]["seed42"]["eval_auc"],
            "T_test_seed42": led["T_info"]["raw"]["seed42"]["test_auc"],
            "eval_minus_test": (led["T_info"]["raw"]["seed42"]["eval_auc"]
                                - led["T_info"]["raw"]["seed42"]["test_auc"]),
            "n_each": led["T_info"]["raw"]["seed42"]["n_eval"],
            "note": "eval and test are equal-sized, disjoint, question-grouped halves of "
                    "ONE trained model. Their gap is pure sampling+selection noise at a "
                    "FIXED seed and is the natural yardstick for how much the 3-seed mean "
                    "can move. The chain selected on EVAL, so TEST is the selection-free "
                    "half and the LOWER of the two here."},
        "Delta_beyond_if_T_were_test_only": (led["T_info"]["raw"]["seed42"]["test_auc"]
                                             - led["ledger"]["VA_nl_mean"]),
        "Delta_beyond_if_T_were_eval_plus_test": None,
        "gate_rule": "NOT a stop/go gate on this cell. The 2026-08-06 FREEZE roster routes "
                     "cells with matched Delta_beyond <= .02 to the MAP-FOCUSED dual track "
                     "(Track-B emphasis, Track A still run) rather than excluding them; this "
                     "cell's dispatched Delta_beyond is -.0001, so it enters the campaign as "
                     "a map-focused cell BY THE FROZEN ROSTER, decided before any closure "
                     "number was computed. What round 0 has to settle is the HONEST residual "
                     "under the closure design, which the E-row full-grid arm (T .6439 vs "
                     "VA_nl .5737) suggests may be far from -.0001.",
    }
    held = np.isin(d["dense_split"], ["eval", "test"])
    t_all = float(np.mean([roc_auc_score(y[held], d["dense_seeds"][held, j])
                           for j in range(d["dense_seeds"].shape[1])]))
    rec["Delta_beyond_if_T_were_eval_plus_test"] = t_all - led["ledger"]["VA_nl_mean"]
    rec["T_eval_plus_test_seeds_on_disk"] = t_all
    rec["seeds_on_disk"] = list(d["dense_seed_ids"])
    rec["group_bootstrap_T_eval_vs_test_note"] = (
        "the eval/test gap of "
        f"{rec['same_seed_split_half_width']['eval_minus_test']:+.4f} at one seed already "
        "spans most of the distance between the dispatched +.0366 and the .02 gate, so the "
        "3-seed mean is the deciding read and no proceed/stop call is made from seed 42.")
    out["gate_uncertainty"] = rec
    return rec


def main():
    sk = C.sklearn_guard()
    d = C.load()
    sp = json.loads((HERE / "mathse_accepted_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    monfull = np.array([r["in_monitor_full"] for r in sp["rows"]])
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])
    ev, te = d["dense_split"] == "eval", d["dense_split"] == "test"
    y, g = d["y"], d["groups"]

    out = {"cell": "mathse_accepted", "sklearn": sk, "splits": sp["summary"],
           "alignment_gate": d["alignment_gate"],
           "readout_tiers": {
               "TIER1_GOVERNING": "pooled AUC on MONITOR (the tier the Layer-1 gate "
                                  "quantity lives on)",
               "TIER2_SECONDARY": "n-weighted within-QUESTION AUC (matches the "
                                  "within-question median-split y-definition)",
               "TIER3_DIAGNOSTIC": "eval-only / test-only / HONEST same-rows level"},
           "layer1_reference": d["layer1"]["ledger"]}

    # ------------------------------------------------------------------- T ----
    out["T"] = {"HONEST_eval_plus_test": C.T_by_seed(d, held),
                "MONITOR": C.T_by_seed(d, monm),
                "mining_slice_M": C.T_by_seed(d, fitm & held),
                "eval_only": C.T_by_seed(d, ev), "test_only": C.T_by_seed(d, te),
                "convention": "T = mean over dense seeds of AUC (mirrors VA_nl's "
                              "mean-over-seeds convention). The seed ENSEMBLE AUC is "
                              "reported but never quoted as T.",
                "selection_note": "the dense chain selected on EVAL, so TEST is the "
                                  "selection-free half."}

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
        "VA_nl_MONITOR_within_question": within_question_auc(y[monm], r0["nl_mon"], g[monm])[0],
        "grid_picks": r0["picks"],
        "impute_note": "this cell's Layer-1 linear leg uses SimpleImputer(median) inside "
                       "each fold, the same convention closure_core.clean_fit applies, so "
                       "there is no const-0.5 / median-impute fork here (unlike press).",
    }
    out["state0_bank"]["VA_nl_MONITOR_seed_spread"] = float(
        max(out["state0_bank"]["VA_nl_MONITOR_per_seed"])
        - min(out["state0_bank"]["VA_nl_MONITOR_per_seed"]))

    # ------------------------------------------------------------- Deltas -----
    out["round0_delta_TIER1_pooled"] = {}
    for name, m in (("MONITOR", monm), ("HONEST", held), ("mining_slice_M", fitm & held),
                    ("eval_only", ev), ("test_only", te)):
        dl, t, v, per = delta_per_seed(d, m, va0)
        out["round0_delta_TIER1_pooled"][name] = {
            "n": int(m.sum()), "n_questions": int(len({str(x) for x in g[m]})),
            "T": t, "T_per_seed": per, "VA_nl": v, "Delta_beyond": dl}
    out["round0_delta_TIER2_within_question"] = {}
    for name, m in (("MONITOR", monm), ("HONEST", held)):
        dl, t, v, info = delta_within_question(d, m, va0)
        out["round0_delta_TIER2_within_question"][name] = {
            "T": t, "VA_nl": v, "Delta_beyond": dl, **info}

    out["round0_delta_bootstrap_MONITOR"] = L.group_boot_ci(
        y[monm], d["dense"][monm], va0[monm], np.array([str(x) for x in g[monm]]))
    out["round0_delta_bootstrap_note"] = (
        "question-cluster paired bootstrap of AUC(dense seed-mean) - AUC(VA_nl) on MONITOR; "
        "read for WIDTH. With one dense seed on disk the seed-mean IS seed 42.")

    out["jackknife_MONITOR"] = jackknife_questions(d, monm, va0)

    # ------------------------------------------------------------ swap --------
    out["swap_baseline_MONITOR"] = swap_pair(y[monm], d["dense"][monm], va0[monm])
    out["swap_baseline_HONEST"] = swap_pair(y[held], d["dense"][held], va0[held])

    gate_uncertainty(d, out)

    np.savez_compressed(HERE / "mathse_accepted_r0_preds.npz",
                        va_nl=va0, va_lin=lin0, dense=d["dense"],
                        dense_seeds=d["dense_seeds"], y=y,
                        groups=np.array([str(x) for x in g], dtype=object),
                        split=split, held=held, monitor_full=monfull)
    (HERE / "mathse_accepted_r0_context.json").write_text(json.dumps(out, indent=1, default=float))
    slim = {k: v for k, v in out.items() if k != "jackknife_MONITOR"}
    slim["jackknife_MONITOR"] = {k: v for k, v in out["jackknife_MONITOR"].items()
                                 if k != "extremes"}
    print(json.dumps(slim, indent=1, default=float))


if __name__ == "__main__":
    main()
