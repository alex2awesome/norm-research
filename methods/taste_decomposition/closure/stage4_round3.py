#!/usr/bin/env python3
"""Layer-3 closure ROUND 3 readout + full closure curve (peer VERDICT).

Computes rounds 0, 1 and 2 in ONE pass with the identical estimator and the
identical split, so the curve Delta_0 -> Delta_1 -> Delta_2 is internally
consistent and the saturation rule (2 consecutive MONITOR VA_nl gains < eps=.005)
is evaluated on comparable numbers.

  round 0 : V + the 154-criterion A bank
  round 1 : round 0 + the 14 A-routed round-1 criteria
  round 2 : round 1 + the A-routed round-2 criteria
  round 3 : round 2 + the A-routed round-3 criteria
  B set   : round-1 + round-2 + round-3 nuisances

Everything is fit inside FIT+MINE (column screen, medians, scaler, grid choice) and
read on MONITOR.  VA_nl = mean over seeds {0,1,2}, spread reported.

Per the prereg amendment (2026-08-05): closure-split Delta LEVELS are protocol
specific and are not comparable to the Layer-1 Delta_beyond; this script therefore
reports (a) round-over-round CHANGES on MONITOR, which is what the stopping rule is
defined on, and (b) the same-rows HONEST level on the 1,244 dense-held-out rows.

CPU only.  Usage: python stage4_round2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import closure_lib as L
from stage4_readout import build_blocks, fit_block, group_boot_ci

HERE = Path(__file__).resolve().parent


def load_round_blocks(r):
    """Round-r A/B score columns, split by that round's final routing.

    PREREG GATE: criteria flagged `collapsed` by that round's score report are
    DROPPED before use ("guided-JSON collapse check on every criterion's score
    distribution before use").  The dropped ids are returned so the readout can
    report them.
    """
    z = np.load(HERE / f"round{r}_scores.npz", allow_pickle=True)
    cids = [str(s) for s in z["crit_ids"]]
    X = z["X"]
    routing = json.loads((HERE / f"round{r}_routing_final.json").read_text())
    rep_path = HERE / f"round{r}_score_report.json"
    collapsed = set()
    if rep_path.exists():
        rep = json.loads(rep_path.read_text())
        collapsed = {k for k, v in rep["per_criterion"].items() if v.get("collapsed")}
    a_ids = [x["blind_id"] for x in routing["final"]
             if x["final_route"] == "A" and x["blind_id"] not in collapsed]
    b_ids = [x["blind_id"] for x in routing["final"]
             if x["final_route"] == "B" and x["blind_id"] not in collapsed]
    dropped = sorted(collapsed)
    if dropped:
        print(f"[collapse gate] round {r}: dropped {dropped}")
    return (X[:, [cids.index(i) for i in a_ids]], X[:, [cids.index(i) for i in b_ids]],
            a_ids, b_ids, dropped)


def main():
    pop, split, dsplit, XA1, XB1, a1_ids, b1_ids, summary = build_blocks()
    XA2, XB2, a2_ids, b2_ids, drop2 = load_round_blocks(2)
    XA3, XB3, a3_ids, b3_ids, drop3 = load_round_blocks(3)
    y, nt = pop["y"], pop["ntitle"]
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(dsplit, ["eval", "test"])
    same = monm & held
    same_in_mon = same[monm]
    ymon, gmon = y[monm], nt[monm]

    dense = pd.read_csv(HERE / "peer_verdict_dense_preds.csv").set_index("i").loc[
        np.arange(len(y)), "dense_prob"
    ].values

    banks = {
        "round0": [pop["V"], pop["A"]],
        "round1": [pop["V"], pop["A"], XA1],
        "round2": [pop["V"], pop["A"], XA1, XA2],
        "round3": [pop["V"], pop["A"], XA1, XA2, XA3],
    }
    fits = {k: fit_block(v, fitm, monm, y, nt) for k, v in banks.items()}
    rb = fit_block([XB1, XB2, XB3], fitm, monm, y, nt)

    T_same = L.auc(y[same], dense[same])
    T_held = L.auc(y[held], dense[held])
    T_mon_all = L.auc(ymon, dense[monm])

    res = {
        "cell": "peer-review verdict (Layer-3 closure pilot, ROUND 3, EXPLORATORY, pre-GEPA)",
        "prereg_amendment": "2026-08-05: quote round-over-round changes + same-rows honest level only",
        "routing_round3": {"n_A": len(a3_ids), "A_ids": a3_ids, "n_B": len(b3_ids), "B_ids": b3_ids},
        "collapse_gate_dropped": {"round2": drop2, "round3": drop3},
        "T": {
            "registry_dense_eval_split": 0.753,
            "same_rows_dense_heldout_1244": T_held,
            "same_rows_MONITOR_heldout_249": T_same,
            "same_rows_MONITOR_all_CONTAMINATED": T_mon_all,
        },
        "rounds": {},
    }

    for k, r in fits.items():
        va_full = np.full(len(y), np.nan)
        va_full[fitm] = r["oof_nl_fitmine"]
        va_full[monm] = r["nl_mon"]
        res["rounds"][k] = {
            "n_features": r["n_features"],
            "VA_lin_MONITOR_all": L.auc(ymon, r["lin_mon"]),
            "VA_nl_MONITOR_all": L.auc(ymon, r["nl_mon"]),
            "VA_nl_MONITOR_all_per_seed": [L.auc(ymon, p) for p in r["nl_mon_seeds"]],
            "VA_lin_MONITOR_samerows": L.auc(ymon[same_in_mon], r["lin_mon"][same_in_mon]),
            "VA_nl_MONITOR_samerows": L.auc(ymon[same_in_mon], r["nl_mon"][same_in_mon]),
            "VA_lin_OOF_fitmine_MININGCONTAM": L.auc(y[fitm], r["oof_lin_fitmine"]),
            "VA_nl_OOF_fitmine_MININGCONTAM": L.auc(y[fitm], r["oof_nl_fitmine"]),
            "VA_nl_honest_level_heldout1244": L.auc(y[held], va_full[held]),
            "Delta_samerows_249": T_same - L.auc(ymon[same_in_mon], r["nl_mon"][same_in_mon]),
            "Delta_honest_level_heldout1244": T_held - L.auc(y[held], va_full[held]),
        }
        res["rounds"][k]["VA_nl_seed_spread_MONITOR_all"] = float(
            max(res["rounds"][k]["VA_nl_MONITOR_all_per_seed"])
            - min(res["rounds"][k]["VA_nl_MONITOR_all_per_seed"])
        )

    R = res["rounds"]
    eps = 0.005
    curve = {"epsilon": eps}
    for a, b, tag in (("round0", "round1", "r1"), ("round1", "round2", "r2"),
                      ("round2", "round3", "r3")):
        gain = R[b]["VA_nl_MONITOR_all"] - R[a]["VA_nl_MONITOR_all"]
        curve[f"VA_nl_gain_MONITOR_{tag}"] = gain
        curve[f"VA_lin_gain_MONITOR_{tag}"] = R[b]["VA_lin_MONITOR_all"] - R[a]["VA_lin_MONITOR_all"]
        curve[f"gain_ci_MONITOR_{tag}"] = group_boot_ci(
            ymon, fits[b]["nl_mon"], fits[a]["nl_mon"], gmon
        )
        curve[f"sub_epsilon_{tag}"] = bool(gain < eps)
    RK = ("round0", "round1", "round2", "round3")
    curve["Delta_curve_samerows_249"] = [R[k]["Delta_samerows_249"] for k in RK]
    curve["Delta_curve_honest_heldout1244"] = [R[k]["Delta_honest_level_heldout1244"] for k in RK]
    curve["VA_nl_curve_MONITOR"] = [R[k]["VA_nl_MONITOR_all"] for k in RK]
    # saturation = 2 CONSECUTIVE sub-epsilon rounds; count the trailing run only
    flags = [curve["sub_epsilon_r1"], curve["sub_epsilon_r2"], curve["sub_epsilon_r3"]]
    run = 0
    for f in flags:
        run = run + 1 if f else 0
    curve["sub_epsilon_flags_r1_r2_r3"] = flags
    curve["consecutive_sub_epsilon_trailing"] = run
    curve["SATURATION_DECLARED"] = bool(run >= 2)
    res["closure_curve"] = curve

    # ------------------------------------------------------ Track B (r1 + r2) ---
    # round-1 and round-2 blind ids BOTH run P01..P25, so they collide as dict keys
    # (5 ids are shared).  Tag by round -- the model fits use the arrays positionally
    # and were always correct, but the per-feature REPORTING needs unique labels.
    all_b = ([f"r1:{i}" for i in b1_ids] + [f"r2:{i}" for i in b2_ids]
             + [f"r3:{i}" for i in b3_ids])
    tb = {
        "n_B_criteria_total": len(all_b),
        "n_B_features_after_screen": rb["n_features"],
        "spurious_alone_AUC_linear_MONITOR": L.auc(ymon, rb["lin_mon"]),
        "spurious_alone_AUC_histgb_MONITOR": L.auc(ymon, rb["nl_mon"]),
        "per_feature_alone_AUC_MONITOR": {},
    }
    XBall = np.column_stack([XB1, XB2, XB3])
    keepB, medB = L.clean_fit(XBall[fitm])
    XBmon = L.clean_apply(XBall[monm], keepB, medB)
    XBfull = L.clean_apply(XBall, keepB, medB)
    kept = [all_b[j] for j in keepB]
    for k, cid in enumerate(kept):
        tb["per_feature_alone_AUC_MONITOR"][cid] = L.auc(ymon, XBmon[:, k])

    r2 = fits["round3"]
    va_full = np.full(len(y), np.nan)
    va_full[fitm] = r2["oof_nl_fitmine"]
    va_full[monm] = r2["nl_mon"]
    jb_full = np.full(len(y), np.nan)
    jb_full[fitm] = rb["oof_nl_fitmine"]
    jb_full[monm] = rb["nl_mon"]

    def strat(mask, va, dn, yy, joint, feats, names, q):
        out = {"n": int(mask.sum()), "q": q,
               "pooled_T": L.auc(yy, dn), "pooled_VA": L.auc(yy, va)}
        out["pooled_Delta"] = out["pooled_T"] - out["pooled_VA"]
        st = L.decile_strata(joint, q=q)
        tA, iA = L.stratified_auc(yy, dn, st, min_n=20)
        vA, _ = L.stratified_auc(yy, va, st, min_n=20)
        out["joint_B_score"] = {"T_adj": tA, "VA_adj": vA, "Delta_adj": tA - vA, **iA}
        out["by_feature"] = {}
        for k, nm in enumerate(names):
            st = L.decile_strata(feats[:, k], q=q)
            tA, iA = L.stratified_auc(yy, dn, st, min_n=20)
            vA, _ = L.stratified_auc(yy, va, st, min_n=20)
            out["by_feature"][nm] = {"T_adj": tA, "VA_adj": vA, "Delta_adj": tA - vA, **iA}
        return out

    tb["stratified_dense_heldout_1244_q10"] = strat(
        held, va_full[held], dense[held], y[held], jb_full[held], XBfull[held], kept, 10)
    tb["stratified_MONITOR_samerows_q5"] = strat(
        same, r2["nl_mon"][same_in_mon], dense[same], ymon[same_in_mon],
        rb["nl_mon"][same_in_mon], XBmon[same_in_mon], kept, 5)
    res["track_B"] = tb

    n_r3 = len(a3_ids) + len(b3_ids) + len(drop3)  # all scored; gate drops post-hoc
    res["judge_calls"] = {
        "round1": 151650, "round2": 151650,
        "round3_population_prompts": int(len(y) * n_r3),
        "round3_anchor_prompts": 36 * n_r3,
        "cumulative": 151650 * 2 + int(len(y) * n_r3) + 36 * n_r3,
    }
    res["score_report_round3"] = json.loads((HERE / "round3_score_report.json").read_text())

    (HERE / "round3_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"T": res["T"], "closure_curve": curve,
                      "rounds": {k: {kk: vv for kk, vv in v.items() if "per_seed" not in kk}
                                 for k, v in R.items()}}, indent=2))
    print("wrote", HERE / "round3_results.json")


if __name__ == "__main__":
    main()
