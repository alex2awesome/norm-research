#!/usr/bin/env python3
"""Layer-3 closure round-1 readout (peer VERDICT), prereg step 6 + Track-B discounting.

Round 0 = the original bank (17 V features + 154 A criteria).
Round 1 = round 0 PLUS the A-routed round-1 criteria (routing = blind audit, with the
frontier arbiter's ruling applied to the single disagreement).
B-routed criteria are DECLARED NUISANCES and never enter the bank.

Everything is fit inside FIT+MINE (column screen, imputation medians, scaler, grid
selection) and evaluated on MONITOR.  VA_nl = mean AUC over seeds {0,1,2} with the
spread reported (FREEZE CHANGE 1).

Honesty note carried through every table: 943 of the 1,192 MONITOR rows were in the
dense model's TRAIN split, so `T` on all-MONITOR is in-sample-contaminated and is
printed only for transparency.  The same-rows Delta uses MONITOR INTERSECT
dense-held-out (249 rows), where both T and VA are honest.

CPU only.  Usage: python stage4_readout.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import closure_lib as L

HERE = Path(__file__).resolve().parent
N_BOOT = 2000


def group_boot_ci(y, pa, pb, groups, n=N_BOOT, seed=0):
    """Group-level paired bootstrap of AUC(pa) - AUC(pb) (FREEZE CHANGE 3)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        if len(set(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    out = np.array(out)
    return {
        "lo": float(np.percentile(out, 2.5)),
        "hi": float(np.percentile(out, 97.5)),
        "p_gt0": float((out > 0).mean()),
    }


def build_blocks():
    pop = L.load_population()
    summary, split, dsplit, mining = L.load_splits()
    z = np.load(HERE / "round1_scores.npz", allow_pickle=True)
    assert (z["i"] == np.arange(len(pop["y"]))).all(), "score rows out of alignment"
    Xnew = z["X"]
    cids = [str(s) for s in z["crit_ids"]]

    routing = json.loads((HERE / "round1_routing_final.json").read_text())
    a_ids = [r["blind_id"] for r in routing["final"] if r["final_route"] == "A"]
    b_ids = [r["blind_id"] for r in routing["final"] if r["final_route"] == "B"]
    ai = [cids.index(i) for i in a_ids]
    bi = [cids.index(i) for i in b_ids]
    return pop, split, dsplit, Xnew[:, ai], Xnew[:, bi], a_ids, b_ids, summary


def fit_block(raw_blocks, fitmask, monmask, y, nt):
    """clean_fit each raw block on FIT+MINE, concat, refit, predict MONITOR."""
    fit_parts, mon_parts = [], []
    for M in raw_blocks:
        if M.shape[1] == 0:
            continue
        keep, med = L.clean_fit(M[fitmask])
        if len(keep) == 0:
            continue
        fit_parts.append(L.clean_apply(M[fitmask], keep, med))
        mon_parts.append(L.clean_apply(M[monmask], keep, med))
    Xfit = np.column_stack(fit_parts)
    Xmon = np.column_stack(mon_parts)
    lin_mon, nl_mon, picks = L.fit_predict_monitor(Xfit, y[fitmask], nt[fitmask], Xmon)
    oof_lin = L.linear_oof(Xfit, y[fitmask], nt[fitmask])
    oof_nl = np.mean([L.gbm_oof(Xfit, y[fitmask], nt[fitmask], seed=s)[0] for s in L.SEEDS], axis=0)
    return {
        "n_features": int(Xfit.shape[1]),
        "lin_mon": lin_mon,
        "nl_mon_seeds": nl_mon,
        "nl_mon": nl_mon.mean(axis=0),
        "picks": picks,
        "oof_lin_fitmine": oof_lin,
        "oof_nl_fitmine": oof_nl,
    }


def main():
    pop, split, dsplit, XA, XB, a_ids, b_ids, summary = build_blocks()
    y, nt = pop["y"], pop["ntitle"]
    fitm = split == "fit_mine"
    monm = split == "monitor"
    held = np.isin(dsplit, ["eval", "test"])
    same = monm & held  # honest same-rows set

    dp = pd.read_csv(HERE / "peer_verdict_dense_preds.csv").set_index("i")
    dense = dp.loc[np.arange(len(y)), "dense_prob"].values

    r0 = fit_block([pop["V"], pop["A"]], fitm, monm, y, nt)
    r1 = fit_block([pop["V"], pop["A"], XA], fitm, monm, y, nt)
    rb = fit_block([XB], fitm, monm, y, nt)

    ymon = y[monm]
    gmon = nt[monm]
    same_in_mon = same[monm]

    def block_report(r, tag):
        rep = {
            "n_features": r["n_features"],
            "VA_lin_MONITOR_all": L.auc(ymon, r["lin_mon"]),
            "VA_nl_MONITOR_all": L.auc(ymon, r["nl_mon"]),
            "VA_nl_MONITOR_all_per_seed": [L.auc(ymon, p) for p in r["nl_mon_seeds"]],
            "VA_lin_MONITOR_samerows": L.auc(ymon[same_in_mon], r["lin_mon"][same_in_mon]),
            "VA_nl_MONITOR_samerows": L.auc(ymon[same_in_mon], r["nl_mon"][same_in_mon]),
            "VA_nl_MONITOR_samerows_per_seed": [
                L.auc(ymon[same_in_mon], p[same_in_mon]) for p in r["nl_mon_seeds"]
            ],
            "VA_lin_OOF_fitmine": L.auc(y[fitm], r["oof_lin_fitmine"]),
            "VA_nl_OOF_fitmine": L.auc(y[fitm], r["oof_nl_fitmine"]),
            "grid_picks": r["picks"],
        }
        rep["VA_nl_seed_spread_MONITOR_all"] = float(
            max(rep["VA_nl_MONITOR_all_per_seed"]) - min(rep["VA_nl_MONITOR_all_per_seed"])
        )
        return rep

    res = {
        "cell": "peer-review verdict (Layer-3 closure pilot, round 1, EXPLORATORY, pre-GEPA)",
        "splits": summary,
        "routing": {"n_A": len(a_ids), "A_ids": a_ids, "n_B": len(b_ids), "B_ids": b_ids},
        "T": {
            "registry_dense_eval_split": 0.753,
            "same_rows_all_population_CONTAMINATED": L.auc(y, dense),
            "same_rows_dense_heldout_1244": L.auc(y[held], dense[held]),
            "same_rows_MONITOR_all_CONTAMINATED": L.auc(ymon, dense[monm]),
            "same_rows_MONITOR_heldout_249": L.auc(y[same], dense[same]),
            "n_MONITOR_heldout": int(same.sum()),
            "n_MONITOR_dense_train_contaminated": int((monm & (dsplit == "train")).sum()),
        },
        "round0": block_report(r0, "round0"),
        "round1": block_report(r1, "round1"),
    }

    T_same = res["T"]["same_rows_MONITOR_heldout_249"]
    T_all = res["T"]["same_rows_MONITOR_all_CONTAMINATED"]
    for r, tag in ((res["round0"], "round0"), (res["round1"], "round1")):
        r["Delta_beyond_samerows"] = T_same - r["VA_nl_MONITOR_samerows"]
        r["Delta_beyond_MONITOR_all_CONTAMINATED_T"] = T_all - r["VA_nl_MONITOR_all"]
        r["Delta_total_samerows"] = T_same - r["VA_lin_MONITOR_samerows"]
        r["Delta_interact_MONITOR_all"] = r["VA_nl_MONITOR_all"] - r["VA_lin_MONITOR_all"]

    res["closure"] = {
        "VA_nl_gain_MONITOR_all": res["round1"]["VA_nl_MONITOR_all"] - res["round0"]["VA_nl_MONITOR_all"],
        "VA_nl_gain_MONITOR_samerows": res["round1"]["VA_nl_MONITOR_samerows"]
        - res["round0"]["VA_nl_MONITOR_samerows"],
        "VA_lin_gain_MONITOR_all": res["round1"]["VA_lin_MONITOR_all"] - res["round0"]["VA_lin_MONITOR_all"],
        "Delta_0_samerows": res["round0"]["Delta_beyond_samerows"],
        "Delta_1_samerows": res["round1"]["Delta_beyond_samerows"],
        "epsilon": 0.005,
        "gain_ci_MONITOR_all": group_boot_ci(ymon, r1["nl_mon"], r0["nl_mon"], gmon),
        "gain_ci_MONITOR_samerows": group_boot_ci(
            ymon[same_in_mon], r1["nl_mon"][same_in_mon], r0["nl_mon"][same_in_mon],
            gmon[same_in_mon]
        ),
    }

    # ------------------------------------------------------- Track B discounting -
    b_lin = L.auc(ymon, rb["lin_mon"])
    b_nl = L.auc(ymon, rb["nl_mon"])
    tb = {
        "n_B_features_after_screen": rb["n_features"],
        "spurious_alone_AUC_linear_MONITOR": b_lin,
        "spurious_alone_AUC_histgb_MONITOR": b_nl,
        "spurious_alone_AUC_linear_OOF_fitmine": L.auc(y[fitm], rb["oof_lin_fitmine"]),
        "spurious_alone_AUC_histgb_OOF_fitmine": L.auc(y[fitm], rb["oof_nl_fitmine"]),
        "per_feature_alone_AUC_MONITOR": {},
    }
    keepB, medB = L.clean_fit(XB[fitm])
    XBmon = L.clean_apply(XB[monm], keepB, medB)
    for k, j in enumerate(keepB):
        tb["per_feature_alone_AUC_MONITOR"][b_ids[j]] = L.auc(ymon, XBmon[:, k])

    # stratified readouts.  Primary = honest same-rows set (n=249) -> quintiles,
    # because deciles there leave ~25 rows per stratum.  Secondary = the full
    # dense-held-out set (n=1,244), where MONITOR rows use held-out VA predictions
    # and FIT+MINE rows use grouped-OOF VA predictions (both out-of-sample).
    def strat_table(mask_rows, va_pred, dense_pred, yy, joint, per_feat, q):
        out = {"n": int(mask_rows.sum()), "q": q, "by_feature": {}}
        out["pooled_T"] = L.auc(yy, dense_pred)
        out["pooled_VA"] = L.auc(yy, va_pred)
        out["pooled_Delta"] = out["pooled_T"] - out["pooled_VA"]
        st = L.decile_strata(joint, q=q)
        tA, iA = L.stratified_auc(yy, dense_pred, st, min_n=20)
        vA, _ = L.stratified_auc(yy, va_pred, st, min_n=20)
        out["joint_B_score"] = {"T_adj": tA, "VA_adj": vA, "Delta_adj": tA - vA, **iA}
        for k, name in enumerate(per_feat[1]):
            st = L.decile_strata(per_feat[0][:, k], q=q)
            tA, iA = L.stratified_auc(yy, dense_pred, st, min_n=20)
            vA, _ = L.stratified_auc(yy, va_pred, st, min_n=20)
            out["by_feature"][name] = {"T_adj": tA, "VA_adj": vA, "Delta_adj": tA - vA, **iA}
        return out

    b_names_kept = [b_ids[j] for j in keepB]
    joint_mon = rb["nl_mon"]
    tb["stratified_MONITOR_samerows_q5"] = strat_table(
        same, r1["nl_mon"][same_in_mon], dense[same], ymon[same_in_mon], joint_mon[same_in_mon],
        (XBmon[same_in_mon], b_names_kept), 5,
    )
    tb["stratified_MONITOR_all_q10_CONTAMINATED_T"] = strat_table(
        monm, r1["nl_mon"], dense[monm], ymon, joint_mon, (XBmon, b_names_kept), 10
    )

    # pooled honest set: FIT+MINE rows use OOF VA, MONITOR rows use held-out VA
    va_full = np.full(len(y), np.nan)
    va_full[fitm] = r1["oof_nl_fitmine"]
    va_full[monm] = r1["nl_mon"]
    XBfull = L.clean_apply(XB, keepB, medB)
    jb_full = np.full(len(y), np.nan)
    jb_full[fitm] = rb["oof_nl_fitmine"]
    jb_full[monm] = rb["nl_mon"]
    tb["stratified_dense_heldout_1244_q10"] = strat_table(
        held, va_full[held], dense[held], y[held], jb_full[held],
        (XBfull[held], b_names_kept), 10,
    )
    res["track_B"] = tb

    res["judge_calls"] = {
        "round1_population_prompts": int(len(y) * 25),
        "round1_anchor_prompts": 36 * 25,
        "cumulative_round1": int(len(y) * 25) + 36 * 25,
    }
    res["score_report"] = json.loads((HERE / "round1_score_report.json").read_text())

    (HERE / "round1_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k not in ("score_report", "splits")}, indent=2)[:8000])
    print("wrote", HERE / "round1_results.json")


if __name__ == "__main__":
    main()
