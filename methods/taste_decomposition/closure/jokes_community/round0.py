#!/usr/bin/env python3
"""ROUND 0 for the reddit-jokes community Layer-3 articulation-closure campaign.

Produces, in one pass, everything the frozen prereg requires before round 1 runs:

  * the OOF ALIGNMENT GATE (registry 2026-08-10) -- refuses to proceed otherwise;
  * READOUT TIERS declared in advance (below), so the governing statistic is not chosen
    after seeing a gain;
  * T on every population as the MEAN OVER DENSE SEEDS OF THE AUC (the same convention
    VA_nl uses), with the seed ensemble reported and never quoted -- this cell's master
    ledger quotes the ENSEMBLE (.7469 at the 3,163 E rows); the campaign quotes the
    mean-of-AUCs, and the two are never mixed in one figure;
  * the round-0 bank state (V + A) fitted under the frozen closure spec: grouped-OOF
    inside FIT+MINE, refit-and-predict on MONITOR;
  * the closure-protocol round-0 residual on each tier, next to the Layer-1 number it is
    NOT comparable to (prereg AMENDMENT 1);
  * the eval-only / test-only split of the residual (the dense chain selected on EVAL, so
    TEST is the selection-free half);
  * the swap baseline (C+, C-);
  * a leave-one-topic-out jackknife of the residual over held-out topics, and BOTH the
    group-cluster and item-level bootstrap bands (this cell has only ~6 MONITOR topics,
    so the group band is coarse by construction; it is still the quoted one).

READOUT TIERS (declared BEFORE any round; the code asserts nothing about which one
"wins"):
  TIER 1, GOVERNING -- pooled AUC on MONITOR.  The stopping rule reads this tier.
  TIER 2, SECONDARY -- n-weighted within-TOPIC AUC.  y is a quartile split taken inside
    (length_bin x format x topic) strata, so the within-topic readout is the one that
    matches the y-definition; it is reported every round but never substituted for
    tier 1.
  TIER 3, DIAGNOSTIC -- eval-only vs test-only, and HONEST (= M u MONITOR).  HONEST is
    VA-honest only where VA came from the OOF side; it is quoted as the same-rows level,
    not as a round-over-round statistic.

CPU only.  Usage: OMP_NUM_THREADS=6 python3 round0.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import cells as C
import closure_core as L
from era_line import within_topic_auc
from readout import swap_pair

HERE = Path(__file__).resolve().parent


def delta_per_seed(d, mask, va):
    y = d["y"][mask]
    per = [float(roc_auc_score(y, d["dense_seeds"][mask, j]))
           for j in range(d["dense_seeds"].shape[1])]
    t = float(np.mean(per))
    v = float(roc_auc_score(y, va[mask]))
    return t - v, t, v, per


def delta_within_topic(d, mask, va):
    y, g = d["y"][mask], d["groups"][mask]
    per = [within_topic_auc(y, d["dense_seeds"][mask, j], g)[0]
           for j in range(d["dense_seeds"].shape[1])]
    t = float(np.mean(per))
    v, info = within_topic_auc(y, va[mask], g)
    return t - v, t, float(v), info


def item_boot_ci(y, pa, pb, n=2000, seed=0):
    """Item-level paired bootstrap, reported BESIDE the group band because this cell has
    only ~6 MONITOR grouping units.  The group band stays the quoted one."""
    rng = np.random.default_rng(seed)
    out = []
    idx0 = np.arange(len(y))
    for _ in range(n):
        idx = rng.choice(idx0, size=len(idx0), replace=True)
        if len(set(y[idx].tolist())) < 2:
            continue
        out.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    out = np.array(out)
    return {"lo": float(np.percentile(out, 2.5)), "hi": float(np.percentile(out, 97.5)),
            "p_gt0": float((out > 0).mean()), "mean": float(out.mean()),
            "note": "ITEM-level; anticonservative w.r.t. topic clustering, reported for "
                    "width only, never quoted in place of the group band"}


def jackknife_topics(d, mask, va):
    y, g = d["y"], np.array([str(x) for x in d["groups"]])
    qs = sorted({q for q in g[mask]})
    rows = []
    for q in qs:
        m = mask & (g != q)
        if len(set(y[m].tolist())) < 2:
            continue
        dl, t, v, _ = delta_per_seed(d, m, va)
        rows.append({"dropped_topic": q, "n_remaining": int(m.sum()), "Delta": dl})
    dv = np.array([r["Delta"] for r in rows])
    n = len(dv)
    se = float(np.sqrt((n - 1) / n * ((dv - dv.mean()) ** 2).sum())) if n > 1 else None
    pooled = delta_per_seed(d, mask, va)[0]
    worst = max(rows, key=lambda r: abs(r["Delta"] - pooled))
    return {"pooled_Delta": pooled, "n_topics": n,
            "jackknife_mean_Delta": float(dv.mean()), "jackknife_SE": se,
            "jackknife_range": [float(dv.min()), float(dv.max())],
            "jackknife_pseudo_CI95": [float(dv.mean() - 1.96 * se),
                                      float(dv.mean() + 1.96 * se)] if se else None,
            "most_influential_topic": worst["dropped_topic"],
            "Delta_without_it": worst["Delta"],
            "all": sorted(rows, key=lambda r: r["Delta"])}


def main():
    sk = C.sklearn_guard()
    d = C.load()
    sp = json.loads((HERE / "jokes_community_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    monfull = np.array([r["in_monitor_full"] for r in sp["rows"]])
    fitm, monm = split == "fit_mine", split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])
    ev, te = d["dense_split"] == "eval", d["dense_split"] == "test"
    y, g = d["y"], d["groups"]

    out = {"cell": "jokes_community", "sklearn": sk, "splits": sp["summary"],
           "alignment_gate": d["alignment_gate"],
           "readout_tiers": {
               "TIER1_GOVERNING": "pooled AUC on MONITOR (the tier the stopping rule reads)",
               "TIER2_SECONDARY": "n-weighted within-TOPIC AUC (matches the "
                                  "within-stratum quartile y-definition)",
               "TIER3_DIAGNOSTIC": "eval-only / test-only / HONEST same-rows level"},
           "layer1_reference": d["layer1"]["ledger"],
           "master_ledger_reference": {
               "n_E": 3163,
               "T_ENSEMBLE_at_E": 0.7468591396954822,
               "VA_nl_at_E": 0.6888412257515344,
               "note": "results/vat_fullgrid_jokes_community.json. Its T is the SEED "
                       "ENSEMBLE AUC and its VA_nl is the Layer-1 OOF restricted to the E "
                       "rows -- a different protocol from this campaign's MONITOR readout. "
                       "The two levels are never differenced or plotted together."}}

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
        "VA_nl_MONITOR_within_topic": within_topic_auc(y[monm], r0["nl_mon"], g[monm])[0],
        "grid_picks": r0["picks"],
        "impute_note": "this cell's Layer-1 legs median-impute with a missingness "
                       "indicator inside each fold, the same convention "
                       "closure_core.clean_fit applies, so there is no const-0.5 / "
                       "median-impute fork here (unlike press).",
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
            "n": int(m.sum()), "n_topics": int(len({str(x) for x in g[m]})),
            "T": t, "T_per_seed": per, "VA_nl": v, "Delta_beyond": dl}
    out["round0_delta_TIER2_within_topic"] = {}
    for name, m in (("MONITOR", monm), ("HONEST", held)):
        dl, t, v, info = delta_within_topic(d, m, va0)
        out["round0_delta_TIER2_within_topic"][name] = {
            "T": t, "VA_nl": v, "Delta_beyond": dl, **info}

    out["round0_delta_bootstrap_MONITOR_group"] = L.group_boot_ci(
        y[monm], d["dense"][monm], va0[monm], np.array([str(x) for x in g[monm]]))
    out["round0_delta_bootstrap_MONITOR_item"] = item_boot_ci(
        y[monm], d["dense"][monm], va0[monm])
    out["round0_delta_bootstrap_note"] = (
        "topic-cluster paired bootstrap of AUC(dense seed-mean) - AUC(VA_nl) on MONITOR, "
        "read for WIDTH. MONITOR holds ~6 topics, so the group band is coarse; the item "
        "band is printed beside it and is never quoted in its place.")

    out["jackknife_MONITOR"] = jackknife_topics(d, monm, va0)

    # ------------------------------------------------------------ swap --------
    out["swap_baseline_MONITOR"] = swap_pair(y[monm], d["dense"][monm], va0[monm])
    out["swap_baseline_HONEST"] = swap_pair(y[held], d["dense"][held], va0[held])

    np.savez_compressed(HERE / "jokes_community_r0_preds.npz",
                        va_nl=va0, va_lin=lin0, dense=d["dense"],
                        dense_seeds=d["dense_seeds"], y=y,
                        groups=np.array([str(x) for x in g], dtype=object),
                        split=split, held=held, monitor_full=monfull)
    (HERE / "jokes_community_r0_context.json").write_text(
        json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
