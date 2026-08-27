#!/usr/bin/env python3
"""Battery readout for the adversarial-debiasing pilot (V1-V4 + final run).

Every comparison is PAIRED (all arms share the identical rows and the identical
docket-grouped split; only the text overlay / the named channels differ), so the
uncertainty on a difference is estimated by a DOCKET-LEVEL bootstrap -- resample
dockets, not rows (FREEZE CHANGE 3 discipline).

Gates, declared before the runs finished:
  V1  auc_eval(R01 planted vanilla) - auc_eval(R00 unplanted vanilla) >= .02
  V2  |auc_eval(GRL planted) - auc_eval(R00)| <= .005
      AND plant-probe AUC <= .55 on the GRL arm, >= .75 on the vanilla planted arm
  V3a |ablation delta(R07 debiased) - ablation delta(R06 vanilla)| < .005
      (delta = auc_eval - auc_eval_with_the_real-signal-token_stripped)
  V3b |auc_eval(R09 date-debiased) - auc_eval(R08 vanilla)| < .005
  V4  sign(vanilla - length-debiased) matches the two frozen instruments and the
      magnitude is within a factor of ~2
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = HERE / "runs"
RES = HERE / "results"
RES.mkdir(exist_ok=True)
N_BOOT = 2000
RNG = np.random.default_rng(0)

# frozen instruments (Layer 2 appendix, notes/2026-08-06__layer2_robustness.md S3)
VANL_LENGTH_DROP = 0.073          # VA_nl pooled .726 -> length-stratified .653


def load(tag):
    d = RUNS / tag
    if not (d / "result.json").exists():
        return None
    r = json.loads((d / "result.json").read_text())
    r["preds"] = pd.read_csv(d / "preds_slim.csv")
    p = d / "probe.json"
    r["probe"] = json.loads(p.read_text())["probes"] if p.exists() else {}
    return r


def auc_on(df, col="prob", split="eval"):
    s = df[df["split"] == split]
    return float(roc_auc_score(s["judgement"], s[col]))


def boot_diff(a_df, b_df, split="eval", col_a="prob", col_b="prob", n=N_BOOT):
    """Docket-level bootstrap of auc(a) - auc(b) on `split`.  a and b must be the
    same rows in the same order (asserted)."""
    A = a_df[a_df["split"] == split].reset_index(drop=True)
    B = b_df[b_df["split"] == split].reset_index(drop=True)
    assert (A["doc_id"].values == B["doc_id"].values).all(), "arms are not row-aligned"
    dock = A["docket"].values
    uniq = np.unique(dock)
    idx_by = {d: np.flatnonzero(dock == d) for d in uniq}
    y, pa, pb = A["judgement"].values, A[col_a].values, B[col_b].values
    point = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    out = []
    for _ in range(n):
        pick = RNG.choice(uniq, len(uniq), replace=True)
        ii = np.concatenate([idx_by[d] for d in pick])
        if len(set(y[ii])) < 2:
            continue
        out.append(roc_auc_score(y[ii], pa[ii]) - roc_auc_score(y[ii], pb[ii]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"diff": float(point), "ci95": [float(lo), float(hi)], "n_boot": len(out)}


def same_regime_length_instruments(preds, split="eval"):
    """The two ADOPTED control instruments (Layer-2 nuisance stratification and the
    freeze's matched sampling / stacked increment) recomputed on THIS pilot's own
    vanilla model, on the same docket-grouped split the GRL gate is read on.

    This is the apples-to-apples comparison V4 needs: the archived same-rows dense
    predictions were produced by a chain trained on a RANDOM row split, so its
    dockets leak across train/eval and its T (.8167) lives in a different regime
    from a docket-disjoint one.  Those archived numbers are still reported, as
    historical context, by t_analog_length_drop()."""
    import sys
    sys.path.insert(0, str(REPO / "methods/taste_decomposition/closure/nc_responded"))
    from track_b_discount import oof_score, _lin, matched_auc, stacked_increment  # noqa: E402

    z = np.load(HERE / "build/nuisance.npz", allow_pickle=True)
    ids = np.array([str(s) for s in z["doc_id"]])
    names = [str(s) for s in z["names"]]
    lut = {d: (z["raw"][i, names.index("char_len")], z["raw"][i, names.index("log_token")])
           for i, d in enumerate(ids)}
    s = preds[preds["split"] == split].reset_index(drop=True)
    X = np.array([lut[str(d)] for d in s["doc_id"]])
    y, p, g = s["judgement"].values, s["prob"].values, s["docket"].values
    pooled = float(roc_auc_score(y, p))

    dec = pd.qcut(X[:, 0], 10, labels=False, duplicates="drop")
    aucs, ns = [], []
    for u in np.unique(dec):
        m = dec == u
        if m.sum() < 20 or len(set(y[m])) < 2:
            continue
        aucs.append(roc_auc_score(y[m], p[m]))
        ns.append(int(m.sum()))
    strat = float(np.average(aucs, weights=ns)) if aucs else None

    b = oof_score(X, y, g, _lin)
    rng = np.random.default_rng(0)
    matched = {}
    for cal in (0.01, 0.02, 0.05):
        mv, used = matched_auc(y, p, b, rng, caliper=cal)
        matched[f"caliper_{cal}"] = {"T_matched": mv, "n_pairs": used, "drop": pooled - mv}
    rev = float(roc_auc_score(y, oof_score(np.column_stack([p, X]), y, g, _lin))
                - roc_auc_score(y, oof_score(p.reshape(-1, 1), y, g, _lin)))
    return {"split": split, "n": int(len(s)), "pooled_auc": pooled,
            "length_stratified_auc": strat,
            "stratified_drop": (pooled - strat) if strat is not None else None,
            "n_qualifying_deciles": len(aucs),
            "length_alone_oof_auc": float(roc_auc_score(y, b)),
            "matched_sampling": matched,
            "stacked_increment_length_over_T": rev}


def t_analog_length_drop():
    """Layer-2 length-stratified drop computed on the FROZEN dense (T) predictions
    for this cell -- the T analogue of the VA_nl -.073."""
    d = pd.read_csv(REPO / "methods/taste_decomposition/closure/samerows_preds/nc_responded_dense_preds_slim.csv")
    z = np.load(HERE / "build/nuisance.npz", allow_pickle=True)
    ids = np.array([str(s) for s in z["doc_id"]])
    names = [str(s) for s in z["names"]]
    cl = dict(zip(ids, z["raw"][:, names.index("char_len")]))
    d["char_len"] = d["doc_id"].astype(str).map(cl)
    sub = d[d["dense_split"].isin(["eval", "test"])]
    y, p = sub["judgement"].values, sub["dense_prob"].values
    dec = pd.qcut(sub["char_len"], 10, labels=False, duplicates="drop").values
    aucs, ns = [], []
    for u in np.unique(dec):
        m = dec == u
        if m.sum() < 20 or len(set(y[m])) < 2:
            continue
        aucs.append(roc_auc_score(y[m], p[m]))
        ns.append(int(m.sum()))
    pooled = float(roc_auc_score(y, p))
    strat = float(np.average(aucs, weights=ns))
    return {"n": int(len(sub)), "pooled_T": pooled, "length_stratified_T": strat,
            "drop": pooled - strat, "n_qualifying_deciles": len(aucs),
            "source": "closure/samerows_preds/nc_responded_dense_preds_slim.csv, dense_split in {eval,test}"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["pooled", "bn"], default="pooled",
                    help="bn = bottleneck-architecture battery (B-tags); gate keys stay R-form")
    args = ap.parse_args()
    tags = ["R00_vanilla_real", "R01_vanilla_planted",
            "R02_grl_planted_full_l0.1", "R03_grl_planted_full_l0.5", "R04_grl_planted_full_l1.0",
            "R05_grl_planted_plantonly", "R06_vanilla_realtok", "R07_grl_realtok_standard",
            "R08_vanilla_v3b", "R09_grl_v3b_date", "R10_grl_length_real", "R11_grl_full_real"]
    pre = "B" if args.arch == "bn" else "R"
    R = {t: load(pre + t[1:]) for t in tags}
    have = {t: r for t, r in R.items() if r is not None}
    out = {"arch": args.arch, "runs_present": [r["tag"] for r in have.values()], "n_boot": N_BOOT}

    # ---- headline table -----------------------------------------------------
    rows = []
    for t, r in have.items():
        rows.append({
            "tag": r["tag"], "n": r["n"], "lambda": r["lambda_adv"],
            "channels": ",".join(r["config"]["channels"]) or "-",
            "auc_eval": r["auc_eval"], "auc_test": r["auc_test"], "auc_evaltest": r["auc_evaltest"],
            "best_step": r["best_checkpoint"]["step"],
            "probe_plant": (r["probe"].get("plant") or {}).get("auc_eval_mean"),
            "probe_realtok": (r["probe"].get("realtok") or {}).get("auc_eval_mean"),
            "probe_char_len": (r["probe"].get("char_len") or {}).get("auc_eval_mean"),
            "probe_docket_year": (r["probe"].get("docket_year") or {}).get("auc_eval_mean"),
            "ablation_delta_eval": (r.get("ablation") or {}).get("delta_eval"),
            "runtime_min": round(r["runtime_sec"] / 60, 1),
        })
    out["table"] = rows

    g = {}
    B = have.get("R00_vanilla_real")

    # ---- V1 -----------------------------------------------------------------
    if B and "R01_vanilla_planted" in have:
        d = boot_diff(have["R01_vanilla_planted"]["preds"], B["preds"])
        g["V1"] = {"statistic": "auc_eval(planted vanilla) - auc_eval(unplanted vanilla)",
                   "threshold": ">= .02", **d, "PASS": bool(d["diff"] >= 0.02)}

    # ---- V2 -----------------------------------------------------------------
    v2 = {}
    van_probe = (have.get("R01_vanilla_planted", {}).get("probe", {}).get("plant") or {}).get("auc_eval_mean")
    for t in ("R02_grl_planted_full_l0.1", "R03_grl_planted_full_l0.5", "R04_grl_planted_full_l1.0",
              "R05_grl_planted_plantonly"):
        if t not in have or B is None:
            continue
        d = boot_diff(have[t]["preds"], B["preds"])
        pp = (have[t]["probe"].get("plant") or {}).get("auc_eval_mean")
        v2[t] = {"vs_unplanted_baseline": d, "probe_plant_auc": pp,
                 "auc_gate_PASS": bool(abs(d["diff"]) <= 0.005),
                 "probe_gate_PASS": bool(pp is not None and pp <= 0.55
                                         and van_probe is not None and van_probe >= 0.75)}
        if "R01_vanilla_planted" in have:
            v2[t]["vs_planted_vanilla"] = boot_diff(have[t]["preds"], have["R01_vanilla_planted"]["preds"])
    if v2:
        g["V2"] = {"vanilla_planted_probe_plant_auc": van_probe, "arms": v2}

    # ---- V3a ----------------------------------------------------------------
    if "R06_vanilla_realtok" in have and "R07_grl_realtok_standard" in have:
        a, b = have["R07_grl_realtok_standard"], have["R06_vanilla_realtok"]
        da = boot_diff(a["preds"], a["preds"], col_a="prob", col_b="prob_ablated")
        db = boot_diff(b["preds"], b["preds"], col_a="prob", col_b="prob_ablated")
        g["V3a"] = {
            "statistic": "ablation delta (auc - auc_token_stripped), debiased vs vanilla",
            "vanilla_delta": db, "debiased_delta": da,
            "difference_of_deltas": float(da["diff"] - db["diff"]),
            "threshold": "< .005",
            "PASS": bool(abs(da["diff"] - db["diff"]) < 0.005),
            "probe_realtok_vanilla": (b["probe"].get("realtok") or {}).get("auc_eval_mean"),
            "probe_realtok_debiased": (a["probe"].get("realtok") or {}).get("auc_eval_mean"),
            "auc_diff_debiased_minus_vanilla": boot_diff(a["preds"], b["preds"]),
        }

    # ---- V3b ----------------------------------------------------------------
    if "R08_vanilla_v3b" in have and "R09_grl_v3b_date" in have:
        d = boot_diff(have["R09_grl_v3b_date"]["preds"], have["R08_vanilla_v3b"]["preds"])
        g["V3b"] = {"statistic": "auc_eval(date-debiased) - auc_eval(vanilla), year-balanced subsample",
                    "threshold": "|diff| < .005", **d, "PASS": bool(abs(d["diff"]) < 0.005),
                    "probe_docket_year_vanilla": (have["R08_vanilla_v3b"]["probe"].get("docket_year") or {}).get("auc_eval_mean"),
                    "probe_docket_year_debiased": (have["R09_grl_v3b_date"]["probe"].get("docket_year") or {}).get("auc_eval_mean")}

    # ---- V4 -----------------------------------------------------------------
    if B and "R10_grl_length_real" in have:
        d = boot_diff(B["preds"], have["R10_grl_length_real"]["preds"])
        try:
            same = same_regime_length_instruments(B["preds"])
        except Exception as e:      # closure/ helpers absent on sk3 -- run V4 locally
            same = {"error": f"{type(e).__name__}: {e}", "stratified_drop": None}
        try:
            archived_t = t_analog_length_drop()
        except Exception as e:
            archived_t = {"error": f"{type(e).__name__}: {e}"}
        inst = {"GRL_removal": {"value": d["diff"], "ci95": d["ci95"],
                                "definition": "auc_eval(vanilla) - auc_eval(length-debiased)"},
                "same_regime_on_this_pilots_vanilla": same,
                "archived_layer2_stratified_T": archived_t,
                "archived_layer2_stratified_VA_nl": {"drop": VANL_LENGTH_DROP,
                                                     "source": "notes/2026-08-06__layer2_robustness.md S3 (frozen)"}}
        ref = same["stratified_drop"]
        ratio = d["diff"] / ref if ref else float("nan")
        g["V4"] = {"instruments": inst,
                   "reference": "same-regime length-stratified drop on this pilot's own vanilla eval preds",
                   "same_sign": bool(ref is not None and d["diff"] * ref > 0),
                   "ratio_GRL_over_stratified": float(ratio),
                   "PASS": bool(ref is not None and d["diff"] * ref > 0 and 0.5 <= abs(ratio) <= 2.0),
                   "probe_char_len_vanilla": (B["probe"].get("char_len") or {}).get("auc_eval_mean"),
                   "probe_char_len_debiased": (have["R10_grl_length_real"]["probe"].get("char_len") or {}).get("auc_eval_mean")}

    # ---- final --------------------------------------------------------------
    if B and "R11_grl_full_real" in have:
        d = boot_diff(B["preds"], have["R11_grl_full_real"]["preds"])
        g["FINAL"] = {"statistic": "auc_eval(vanilla T) - auc_eval(full-nuisance debiased T)",
                      **d,
                      "auc_vanilla_eval": B["auc_eval"],
                      "auc_debiased_eval": have["R11_grl_full_real"]["auc_eval"],
                      "probes_debiased": have["R11_grl_full_real"]["probe"],
                      "probes_vanilla": B["probe"]}

    out["gates"] = g
    fname = "battery.json" if args.arch == "pooled" else "battery_bn.json"
    (RES / fname).write_text(json.dumps(out, indent=2, default=float))
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print(json.dumps(g, indent=2, default=float)[:6000])


if __name__ == "__main__":
    main()
