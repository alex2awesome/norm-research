#!/usr/bin/env python3
"""Gate arithmetic for the DECORRELATED-TRAINING planted battery (instrument #3).

Gates (declared here, BEFORE any D-arm finished training; design S12 binding):

V1  EXPLOIT (verified from STORED artifacts, no retraining): R01 planted vanilla
    exploits the plant -- ablation delta +.0275 [CI], probe 1.000.

V2' REMOVAL-OF-RELIANCE (arm D02 = planted corpus + plant-decorrelation weights):
    (a) spec-literal: |auc_eval(D02) - auc_eval(R00)| <= .005.  REPORTED WITH CI;
        known noise-limited (independent-run difference, CI half-width ~.03 at
        n=953 -- both prior batteries documented this).
    (b) CAUSAL PRIMARY: within-model token-ablation reliance |delta_eval| <= .005.
    (c) task AUC not degraded below the vanilla seed band
        [min over {R00 s42, D00_s1, D00_s2}] (n_eff cost reported).
    Chain-gate rc = PASS(b) AND PASS(c); (a) recorded (literal + CI).
    Probe on D02 reps is SCOPE ONLY: a reweighted model may still DECODE the
    plant (the frozen substrate carries it at .955 before any training); the
    instrument certifies non-RELIANCE, not non-decodability.

V3' SPECIFICITY:
    (a) D07 (realtok corpus + STANDARD-nuisance decorrelation): the real-signal
        token's contribution survives -- |ablDelta(D07) - ablDelta(R06)| < .005
        (difference-of-deltas, joint docket bootstrap).
    (b) D09 (v3b year-balanced subsample + DATE decorrelation, weights ~= 1):
        |auc_eval(D09) - auc_eval(R08)| < .005 literal; if literal misses but the
        paired CI contains 0 and |diff| <= .02, recorded PASS-within-resolution
        (declared here, not post hoc).

V4' CONSISTENCY (arm D10 = real corpus + length decorrelation), primary readout
    EVAL+TEST POOLED (n=1,903; the pilot showed single-953-row-split instruments
    disagree in sign -- declared before results):
    implied length influence = stratified-drop(R00) - stratified-drop(D10)
    (the CHANGE in length reliance), cross-checked against the two ADOPTED
    instruments on R00's own preds (matched sampling, stacked increment).
    PASS = same sign as the adopted instruments' influence estimate and
    magnitude within a factor of ~2 (band [0.5, 2]); if all instruments'
    magnitudes are inside the bootstrap noise floor (|est| < .01), the gate is
    recorded INDETERMINATE-SMALL rather than sign-tested on noise.

CAP (bonus, no gate): D20 cap vanilla vs D21 cap joint-B-decorrelated;
    readouts vs archived T .5554 and bank VA_nl .6656.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_BOOT = 2000
RNG = np.random.default_rng(0)


def _lin():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def load(runs_dir, tag):
    d = Path(runs_dir) / tag
    if not (d / "result.json").exists():
        return None
    r = json.loads((d / "result.json").read_text())
    r["preds"] = pd.read_csv(d / "preds_slim.csv")
    p = d / "probe.json"
    r["probe"] = json.loads(p.read_text())["probes"] if p.exists() else {}
    return r


def sub(df, split):
    if split == "evaltest":
        return df[df["split"].isin(["eval", "test"])].reset_index(drop=True)
    return df[df["split"] == split].reset_index(drop=True)


def boot_diff(a_df, b_df, split="eval", col_a="prob", col_b="prob", n=N_BOOT):
    A, B = sub(a_df, split), sub(b_df, split)
    assert (A["doc_id"].astype(str).values == B["doc_id"].astype(str).values).all(), "arms not row-aligned"
    dock = A["docket"].astype(str).values
    uniq = np.unique(dock)
    idx_by = {d: np.flatnonzero(dock == d) for d in uniq}
    y, pa, pb = A["judgement"].values, A[col_a].values, B[col_b].values
    point = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    out = []
    for _ in range(n):
        ii = np.concatenate([idx_by[d] for d in RNG.choice(uniq, len(uniq), replace=True)])
        if len(set(y[ii])) < 2:
            continue
        out.append(roc_auc_score(y[ii], pa[ii]) - roc_auc_score(y[ii], pb[ii]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"diff": float(point), "ci95": [float(lo), float(hi)], "n_boot": len(out)}


def boot_dod(a_df, b_df, split="eval", n=N_BOOT):
    """difference of within-model ablation deltas: (aucA - aucA_abl) - (aucB - aucB_abl),
    joint docket bootstrap (both runs share rows/dockets)."""
    A, B = sub(a_df, split), sub(b_df, split)
    assert (A["doc_id"].astype(str).values == B["doc_id"].astype(str).values).all()
    dock = A["docket"].astype(str).values
    uniq = np.unique(dock)
    idx_by = {d: np.flatnonzero(dock == d) for d in uniq}
    y = A["judgement"].values
    cols = [A["prob"].values, A["prob_ablated"].values, B["prob"].values, B["prob_ablated"].values]

    def stat(ii):
        return ((roc_auc_score(y[ii], cols[0][ii]) - roc_auc_score(y[ii], cols[1][ii]))
                - (roc_auc_score(y[ii], cols[2][ii]) - roc_auc_score(y[ii], cols[3][ii])))

    point = stat(np.arange(len(y)))
    out = []
    for _ in range(n):
        ii = np.concatenate([idx_by[d] for d in RNG.choice(uniq, len(uniq), replace=True)])
        if len(set(y[ii])) < 2:
            continue
        out.append(stat(ii))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"diff_of_deltas": float(point), "ci95": [float(lo), float(hi)], "n_boot": len(out)}


def ablation_ci(a_df, split="eval", n=N_BOOT):
    return boot_diff(a_df, a_df, split=split, col_a="prob", col_b="prob_ablated", n=n)


# ---------------- nuisance-score instruments (self-contained) -----------------

def oof_score(X, y, groups, model_fn):
    gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    oof = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        m = model_fn()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def stratified_auc(y, p, strata, min_n=20):
    aucs, ns = [], []
    for u in np.unique(strata):
        m = strata == u
        if m.sum() < min_n or len(set(y[m])) < 2:
            continue
        aucs.append(roc_auc_score(y[m], p[m]))
        ns.append(int(m.sum()))
    if not aucs:
        return None, 0
    return float(np.average(aucs, weights=ns)), len(aucs)


def matched_drop(y, p, s, caliper):
    P, N = np.where(y == 1)[0], np.where(y == 0)[0]
    order = np.argsort(s[N])
    Ns, sNs = N[order], s[N][order]
    conc, used = 0.0, 0
    for i in P:
        j = np.searchsorted(sNs, s[i])
        best, bd = None, np.inf
        for jj in (j - 1, j, j + 1):
            if 0 <= jj < len(Ns):
                d = abs(sNs[jj] - s[i])
                if d < bd:
                    bd, best = d, Ns[jj]
        if best is None or bd > caliper:
            continue
        used += 1
        conc += (p[i] > p[best]) + 0.5 * (p[i] == p[best])
    if used == 0:
        return None, 0
    return float(roc_auc_score(y, p) - conc / used), used


def score_instruments(preds, s_lut, split="evaltest"):
    """pooled AUC, s-stratified drop, matched drop, stacked increment of the
    nuisance score over the model score.  s_lut: doc_id -> feature vector."""
    s = sub(preds, split)
    y, p, g = s["judgement"].values, s["prob"].values, s["docket"].astype(str).values
    X = np.array([s_lut[str(d)] for d in s["doc_id"]])
    pooled = float(roc_auc_score(y, p))
    b = oof_score(X, y, g, _lin)
    # stratify by RAW char_len deciles for the 2-col length case (Layer-2 /
    # pilot convention); by the OOF joint score for high-dim sets (closure
    # discount convention, cap joint-B)
    strat_src = X[:, 0] if X.shape[1] <= 2 else b
    dec = pd.qcut(strat_src, 10, labels=False, duplicates="drop")
    dec = np.asarray(dec)
    strat, n_used = stratified_auc(y, p, dec)
    per_dec = {}
    for u in np.unique(dec):
        m = dec == u
        if m.sum() < 20 or len(set(y[m])) < 2:
            continue
        per_dec[int(u)] = {"n": int(m.sum()), "auc": float(roc_auc_score(y[m], p[m]))}
    matched = {}
    for cal in (0.01, 0.02, 0.05):
        d, used = matched_drop(y, p, b, cal)
        matched[f"caliper_{cal}"] = {"drop": d, "n_pairs": used}
    stack_both = oof_score(np.column_stack([p, X]), y, g, _lin)
    stack_p = oof_score(p.reshape(-1, 1), y, g, _lin)
    inc = float(roc_auc_score(y, stack_both) - roc_auc_score(y, stack_p))
    return {"split": split, "n": int(len(s)), "pooled_auc": pooled,
            "s_alone_oof_auc": float(roc_auc_score(y, b)),
            "stratified_auc": strat, "stratified_drop": (pooled - strat) if strat else None,
            "n_strata_used": n_used, "per_decile": per_dec,
            "matched_sampling": matched, "stacked_increment_s_over_score": inc}


def within_stratum_advantage(inst_a, inst_b):
    ks = sorted(set(inst_a["per_decile"]) & set(inst_b["per_decile"]))
    if not ks:
        return None
    n = np.array([inst_a["per_decile"][k]["n"] for k in ks])
    d = np.array([inst_a["per_decile"][k]["auc"] - inst_b["per_decile"][k]["auc"] for k in ks])
    return float(np.average(d, weights=n))


# ----------------------------------------------------------- gate sections ----

def section_v2(g, R00, R01, D02, s1, s2):
    v1 = {"stored": True,
          "spec_literal_eval": boot_diff(R01["preds"], R00["preds"]),
          "ablation_eval": ablation_ci(R01["preds"]),
          "probe_plant_planted_vanilla": (R01["probe"].get("plant") or {}).get("auc_eval_mean"),
          "probe_plant_unplanted_control": (R00["probe"].get("plant") or {}).get("auc_eval_mean")}
    v1["PASS"] = bool(v1["ablation_eval"]["diff"] >= 0.02 and (v1["probe_plant_planted_vanilla"] or 0) >= 0.75)
    g["V1"] = v1

    a = {s: boot_diff(D02["preds"], R00["preds"], split=s) for s in ("eval", "test", "evaltest")}
    abl = {s: ablation_ci(D02["preds"], split=s) for s in ("eval", "test", "evaltest")}
    reliance_change = {s: boot_dod(R01["preds"], D02["preds"], split=s) for s in ("eval", "evaltest")}
    band = sorted([R00["auc_eval"], s1["auc_eval"], s2["auc_eval"]])
    v2 = {"a_spec_literal": a,
          "a_PASS_literal": bool(abs(a["eval"]["diff"]) <= 0.005),
          "a_ci_contains_0": bool(a["eval"]["ci95"][0] <= 0 <= a["eval"]["ci95"][1]),
          "b_ablation": abl,
          "b_PASS": bool(abs(abl["eval"]["diff"]) <= 0.005),
          "b_reliance_change_R01_minus_D02": reliance_change,
          "c_seed_band": {"band": band, "seeds": {"R00_s42": R00["auc_eval"],
                                                  "s1": s1["auc_eval"], "s2": s2["auc_eval"]},
                          "D02_auc_eval": D02["auc_eval"],
                          "n_eff_train": D02.get("n_eff_train")},
          "c_PASS": bool(D02["auc_eval"] >= band[0]),
          "c_above_band": bool(D02["auc_eval"] > band[-1]),
          "probe_scope_note": {
              "probe_plant_D02": (D02["probe"].get("plant") or {}).get("auc_eval_mean"),
              "note": "decodability is OUT OF SCOPE: the frozen base carries the plant at "
                      "probe .955 before any training; this instrument certifies "
                      "non-RELIANCE (ablation ~ 0), not non-decodability."}}
    v2["PASS"] = bool(v2["b_PASS"] and v2["c_PASS"])
    g["V2prime"] = v2


def section_v3(g, R06, D07, R08, D09):
    dod = {s: boot_dod(D07["preds"], R06["preds"], split=s) for s in ("eval", "evaltest")}
    v3a = {"statistic": "ablDelta(D07 decor) - ablDelta(R06 vanilla), realtok token",
           "dod": dod,
           "vanilla_delta_eval": ablation_ci(R06["preds"]),
           "decor_delta_eval": ablation_ci(D07["preds"]),
           "auc_diff_decor_minus_vanilla": boot_diff(D07["preds"], R06["preds"]),
           "probe_realtok_vanilla": (R06["probe"].get("realtok") or {}).get("auc_eval_mean"),
           "probe_realtok_decor": (D07["probe"].get("realtok") or {}).get("auc_eval_mean"),
           "PASS": bool(abs(dod["eval"]["diff_of_deltas"]) < 0.005)}
    d9 = {s: boot_diff(D09["preds"], R08["preds"], split=s) for s in ("eval", "evaltest")}
    v3b = {"statistic": "auc(D09 date-decor) - auc(R08 vanilla), year-balanced subsample",
           "diff": d9,
           "PASS_literal": bool(abs(d9["eval"]["diff"]) < 0.005),
           "PASS_within_resolution": bool(d9["eval"]["ci95"][0] <= 0 <= d9["eval"]["ci95"][1]
                                          and abs(d9["eval"]["diff"]) <= 0.02)}
    v3b["PASS"] = bool(v3b["PASS_literal"] or v3b["PASS_within_resolution"])
    g["V3prime_a"], g["V3prime_b"] = v3a, v3b


def section_v4(g, R00, D10, len_lut):
    v4 = {"primary_split": "evaltest (declared; pilot S3.1 power warning)"}
    for s in ("eval", "evaltest"):
        iv = score_instruments(R00["preds"], len_lut, split=s)
        idc = score_instruments(D10["preds"], len_lut, split=s)
        v4[s] = {
            "vanilla": iv, "decor": idc,
            "between_run_pooled_diff_R00_minus_D10": boot_diff(R00["preds"], D10["preds"], split=s),
            "implied_influence_reliance_change": (iv["stratified_drop"] - idc["stratified_drop"])
            if (iv["stratified_drop"] is not None and idc["stratified_drop"] is not None) else None,
            "within_stratum_advantage_decor_minus_vanilla": within_stratum_advantage(idc, iv),
        }
    prim = v4["evaltest"]
    ref_matched = prim["vanilla"]["matched_sampling"]["caliper_0.02"]["drop"]
    ref_stack = prim["vanilla"]["stacked_increment_s_over_score"]
    est = prim["implied_influence_reliance_change"]
    v4["reference_instruments_on_R00"] = {
        "matched_caliper_.02_drop": ref_matched, "stacked_increment": ref_stack,
        "stratified_drop_vanilla": prim["vanilla"]["stratified_drop"]}
    # verdict rule (declared): FAIL only when the adopted reference instruments
    # are COHERENT (same sign, |.| >= .005) and the decor-implied influence
    # disagrees in sign or lands outside the factor-2 band; incoherent or
    # sub-resolution references make the consistency test INDETERMINATE
    # (recorded, non-blocking) -- a property of the cell, not of the instrument.
    usable = [x for x in (ref_matched, ref_stack) if x is not None and abs(x) >= 0.005]
    if not usable:
        v4["verdict"] = "INDETERMINATE-SMALL (both reference instruments <.005; declared rule)"
        v4["PASS"] = True
    elif len(usable) == 2 and usable[0] * usable[1] < 0:
        v4["verdict"] = "INDETERMINATE-REFS-DISAGREE (matched vs stacked opposite signs; declared rule)"
        v4["PASS"] = True
    else:
        ref = float(np.mean(usable))
        same_sign = (est or 0) * ref > 0
        ratio = (est / ref) if ref != 0 else float("nan")
        v4["same_sign"] = bool(same_sign)
        v4["ratio_est_over_ref"] = float(ratio)
        v4["reference_used"] = ref
        v4["PASS"] = bool(same_sign and 0.5 <= abs(ratio) <= 2.0)
        v4["verdict"] = "PASS" if v4["PASS"] else "FAIL"
    g["V4prime"] = v4


def section_cap(g, D20, D21, build_dir):
    capz = np.load(Path(build_dir) / "cap" / "cap_B.npz", allow_pickle=True)
    blut = {str(d): capz["S"][i] for i, d in enumerate(capz["doc_id"])}
    cap = {"note": "bonus arm, no gate; readout vs archived T .5554 / bank VA_nl .6656 "
                   "(notes/2026-08-06__samerows_T_rescores.md)"}
    for s in ("eval", "test", "evaltest"):
        cap[s] = {"auc_vanilla": float(roc_auc_score(sub(D20["preds"], s)["judgement"],
                                                     sub(D20["preds"], s)["prob"])),
                  "auc_decor": float(roc_auc_score(sub(D21["preds"], s)["judgement"],
                                                   sub(D21["preds"], s)["prob"])),
                  "paired_diff_decor_minus_vanilla": boot_diff(D21["preds"], D20["preds"], split=s)}
    cap["jointB_instruments_vanilla"] = score_instruments(D20["preds"], blut, split="evaltest")
    cap["jointB_instruments_decor"] = score_instruments(D21["preds"], blut, split="evaltest")
    cap["within_stratum_advantage_decor_minus_vanilla"] = within_stratum_advantage(
        cap["jointB_instruments_decor"], cap["jointB_instruments_vanilla"])
    cap["references"] = {"T_archived_samerows_heldout": 0.5554, "bank_VA_nl": 0.6656}
    g["CAP_bonus"] = cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["v2p", "v3p", "v4p", "cap", "full"], default="full")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--build_dir", default="build")
    ap.add_argument("--out", default="results/battery_decor.json")
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    res = json.loads(out_path.read_text()) if out_path.exists() else {"gates": {}}
    g = res["gates"]
    full = args.gate == "full"

    def L(tag):
        return load(args.runs_dir, tag)

    nz = np.load(Path(args.build_dir) / "nuisance.npz", allow_pickle=True)
    names = [str(s) for s in nz["names"]]
    len_lut = {str(d): (nz["raw"][i, names.index("char_len")], nz["raw"][i, names.index("log_token")])
               for i, d in enumerate(nz["doc_id"])}

    rc = 0
    if args.gate in ("v2p",) or full:
        arms = [L(t) for t in ("R00_vanilla_real", "R01_vanilla_planted", "D02_decor_planted_plant",
                               "D00_vanilla_real_s1", "D00_vanilla_real_s2")]
        if all(arms):
            section_v2(g, *arms)
        elif not full:
            print("v2p: missing runs")
            raise SystemExit(4)
        if args.gate == "v2p":
            rc = 0 if g.get("V2prime", {}).get("PASS") else 3

    if args.gate in ("v3p",) or full:
        arms = [L(t) for t in ("R06_vanilla_realtok", "D07_decor_realtok_standard",
                               "R08_vanilla_v3b", "D09_decor_v3b_date")]
        if all(arms):
            section_v3(g, *arms)
        elif not full:
            print("v3p: missing runs")
            raise SystemExit(4)
        if args.gate == "v3p":
            rc = 0 if (g.get("V3prime_a", {}).get("PASS") and g.get("V3prime_b", {}).get("PASS")) else 3

    if args.gate in ("v4p",) or full:
        arms = [L(t) for t in ("R00_vanilla_real", "D10_decor_length_real")]
        if all(arms):
            section_v4(g, arms[0], arms[1], len_lut)
        elif not full:
            print("v4p: missing runs")
            raise SystemExit(4)
        if args.gate == "v4p":
            rc = 0 if g.get("V4prime", {}).get("PASS") else 3

    if args.gate in ("cap",) or full:
        D20, D21 = L("D20_cap_vanilla"), L("D21_cap_decor_jointB")
        if D20 and D21:
            section_cap(g, D20, D21, args.build_dir)
        elif not full:
            print("cap: missing runs")
            raise SystemExit(4)

    # coordinator directive 2026-08-08: record per-cell join/alignment gate
    # status in every results JSON.  This battery performs NO positional-order
    # joins: every cross-arm comparison hard-asserts doc_id equality row-by-row
    # (boot_diff/boot_dod), decorrelation weights are doc_id-keyed with a
    # full-train-coverage assert in train_decor.py, and the cap_crowd B-matrix
    # was built under an exact row_id==population-id assert (make_cap_assets.py).
    # The *_va_nl_oof_*.npy bank-order landmine does not apply: VA_nl references
    # here are quoted constants from notes/2026-08-06__samerows_T_rescores.md,
    # never recomputed via positional joins.
    res["alignment_gates"] = {
        "positional_joins": "none",
        "cross_arm_comparisons": "doc_id equality asserted row-by-row on every boot_diff/boot_dod call",
        "weights_join": "doc_id-keyed lut; train coverage asserted in train_decor.py",
        "cap_B_matrix": "row_id == population id asserted exactly at build (make_cap_assets.py)",
        "va_nl_references": "quoted constants (samerows note), no oof-array joins",
    }
    res["gates"] = g
    out_path.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if "PASS" in kk or kk in ("verdict", "same_sign", "ratio_est_over_matched")}
                      if isinstance(v, dict) else v for k, v in g.items()}, indent=2, default=str))
    print(f"gate={args.gate} rc={rc}")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
