#!/usr/bin/env python3
"""Style Invitational v2: Layer-1 ledger + the LENGTH-SURVIVAL acceptance test.

This cell's rebuild exists because v1 was declared terminal as "bank = length
model (0/32 rubrics survive length strata)". The acceptance test is therefore
not an appendix here -- it is the point. Every criterion and every fitted block
is reported POOLED and INSIDE LENGTH STRATA, on the parse-artifact-free
population.

REUSE: all estimators imported, never reimplemented --
  * `layer1_gemma_cells` (L): outer_folds, linear_oof_family1, gbm_oof_family1
  * `scaleupC_layer1` (SC): load_scaleupC_bank(out=...), dense_T,
    group_bootstrap_delta, group_bootstrap_auc, run_cell
  * stratification follows `closure/maps_hw_si/closure_core.py`
    (decile_strata + n-weighted stratified_auc, min_n=20) -- the SAME convention
    the v1 verdict was produced under, so v1 and v2 numbers are comparable.

  python3 methods/taste_decomposition/si_v2_layer1.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
import layer1_gemma_cells as L  # noqa: E402
import scaleupC_layer1 as SC  # noqa: E402

REPO = SC.REPO
RESULTS = SC.RESULTS_DIR
SI_OUT = Path("outputs/va_gemma_banks_si_v2")
V2 = REPO / "datasets/humor/style_invitational/va_v2"
DENSE = V2 / "dense_standard_si_clean"
SLUG = "si_v2_toptier"
COLLAPSE_MODAL_MAX = 0.98

# v1 numbers, for the side-by-side. Source: notes/2026-08-08__maps_hw_si.md S4/S4.1
# and datasets/humor/style_invitational/va/RESULTS_gemma.md.
V1 = {"population_n": 9637, "n_criteria": 32,
      "V_nl": 0.6315, "A_nl": 0.6131, "VA_nl": 0.6401, "T": 0.6490,
      "A_lin": 0.6090, "V_lin": 0.6227, "VA_lin": 0.6161,
      "n_criteria_pooled_ge_.05": 2, "n_criteria_within_V_strata_ge_.05": 0,
      "median_abs_dev_pooled": 0.0120, "median_abs_dev_within_V": 0.0091,
      "bank_within_length_strata": 0.5409, "T_within_length_strata": 0.5889}


def modal_share(col):
    v = col.copy()
    fin = np.isfinite(v)
    v[~fin] = float(np.nanmedian(v)) if fin.any() else 0.5
    _, cnts = np.unique(v, return_counts=True)
    return float(cnts.max() / len(v))


def decile_strata(x, q=10):
    x = np.asarray(x, dtype=float)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, q + 1)))
    if len(edges) < 3:
        return np.zeros(len(x), dtype=int)
    return np.clip(np.digitize(x, edges[1:-1], right=True), 0, len(edges) - 2)


def stratified_auc(y, p, strata, min_n=20):
    """n-weighted AUC within strata -- the closure_core convention v1 used."""
    y, p, strata = np.asarray(y), np.asarray(p), np.asarray(strata)
    num = tot = 0.0
    used = 0
    for s in np.unique(strata):
        m = strata == s
        if m.sum() < min_n or len(set(y[m])) < 2:
            continue
        num += m.sum() * roc_auc_score(y[m], p[m])
        tot += m.sum()
        used += 1
    if tot == 0:
        return float("nan"), 0
    return float(num / tot), used


def within_group_auc(y, groups, pred):
    tot = w = 0.0
    per = []
    for q in np.unique(groups):
        m = groups == q
        yy = y[m]
        if yy.min() == yy.max():
            continue
        a = roc_auc_score(yy, pred[m])
        n = int(yy.sum() * (len(yy) - yy.sum()))
        tot += n * a
        w += n
        per.append(a)
    if not per:
        return None
    return {"pair_weighted": float(tot / w), "unweighted_mean": float(np.mean(per)),
            "n_mixed_groups": len(per), "n_pairs": int(w)}


def cell_si_v2():
    out = REPO / SI_OUT
    meta, A, V, groups, shard, ids = SC.load_scaleupC_bank("si_v2", out=out)
    y = np.array(meta["ys"]["top_tier"], dtype=float).astype(int)
    meta = dict(meta)
    a_names = list(meta["a_names"])

    gate = {"modal_max": COLLAPSE_MODAL_MAX, "dropped": [],
            "modal_by_criterion": {}}
    shares = np.array([modal_share(A[:, c]) for c in range(A.shape[1])])
    for nm, s in zip(a_names, shares):
        gate["modal_by_criterion"][nm] = round(float(s), 4)
    keep_c = shares <= COLLAPSE_MODAL_MAX
    gate["dropped"] = [{"criterion": nm, "modal_share": round(float(s), 4)}
                       for nm, s in zip(a_names, shares) if s > COLLAPSE_MODAL_MAX]
    A = A[:, keep_c]
    a_names = [nm for nm, k in zip(a_names, keep_c) if k]
    meta["a_names"] = a_names
    gate["kept_n"] = int(A.shape[1])
    print(f"[collapse gate] dropped {len(gate['dropped'])} of {len(shares)} "
          f"-> A has {A.shape[1]} columns")
    for d in gate["dropped"]:
        print(f"    DROPPED {d['criterion']}  modal={d['modal_share']}")

    T, Tinfo = SC.dense_T(DENSE)
    return dict(
        title="Style Invitational top-tier curation (v2 mature bank, "
              "parse-artifact-free population)",
        A=A, V=V, y=y, groups=groups, ids=ids, meta=meta, shard_of=shard,
        group_column="week_id", T=T, T_info=Tinfo,
        matrix=f"{SI_OUT}/si_v2_shard*.npz",
        dense_dir=str(DENSE), prior_published=V1, collapse_gate=gate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "si_v2_ledger.json"))
    a = ap.parse_args()

    SC.SCALEUPC_OUT = REPO / SI_OUT
    SC.CELLS[SLUG] = cell_si_v2
    res = SC.run_cell(SLUG)

    d = cell_si_v2()
    y, groups, ids, A, V = d["y"], d["groups"], d["ids"], d["A"], d["V"]
    a_names = list(d["meta"]["a_names"])
    rub = [json.loads(l) for l in open(V2 / "rubrics.jsonl") if l.strip()]
    track = {r["name"]: r["track"] for r in rub}
    orient = {r["name"]: r["orientation"] for r in rub}
    is_real = np.array([track.get(n, "A") == "A" for n in a_names])

    pop = pd.read_csv(V2 / "population.csv.gz")
    pop["row_id"] = pop.row_id.astype(str)
    P = pop.set_index("row_id")
    order = [str(i) for i in ids]
    char_len = P.loc[order, "char_len"].values.astype(float)
    split = P.loc[order, "split"].values
    tier = P.loc[order, "tier"].values

    folds = L.outer_folds(len(y), groups, n_splits=5)
    extra = {"n": int(len(y)), "pos_rate": float(y.mean()),
             "n_criteria_scored": int(A.shape[1]),
             "n_real": int(is_real.sum()), "n_surface": int((~is_real).sum())}

    # ---- the two stratifications -------------------------------------------
    st_len = decile_strata(char_len, q=10)
    v_auc, v_oof = L.linear_oof_family1(V, y, groups, folds)
    st_v = decile_strata(v_oof, q=10)
    extra["length_nuisance"] = {
        "charlen_alone_pooled": float(roc_auc_score(y, char_len)),
        "charlen_within_week": within_group_auc(y, groups, char_len),
        "joint_V_lin_pooled": v_auc,
        "week_identity_alone": float(roc_auc_score(
            y, pd.Series(y).groupby(pd.Series(groups)).transform("mean").values)),
        "v1_comparison_charlen_pooled_all_rows": 0.6227}

    # ---- PER-CRITERION survival: the acceptance test -----------------------
    rows = []
    for k, nm in enumerate(a_names):
        col = A[:, k].copy()
        fin = np.isfinite(col)
        col[~fin] = float(np.nanmedian(col)) if fin.any() else 0.5
        pooled = float(roc_auc_score(y, col))
        s_len, n_len = stratified_auc(y, col, st_len)
        s_v, n_v = stratified_auc(y, col, st_v)
        o = orient.get(nm, "positive")
        # DIRECTION VALIDITY: a negatively-oriented criterion (1.0 = flaw) should
        # score BELOW .5. If it scores above, either the judge is reading it
        # backwards or the theory behind it is wrong -- either way the criterion
        # is not measuring what it claims, and that has to be visible.
        if o == "negative":
            as_intended = bool(pooled < 0.5)
        elif o == "positive":
            as_intended = bool(pooled > 0.5)
        else:
            as_intended = None  # surface probes carry no intended direction
        rows.append({"criterion": nm, "track": track.get(nm, "A"),
                     "orientation": o,
                     "alone_AUC_pooled": pooled,
                     "alone_AUC_within_length_strata": s_len,
                     "alone_AUC_within_V_strata": s_v,
                     "direction_as_intended": as_intended,
                     "na_rate": float((~fin).mean()),
                     "modal_share": modal_share(A[:, k]),
                     "shrinkage_length": abs(pooled - .5) - abs(s_len - .5)})
    rows.sort(key=lambda r: -abs(r["alone_AUC_pooled"] - .5))
    surv = {
        "n_criteria": len(rows),
        "n_pooled_ge_.05": sum(1 for r in rows if abs(r["alone_AUC_pooled"] - .5) >= .05),
        "n_within_length_ge_.05": sum(
            1 for r in rows if abs(r["alone_AUC_within_length_strata"] - .5) >= .05),
        "n_within_V_ge_.05": sum(
            1 for r in rows if abs(r["alone_AUC_within_V_strata"] - .5) >= .05),
        "median_abs_dev_pooled": float(np.median(
            [abs(r["alone_AUC_pooled"] - .5) for r in rows])),
        "median_abs_dev_within_length": float(np.median(
            [abs(r["alone_AUC_within_length_strata"] - .5) for r in rows])),
        "median_abs_dev_within_V": float(np.median(
            [abs(r["alone_AUC_within_V_strata"] - .5) for r in rows])),
        "direction_validity": {
            "n_with_intended_direction": sum(
                1 for r in rows if r["direction_as_intended"] is not None),
            "n_as_intended": sum(1 for r in rows if r["direction_as_intended"]),
            "negative_criteria_wrong_way": [
                r["criterion"] for r in rows
                if r["orientation"] == "negative" and r["direction_as_intended"] is False],
            "positive_criteria_wrong_way": [
                r["criterion"] for r in rows
                if r["orientation"] == "positive" and r["direction_as_intended"] is False]},
        "criteria": rows}
    extra["criterion_survival"] = surv
    print(f"\n[SURVIVAL] |AUC-.5|>=.05: {surv['n_pooled_ge_.05']} pooled -> "
          f"{surv['n_within_length_ge_.05']} within LENGTH strata -> "
          f"{surv['n_within_V_ge_.05']} within V strata   "
          f"(v1 was {V1['n_criteria_pooled_ge_.05']} -> "
          f"{V1['n_criteria_within_V_strata_ge_.05']})")
    print(f"           median |AUC-.5| {surv['median_abs_dev_pooled']:.4f} -> "
          f"{surv['median_abs_dev_within_length']:.4f} (length) -> "
          f"{surv['median_abs_dev_within_V']:.4f} (V)   "
          f"(v1 {V1['median_abs_dev_pooled']:.4f} -> {V1['median_abs_dev_within_V']:.4f})")
    for r in rows[:10]:
        print(f"    {r['alone_AUC_pooled']:.3f} -> {r['alone_AUC_within_length_strata']:.3f} "
              f"[{r['orientation'][:3]}] {r['criterion'][:46]}")

    # ---- fitted blocks: pooled / within-length / within-week ---------------
    mats = {"V": V, "A": A, "VA": np.column_stack([V, A]),
            "A_real": A[:, is_real], "A_surface": A[:, ~is_real]}
    preds, table = {}, {}
    for k, M in mats.items():
        auc, oof = L.linear_oof_family1(M, y, groups, folds)
        preds[k + "_lin"] = oof
        sl, _ = stratified_auc(y, oof, st_len)
        table[k + "_lin"] = {"pooled": auc, "within_length_strata": sl,
                             "within_week": within_group_auc(y, groups, oof)}
        print(f"  {k+'_lin':12s} pooled {auc:.4f}  within-LENGTH {sl:.4f}  "
              f"within-week {table[k+'_lin']['within_week']['pair_weighted']:.4f}")
    for k in ["V", "VA", "A_real"]:
        oofs = [L.gbm_oof_family1(mats[k], y, groups, folds, s)["oof"]
                for s in L.GBM_SEEDS]
        mo = np.mean(oofs, axis=0)
        preds[k + "_nl"] = mo
        sl, _ = stratified_auc(y, mo, st_len)
        table[k + "_nl"] = {"pooled_auc_of_seed_mean_oof": float(roc_auc_score(y, mo)),
                            "within_length_strata": sl,
                            "within_week": within_group_auc(y, groups, mo)}
        print(f"  {k+'_nl':12s} pooled {table[k+'_nl']['pooled_auc_of_seed_mean_oof']:.4f}"
              f"  within-LENGTH {sl:.4f}")
    extra["blocks"] = table

    # ---- same-rows Delta_beyond -------------------------------------------
    Tinfo = d["T_info"] or {}
    raw = Tinfo.get("raw", {})
    runs = raw.get("runs", raw) if isinstance(raw, dict) else {}
    same = {}
    for leg in ["eval", "test"]:
        m = split == leg
        aucs = [float(v[f"{leg}_auc"]) for v in runs.values()
                if isinstance(v, dict) and f"{leg}_auc" in v]
        if m.sum() < 50 or len(np.unique(y[m])) < 2 or not aucs:
            continue
        legT = float(np.mean(aucs))
        vanl = float(roc_auc_score(y[m], preds["VA_nl"][m]))
        sl, _ = stratified_auc(y[m], preds["VA_nl"][m], st_len[m])
        same[leg] = {"n": int(m.sum()), "n_pos": int(y[m].sum()),
                     "VA_lin": float(roc_auc_score(y[m], preds["VA_lin"][m])),
                     "VA_nl": vanl, "T_seed_mean": legT,
                     "T_seeds": aucs, "T_seed_spread": float(max(aucs) - min(aucs)),
                     "Delta_beyond": legT - vanl,
                     "VA_nl_within_length_strata": sl}
    extra["same_rows"] = same

    # ---- ids-carried OOF + <1e-9 reproduction ------------------------------
    repro = {"tol": 1e-9, "checks": {}, "all_pass": True}
    for k, target in [("V_lin", res["linear"]["V"]), ("A_lin", res["linear"]["A"]),
                      ("VA_lin", res["linear"]["VA"])]:
        got = float(roc_auc_score(y, preds[k]))
        dd = abs(got - target)
        repro["checks"][k] = {"ledger": target, "recomputed": got, "abs_diff": dd,
                              "pass": bool(dd < 1e-9)}
        if dd >= 1e-9:
            repro["all_pass"] = False
    extra["oof_reproduction"] = repro
    print(f"[repro <1e-9] all_pass={repro['all_pass']}")

    oof_path = RESULTS / "si_v2_oof_with_ids.npz"
    np.savez_compressed(
        oof_path, ids=np.array([str(i) for i in ids], dtype=object),
        groups=np.array([str(g) for g in groups], dtype=object), y=y,
        char_len=char_len, tier=np.array([str(t) for t in tier], dtype=object),
        split=np.array([str(s) for s in split], dtype=object),
        **{k: v for k, v in preds.items()})
    extra["oof_artifact"] = str(oof_path)
    extra["v1_comparison"] = V1

    res["collapse_gate"] = d["collapse_gate"]
    res["si_v2_extras"] = extra
    Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    print("\nwrote", a.out)
    lg = res["ledger"]
    print("=== LEDGER ===")
    for k in ("V_lin", "V_nl_mean", "A_lin", "VA_lin", "VA_nl_mean", "T",
              "Delta_interact", "Delta_total", "Delta_beyond"):
        if lg.get(k) is not None:
            print(f"  {k:16s} {lg[k]:+.4f}")


if __name__ == "__main__":
    main()
