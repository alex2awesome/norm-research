#!/usr/bin/env python3
"""SO answer-votes closure: ROUND 0 — baseline, power check, swap baseline, census input.

Round 0 mines nothing. It establishes what every later round is measured against
and, critically, whether the frozen stopping rule can be resolved at all on this
cell's MONITOR (n = 501, the honest consequence of MONITOR ⊂ dense-held-out with
only 2,440 dense-held-out rows).

THE POWER QUESTION, answered here rather than discovered at round 3: the
stopping rule is 2 consecutive rounds with MONITOR VA_nl gain < ε = .005. Two
different sampling quantities matter and are both computed:
  * the LEVEL noise -- group-bootstrap CI of VA_nl on MONITOR. If this is wide
    (it will be), absolute Δ_r levels are not quotable round-to-round.
  * the PAIRED noise -- the round-over-round GAIN is a paired statistic on
    identical rows, so its SE is much smaller than the level's. This is the
    quantity ε actually applies to, and it is estimated here by a paired
    group bootstrap of (VA_nl with a criterion added) − (VA_nl without), using
    a held-out real criterion as the perturbation stand-in.
If ε sits below the paired noise floor, the campaign's "sub-ε" rounds are
unreadable and that must be declared BEFORE any mining, not after.

Outputs: so_votes_round0.json + splits.npz
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import cells_so as C

HERE = Path(__file__).resolve().parent
T_VIEWS = ("qtrunc", "title", "abank")


def group_boot_auc(y, g, p, n=2000, seed=11):
    rng = np.random.default_rng(seed)
    uq = np.unique(g)
    idx = {q: np.flatnonzero(g == q) for q in uq}
    out = []
    for _ in range(n):
        qs = rng.choice(uq, size=len(uq), replace=True)
        i = np.concatenate([idx[q] for q in qs])
        if len(set(y[i])) < 2:
            continue
        out.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"auc": float(roc_auc_score(y, p)), "ci95": [float(lo), float(hi)],
            "sd": float(np.std(out)), "n_boot": len(out)}


def group_boot_paired(y, g, pa, pb, n=2000, seed=13):
    """Paired bootstrap of AUC(pa) - AUC(pb) on the same rows."""
    rng = np.random.default_rng(seed)
    uq = np.unique(g)
    idx = {q: np.flatnonzero(g == q) for q in uq}
    out = []
    for _ in range(n):
        qs = rng.choice(uq, size=len(uq), replace=True)
        i = np.concatenate([idx[q] for q in qs])
        if len(set(y[i])) < 2:
            continue
        out.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"delta": float(roc_auc_score(y, pa) - roc_auc_score(y, pb)),
            "ci95": [float(lo), float(hi)], "sd": float(np.std(out)),
            "n_boot": len(out)}


def main():
    d = C.load()
    split, rep = C.build_splits(d)
    y, g = d["y"], np.array([str(x) for x in d["groups"]])
    fit = split == "fit_mine"
    mon = split == "monitor"
    np.savez_compressed(HERE / "splits.npz", split=split.astype(str),
                        ids=np.array([str(i) for i in d["ids"]], dtype=object),
                        groups=g, y=y)

    res = {"cell": "so_votes", "round": 0, "splits": rep,
           "collapse_gate": d["collapse_gate"],
           "alignment_gates_failed": [x for x in d["alignment_gates"]
                                      if not x.get("gate_pass", True)],
           "T_convention": {
               "primary": C.T_PRIMARY,
               "why": "question-inclusive AND answer-preserving; the audit showed "
                      "the abank control was displacement-blinded on 6.9% of rows "
                      "and the title view never saw the question the judge saw",
               "all_views_reported": True}}

    # ---- incoming bank baseline, fitted inside FIT+MINE only ---------------
    allA = list(range(d["A"].shape[1]))
    base = C.fit_va(d, allA, fit)
    # apply the fitted stack out-of-sample to MONITOR: refit on all of fit_mine
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = np.column_stack([d["V"], d["A"]])

    def fit_apply(Xm, seeds=(0, 1, 2)):
        lin = Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)),
                        ("s", StandardScaler()),
                        ("m", LogisticRegression(C=1, max_iter=2000,
                                                 solver="liblinear",
                                                 random_state=20260728))])
        lin.fit(Xm[fit], y[fit])
        p_lin = lin.predict_proba(Xm[mon])[:, 1]
        ps = []
        for s in seeds:
            im = SimpleImputer(strategy="median", add_indicator=True).fit(Xm[fit])
            gb = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=.06,
                                                max_iter=400, early_stopping=True,
                                                random_state=s)
            gb.fit(im.transform(Xm[fit]), y[fit])
            ps.append(gb.predict_proba(im.transform(Xm[mon]))[:, 1])
        return p_lin, np.mean(ps, axis=0), ps

    p_lin, p_nl, p_nl_seeds = fit_apply(X)
    res["monitor_baseline"] = {
        "n_monitor": int(mon.sum()),
        "VA_lin": group_boot_auc(y[mon], g[mon], p_lin),
        "VA_nl": group_boot_auc(y[mon], g[mon], p_nl),
        "VA_nl_seed_aucs": [float(roc_auc_score(y[mon], p)) for p in p_nl_seeds],
        "within_question": C.within_group_auc(y[mon], g[mon], p_nl)[0],
        "fit_internal_oof_nl": base["nl_auc_mean"]}

    # ---- T on MONITOR, all conventions -------------------------------------
    tt = {}
    for v in T_VIEWS:
        for key in (f"{v}_mean", f"{v}_s42"):
            if key in d["dense"]:
                col = d["dense"][key]
                ok = np.isfinite(col) & mon
                if ok.sum() < 50:
                    continue
                tt[key] = {"n": int(ok.sum()),
                           **group_boot_auc(y[ok], g[ok], col[ok])}
    res["monitor_T"] = tt
    # Delta_0 on MONITOR. p_lin/p_nl are indexed over the MONITOR subset in
    # MONITOR order, so the dense column must be subset the same way and any
    # NaN mask applied WITHIN that subset (not against the full population).
    y_m, g_m = y[mon], g[mon]
    res["Delta_0"] = {}
    for v in T_VIEWS:
        for key in (f"{v}_s42", f"{v}_mean"):
            if key not in d["dense"]:
                continue
            tm = d["dense"][key][mon]
            ok = np.isfinite(tm)
            if ok.sum() < 50 or len(set(y_m[ok])) < 2:
                continue
            res["Delta_0"][key] = {
                "n": int(ok.sum()),
                "T": float(roc_auc_score(y_m[ok], tm[ok])),
                "VA_nl": float(roc_auc_score(y_m[ok], p_nl[ok])),
                "Delta_beyond": float(roc_auc_score(y_m[ok], tm[ok])
                                      - roc_auc_score(y_m[ok], p_nl[ok])),
                "paired_boot": group_boot_paired(y_m[ok], g_m[ok], tm[ok], p_nl[ok]),
                "primary": key == f"{C.T_PRIMARY}_s42"}

    # ---- THE POWER CHECK ---------------------------------------------------
    # Perturbation stand-in: drop the single strongest real criterion from the
    # bank and re-fit. The resulting paired gain is the scale of "one useful
    # criterion", which is what a mining round hopes to add.
    from sklearn.metrics import roc_auc_score as _auc
    alone = [(abs(_auc(y[fit], np.nan_to_num(d["A"][fit, k],
                                             nan=np.nanmedian(d["A"][fit, k]))) - .5), k)
             for k in range(d["A"].shape[1])]
    alone.sort(reverse=True)
    k_top = alone[0][1]
    keep = [k for k in allA if k != k_top]
    X_minus = np.column_stack([d["V"], d["A"][:, keep]])
    _, p_nl_minus, _ = fit_apply(X_minus)
    res["power_check"] = {
        "perturbation": f"drop strongest criterion '{d['a_names'][k_top]}' "
                        f"(alone |AUC-.5| = {alone[0][0]:.4f}) and refit",
        "paired_gain_full_vs_minus1": group_boot_paired(y[mon], g[mon], p_nl, p_nl_minus),
        "epsilon": 0.005,
        "level_sd_VA_nl": res["monitor_baseline"]["VA_nl"]["sd"],
        "seed_spread_VA_nl": float(max(res["monitor_baseline"]["VA_nl_seed_aucs"])
                                   - min(res["monitor_baseline"]["VA_nl_seed_aucs"]))}
    pg = res["power_check"]["paired_gain_full_vs_minus1"]
    res["power_check"]["verdict"] = (
        "epsilon RESOLVABLE" if pg["sd"] < 0.005 else
        "epsilon BELOW THE PAIRED NOISE FLOOR — sub-eps rounds are unreadable "
        "on this MONITOR; report the curve but do NOT read a stopping decision "
        "off it without the wider-MONITOR sensitivity arm")

    # ---- position line + trivial channels on MONITOR -----------------------
    res["trivial_channels_monitor"] = {
        "position_alone": float(roc_auc_score(y[mon], -d["position"][mon])),
        "charlen_alone": float(roc_auc_score(y[mon], d["char_len"][mon])),
        "q_viewcount_alone": float(roc_auc_score(
            y[mon], np.nan_to_num(d["q_viewcount"][mon], nan=0.0))),
        "question_identity_alone": float(roc_auc_score(
            y[mon], __import__("pandas").Series(y[mon]).groupby(
                __import__("pandas").Series(g[mon])).transform("mean").values))}

    # ---- swap baseline (C+/C-) --------------------------------------------
    prim = f"{C.T_PRIMARY}_s42"
    tcol = d["dense"].get(prim)
    if tcol is not None:
        tm = tcol[mon]
        okm = np.isfinite(tm)
        y_ok = y_m[okm]
        tb = tm[okm] > np.median(tm[okm])
        vb = p_nl[okm] > np.median(p_nl[okm])
        okp = np.zeros(len(y), bool)
        okp[np.flatnonzero(mon)[okm]] = True
        res["swap_baseline"] = {
            "note": "C+ = dense right & bank wrong; C- = bank right & dense wrong "
                    "(median-split agreement classes on MONITOR)",
            "n_Cplus": int(((tb == y_ok.astype(bool)) & (vb != y_ok.astype(bool))).sum()),
            "n_Cminus": int(((vb == y_ok.astype(bool)) & (tb != y_ok.astype(bool))).sum()),
            "n_both_right": int(((tb == y_ok.astype(bool)) & (vb == y_ok.astype(bool))).sum()),
            "n_both_wrong": int(((tb != y_ok.astype(bool)) & (vb != y_ok.astype(bool))).sum())}

    (HERE / "so_votes_round0.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in
                      ["splits", "monitor_baseline", "monitor_T", "power_check",
                       "trivial_channels_monitor", "swap_baseline"]
                      if k in res}, indent=1, default=str))


if __name__ == "__main__":
    main()
