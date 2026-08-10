#!/usr/bin/env python3
"""Round readout for the code_v3 cell: Track-A closure curve, spurious map, discount,
stacked increment, swap pair, B-side missing mass.

Adapted from maps_hw_si/readout.py with ONE structural change, which is the whole point
of §2 of the note: **every AUC in this file is a WITHIN-REPO n-weighted AUC**, never a
pooled one. Pooled AUC on this corpus is composition-dominated (the pooled eval readout
says the articulated stack beats dense by .093 while the pooled test readout says dense
wins by .157; within repo both splits agree that dense wins by +.04 to +.06).

Consequences of that change, each recorded:
  * the DISCOUNT is a within-repo × nuisance-stratum readout: cells are (repository,
    decile of the nuisance score), each cell needs >= 10 rows and both classes, and the
    n-weighted mean over cells is the discounted AUC.  This is the same estimator the
    position line uses in §4 of the note, generalised to any nuisance score.
  * the STACKED INCREMENT fits its logistic stack grouped-OOF by repository and then
    reads the result within repo, so neither the fit nor the readout can be carried by
    between-repo composition.
  * the SWAP pair algebra counts only (merged, closed-unmerged) pairs that lie inside a
    single repository.

Populations:
  MONITOR  -- the frozen decision population (54 repos, 29 scorable), never seen by a
              proposer; VA-honest AND T-honest (the whole cell population is
              dense-held-out).
  HONEST   -- the full population: OOF inside FIT+MINE, refit-predicted on MONITOR.
              Better powered, mildly mining-contaminated, therefore conservative.
  eval / test -- the two dense-split replicates; test is the selection-free reading.

CPU only.  Usage: python readout_code.py --round 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "maps_hw_si"))

import cells as C                                            # noqa: E402
import cells_code as CC                                      # noqa: E402
import closure_core as L                                     # noqa: E402
import stage1_slice_code as S1                               # noqa: E402

TAG = "code_v3"
EPS = 0.005
MIN_CELL = 10


# ------------------------------------------------------------- estimators ---
def wauc(y, p, g, mask=None):
    return CC.within_repo_auc(y, p, g, mask)["nwtd"]


def wdelta(y, pa, pb, g, mask=None):
    return CC.within_repo_delta(y, pa, pb, g, mask)


def within_repo_strat_auc(y, p, g, x, mask=None, q=10, min_cell=MIN_CELL):
    """n-weighted AUC over (repository x decile-of-x) cells -- the within-repo discount."""
    from sklearn.metrics import roc_auc_score
    y, p, g, x = map(np.asarray, (y, p, g, x))
    if mask is not None:
        y, p, g, x = y[mask], p[mask], g[mask], x[mask]
    num = tot = 0.0
    used = 0
    for r in np.unique(g):
        m = g == r
        if m.sum() < CC.MIN_REPO_N or len(set(y[m])) < 2:
            continue
        xr = x[m]
        edges = np.unique(np.quantile(xr, np.linspace(0, 1, q + 1)))
        st = (np.zeros(len(xr), int) if len(edges) < 3
              else np.clip(np.digitize(xr, edges[1:-1], right=True), 0, len(edges) - 2))
        for s in np.unique(st):
            k = st == s
            if k.sum() < min_cell or len(set(y[m][k])) < 2:
                continue
            num += k.sum() * roc_auc_score(y[m][k], p[m][k])
            tot += k.sum()
            used += 1
    return (num / tot if tot else float("nan")), {"n_cells": used, "n_rows": int(tot)}


def stack_oof(cols, y, groups, n_splits=5):
    X = np.column_stack(cols)
    folds = list(GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(
        np.zeros(len(y)), groups=groups))
    oof = np.zeros(len(y))
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def swap_pair_within_repo(y, dense, va, g, mask=None):
    y, dense, va, g = map(np.asarray, (y, dense, va, g))
    m = np.ones(len(y), bool) if mask is None else mask
    cp_n = cp_d = cm_n = cm_d = npairs = 0
    for r in np.unique(g[m]):
        s = m & (g == r)
        if s.sum() < CC.MIN_REPO_N or len(set(y[s])) < 2:
            continue
        yi, pi, vi = y[s], dense[s], va[s]
        P, N = np.where(yi == 1)[0], np.where(yi == 0)[0]
        dok = (pi[P][:, None] - pi[N][None, :]) > 0
        bok = (vi[P][:, None] - vi[N][None, :]) > 0
        npairs += dok.size
        cp_d += int(dok.sum()); cp_n += int((bok & dok).sum())
        cm_d += int((~dok).sum()); cm_n += int((bok & ~dok).sum())
    return {"n_within_repo_pairs": npairs,
            "w_plus": cp_d / npairs if npairs else None,
            "C_plus": cp_n / cp_d if cp_d else None,
            "C_minus": cm_n / cm_d if cm_d else None,
            "spearman_bank_vs_dense": float(spearmanr(va[m], dense[m]).statistic)}


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="1")
    a = ap.parse_args()
    tag = f"{TAG}_r{a.round}"

    d = C.load()
    z = np.load(HERE / "splits.npz", allow_pickle=True)
    fitm, monm = z["fitmask"], z["monmask"]
    y, g, dense = d["y"], d["groups"], d["dense"]
    tiers = {"MONITOR": monm, "HONEST": np.ones(len(y), bool),
             "eval": d["split"] == "eval", "test": d["split"] == "test"}

    zz = np.load(HERE / f"{tag}_scores.npz", allow_pickle=True)
    pos = {str(s): k for k, s in enumerate(zz["row_ids"])}
    Xnew = zz["X"][[pos[i] for i in d["ids"]]]
    cids = [str(s) for s in zz["a_ids"]]
    routing = json.loads((HERE / f"{tag}_routing_final.json").read_text())
    A_ids = [r["blind_id"] for r in routing["final"] if r["final_route"] == "A"]
    B_rows = [r for r in routing["final"] if r["final_route"] == "B"]
    B_ids = [r["blind_id"] for r in B_rows]
    B_mixed = {r["blind_id"]: bool(r.get("mixed")) for r in B_rows}
    B_parent = {r["blind_id"]: r.get("upstream_parent", "surface-only") for r in B_rows}
    NAME = {r["blind_id"]: r["name"] for r in routing["final"]}

    def blockify(ids):
        if not ids:
            return np.zeros((len(y), 0))
        X = Xnew[:, [cids.index(i) for i in ids]]
        return np.column_stack([X, (~np.isnan(X)).astype(float)])

    XA, XB = blockify(A_ids), blockify(B_ids)
    prev_blocks, prev_tags = S1.current_blocks(d, a.round)

    r_prev = L.fit_block(prev_blocks, fitm, monm, y, g)
    r_new = L.fit_block(prev_blocks + ([XA] if XA.shape[1] else []), fitm, monm, y, g)
    rb = L.fit_block([XB], fitm, monm, y, g) if XB.shape[1] else None
    strict_cols = [k for k, b in enumerate(B_ids) if not B_mixed.get(b)]
    rb_strict = (L.fit_block([blockify([B_ids[k] for k in strict_cols])], fitm, monm, y, g)
                 if 0 < len(strict_cols) < len(B_ids) else None)

    def full(r):
        v = np.full(len(y), np.nan)
        v[fitm] = r["oof_nl_fitmine"]
        v[monm] = r["nl_mon"]
        return v

    va_prev, va_new = full(r_prev), full(r_new)
    jb = full(rb) if rb is not None else None

    res = {"cell": TAG, "round": a.round, "tag": tag,
           "readout": "ALL AUCs are WITHIN-REPO n-weighted (repos with n>=20 and both "
                      "classes); pooled AUC is never quoted as a residual on this cell",
           "bank_entering_round": prev_tags,
           "routing": {"n_A": len(A_ids), "n_B": len(B_ids),
                       "misrouting_rate": routing["misrouting_rate"],
                       "probe_pass": routing["probe_pass"],
                       "n_mixed_B": routing["n_mixed_B"]},
           "score_report": (json.loads((HERE / f"{tag}_scores.report.json").read_text())
                            if (HERE / f"{tag}_scores.report.json").exists() else None)}

    # ------------------------------------------------- Track A closure curve --
    ta = {"epsilon": EPS}
    for tier, m in tiers.items():
        prev_seeds = [wauc(y, p_, g, m) for p_ in
                      ([full({"oof_nl_fitmine": r_prev["oof_nl_fitmine"], "nl_mon": s})
                        for s in r_prev["nl_mon_seeds"]] if tier == "MONITOR" else [])]
        ta[tier] = {
            "T_within": wauc(y, dense, g, m),
            "VA_nl_within_prev": wauc(y, va_prev, g, m),
            "VA_nl_within_new": wauc(y, va_new, g, m),
            "VA_lin_within_new": wauc(y, _lin_full(r_new, fitm, monm, len(y)), g, m),
            "gain": wauc(y, va_new, g, m) - wauc(y, va_prev, g, m),
            "Delta_prev": wdelta(y, dense, va_prev, g, m),
            "Delta_new": wdelta(y, dense, va_new, g, m),
        }
        if prev_seeds:
            ta[tier]["VA_nl_within_prev_per_seed"] = prev_seeds
            ta[tier]["VA_nl_within_prev_seed_spread"] = float(max(prev_seeds) - min(prev_seeds))
    ta["sub_epsilon_signed_MONITOR"] = bool(ta["MONITOR"]["gain"] < EPS)
    ta["sub_epsilon_magnitude_MONITOR"] = bool(abs(ta["MONITOR"]["gain"]) < EPS)
    res["track_A"] = ta

    # ------------------------------------------------------- spurious map ----
    if rb is not None:
        keepB, medB = L.clean_fit(XB[fitm])
        XBall = L.clean_apply(XB, keepB, medB)
        kept_ids = [(B_ids + [b + "_applied" for b in B_ids])[j] for j in keepB]
        mp = []
        for k, bid in enumerate(kept_ids):
            base = bid.replace("_applied", "")
            mp.append({"channel_id": bid, "name": NAME.get(base, base),
                       "upstream_parent": B_parent.get(base, "surface-only"),
                       "mixed": B_mixed.get(base, False),
                       "alone_within_HONEST": wauc(y, XBall[:, k], g),
                       "alone_within_MONITOR": wauc(y, XBall[:, k], g, monm),
                       "alone_within_FITMINE": wauc(y, XBall[:, k], g, fitm)})
        mp.sort(key=lambda r: -abs(r["alone_within_HONEST"] - .5))
        res["spurious_map"] = {"channels": mp,
                               "n_dropped_by_screen": int(XB.shape[1] - len(keepB))}

        # ------------------------------------------------------- discount ----
        def discount(rbx, label):
            j = full(rbx)
            out = {"label": label, "n_B_features_after_screen": rbx["n_features"],
                   "spurious_alone_within_HONEST_histgb": wauc(y, j, g),
                   "spurious_alone_within_MONITOR_histgb": wauc(y, rbx["nl_mon"], g, monm),
                   "spurious_alone_within_HONEST_linear":
                       wauc(y, _lin_full(rbx, fitm, monm, len(y)), g)}
            for tier, m in tiers.items():
                tA, iA = within_repo_strat_auc(y, dense, g, j, m)
                vA, _ = within_repo_strat_auc(y, va_new, g, j, m)
                out[tier] = {"T_adj": tA, "VA_adj": vA, "Delta_adj": tA - vA,
                             "Delta_undiscounted": wdelta(y, dense, va_new, g, m)["delta_nwtd"],
                             **iA}
            out["matched_sampling_triggered"] = bool(
                out["spurious_alone_within_HONEST_histgb"] > .65)
            return out

        res["discount_ALL_B"] = discount(rb, "all channels")
        if rb_strict is not None:
            res["discount_STRICT_no_mixed"] = discount(rb_strict, "mixed excluded")
        res["discount_note"] = (
            "FREEZE ADDENDUM 2 sensitivity band: ALL_B discounts every named channel "
            "including MIXED ones; STRICT discounts only the unmixed ones. The truth for "
            "Delta_adj lies between the two. Delta_adj is NOT an effect size: stratifying "
            "on a strong nuisance score costs VA more than it costs T.")

        # ----------------------------------------------- stacked increment ---
        def stack_block(m):
            yy, gg = y[m], g[m]
            jbm, dn, va = jb[m], dense[m], va_new[m]
            s_bd = stack_oof([jbm, dn], yy, gg)
            s_bv = stack_oof([jbm, va], yy, gg)
            s_all = stack_oof([jbm, dn, va], yy, gg)
            W = lambda v: CC.within_repo_auc(yy, v, gg)["nwtd"]      # noqa: E731
            return {"n": int(m.sum()), "within_jointB": W(jbm),
                    "within_dense": W(dn), "within_bank": W(va),
                    "within_stack_B_dense": W(s_bd),
                    "dense_increment_over_B": W(s_bd) - W(jbm),
                    "within_stack_B_bank": W(s_bv),
                    "bank_increment_over_B": W(s_bv) - W(jbm),
                    "within_stack_all": W(s_all),
                    "dense_increment_over_B_plus_bank": W(s_all) - W(s_bv)}
        res["stacked_increment"] = {t: stack_block(m) for t, m in tiers.items()}

    # ------------------------------------------------------------- swap ------
    res["swap_pair"] = {}
    for tier, m in tiers.items():
        s0 = swap_pair_within_repo(y, dense, va_prev, g, m)
        s1 = swap_pair_within_repo(y, dense, va_new, g, m)
        res["swap_pair"][tier] = {
            "entering_round": s0, "after_round": s1,
            "dC_plus": (s1["C_plus"] - s0["C_plus"]) if s0["C_plus"] is not None else None,
            "dC_minus": (s1["C_minus"] - s0["C_minus"]) if s0["C_minus"] is not None else None,
            "d_rho": s1["spearman_bank_vs_dense"] - s0["spearman_bank_vs_dense"]}
        sp = res["swap_pair"][tier]
        sp["swap_signature"] = bool(sp["dC_plus"] and sp["dC_minus"]
                                    and sp["dC_plus"] > 0 > sp["dC_minus"])

    sp_path = HERE / f"{tag}_species.json"
    if sp_path.exists():
        s = json.loads(sp_path.read_text())
        res["missing_mass"] = {t: s["tracks"][t]["good_turing"] for t in s.get("tracks", {})}

    (HERE / f"{tag}_results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("score_report", "spurious_map")}, indent=1,
                     default=float)[:6000])
    if "spurious_map" in res:
        print("\nSPURIOUS MAP (within-repo alone AUC, HONEST population)")
        for c in res["spurious_map"]["channels"][:20]:
            print(f"  {c['alone_within_HONEST']:.3f}  {'MIXED' if c['mixed'] else '     '}  "
                  f"{c['name'][:56]:56s} <- {str(c['upstream_parent'])[:38]}")
    print("wrote", HERE / f"{tag}_results.json")


def _lin_full(r, fitm, monm, n):
    v = np.full(n, np.nan)
    v[fitm] = r["oof_lin_fitmine"]
    v[monm] = r["lin_mon"]
    return v


if __name__ == "__main__":
    main()
