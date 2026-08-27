#!/usr/bin/env python3
"""Round readout for the map-focused batch: SPURIOUS MAP (headline), discount,
stacked increment, Track-A closure (secondary), B-side missing mass.

Populations, both reported for every quantity:
  MONITOR   -- the frozen decision population for the closure rule (Delta_r).
               Every MONITOR row is dense-held-out by construction, so T is honest.
  HONEST    -- the full dense-held-out population (MONITOR union the mining slice
               M).  VA / joint-B predictions are grouped-OOF on the FIT+MINE side
               and held-out on the MONITOR side, so every prediction is
               out-of-sample.  This is the pilot's own primary discount population
               (`stratified_dense_heldout_1244_q10` in closure/stage4_readout.py)
               and it is 2-4x larger than MONITOR here, which is why the MAP is
               quoted on it.

Readouts
  1. SPURIOUS MAP        per-channel alone-AUC (raw judged score, no fitting),
                         with upstream parent + mixed flag (FREEZE ADDENDUM 2).
  2. DISCOUNT            joint spurious-alone AUC (linear + HistGB); decile-
                         stratified T_adj / VA_adj / Delta_adj by the joint B score
                         and by each channel.  Reported twice: ALL B channels, and
                         STRICT (mixed channels dropped) -- the Addendum-2
                         sensitivity band.
  3. STACKED INCREMENT   AUC(joint B) vs AUC(logistic stack of B + dense) and
                         vs AUC(stack of B + bank VA_nl); grouped-OOF inside the
                         readout population.  Stratification-free.
  4. TRACK A             VA_nl round r vs round r-1 on MONITOR and on HONEST,
                         with group-level paired bootstrap CIs; Delta_beyond.
  5. B-SIDE MISSING MASS copied through from <tag>_species.json.

CPU only.  Usage: python readout.py --cell cap_crowd --round 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scipy.stats import spearmanr

import cells as C
import closure_core as L

def swap_pair(y, dense, va, groups=None):
    """(C+, C-) pair algebra, the freeze's per-round swap readout.

    C+ = P(bank orders a discordant-label pair correctly | the dense model does)
    C- = P(bank orders it correctly | the dense model does NOT).
    A round that raises C+ while pushing C- toward .5 is buying rank agreement with
    the dense model by inheriting its errors -- the swap signature.
    """
    y = np.asarray(y)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    dp = dense[pos][:, None] - dense[neg][None, :]
    vp = va[pos][:, None] - va[neg][None, :]
    dense_ok = dp > 0
    bank_ok = vp > 0
    n_ok, n_bad = int(dense_ok.sum()), int((~dense_ok).sum())
    return {
        "n_pairs": int(dense_ok.size),
        "w_plus": n_ok / dense_ok.size,
        "C_plus": float(bank_ok[dense_ok].mean()) if n_ok else None,
        "C_minus": float(bank_ok[~dense_ok].mean()) if n_bad else None,
        "spearman_bank_vs_dense": float(spearmanr(va, dense).statistic),
    }


HERE = Path(__file__).resolve().parent


def stack_oof(cols, y, groups, n_splits=5):
    """Grouped-OOF logistic stack of already-out-of-sample score columns."""
    X = np.column_stack(cols)
    folds = list(GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(
        np.zeros(len(y)), groups=groups))
    oof = np.zeros(len(y))
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def strat_table(y, va, dense, joint, feats, names, q):
    out = {"pooled_T": L.auc(y, dense), "pooled_VA": L.auc(y, va)}
    out["pooled_Delta"] = out["pooled_T"] - out["pooled_VA"]
    st = L.decile_strata(joint, q=q)
    tA, iA = L.stratified_auc(y, dense, st, min_n=20)
    vA, _ = L.stratified_auc(y, va, st, min_n=20)
    out["joint_B_score"] = {"T_adj": tA, "VA_adj": vA, "Delta_adj": tA - vA, **iA}
    out["by_channel"] = {}
    for k, nm in enumerate(names):
        st = L.decile_strata(feats[:, k], q=q)
        tA, iA = L.stratified_auc(y, dense, st, min_n=20)
        vA, _ = L.stratified_auc(y, va, st, min_n=20)
        out["by_channel"][nm] = {"T_adj": tA, "VA_adj": vA, "Delta_adj": tA - vA, **iA}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--round", default="1")
    a = ap.parse_args()
    tag = f"{a.cell}_r{a.round}"

    d = C.load(a.cell)
    sp = json.loads((HERE / f"{a.cell}_splits.json").read_text())
    split = np.array([r["split"] for r in sp["rows"]])
    y, groups = d["y"], d["groups"]
    dense = d["dense"]
    fitm = split == "fit_mine"
    monm = split == "monitor"
    held = np.isin(d["dense_split"], ["eval", "test"])          # HONEST population
    mon_in_held = monm[held]

    # ---- new scores + routing ------------------------------------------------
    z = np.load(HERE / f"{tag}_scores.npz", allow_pickle=True)
    assert (z["i"] == np.arange(len(y))).all(), "score rows out of alignment"
    Xnew, cids = z["X"], [str(s) for s in z["crit_ids"]]
    routing = json.loads((HERE / f"{tag}_routing_final.json").read_text())
    A_ids = [r["blind_id"] for r in routing["final"] if r["final_route"] == "A"]
    B_rows = [r for r in routing["final"] if r["final_route"] == "B"]
    B_ids = [r["blind_id"] for r in B_rows]
    B_mixed = {r["blind_id"]: bool(r.get("mixed")) for r in B_rows}
    B_parent = {r["blind_id"]: r.get("upstream_parent", "surface-only") for r in B_rows}
    B_name = {r["blind_id"]: r["name"] for r in routing["final"]}
    XA = Xnew[:, [cids.index(i) for i in A_ids]] if A_ids else np.zeros((len(y), 0))
    XB = Xnew[:, [cids.index(i) for i in B_ids]] if B_ids else np.zeros((len(y), 0))

    # ---- prior-round A blocks (bank state entering this round) ---------------
    import stage1_slice as S1
    prev_blocks, prev_tags = S1.current_blocks(d, a.round)

    res = {"cell": a.cell, "round": a.round, "tag": tag,
           "splits": sp["summary"],
           "bank_entering_round": prev_tags,
           "routing": {"n_A": len(A_ids), "n_B": len(B_ids),
                       "misrouting_rate": routing["misrouting_rate"],
                       "probe_pass": routing["probe_pass"],
                       "n_mixed_B": routing["n_mixed_B"]},
           "score_report": json.loads((HERE / f"{tag}_score_report.json").read_text())
           if (HERE / f"{tag}_score_report.json").exists() else None}

    # ======================================================= Track A (closure) =
    r_prev = L.fit_block(prev_blocks, fitm, monm, y, groups)
    r_new = L.fit_block(prev_blocks + [XA], fitm, monm, y, groups)
    rb = L.fit_block([XB], fitm, monm, y, groups)
    rb_strict = None
    strict_idx = [k for k, b in enumerate(B_ids) if not B_mixed.get(b)]
    if 0 < len(strict_idx) < len(B_ids):
        rb_strict = L.fit_block([XB[:, strict_idx]], fitm, monm, y, groups)

    ymon, gmon = y[monm], groups[monm]

    def va_full(r):
        v = np.full(len(y), np.nan)
        v[fitm] = r["oof_nl_fitmine"]
        v[monm] = r["nl_mon"]
        return v

    va_prev_f, va_new_f = va_full(r_prev), va_full(r_new)
    jb_f = va_full(rb)

    def blk(r, name):
        return {
            "n_features": r["n_features"],
            "VA_lin_MONITOR": L.auc(ymon, r["lin_mon"]),
            "VA_nl_MONITOR": L.auc(ymon, r["nl_mon"]),
            "VA_nl_MONITOR_per_seed": [L.auc(ymon, p) for p in r["nl_mon_seeds"]],
            "VA_nl_MONITOR_seed_spread": float(max(L.auc(ymon, p) for p in r["nl_mon_seeds"])
                                               - min(L.auc(ymon, p) for p in r["nl_mon_seeds"])),
            "VA_nl_HONEST": L.auc(y[held], va_full(r)[held]),
            "VA_nl_OOF_fitmine": L.auc(y[fitm], r["oof_nl_fitmine"]),
            "grid_picks": r["picks"],
        }

    T_mon = L.auc(ymon, dense[monm])
    T_hon = L.auc(y[held], dense[held])
    res["T"] = {"T_MONITOR": T_mon, "n_MONITOR": int(monm.sum()),
                "T_HONEST_dense_heldout": T_hon, "n_HONEST": int(held.sum()),
                "T_note": "dense probabilities exist only on the dense-held-out rows "
                          "for these two cells (3-seed dense-standard chain); the row-level "
                          "score is the seed ensemble"}
    res["track_A"] = {
        "state_entering_round": blk(r_prev, "prev"),
        f"state_after_round_{a.round}": blk(r_new, "new"),
        "gain_MONITOR": L.auc(ymon, r_new["nl_mon"]) - L.auc(ymon, r_prev["nl_mon"]),
        "gain_HONEST": L.auc(y[held], va_new_f[held]) - L.auc(y[held], va_prev_f[held]),
        "gain_ci_MONITOR": L.group_boot_ci(ymon, r_new["nl_mon"], r_prev["nl_mon"], gmon),
        "gain_ci_HONEST": L.group_boot_ci(y[held], va_new_f[held], va_prev_f[held],
                                          groups[held]),
        "epsilon": 0.005,
        "Delta_beyond_MONITOR_prev": T_mon - L.auc(ymon, r_prev["nl_mon"]),
        "Delta_beyond_MONITOR_new": T_mon - L.auc(ymon, r_new["nl_mon"]),
        "Delta_beyond_HONEST_prev": T_hon - L.auc(y[held], va_prev_f[held]),
        "Delta_beyond_HONEST_new": T_hon - L.auc(y[held], va_new_f[held]),
    }

    # ============================================================ SPURIOUS MAP =
    keepB, medB = L.clean_fit(XB[fitm])
    XBall = L.clean_apply(XB, keepB, medB)
    kept_ids = [B_ids[j] for j in keepB]
    mp = []
    for k, bid in enumerate(kept_ids):
        col = XBall[:, k]
        mp.append({
            "channel_id": bid, "name": B_name[bid],
            "upstream_parent": B_parent[bid], "mixed": B_mixed.get(bid, False),
            "alone_AUC_HONEST": L.auc(y[held], col[held]),
            "alone_AUC_MONITOR": L.auc(ymon, col[monm]),
            "alone_AUC_FITMINE": L.auc(y[fitm], col[fitm]),
            "mean": float(np.nanmean(Xnew[:, cids.index(bid)])),
            "na_rate": float(np.isnan(Xnew[:, cids.index(bid)]).mean()),
        })
    # CELL-SPECIFIC ANNOTATION (recorded in notes/2026-08-10__closure_mathse_accepted.md 3.1):
    # this cell's V block is 28 NAMED surface features, so length / LaTeX / formatting
    # channels are ALREADY IN THE ARTICULATED BANK. A channel that duplicates a v_*
    # column cannot be discounted off Delta without also being discounted off VA. Each
    # mapped channel therefore carries its strongest rank correlation with the V block,
    # so "already articulated" is visible instead of being double-counted.
    from scipy.stats import spearmanr as _sr
    Vb, vnames = d["V"], d["v_names"]
    for k, rec in enumerate(mp):
        col = XBall[:, kept_ids.index(rec["channel_id"])]
        best, bestn = 0.0, None
        for j, vn in enumerate(vnames):
            m = np.isfinite(col) & np.isfinite(Vb[:, j].astype(float))
            if m.sum() < 200:
                continue
            r = _sr(col[m], Vb[m, j].astype(float)).statistic
            if np.isfinite(r) and abs(r) > abs(best):
                best, bestn = float(r), vn
        rec["max_abs_rho_with_V_block"] = round(abs(best), 4)
        rec["closest_V_column"] = bestn
        rec["ALREADY_ARTICULATED"] = bool(abs(best) >= 0.70)

    dropped = [b for b in B_ids if b not in kept_ids]
    mp.sort(key=lambda r: -abs(r["alone_AUC_HONEST"] - .5))
    res["spurious_map"] = {"channels": mp,
                           "n_dropped_by_degeneracy_screen": len(dropped),
                           "dropped_ids": dropped}

    # A-side alone-AUCs, for symmetry (secondary)
    keepA, medA = (L.clean_fit(XA[fitm]) if XA.shape[1] else (np.array([], int), np.array([])))
    if len(keepA):
        XAall = L.clean_apply(XA, keepA, medA)
        res["track_A_alone_AUC"] = [
            {"criterion_id": A_ids[j], "name": B_name[A_ids[j]],
             "alone_AUC_HONEST": L.auc(y[held], XAall[held, k])}
            for k, j in enumerate(keepA)]
        res["track_A_alone_AUC"].sort(key=lambda r: -abs(r["alone_AUC_HONEST"] - .5))

    # =============================================================== DISCOUNT =
    def discount(rbx, cols_idx, label):
        out = {"n_B_features_after_screen": rbx["n_features"],
               "spurious_alone_AUC_linear_MONITOR": L.auc(ymon, rbx["lin_mon"]),
               "spurious_alone_AUC_histgb_MONITOR": L.auc(ymon, rbx["nl_mon"]),
               "spurious_alone_AUC_linear_OOF_fitmine": L.auc(y[fitm], rbx["oof_lin_fitmine"]),
               "spurious_alone_AUC_histgb_HONEST": L.auc(y[held], va_full(rbx)[held])}
        feats = XBall[:, cols_idx] if cols_idx is not None else XBall
        nms = [kept_ids[k] + " | " + B_name[kept_ids[k]]
               for k in (cols_idx if cols_idx is not None else range(len(kept_ids)))]
        jf = va_full(rbx)
        out["stratified_HONEST_q10"] = strat_table(
            y[held], va_new_f[held], dense[held], jf[held], feats[held], nms, 10)
        out["stratified_MONITOR_q5"] = strat_table(
            ymon, r_new["nl_mon"], dense[monm], jf[monm], feats[monm], nms, 5)
        return out

    res["discount_ALL_B"] = discount(rb, None, "all")
    if rb_strict is not None:
        keep_map = [k for k, bid in enumerate(kept_ids) if not B_mixed.get(bid)]
        res["discount_STRICT_no_mixed"] = discount(rb_strict, keep_map, "strict")
        res["discount_note"] = ("FREEZE ADDENDUM 2 sensitivity band: ALL_B discounts every "
                                "named channel including the MIXED ones (channels whose "
                                "conjectured upstream parent plausibly causes real quality "
                                "too); STRICT discounts only the unmixed ones. The truth "
                                "for Delta_adj lies between the two.")
    else:
        res["discount_note"] = ("no mixed/unmixed split possible this round "
                                f"({routing['n_mixed_B']} of {len(B_ids)} channels mixed)")

    # ===================================================== STACKED INCREMENT ==
    def stack_block(mask, jointp, vap, tagp):
        yy, gg = y[mask], groups[mask]
        jb, dn, va = jointp[mask], dense[mask], vap[mask]
        s_bd = stack_oof([jb, dn], yy, gg)
        s_bv = stack_oof([jb, va], yy, gg)
        s_all = stack_oof([jb, dn, va], yy, gg)
        return {
            "n": int(mask.sum()),
            "AUC_jointB": L.auc(yy, jb),
            "AUC_dense": L.auc(yy, dn),
            "AUC_bank_VA_nl": L.auc(yy, va),
            "AUC_stack_B_plus_dense": L.auc(yy, s_bd),
            "dense_increment_over_B": L.auc(yy, s_bd) - L.auc(yy, jb),
            "AUC_stack_B_plus_bank": L.auc(yy, s_bv),
            "bank_increment_over_B": L.auc(yy, s_bv) - L.auc(yy, jb),
            "AUC_stack_B_plus_dense_plus_bank": L.auc(yy, s_all),
            "dense_increment_over_B_plus_bank": L.auc(yy, s_all) - L.auc(yy, s_bv),
            "bank_increment_over_B_plus_dense": L.auc(yy, s_all) - L.auc(yy, s_bd),
            "ci_dense_increment": L.group_boot_ci(yy, s_bd, jb, gg),
            "ci_bank_increment": L.group_boot_ci(yy, s_bv, jb, gg),
        }

    res["swap_pair_HONEST"] = {
        "entering_round": swap_pair(y[held], dense[held], va_prev_f[held]),
        f"after_round_{a.round}": swap_pair(y[held], dense[held], va_new_f[held]),
    }
    sp0, sp1 = res["swap_pair_HONEST"]["entering_round"], res["swap_pair_HONEST"][f"after_round_{a.round}"]
    if sp0 and sp1:
        res["swap_pair_HONEST"]["delta"] = {
            "dC_plus": sp1["C_plus"] - sp0["C_plus"],
            "dC_minus": sp1["C_minus"] - sp0["C_minus"],
            "d_rho": sp1["spearman_bank_vs_dense"] - sp0["spearman_bank_vs_dense"],
            "swap_signature": bool(sp1["C_plus"] > sp0["C_plus"] and sp1["C_minus"] < sp0["C_minus"]),
        }

    res["stacked_increment_HONEST"] = stack_block(held, jb_f, va_new_f, "honest")
    res["stacked_increment_MONITOR"] = stack_block(monm, jb_f, va_new_f, "monitor")

    # ====================================================== B-side missing mass
    sp_path = HERE / f"{tag}_species.json"
    if sp_path.exists():
        s = json.loads(sp_path.read_text())
        res["missing_mass"] = {t: s["tracks"][t]["good_turing"] for t in s["tracks"]}
        res["n_species"] = {t: s["tracks"][t]["n_species"] for t in s["tracks"]}

    (HERE / f"{tag}_results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("score_report", "splits", "spurious_map",
                                   "discount_ALL_B", "discount_STRICT_no_mixed")},
                     indent=1, default=float)[:6000])
    print("\nSPURIOUS MAP (alone AUC on HONEST population)")
    for c in res["spurious_map"]["channels"]:
        print(f"  {c['alone_AUC_HONEST']:.3f}  {'MIXED' if c['mixed'] else '     '}  "
              f"{c['name'][:58]:58s}  <- {c['upstream_parent'][:40]}")
    print("wrote", HERE / f"{tag}_results.json")


if __name__ == "__main__":
    main()
