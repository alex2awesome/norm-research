#!/usr/bin/env python3
"""Per-round readout for the CW-community confirmatory closure campaign.

Consumes:
  round{r}_scores.npz        Gemma scores for this round's 25 criteria
  round{r}_routing.json      blind-audit routing (final track per criterion)
  round{r-1}_state.npz       the incoming bank state

Produces (round{r}_results.json + round{r}_state.npz):
  * Delta_r on MONITOR (the frozen decision readout) + population + TEST levels
  * (Delta C+, Delta C-) swap pair vs the previous round
  * Track-B discount table: spurious-alone AUC (linear + HistGB), then EITHER
    decile stratification OR matched sampling once spurious-alone > .65
  * stacked-increment readout (AUC of a joint-B-model vs a stack of B-model +
    dense; same for the bank) -- the stratification-free control from the
    FREEZE ADDENDUM
  * collapse-gate and sign-contradiction bookkeeping

CPU only.  Usage: python round_readout.py --round 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import closure_lib_cw as C

HERE = Path(__file__).resolve().parent
MATCH_TRIGGER = 0.65


# ---------------------------------------------------------------- helpers ----
def grouped_oof_prob(X, y, groups, kind="linear", seed=0):
    """Grouped-OOF probability over ALL rows (used for nuisance-alone readouts)."""
    folds = list(GroupKFold(n_splits=min(5, len(np.unique(groups))))
                 .split(np.zeros(len(y)), groups=groups))
    p = np.full(len(y), np.nan)
    for tr, te in folds:
        if kind == "linear":
            m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                              StandardScaler(),
                              LogisticRegression(C=1.0, max_iter=2000))
            m.fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        else:
            imp = SimpleImputer(strategy="median", add_indicator=True)
            Xtr, Xte = imp.fit_transform(X[tr]), imp.transform(X[te])
            m = HistGradientBoostingClassifier(max_leaf_nodes=15, learning_rate=0.06,
                                               max_iter=400, early_stopping=True,
                                               validation_fraction=0.1,
                                               n_iter_no_change=20, random_state=seed)
            m.fit(Xtr, y[tr])
            p[te] = m.predict_proba(Xte)[:, 1]
    return p


def matched_pair_auc(y, score, control, caliper=0.02, seed=0):
    """Matched-sampling readout: pair each positive with the closest-control
    negative (percentile caliper), then report the fraction of matched pairs the
    score orders correctly.  Holds the nuisance approximately constant instead of
    conditioning on coarse strata."""
    y = np.asarray(y)
    r = np.argsort(np.argsort(control)) / max(1, len(control) - 1)
    P = np.where(y == 1)[0]
    N = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    P = P[rng.permutation(len(P))]
    used = np.zeros(len(N), bool)
    rN = r[N]
    conc, ties, n = 0.0, 0, 0
    for i in P:
        d = np.abs(rN - r[i])
        d[used] = np.inf
        j = int(np.argmin(d))
        if not np.isfinite(d[j]) or d[j] > caliper:
            continue
        used[j] = True
        n += 1
        a, b = score[i], score[N[j]]
        if a > b:
            conc += 1
        elif a == b:
            conc += 0.5
            ties += 1
    return (float(conc / n) if n else float("nan"), {"n_pairs": int(n),
                                                     "n_ties": int(ties),
                                                     "caliper": caliper})


def swap_pair(y, bank, dense):
    """(C+, C-) = bank concordance conditional on the dense model being right /
    wrong on a (pos,neg) pair.  Exact pair algebra, no sampling."""
    P = np.where(y == 1)[0]
    N = np.where(y == 0)[0]
    D = np.sign(dense[P][:, None] - dense[N][None, :])
    S = np.sign(bank[P][:, None] - bank[N][None, :])
    conc = (S > 0).astype(float) + 0.5 * (S == 0)
    out = {}
    for nm, m in (("plus", D > 0), ("minus", D < 0), ("tie", D == 0)):
        w = m.sum()
        out[f"w_{nm}"] = float(w / D.size)
        out[f"C_{nm}"] = float(conc[m].mean()) if w else float("nan")
    return out


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    r = a.round
    prev = np.load(HERE / f"round{r-1}_state.npz", allow_pickle=True)
    VA0 = prev["VA"]
    y = prev["y"]
    groups = np.array([str(g) for g in prev["groups"]])
    split = np.array([str(s) for s in prev["split"]])
    ids = np.array([str(s) for s in prev["ids"]])
    T = prev["T"]
    nuis0 = prev["NUIS"] if "NUIS" in prev.files else np.zeros((len(y), 0))
    nuis0_names = ([str(s) for s in prev["nuis_names"]]
                   if "nuis_names" in prev.files else [])
    bank_names0 = [str(s) for s in prev["bank_names"]] if "bank_names" in prev.files \
        else [f"col{j}" for j in range(VA0.shape[1])]

    sc = np.load(HERE / f"round{r}_scores.npz", allow_pickle=True)
    sid = np.array([str(s) for s in sc["ids"]])
    assert list(sid) == list(ids), "score row order does not match state"
    X = sc["X"]
    cids = [str(s) for s in sc["cids"]]
    names = [str(s) for s in sc["names"]]
    rep = json.loads((HERE / f"round{r}_scores.report.json").read_text())
    collapsed = {c["cid"] for c in rep["collapse"] if c["COLLAPSED"]}

    routing = json.loads((HERE / f"round{r}_routing.json").read_text())
    final_track = {d["cid"]: d["final_track"] for d in routing["decisions"]}

    keepA = [j for j, c in enumerate(cids)
             if final_track.get(c) == "A" and c not in collapsed]
    keepB = [j for j, c in enumerate(cids)
             if final_track.get(c) == "B" and c not in collapsed]
    print(f"[r{r}] {len(cids)} scored, {len(collapsed)} collapsed -> "
          f"A={len(keepA)} B={len(keepB)}")

    # sign check: alone-AUC direction vs the stated quality rationale (report-only
    # trigger per the freeze -- criteria whose alone-AUC is < .5 are flagged)
    fmm = split == "fit_mine"

    def _alone(mask):
        d = {}
        for j, c in enumerate(cids):
            col = X[mask, j].copy()
            fin = np.isfinite(col)
            if fin.sum() < 20 or len(np.unique(col[fin])) < 2:
                d[c] = float("nan")
                continue
            col[~fin] = np.nanmedian(col)
            d[c] = float(roc_auc_score(y[mask], col))
        return d

    # DESIGN-DECISION alone-AUC (the sign-contradiction re-audit trigger) is read on
    # FIT+MINE only -- MONITOR is never read for a design decision.  The population
    # figure is reported for the spurious map, which is a readout, not a decision.
    alone_fm = _alone(fmm)
    alone = _alone(np.ones(len(y), bool))

    # TWO-SIDED, NOISE-SCALED BAND (adopted for rounds 6+ after the round-5 finding that
    # `alone-AUC < .5` fires on nulls).  Hanley-McNeil standard error at AUC = .5 on the
    # FIT+MINE class counts; a criterion trips the trigger only if it sits more than
    # 2 SE BELOW chance, i.e. the sign is evidence rather than noise.
    n1 = int(y[fmm].sum())
    n0 = int((~y[fmm].astype(bool)).sum())
    a0 = 0.5
    q1, q2 = a0 / (2 - a0), 2 * a0 ** 2 / (1 + a0)
    se = float(np.sqrt((a0 * (1 - a0) + (n1 - 1) * (q1 - a0 ** 2)
                        + (n0 - 1) * (q2 - a0 ** 2)) / (n1 * n0)))
    lo, hi = 0.5 - 2 * se, 0.5 + 2 * se
    sign_contradicting = [c for c in cids
                          if final_track.get(c) == "A"
                          and alone_fm[c] == alone_fm[c] and alone_fm[c] < lo]
    sign_null_band = [c for c in cids
                      if final_track.get(c) == "A"
                      and alone_fm[c] == alone_fm[c] and lo <= alone_fm[c] < 0.5]

    # ------------------------------------------------- bank update + Delta ----
    VA1 = np.column_stack([VA0, X[:, keepA]]) if keepA else VA0
    bank_names1 = bank_names0 + [names[j] for j in keepA]
    res1 = C.fit_block(VA1, y, groups, split)
    # the incoming state's own readouts are reconstructed from its saved per-seed
    # held-out predictions -- identical numbers, half the compute, and it guarantees
    # the round-over-round Delta compares the SAME fit the previous round reported
    if "held_seed_preds" in prev.files:
        res0 = {"_pop_nl": prev["pop_nl"], "_pop_lin": prev["pop_lin"],
                "va_nl_population": float(roc_auc_score(y, prev["pop_nl"])),
                "va_lin_population": float(roc_auc_score(y, prev["pop_lin"])),
                **C.monitor_stats_from_saved(y, split, prev["held_seed_preds"],
                                             prev["held_mask"].astype(bool))}
    else:
        res0 = C.fit_block(VA0, y, groups, split)

    mon = split == "monitor"
    tst = split == "test"
    out = {
        "round": r,
        "n_scored": len(cids), "n_collapsed": len(collapsed),
        "n_A_kept": len(keepA), "n_B_kept": len(keepB),
        "collapsed_cids": sorted(collapsed),
        "alone_auc_population": alone,
        "alone_auc_fitmine": alone_fm,
        "sign_contradicting_A": sign_contradicting,
        "sign_null_band_A": sign_null_band,
        "sign_band": {"rule": "two-sided Hanley-McNeil band at AUC=.5 on FIT+MINE; "
                              "trigger fires only below .5 - 2 SE (adopted round 6+)",
                      "se": se, "lo": lo, "hi": hi, "n_pos": n1, "n_neg": n0},
        "anchor_battery": rep["anchor_battery"],
        "bank_size_before": int(VA0.shape[1]), "bank_size_after": int(VA1.shape[1]),
        "prev": {k: v for k, v in res0.items() if not k.startswith("_")},
        "prev_source": ("reconstructed from saved per-seed predictions"
                        if "held_seed_preds" in prev.files else "refit"),
        "curr": {k: v for k, v in res1.items() if not k.startswith("_")},
    }
    out["Delta_VA_nl_monitor"] = res1["va_nl_monitor"] - res0["va_nl_monitor"]
    out["Delta_VA_nl_population"] = res1["va_nl_population"] - res0["va_nl_population"]
    out["T_monitor"] = float(roc_auc_score(y[mon], T[mon]))
    out["T_population"] = float(roc_auc_score(y, T))
    out["Delta_beyond_monitor"] = out["T_monitor"] - res1["va_nl_monitor"]
    out["Delta_beyond_population"] = out["T_population"] - res1["va_nl_population"]
    if tst.sum():
        out["T_test"] = float(roc_auc_score(y[tst], T[tst]))
        out["VA_nl_test"] = float(roc_auc_score(y[tst], res1["_pop_nl"][tst]))
        out["Delta_beyond_test_DO_NOT_QUOTE_UNTIL_STOP"] = (
            out["T_test"] - out["VA_nl_test"])
    out["boot_gain_monitor"] = C.group_bootstrap_delta(
        y[mon], res1["_pop_nl"][mon], res0["_pop_nl"][mon], groups[mon])

    # ------------------------------------------------------------ swap pair ---
    sp0 = swap_pair(y, res0["_pop_nl"], T)
    sp1 = swap_pair(y, res1["_pop_nl"], T)
    out["swap"] = {"prev": sp0, "curr": sp1,
                   "dC_plus": sp1["C_plus"] - sp0["C_plus"],
                   "dC_minus": sp1["C_minus"] - sp0["C_minus"],
                   "error_minus_insight_inheritance":
                       (-(sp1["C_minus"] - sp0["C_minus"]))
                       - (sp1["C_plus"] - sp0["C_plus"])}

    # -------------------------------------------------------- Track-B block ---
    # FREEZE ADDENDUM 2: every B channel carries a conjectured upstream parent and a
    # MIXED flag (parent plausibly causes real quality too).  MIXED channels are
    # reported in BOTH bands -- the full-set discount and a strict-set discount that
    # drops them -- never silently routed to one side.
    prov = json.loads((HERE / f"round{r}_proposals_provenance.json").read_text())
    meta_by_cid = {c["cid"]: c for c in prov["criteria"]}
    new_up = [meta_by_cid.get(cids[j], {}).get("upstream_parent") for j in keepB]
    new_mixed = [bool(meta_by_cid.get(cids[j], {}).get("mixed")) for j in keepB]
    up0 = [str(s) for s in prev["nuis_upstream"]] if "nuis_upstream" in prev.files else []
    mx0 = list(prev["nuis_mixed"].astype(bool)) if "nuis_mixed" in prev.files else []

    NUIS_all = np.column_stack([nuis0, X[:, keepB]]) if keepB else nuis0
    nuis_names_all = nuis0_names + [names[j] for j in keepB]
    nuis_up_all = up0 + [u if u else "unspecified" for u in new_up]
    nuis_mixed_all = mx0 + new_mixed

    # FREEZE ADDENDUM 3: a MIXED parent is RETIRED from the readouts (recorded, not
    # deleted) once its decomposed components have been scored in this round.
    retired = set()
    rf = HERE / "retired_parents.json"
    if rf.exists():
        for fam in json.loads(rf.read_text()):
            if fam.get("components_scored_in_round", 10 ** 9) <= r:
                retired |= set(fam["parent_channels_retired"])
    live = [j for j, n in enumerate(nuis_names_all) if n not in retired]
    NUIS = NUIS_all[:, live] if len(live) else NUIS_all[:, :0]
    nuis_names = [nuis_names_all[j] for j in live]
    nuis_up = [nuis_up_all[j] for j in live]
    nuis_mixed = [nuis_mixed_all[j] for j in live]
    retired_now = sorted(retired & set(nuis_names_all))
    disc = {"n_nuisance_cumulative": int(NUIS.shape[1]), "nuisance_names": nuis_names,
            "upstream_parents": dict(zip(nuis_names, nuis_up)),
            "mixed_channels": [n for n, m in zip(nuis_names, nuis_mixed) if m],
            "n_mixed": int(sum(nuis_mixed)),
            "retired_parents_addendum3": retired_now,
            "n_channels_retired": len(retired_now),
            "n_upstream_traced": int(sum(
                1 for u in nuis_up if u and u.lower() not in
                ("surface-only", "unspecified")))}
    if NUIS.shape[1]:
        b_lin = grouped_oof_prob(NUIS, y, groups, "linear")
        b_nl = grouped_oof_prob(NUIS, y, groups, "gbm")
        disc["spurious_alone_linear"] = float(roc_auc_score(y, b_lin))
        disc["spurious_alone_histgb"] = float(roc_auc_score(y, b_nl))
        joint = b_nl if disc["spurious_alone_histgb"] >= disc["spurious_alone_linear"] \
            else b_lin
        disc["joint_used"] = ("histgb" if joint is b_nl else "linear")
        disc["per_channel_alone_auc"] = {
            nuis_names[j]: (float(roc_auc_score(
                y, np.where(np.isfinite(NUIS[:, j]), NUIS[:, j],
                            np.nanmedian(NUIS[:, j]))))
                if np.isfinite(NUIS[:, j]).sum() > 20
                and len(np.unique(NUIS[np.isfinite(NUIS[:, j]), j])) > 1
                else None)
            for j in range(NUIS.shape[1])}

        va_pop = res1["_pop_nl"]
        s_alone = max(disc["spurious_alone_linear"], disc["spurious_alone_histgb"])
        disc["estimator"] = "matched_sampling" if s_alone > MATCH_TRIGGER else "deciles"

        def discount_with(control, label):
            if s_alone > MATCH_TRIGGER:
                tv, ti = matched_pair_auc(y, T, control)
                vv, vi = matched_pair_auc(y, va_pop, control)
                info = {"T": ti, "VA": vi}
            else:
                st = C.decile_strata(control)
                tv, ti = C.stratified_auc(y, T, st)
                vv, vi = C.stratified_auc(y, va_pop, st)
                info = {"T": ti, "VA": vi}
            return {"label": label, "T_adj": tv, "VA_adj": vv,
                    "Delta_adj": tv - vv, "info": info}

        full = discount_with(joint, "all_named_channels")
        disc.update({"T_adj": full["T_adj"], "VA_adj": full["VA_adj"],
                     "Delta_adj": full["Delta_adj"],
                     "discount_band": {"full_set": full}})
        # strict band: drop MIXED channels (their upstream parent plausibly causes
        # real quality, so discounting on them can over-discount)
        strict_idx = [j for j in range(NUIS.shape[1]) if not nuis_mixed[j]]
        if strict_idx and len(strict_idx) < NUIS.shape[1]:
            js = grouped_oof_prob(NUIS[:, strict_idx], y, groups,
                                  disc["joint_used"] if disc["joint_used"] != "linear"
                                  else "linear")
            disc["spurious_alone_strict"] = float(roc_auc_score(y, js))
            disc["discount_band"]["strict_set_no_mixed"] = discount_with(
                js, "surface_and_unmixed_only")
        disc["undiscounted"] = {"T": float(roc_auc_score(y, T)),
                                "VA": float(roc_auc_score(y, va_pop)),
                                "Delta": float(roc_auc_score(y, T)
                                               - roc_auc_score(y, va_pop))}

        # ------------------------------------- stacked-increment (addendum) ---
        def stack_auc(base, extra):
            Z = np.column_stack([base, extra])
            return float(roc_auc_score(y, grouped_oof_prob(Z, y, groups, "linear")))
        disc["stacked_increment"] = {
            "AUC_joint_B": float(roc_auc_score(y, joint)),
            "AUC_B_plus_dense": stack_auc(joint, T),
            "AUC_bank": float(roc_auc_score(y, va_pop)),
            "AUC_bank_plus_dense": stack_auc(va_pop, T),
        }
        disc["stacked_increment"]["dense_increment_over_all_named_channels"] = (
            disc["stacked_increment"]["AUC_B_plus_dense"]
            - disc["stacked_increment"]["AUC_joint_B"])
        disc["stacked_increment"]["dense_increment_over_bank"] = (
            disc["stacked_increment"]["AUC_bank_plus_dense"]
            - disc["stacked_increment"]["AUC_bank"])
    out["discount"] = disc

    np.savez_compressed(HERE / f"round{r}_state.npz", VA=VA1, y=y, groups=groups,
                        split=split.astype(object), ids=ids.astype(object), T=T,
                        NUIS=NUIS_all, pop_nl=res1["_pop_nl"], pop_lin=res1["_pop_lin"],
                        bank_names=np.array(bank_names1, dtype=object),
                        nuis_names=np.array(nuis_names_all, dtype=object),
                        nuis_upstream=np.array(nuis_up_all, dtype=object),
                        nuis_mixed=np.array(nuis_mixed_all, dtype=bool),
                        held_seed_preds=res1["_held_seed_preds"],
                        held_mask=res1["_held_mask"])
    (HERE / f"round{r}_results.json").write_text(json.dumps(out, indent=1, default=str))
    brief = {k: out[k] for k in
             ("round", "n_A_kept", "n_B_kept", "n_collapsed",
              "Delta_VA_nl_monitor", "Delta_VA_nl_population",
              "Delta_beyond_monitor", "Delta_beyond_population")}
    brief["VA_nl_monitor"] = res1["va_nl_monitor"]
    brief["swap"] = {k: out["swap"][k] for k in ("dC_plus", "dC_minus")}
    brief["spurious_alone"] = disc.get("spurious_alone_histgb")
    brief["Delta_adj"] = disc.get("Delta_adj")
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
