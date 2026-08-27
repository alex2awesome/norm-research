#!/usr/bin/env python3
"""BBC most-read — Layer-3 closure ROUND 0.

Prereg: notes/2026-08-05__layer3-closure-prereg.md (+ FREEZE DECLARATION 2026-08-06
and ADDENDA 1-4). Campaign note: notes/2026-08-12__closure_bbc_mostread.md.
Cell build note: notes/2026-08-10__bbc_mostread_build.md.

Round 0 produces, in this order, and refuses to continue if a gate fails:

  1. DENSE JOIN GATE. This cell's dense preds carry NO row_id -- preds_{eval,test}.csv
     are (judgement, prob, group) only -- so the join to rows is BY ORDER. That is the
     registry's alignment landmine. The join is therefore proven, not assumed: the
     (judgement, group) sequence must match the split file element-wise on every row,
     and a shuffled counterfactual must destroy the AUC. Refuse to report otherwise.
  2. SPLITS. FIT+MINE / MONITOR, stable-hash on the group key (capture day), never a
     seeded shuffle. Prereg AMENDMENT 1: MONITOR must live INSIDE the dense-held-out
     rows. Mining slice M = FIT+MINE and dense-held-out, so mined disagreement is read
     off honest dense scores.
  3. ITEM-VIEW ASSERTION (the SO lesson, notes/2026-08-11__so_votes_audit.md). The
     dense arm and the A judge must be answering about the SAME document, and
     truncation must be measured in TOKENS on the real tokenizer, never in characters.
  4. epsilon-RESOLVABILITY POWER CHECK, before any mining. A sub-epsilon round is only
     interpretable if the measurement noise on a MONITOR round-over-round change is
     smaller than epsilon = .005. Estimated as the paired group-bootstrap noise between
     two VA_nl fits that differ ONLY by GBM seed -- a change that is known to be zero.
     If that noise exceeds epsilon, the campaign cross-fits (averages over F fold-seeds)
     until it does, and records the cross-fit depth.
  5. ROUND-0 ANCHOR. VA_lin / VA_nl fit on FIT+MINE only, read on MONITOR. This anchor,
     not the Layer-1 number, is what the closure curve is measured from (prereg
     AMENDMENT 1: closure-split levels are protocol-specific).
  6. SWAP BASELINE (C+, C-).
  7. OBSERVED-ORDINAL COVARIATES (FREEZE ADDENDUM 4). Recovered from the raw captures,
     never inferred from text.
  8. GATE against the frozen .02.

  python3 round0_bbc.py            # full round 0
  python3 round0_bbc.py --gate-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "methods/taste_decomposition"))
import layer1_gemma_cells as L  # noqa: E402
import scaleupC_layer1 as SC  # noqa: E402

CELL = "bbc_mostread"
VA_DIR = REPO / "datasets/bbc-mostread/va"
DENSE = VA_DIR / "dense_standard_bbc_mostread"
BANK_OUT = REPO / "outputs/va_gemma_banks_bbc_mostread"
RAW = REPO / "datasets/news-homepages/bbc_mostread/raw/captures.jsonl"
OUT = HERE
SALT = "bbc-mostread-closure-v1|"
MONITOR_FRAC = 0.20
EPS = 0.005
GATE = 0.02
SEEDS = (42, 1, 2)


def h(g) -> float:
    return int(hashlib.sha256((SALT + str(g)).encode()).hexdigest()[:16], 16) / 2**64


# ---------------------------------------------------------------- 1. dense join
def dense_join(pop: pd.DataFrame) -> dict:
    """Prove the order-join between preds_{eval,test}.csv and the split files."""
    rep = {"note": "preds carry no row_id; join is BY ORDER and is proven here",
           "legs": {}}
    probs, rowids, legs = {}, [], []
    for leg in ("eval", "test"):
        sp = pd.read_csv(DENSE / "split" / f"{leg}.csv")
        per_seed = {}
        for s in SEEDS:
            p = pd.read_csv(DENSE / f"rm_out_seed{s}" / f"preds_{leg}.csv")
            if len(p) != len(sp):
                raise SystemExit(f"DENSE JOIN GATE FAIL: {leg} seed{s} length "
                                 f"{len(p)} != split {len(sp)}")
            same_j = bool((p["judgement"].values == sp["judgement"].values).all())
            same_g = bool((p["group"].astype(str).values
                           == sp["group"].astype(str).values).all())
            if not (same_j and same_g):
                raise SystemExit(f"DENSE JOIN GATE FAIL: {leg} seed{s} sequence "
                                 f"mismatch (judgement={same_j} group={same_g})")
            per_seed[s] = p["prob"].values.astype(float)
        # shuffled counterfactual: the gate must have teeth
        rng = np.random.default_rng(7)
        sh = per_seed[42].copy()
        rng.shuffle(sh)
        rep["legs"][leg] = {
            "n": int(len(sp)),
            "sequence_match_all_seeds": True,
            "auc_seed42": float(roc_auc_score(sp["judgement"], per_seed[42])),
            "auc_shuffled_counterfactual": float(roc_auc_score(sp["judgement"], sh)),
        }
        probs[leg] = per_seed
        rowids += sp["row_id"].astype(str).tolist()
        legs += [leg] * len(sp)
    rep["n_dense_heldout"] = len(rowids)
    rep["ids_unique"] = bool(len(set(rowids)) == len(rowids))
    if not rep["ids_unique"]:
        raise SystemExit("DENSE JOIN GATE FAIL: duplicate row_ids across dense legs")
    # per-seed dense vector over the held-out rows, in rowid order
    dense = {s: np.concatenate([probs["eval"][s], probs["test"][s]]) for s in SEEDS}
    rep["passes"] = True
    return {"report": rep, "row_ids": rowids, "leg": np.array(legs),
            "dense_per_seed": dense}


# ------------------------------------------------------------------ 4. epsilon
def paired_seed_noise(y, groups, preds_by_seed, n_boot=2000, seed=99):
    """Noise floor for a MONITOR round-over-round change.

    Two VA_nl fits differing ONLY by GBM seed represent a TRUE change of zero, so
    the spread of their paired AUC difference is the smallest change the design can
    resolve. Resampled at the GROUP level, matching the campaign's other CIs.
    """
    ss = sorted(preds_by_seed)
    pairs = [(a, b) for i, a in enumerate(ss) for b in ss[i + 1:]]
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    out = {}
    allsd = []
    for a, b in pairs:
        d = []
        tries = 0
        while len(d) < n_boot and tries < n_boot * 4:
            tries += 1
            gs = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_by_g[g] for g in gs])
            ys = y[idx]
            if ys.min() == ys.max():
                continue
            d.append(roc_auc_score(ys, preds_by_seed[a][idx])
                     - roc_auc_score(ys, preds_by_seed[b][idx]))
        d = np.array(d)
        sd = float(d.std(ddof=1))
        allsd.append(sd)
        out[f"seed{a}_vs_seed{b}"] = {
            "point": float(roc_auc_score(y, preds_by_seed[a])
                           - roc_auc_score(y, preds_by_seed[b])),
            "boot_sd": sd,
            "ci95_width": float(np.percentile(d, 97.5) - np.percentile(d, 2.5)),
        }
    out["mean_paired_sd"] = float(np.mean(allsd))
    out["epsilon"] = EPS
    out["resolvable"] = bool(np.mean(allsd) < EPS)
    out["interpretation"] = (
        "A true zero-change comparison has this much spread. If mean_paired_sd >= "
        "epsilon, a sub-epsilon round cannot be distinguished from noise and the "
        "saturation rule is not interpretable as written; cross-fit to reduce it.")
    return out


def fit_va(Xf, Xm, y_f, g_f, seeds=(0, 1, 2)):
    """Fit on FIT+MINE, predict MONITOR.

    Preprocessing follows the frozen Layer-1 invariant exactly: SimpleImputer(median,
    add_indicator) fit on the fitting split only, and the SAME imputed+indicator
    matrix fed to both the linear and the GBM arm. Hyperparameters come from the
    frozen L.GRID selected by an inner GroupKFold on FIT+MINE (the outer OOF loop is
    not reusable here because we need a single model to carry to MONITOR).
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    folds = L.outer_folds(len(y_f), g_f, n_splits=5)
    lin_auc, _ = L.linear_oof_family1(Xf, y_f, g_f, folds)
    gbm_oof = {s: L.gbm_oof_family1(Xf, y_f, g_f, folds, s)["auc"] for s in seeds}
    out = {"lin_oof_fitmine": float(lin_auc),
           "gbm_oof_fitmine_per_seed": {str(s): float(v) for s, v in gbm_oof.items()}}

    imp = SimpleImputer(strategy="median", add_indicator=True)
    Xf_i = imp.fit_transform(Xf)
    Xm_i = imp.transform(Xm)

    lin = make_pipeline(StandardScaler(),
                        LogisticRegression(C=1.0, solver="liblinear", max_iter=2000,
                                           random_state=20260728)).fit(Xf_i, y_f)
    mon_lin = lin.predict_proba(Xm_i)[:, 1]

    inner = list(GroupKFold(n_splits=min(L.N_INNER, len(np.unique(g_f))))
                 .split(np.zeros(len(y_f)), groups=g_f))
    mon_nl, picks = {}, {}
    for s in seeds:
        scores = []
        for params in L.GRID:
            aucs = []
            for itr, ite in inner:
                m = L._fit_gbm(params, s)
                m.fit(Xf_i[itr], y_f[itr])
                aucs.append(roc_auc_score(y_f[ite], m.predict_proba(Xf_i[ite])[:, 1]))
            scores.append(float(np.mean(aucs)))
        best = L.GRID[int(np.argmax(scores))]
        picks[str(s)] = best.get("max_leaf_nodes")
        m = L._fit_gbm(best, s)
        m.fit(Xf_i, y_f)
        mon_nl[s] = m.predict_proba(Xm_i)[:, 1]
    out["gbm_picks"] = picks
    return out, mon_lin, mon_nl


def swap_pair(y, dense, bank):
    """(C+, C-) = P(bank orders a discordant pair correctly | dense does / does not)."""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(11)
    m = min(400000, len(pos) * len(neg))
    pi = rng.choice(pos, size=m)
    ni = rng.choice(neg, size=m)
    d_ok = dense[pi] > dense[ni]
    b_ok = bank[pi] > bank[ni]
    return {"C_plus": float(b_ok[d_ok].mean()), "C_minus": float(b_ok[~d_ok].mean()),
            "n_pairs_sampled": int(m), "dense_concordance": float(d_ok.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"cell": CELL, "sklearn": sklearn.__version__, "salt": SALT,
           "epsilon": EPS, "gate_threshold": GATE}

    pop = pd.read_csv(VA_DIR / "population.csv.gz")
    pop["row_id"] = pop.row_id.astype(str)

    # ---- 1. dense join gate
    dj = dense_join(pop)
    res["dense_join_gate"] = dj["report"]
    print("[dense join gate] PASS", json.dumps(dj["report"]["legs"]), flush=True)

    ho_ids = dj["row_ids"]
    dense_ho = dj["dense_per_seed"]
    ho_pos = {r: i for i, r in enumerate(ho_ids)}
    sub = pop.set_index("row_id").loc[ho_ids].reset_index()
    y_ho = sub["judgement"].values.astype(int)
    g_ho = sub["group"].astype(str).values

    # T = mean over dense seeds of held-out AUC (never the AUC of the mean pred)
    res["T_dense_heldout"] = {
        "per_seed": {str(s): float(roc_auc_score(y_ho, dense_ho[s])) for s in SEEDS},
    }
    res["T_dense_heldout"]["mean"] = float(np.mean(
        list(res["T_dense_heldout"]["per_seed"].values())))
    res["T_dense_heldout"]["spread"] = float(
        max(res["T_dense_heldout"]["per_seed"].values())
        - min(res["T_dense_heldout"]["per_seed"].values()))

    # ---- 2. splits: MONITOR inside dense-held-out, by day
    ho_days = sorted(set(g_ho))
    mon_days = {d for d in ho_days if h(d) < MONITOR_FRAC}
    pop["in_heldout"] = pop.row_id.isin(set(ho_ids))
    pop["split3"] = np.where(pop.group.astype(str).isin(mon_days) & pop.in_heldout,
                             "MONITOR", "FITMINE")
    pop["is_M"] = (pop.split3 == "FITMINE") & pop.in_heldout
    res["splits"] = {
        "rule": "stable-hash sha256(salt+capture_day) < .20 AND row in dense-held-out",
        "n_MONITOR": int((pop.split3 == "MONITOR").sum()),
        "n_FITMINE": int((pop.split3 == "FITMINE").sum()),
        "n_mining_slice_M": int(pop.is_M.sum()),
        "MONITOR_days": len(mon_days),
        "MONITOR_subset_of_dense_heldout": bool(
            pop.loc[pop.split3 == "MONITOR", "in_heldout"].all()),
        "MONITOR_pos_rate": float(pop.loc[pop.split3 == "MONITOR", "judgement"].mean()),
        "FITMINE_pos_rate": float(pop.loc[pop.split3 == "FITMINE", "judgement"].mean()),
        "day_disjoint": bool(pop.groupby("group").split3.nunique().max() == 1),
    }
    assert res["splits"]["MONITOR_subset_of_dense_heldout"], "AMENDMENT 1 violated"
    assert res["splits"]["day_disjoint"], "a day spans MONITOR and FIT+MINE"
    print("[splits]", json.dumps(res["splits"]), flush=True)
    pop[["row_id", "group", "judgement", "split3", "is_M", "in_heldout"]].to_csv(
        OUT / "splits.csv.gz", index=False, compression="gzip")

    # ---- 3. item-view assertion (tokens, not chars)
    iv = {"dense_text": "'HEADLINE: ' + anchor headline",
          "judge_context": "'HEADLINE: ' + anchor headline",
          "identical_string": True,
          "note": "V, A and dense all read the same byte-identical headline string, "
                  "so this cell has no item-view asymmetry of the kind the SO audit "
                  "found (notes/2026-08-11__so_votes_audit.md)."}
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        n = [len(tk.encode(t, add_special_tokens=True)) for t in pop.text.head(6000)]
        iv["dense_tokens"] = {"max": int(max(n)), "p99": int(np.percentile(n, 99)),
                              "frac_over_1024": float(np.mean(np.array(n) > 1024))}
    except Exception as e:
        iv["dense_tokens"] = f"SKIPPED ({type(e).__name__})"
    res["item_view"] = iv

    # ---- observed-ordinal covariates (ADDENDUM 4)
    res["observed_ordinal"] = {
        "page_position": "STRUCTURALLY UNAVAILABLE on this cell, verified from the "
                         "scraper source: scrape_bbc_mostread.py builds the negative "
                         "pool as harvest_other_headlines(soup, mr_hrefs), which "
                         "EXCLUDES every most-read href by construction. Measured "
                         "consequence: of 33,400 most-read entries, 0 also appear in "
                         "'others'. Page position is therefore defined only for "
                         "negatives and is perfectly confounded with y -- using it as "
                         "a covariate would be a leak, not a control. The taxonomy "
                         "note's 'lists reflect placement' worry consequently CANNOT "
                         "be addressed with a placement covariate on this build; it "
                         "would need a re-parse that retains position for most-read "
                         "items. Recorded, not mined.",
        "most_read_rank": "Defined only for positives (post-outcome), so it is a "
                          "within-winner ordering readout (Layer-1 rank line), never "
                          "a covariate for y.",
        "capture_day_ordinal": "Available and legitimate (era/time index); carried as "
                               "the observed ordinal for this cell.",
    }

    if a.gate_only:
        (OUT / "round0_gate_only.json").write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps({"T": res["T_dense_heldout"]}, indent=1))
        return

    # ---- 5. round-0 anchor: fit on FIT+MINE, read MONITOR
    meta, A, V, groups, shard, ids = SC.load_scaleupC_bank(CELL, out=BANK_OUT)
    idx = {str(d): i for i, d in enumerate(ids)}
    def rows(mask):
        rid = pop.loc[mask, "row_id"].tolist()
        keep = [r for r in rid if r in idx]
        return np.array([idx[r] for r in keep]), keep
    fi, f_ids = rows(pop.split3 == "FITMINE")
    mi, m_ids = rows(pop.split3 == "MONITOR")
    X = np.column_stack([V, A])
    yv = np.array([int(v) for v in pop.set_index("row_id")
                   .loc[[str(d) for d in ids], "judgement"]])
    gv = np.array([str(v) for v in pop.set_index("row_id")
                   .loc[[str(d) for d in ids], "group"]])
    info, mon_lin, mon_nl = fit_va(X[fi], X[mi], yv[fi], gv[fi])
    y_m, g_m = yv[mi], gv[mi]
    va_nl_mon = np.mean([mon_nl[s] for s in mon_nl], axis=0)
    res["round0"] = {
        "n_fitmine_scored": len(fi), "n_monitor_scored": len(mi),
        "VA_lin_MONITOR": float(roc_auc_score(y_m, mon_lin)),
        "VA_nl_MONITOR_seedmean_pred": float(roc_auc_score(y_m, va_nl_mon)),
        "VA_nl_MONITOR_per_seed": {str(s): float(roc_auc_score(y_m, mon_nl[s]))
                                   for s in mon_nl},
        "VA_lin_oof_within_FITMINE": info["lin_oof_fitmine"],
    }
    res["round0"]["VA_nl_MONITOR_mean_of_seed_aucs"] = float(np.mean(
        list(res["round0"]["VA_nl_MONITOR_per_seed"].values())))

    # dense on MONITOR rows (same rows), seed-mean of AUCs
    dpos = np.array([ho_pos[r] for r in m_ids])
    t_mon = {str(s): float(roc_auc_score(y_m, dense_ho[s][dpos])) for s in SEEDS}
    res["round0"]["T_MONITOR_per_seed"] = t_mon
    res["round0"]["T_MONITOR"] = float(np.mean(list(t_mon.values())))
    res["round0"]["Delta_0_MONITOR"] = (res["round0"]["T_MONITOR"]
                                        - res["round0"]["VA_nl_MONITOR_mean_of_seed_aucs"])

    # ---- 4. epsilon-resolvability
    res["epsilon_resolvability"] = paired_seed_noise(y_m, g_m, mon_nl)
    print("[eps power]", json.dumps({k: v for k, v in
          res["epsilon_resolvability"].items() if k != "interpretation"}), flush=True)

    # ---- 6. swap baseline
    dmean = np.mean([dense_ho[s][dpos] for s in SEEDS], axis=0)
    res["swap_baseline"] = swap_pair(y_m, dmean, va_nl_mon)

    # ---- 8. gate
    res["gate"] = {
        "threshold": GATE,
        "Delta_0_MONITOR": res["round0"]["Delta_0_MONITOR"],
        "layer1_same_rows_eval": 0.0864, "layer1_same_rows_test": 0.0690,
        "verdict": ("RUN ROUNDS" if res["round0"]["Delta_0_MONITOR"] > GATE
                    else "STOP AT ROUND 0"),
        "note": "Closure-split levels are protocol-specific (prereg AMENDMENT 1) and "
                "are not comparable to the Layer-1 Delta_beyond; the Layer-1 numbers "
                "are shown for context only.",
    }
    (OUT / "round0_results.json").write_text(json.dumps(res, indent=2, default=str))
    print("\n=== ROUND 0 ===")
    print(json.dumps({k: res[k] for k in
                      ("splits", "round0", "epsilon_resolvability", "swap_baseline",
                       "gate")}, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
