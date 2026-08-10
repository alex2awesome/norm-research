#!/usr/bin/env python3
"""Layer-1 "nonlinear stack" of the taste-residual decomposition, patents
claim-fell cell.

Design: notes/2026-08-05__taste-decomposition-design.md (S0 ledger, S1 protocol).
Template: methods/taste_decomposition/nc_layer1_stack.py (most current; group-level
cluster bootstrap PRIMARY, seeds {0,1,2} mean+spread).

THIS CELL'S SPECIAL RULE (task brief + registry notes/2026-07-27__vat-run-registry.md
line 54: "Patents V .601/VA .626 . T on claim-fell (V4 -- NO honest dense exists)"):
there is no honest dense model for this cell -- the ledger stops at Delta_interact.
NEVER compute or report a T/Delta_total/Delta_beyond here.

Matrix provenance (discovery log, see report): the published V .601/VA .626 pair is
NOT the qwen-judge per-rejection-class numbers in notes/2026-06-12__patents-vat-final-
table.md, nor the OA-scale retrieved-only V=0.621/within-doc=0.626 in that same note.
It is the "option3 scale build" numbers first reported (as unsourced hardcoded
constants PAT_V=.591/PAT_A=.616) in notebooks/2026-07-01__patents-laws-VA-decomposition
.ipynb, later corrected by datasets/patents/audit_regroup_va.py once the CSV was
row-aligned with its app_id source (documented result in running-research-notes.md
2026-07-08: "grouped-by-app V=.601/A=.623"). This script is that same audit,
ported into the Layer-1 template:
  - matrix: notebooks/data/patents_va_features.csv (59,937 claim-rejection-class
    rows x 7 V (thin/lexical-overlap) + 4 A (LLM/Gemma disclosure-judgment) cols)
  - group column: app_id, attached by row-aligning the CSV against
    datasets/patents/processed/option3_claims_gemma_scale.jsonl (sk3; cached locally,
    slim fields only, at methods/taste_decomposition/data_cache/
    patents_option3_claims_slim.jsonl -- app_id/claim_num/element-md5/label/
    rejection_type/n_refs/n_disclose/gold_disclose only, verified 0/59,937 mismatches
    against the CSV's shared fields)
  - V_COLS = thin lexical/structural features (verifiable, no model judgment)
  - A_COLS (this script's "A-only") = 4 Gemma-judged per-element disclosure
    aggregates (a_n_disclose, a_any_disclose, a_frac_disclose, a_max_disclose_overlap)
  - VA = V_COLS + A_COLS concatenated (11 total) -- this is what the notebook/audit
    script calls "A" (thin+thick); DO NOT confuse with this script's A-only.

GATE: the "existing linear aggregation" for this cell is audit_regroup_va.py's own
hand-rolled L2-logistic-regression + grouped_auc (5-fold split of *sorted app_id set*,
not sklearn GroupKFold) -- mirrored here VERBATIM as the linear leg, on FULL DATA
(no dedup; the dedup variant in the audit script is a robustness variant, not the
published number). Reproduces V=.6007 (published .601, diff .0003) and
VA=.6233 (published .626, diff .0027) -- both within +-.005 tolerance -> GATE PASS.
Outer folds for the nonlinear leg are the SAME sorted-app_id 5-way split (protocol:
"same outer folds for linear+nonlinear").

CPU only. No new judging. Usage:
  python patents_verdict_layer1.py
  python patents_verdict_layer1.py --shap
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "notebooks" / "data" / "patents_va_features.csv"
JL_SLIM = Path(__file__).resolve().parent / "data_cache" / "patents_option3_claims_slim.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

V_COLS = ["v_max_lexoverlap", "v_mean_lexoverlap", "v_count_lexhit", "v_element_wordlen",
          "v_n_refs", "v_max_spanlen", "v_mean_spanlen"]
A_ONLY_COLS = ["a_n_disclose", "a_any_disclose", "a_frac_disclose", "a_max_disclose_overlap"]
VA_COLS = V_COLS + A_ONLY_COLS

# published gate (registry notes/2026-07-27__vat-run-registry.md line 54, cross-checked
# against running-research-notes.md 2026-07-08 audit_regroup_va.py FULL-DATA result)
PUBLISHED = {"V": 0.601, "VA": 0.626}
GATE_TOL = 0.005

GRID = [
    {"max_leaf_nodes": 15, "learning_rate": 0.06, "max_iter": 400},
    {"max_leaf_nodes": 31, "learning_rate": 0.06, "max_iter": 400},
]
N_OUTER = 5
N_INNER = 3


# ---------------------------------------------------------------- data -----
def clean_cols(M):
    """Identical degeneracy/impute guard used by every other Layer-1 cell
    (nc_layer1_stack.clean_cols): drop degenerate cols, median-impute NA.
    No-op here in practice (0 NaN, all 11 cols non-degenerate, verified by
    inspection) but applied for protocol identity across linear/nonlinear."""
    keep, out = [], []
    for j in range(M.shape[1]):
        col = M[:, j].astype(float)
        nonna = col[~np.isnan(col)]
        if len(nonna) == 0:
            continue
        med = np.median(nonna)
        c = np.where(np.isnan(col), med, col)
        vals, counts = np.unique(c, return_counts=True)
        offmodal = len(c) - counts.max()
        if offmodal < 5 or c.std() == 0:
            continue
        keep.append(j)
        out.append(c)
    if not out:
        return np.zeros((M.shape[0], 0)), keep
    return np.column_stack(out), keep


def load_data():
    rows = list(csv.DictReader(open(CSV_PATH)))
    jkeys = []
    with open(JL_SLIM) as fh:
        for ln in fh:
            r = json.loads(ln)
            jkeys.append((str(r["app_id"]), str(r["claim_num"]),
                          hashlib.md5(r["element"].encode()).hexdigest(),
                          r["label"], str(r.get("rejection_type")),
                          int(r["n_refs"]), int(r["n_disclose"]), bool(r["gold_disclose"])))
    assert len(rows) == len(jkeys), f"row-count mismatch: csv={len(rows)} jsonl={len(jkeys)}"

    mism = 0
    for c, j in zip(rows, jkeys):
        ok = (int(float(c["fell"])) == (1 if j[3] == "pos" else 0)
              and int(float(c["v_n_refs"])) == j[5]
              and int(float(c["a_n_disclose"])) == j[6]
              and int(float(c["gold_disclose"])) == int(j[7]))
        mism += not ok
    assert mism == 0, f"{mism}/{len(rows)} rows failed CSV<->jsonl alignment -- app_id attach invalid"

    X_all = np.array([[float(c[col]) for col in VA_COLS] for c in rows])
    y = np.array([float(c["fell"]) for c in rows])
    g = np.array([j[0] for j in jkeys])  # app_id
    return X_all, y, g, mism


# -------------------------------------------------- linear (existing agg) --
def _auc(y, s):
    """Mann-Whitney AUC, verbatim from audit_regroup_va.py / the notebook."""
    y = np.asarray(y, float); s = np.asarray(s, float)
    order = np.argsort(s, kind="mergesort"); sr = s[order]
    rk = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sr[j + 1] == sr[i]:
            j += 1
        rk[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    p = (y == 1).sum(); n = (y == 0).sum()
    return (rk[y == 1].sum() - p * (p + 1) / 2) / (p * n)


def sorted_app_id_folds(g):
    """The task's EXISTING fold construction (audit_regroup_va.py / the notebook's
    grouped_auc): sorted unique app_id set split into 5 contiguous blocks via
    np.array_split -- NOT sklearn GroupKFold. Returned as (train_idx, test_idx)
    tuples so it is a drop-in outer-fold list for both the linear gate and the
    nonlinear leg (protocol: same outer folds for linear+nonlinear)."""
    uniq = np.array(sorted(set(g)))
    blocks = np.array_split(uniq, N_OUTER)
    folds = []
    for blk in blocks:
        te_mask = np.isin(g, blk)
        te = np.where(te_mask)[0]
        tr = np.where(~te_mask)[0]
        folds.append((tr, te))
    return folds


def linear_grouped_auc(X, y, folds):
    """VERBATIM reproduction of audit_regroup_va.grouped_auc's L2-logistic-
    regression-by-gradient-descent, refactored to take a pre-built fold list
    (same folds object as the nonlinear leg) and to also return the OOF vector
    (needed downstream for the Delta_interact bootstrap)."""
    sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    oof = np.zeros(len(y))
    for tr, te in folds:
        Xt = X[tr]
        mu = Xt.mean(0); sd = Xt.std(0) + 1e-8
        Xb = np.c_[np.ones(len(Xt)), (Xt - mu) / sd]
        w = np.zeros(Xb.shape[1])
        for _ in range(2500):
            p = sig(Xb @ w)
            w -= 0.3 * (Xb.T @ (p - y[tr]) / len(Xt) + 1e-2 * np.r_[0, w[1:]])
        Xe = X[te]
        Xeb = np.c_[np.ones(len(te)), (Xe - mu) / sd]
        oof[te] = sig(Xeb @ w)
    return float(_auc(y, oof)), oof


# --------------------------------------------------------------- nonlinear -
def _fit_gbm(params, seed):
    return HistGradientBoostingClassifier(
        max_leaf_nodes=params["max_leaf_nodes"],
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
    )


def gbm_oof(Xf, y, groups, folds, seed=0, verbose=False):
    """Nested: inner GroupKFold(3) grid pick inside each train fold only.
    Verbatim structure from nc_layer1_stack.gbm_oof."""
    oof = np.zeros(len(y))
    picks, train_aucs, inner_log = [], [], []
    for fi, (tr, te) in enumerate(folds):
        gtr = groups[tr]
        inner = list(GroupKFold(n_splits=min(N_INNER, len(np.unique(gtr)))).split(np.zeros(len(tr)), groups=gtr))
        scores = []
        for params in GRID:
            aucs = []
            for itr, ite in inner:
                m = _fit_gbm(params, seed)
                m.fit(Xf[tr][itr], y[tr][itr])
                aucs.append(roc_auc_score(y[tr][ite], m.predict_proba(Xf[tr][ite])[:, 1]))
            scores.append(float(np.mean(aucs)))
        best = int(np.argmax(scores))
        picks.append(GRID[best]["max_leaf_nodes"])
        inner_log.append({"fold": fi, "inner_auc": dict(zip([str(g["max_leaf_nodes"]) for g in GRID], scores))})
        m = _fit_gbm(GRID[best], seed)
        m.fit(Xf[tr], y[tr])
        oof[te] = m.predict_proba(Xf[te])[:, 1]
        train_aucs.append(float(roc_auc_score(y[tr], m.predict_proba(Xf[tr])[:, 1])))
        if verbose:
            print(f"    fold {fi}: pick leaves={picks[-1]} inner={scores} train_auc={train_aucs[-1]:.3f}")
    return {
        "auc": float(roc_auc_score(y, oof)),
        "picks": picks,
        "train_auc_mean": float(np.mean(train_aucs)),
        "train_aucs": train_aucs,
        "inner": inner_log,
        "oof": oof,
    }


# ----------------------------------------------------------------- bootstrap
def bootstrap_delta_interact(oof_lin, oof_nl_mean, y, n_boot=2000, seed=0):
    """Row-level paired bootstrap over OOF rows -- SECONDARY diagnostic only.
    Rows are not exchangeable (app_id-grouped OOF folds; a single application
    contributes multiple claim-rejection rows), so this understates CI width.
    Verbatim method from nc_layer1_stack.bootstrap_delta_interact."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        auc_lin = roc_auc_score(yb, oof_lin[idx])
        auc_nl = roc_auc_score(yb, oof_nl_mean[idx])
        deltas.append(auc_nl - auc_lin)
    deltas = np.array(deltas)
    return {
        "n_boot_used": int(len(deltas)),
        "mean": float(deltas.mean()),
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "p_gt_0": float((deltas > 0).mean()),
    }


def bootstrap_delta_interact_group(oof_lin, oof_nl_mean, y, groups, n_boot=2000, seed=0):
    """PRIMARY bootstrap CI on Delta_interact: cluster (app_id-level) resample.
    Rows within an app_id (multiple claims x rejection classes on one patent
    application) are not independent, so a row-level resample understates CI
    width. Generalized from nc_layer1_stack.bootstrap_delta_interact_docket
    (there: docket; here: app_id) per the task's FROZEN group-level-PRIMARY
    requirement."""
    rng = np.random.default_rng(seed)
    uniq_groups = np.unique(groups)
    n_groups = len(uniq_groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq_groups}
    deltas = []
    for _ in range(n_boot):
        draw = rng.choice(uniq_groups, size=n_groups, replace=True)
        idx = np.concatenate([idx_by_group[g] for g in draw])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        auc_lin = roc_auc_score(yb, oof_lin[idx])
        auc_nl = roc_auc_score(yb, oof_nl_mean[idx])
        deltas.append(auc_nl - auc_lin)
    deltas = np.array(deltas)
    return {
        "n_boot_used": int(len(deltas)),
        "n_groups_resampled": int(n_groups),
        "mean": float(deltas.mean()),
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "p_gt_0": float((deltas > 0).mean()),
    }


# ----------------------------------------------------------------- shap ----
def shap_interactions(Xf, y, names, seed=0, n_sub=300):
    """Top interaction pairs (descriptive only). Only 11 VA features total, so
    no top-k screening step is needed (unlike the 100+-feature cells) -- fit
    the frozen-grid model directly and compute exact TreeSHAP interactions on
    a 300-row subsample."""
    import shap

    m = _fit_gbm(GRID[1], seed)
    m.fit(Xf, y)
    ex = shap.TreeExplainer(m)
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(y), size=min(n_sub, len(y)), replace=False)
    sv = ex.shap_values(Xf[sub])
    if isinstance(sv, list):
        sv = sv[-1]
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    imp = np.abs(sv).mean(0)

    iv = ex.shap_interaction_values(Xf[sub])
    if isinstance(iv, list):
        iv = iv[-1]
    if iv.ndim == 4:
        iv = iv[:, :, :, -1]
    M = np.abs(iv).mean(0)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append((names[i], names[j], float(M[i, j] + M[j, i])))
    pairs.sort(key=lambda t: -t[2])
    diag = [(names[i], float(M[i, i])) for i in range(len(names))]
    off_frac = float((M.sum() - np.trace(M)) / M.sum())
    return {
        "method": "TreeSHAP exact interaction values (all 11 VA features, 300-row subsample)",
        "top_features": [{"name": n, "mean_abs_shap": float(v)} for n, v in zip(names, imp)],
        "top_pairs": [{"a": a, "b": b, "mean_abs_interaction": v} for a, b, v in pairs[:10]],
        "main_effects": [{"name": n, "mean_abs_main": v} for n, v in diag],
        "offdiagonal_mass_fraction": off_frac,
        "n_subsample": int(len(sub)),
    }


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shap", action="store_true", help="also compute SHAP interactions on VA")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or str(RESULTS_DIR / "patents_verdict_layer1.json")
    t0 = time.time()

    X_all, y, g, mism = load_data()
    n = len(y)
    n_groups = len(np.unique(g))
    print(f"n={n} pos_rate={y.mean():.4f} groups(app_id)={n_groups} align_mismatches={mism}")

    iV = [VA_COLS.index(c) for c in V_COLS]
    iA = [VA_COLS.index(c) for c in A_ONLY_COLS]
    mats_raw = {"V": X_all[:, iV], "A": X_all[:, iA], "VA": X_all}

    # identical degeneracy/impute guard, both legs
    mats, names = {}, {}
    for k, cols in (("V", V_COLS), ("A", A_ONLY_COLS), ("VA", VA_COLS)):
        Mc, keep = clean_cols(mats_raw[k])
        mats[k] = Mc
        names[k] = [cols[j] for j in keep]

    folds = sorted_app_id_folds(g)
    print(f"outer folds: {N_OUTER} (sorted-app_id array_split, task's existing fold construction)")
    for i, (tr, te) in enumerate(folds):
        print(f"  fold {i}: train={len(tr)} test={len(te)} test_groups={len(np.unique(g[te]))}")

    res = {
        "cell": "patents_verdict (claim-fell)",
        "n": int(n),
        "pos_rate": float(y.mean()),
        "n_groups": int(n_groups),
        "group_column": "app_id",
        "matrix": str(CSV_PATH),
        "group_source": str(JL_SLIM),
        "align_mismatches": int(mism),
        "a_bank": "4 Gemma-judged per-element disclosure aggregates (a_n_disclose, "
                  "a_any_disclose, a_frac_disclose, a_max_disclose_overlap); "
                  "NOT the 154/198-rubric A-banks used on other VAT cells",
        "n_features": {k: int(v.shape[1]) for k, v in mats.items()},
        "special_rule": "T EXISTS but Delta_beyond is NOT an articulation residual -- "
                         "see the 2026-08-07 round-0 audit (notes/2026-08-07__closure_patents.md): "
                         "dense .7965 is a sound AUC driven ~70-85% by the printed claim "
                         "ordinal number (+ reference-construction artifact); the A bank here "
                         "is ONE concept in 4 columns. Never quote +.171 as taste. "
                         "Ledger stops at Delta_interact until the RUNBOOK revival "
                         "prerequisites are met.",
        "linear": {},
        "nonlinear": {},
    }

    # ---- linear (existing aggregation) ----
    lin_oof = {}
    for k in ["V", "A", "VA"]:
        auc_v, oof = linear_grouped_auc(mats[k], y, folds)
        res["linear"][k] = auc_v
        lin_oof[k] = oof
        print(f"  linear  {k:2s}: {auc_v:.4f}")

    # ---- nonlinear, seeds 0,1,2 ----
    nl_seed_runs = {"V": {}, "A": {}, "VA": {}}
    for k in ["V", "A", "VA"]:
        for s in (0, 1, 2):
            print(f"  gbm {k} seed {s} ...")
            r = gbm_oof(mats[k], y, g, folds, seed=s, verbose=(s == 0))
            nl_seed_runs[k][s] = r
            print(f"  gbm     {k:2s} seed {s}: {r['auc']:.4f}  (train {r['train_auc_mean']:.4f})")

    for k in ["V", "A", "VA"]:
        aucs = [nl_seed_runs[k][s]["auc"] for s in (0, 1, 2)]
        res["nonlinear"][k] = {
            "seed_aucs": {str(s): nl_seed_runs[k][s]["auc"] for s in (0, 1, 2)},
            "mean_auc": float(np.mean(aucs)),
            "spread": float(max(aucs) - min(aucs)),
            "train_auc_mean_seed0": nl_seed_runs[k][0]["train_auc_mean"],
            "picks_seed0": nl_seed_runs[k][0]["picks"],
        }
        if k == "VA":
            oof_seed0 = nl_seed_runs["VA"][0]["oof"]
            oof_mean = np.mean([nl_seed_runs["VA"][s]["oof"] for s in (0, 1, 2)], axis=0)
            np.save(Path(out_path).with_name("patents_verdict_va_nl_oof_seed0.npy"), oof_seed0)
            np.save(Path(out_path).with_name("patents_verdict_va_nl_oof_mean3.npy"), oof_mean)

    L, N = res["linear"], res["nonlinear"]
    VA_nl_mean = N["VA"]["mean_auc"]
    ledger = {
        "V_lin": L["V"], "V_nl_mean": N["V"]["mean_auc"], "V_nl_spread": N["V"]["spread"],
        "A_lin": L["A"], "A_nl_mean": N["A"]["mean_auc"], "A_nl_spread": N["A"]["spread"],
        "VA_lin": L["VA"], "VA_nl_mean": VA_nl_mean, "VA_nl_spread": N["VA"]["spread"],
        "Delta_interact": VA_nl_mean - L["VA"],
        "V_interact": N["V"]["mean_auc"] - L["V"],
    }
    res["ledger"] = ledger
    res["overfit_gap"] = {k: N[k]["train_auc_mean_seed0"] - N[k]["seed_aucs"]["0"] for k in ["V", "A", "VA"]}

    # bootstrap CI on Delta_interact (nl = mean-of-3-seeds OOF probs).
    oof_nl_mean_va = np.mean([nl_seed_runs["VA"][s]["oof"] for s in (0, 1, 2)], axis=0)
    res["delta_interact_bootstrap_group_PRIMARY"] = bootstrap_delta_interact_group(
        lin_oof["VA"], oof_nl_mean_va, y, g)
    res["delta_interact_bootstrap_row_secondary"] = bootstrap_delta_interact(lin_oof["VA"], oof_nl_mean_va, y)

    # gate against published registry numbers
    res["gate"] = {
        "V": {"published": PUBLISHED["V"], "reproduced": L["V"], "abs_diff": abs(L["V"] - PUBLISHED["V"]),
              "pass": abs(L["V"] - PUBLISHED["V"]) <= GATE_TOL},
        "VA": {"published": PUBLISHED["VA"], "reproduced": L["VA"], "abs_diff": abs(L["VA"] - PUBLISHED["VA"]),
               "pass": abs(L["VA"] - PUBLISHED["VA"]) <= GATE_TOL},
        "n_check": {"published": 59937, "reproduced": int(n), "pass": 59937 == int(n)},
    }

    if args.shap:
        print("  shap ...")
        try:
            res["shap"] = shap_interactions(mats["VA"], y, names["VA"], seed=0)
        except Exception as e:  # pragma: no cover
            res["shap"] = {"error": repr(e)}
            print("  shap FAILED:", e)

    res["runtime_sec"] = time.time() - t0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, indent=2))
    print(json.dumps(res["ledger"], indent=2))
    print(json.dumps(res["gate"], indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
