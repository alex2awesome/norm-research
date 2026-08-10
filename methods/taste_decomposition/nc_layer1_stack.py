#!/usr/bin/env python3
"""Layer-1 "nonlinear stack" of the taste-residual decomposition, N&C cells.

Design: notes/2026-08-05__taste-decomposition-design.md (S0 ledger, S1 protocol).
Pilot precedent + protocol notes: notes/2026-08-05__layer1_peer_verdict_pilot.md
(methods/taste_decomposition/layer1_stack.py). This script is the same protocol
ported to notice-and-comment (N&C) regulatory comments, all three expert y's,
FROZEN GKF design ONLY (verbatim port of
datasets/notice-and-comment/v4/aggregate_nc_multiy.py -- plain GroupKFold(5) on
docket, StandardScaler+LogisticRegression(C=1), clean_cols degeneracy guard).
Never the y_audit/StratGKF design (datasets/notice-and-comment/v4/aggregate_vat_nc.py,
StratifiedGroupKFold + SimpleImputer + class_weight="balanced") -- that design is
estimator-sensitive on the agree-vs-disagree cell (A .612 there vs .524 here) and
mixing the two would confound Delta_interact with an estimator swap, not a
nonlinearity gain.

Three cells (matched shards = nc_scores_shard0..4.npz, pre-GEPA 198-rubric A-bank):
  * responded : matched-with-any-label(1) vs genuinely-unmatched(0);
                universe = matched (X_m) union unmatched (X_u)
  * outcome   : majority MADE(1) vs NONE(0) over a comment's matched response
                labels, ties dropped; universe = matched-labeled only
  * agree     : majority accepted/agree(1) vs disagree(0) response_type;
                universe = matched-labeled only

Grouping unit = docket (matches aggregate_nc_multiy.py exactly).

Runs, on the SAME rows / SAME cleaned matrices / SAME grouped OOF folds per cell:
  * linear gate   : StandardScaler + LogisticRegression(C=1) -> V_lin, A_lin, VA_lin
                    gated against datasets/notice-and-comment/v4/nc_multiy_results.json
                    full_pool numbers (+-.005)
  * nonlinear     : HistGradientBoostingClassifier, frozen grid
                    max_leaf_nodes in {15,31}, learning_rate .06, max_iter 400
                    + early stopping; grid picked by inner GroupKFold(3) inside
                    each train fold ONLY -> V_nl, A_nl, VA_nl, each seeds {0,1,2}
                    -> FREEZE CHANGE 1 (pilot): report mean + spread, not seed 0 alone
  * Delta_interact bootstrap CI: paired 2,000x bootstrap over OOF rows, nl = mean
    of the 3 seeds' OOF probabilities (probability-space average), mirroring the
    pilot's "paired bootstrap over OOF rows" method
  * overfit check : train-fold AUC vs OOF AUC at the selected grid point
  * SHAP          : top-10 interaction pairs (descriptive only), computed ONLY
    for the cell with the largest |Delta_interact| across the three (protocol
    item 4) -- driven by --shap-cell after all three cells have run once

CPU only. No new judging. Usage:
  python nc_layer1_stack.py --cell responded
  python nc_layer1_stack.py --cell outcome
  python nc_layer1_stack.py --cell agree
  python nc_layer1_stack.py --cell agree --shap   # SHAP only if this cell wins |Delta_interact|
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
NC = REPO / "datasets" / "notice-and-comment" / "v4"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

import sys  # noqa: E402
sys.path.insert(0, str(NC))
from aggregate_vat_nc import v_features, V_NAMES  # noqa: E402

MATCHED_SHARDS = sorted(glob.glob(str(NC / "nc_scores_shard*.npz")))
UNMATCHED_NPZ = NC / "nc_scores_unmatched.npz"
SAMPLE_JSONL = NC / "nc_vat_sample.jsonl"
UNMATCHED_JSONL = NC / "nc_unmatched_sample.jsonl"
LABELS_FULL = NC / "nc_vat_sample_labels_full.json"
PUBLISHED_JSON = NC / "nc_multiy_results.json"

AGREE = {"accepted", "agree"}
DISAGREE = {"disagree"}
MADE = {"MADE"}
NONE_ = {"NONE"}

# frozen grid (design section 1)
GRID = [
    {"max_leaf_nodes": 15, "learning_rate": 0.06, "max_iter": 400},
    {"max_leaf_nodes": 31, "learning_rate": 0.06, "max_iter": 400},
]
N_OUTER = 5
N_INNER = 3

# T (dense clean-eval AUC), registry notes/2026-07-27__vat-run-registry.md
# "DENSE CHAIN -- CLEAN-EVAL FINAL" table. agree is UNSTABLE: eval .566 vs
# test .639 diverge (n_eval 505, docket-skewed) -> report Delta_beyond against BOTH.
T_VALUES = {
    "responded": {"eval": 0.808, "test": 0.825},
    "outcome": {"eval": 0.622, "test": 0.623},
    "agree": {"eval": 0.566, "test": 0.639},
}

CELL_TO_JSON_KEY = {
    "responded": "responded-or-not",
    "outcome": "outcome-majority",
    "agree": "agree-vs-disagree",
}


# ---------------------------------------------------------------- data -----
def clean_cols(M):
    """EXACT copy of aggregate_nc_multiy.clean_cols: drop degenerate cols, median-impute NA."""
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


def load_shard_scores(paths):
    """doc_id -> A-score row (198,), doc_id -> docket, doc_id -> agency. Anchors dropped.
    EXACT copy of aggregate_nc_multiy.load_shard_scores."""
    X_by_id, docket_by_id, agency_by_id = {}, {}, {}
    a_names = None
    for p in paths:
        d = np.load(p, allow_pickle=True)
        a_names = [str(x) for x in d["a_names"]]
        for i, did in enumerate(d["doc_id"]):
            did = str(did)
            if str(d["agency"][i]) == "__ANCHOR":
                continue
            X_by_id[did] = d["X"][i]
            docket_by_id[did] = str(d["docket"][i])
            agency_by_id[did] = str(d["agency"][i])
    return X_by_id, docket_by_id, agency_by_id, a_names


def load_jsonl_texts(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["doc_id"]] = r.get("text", "")
    return out


def maj(vals, pos, neg):
    p = sum(v in pos for v in vals)
    n = sum(v in neg for v in vals)
    if p == n:
        return -1
    return int(p > n)


def load_label_ys(path):
    """doc_id -> {'outcome_majority': -1/0/1, 'agree_vs_disagree': -1/0/1}.
    EXACT copy of aggregate_nc_multiy.load_label_ys."""
    labs = json.load(open(path))
    out = {}
    for doc_id, ls in labs.items():
        oc = [l["outcome_collapsed"] for l in ls]
        rt = [l["response_type"] for l in ls]
        out[doc_id] = {
            "outcome_majority": maj(oc, MADE, NONE_),
            "agree_vs_disagree": maj(rt, AGREE, DISAGREE),
        }
    return out


def build_row(doc_id, y, X_by_id, docket_by_id, text_by_id):
    a = X_by_id[doc_id].astype(float)
    v = np.array([v_features(text_by_id.get(doc_id, ""))[n] for n in V_NAMES], dtype=float)
    return y, a, v, docket_by_id[doc_id]


class NCData:
    """Loads all N&C VAT universes once, verbatim per aggregate_nc_multiy.main()."""

    def __init__(self):
        for p in (SAMPLE_JSONL, UNMATCHED_JSONL, LABELS_FULL, UNMATCHED_NPZ):
            assert p.exists(), f"missing {p}"
        assert MATCHED_SHARDS, "missing nc_scores_shard*.npz"

        self.X_m, self.docket_m, self.agency_m, self.a_names = load_shard_scores(MATCHED_SHARDS)
        self.X_u, self.docket_u, self.agency_u, _ = load_shard_scores([str(UNMATCHED_NPZ)])
        self.text_m = load_jsonl_texts(SAMPLE_JSONL)
        self.text_u = load_jsonl_texts(UNMATCHED_JSONL)
        label_ys = load_label_ys(LABELS_FULL)

        overlap = set(self.X_m) & set(self.X_u)
        for did in overlap:
            del self.X_u[did]
            del self.docket_u[did]
            del self.agency_u[did]
        self.n_overlap_dropped = len(overlap)

        self.y_out_by_id = {did: label_ys[did]["outcome_majority"] for did in self.X_m if did in label_ys}
        self.y_agr_by_id = {did: label_ys[did]["agree_vs_disagree"] for did in self.X_m if did in label_ys}
        self.valid_out = {did for did, y in self.y_out_by_id.items() if y in (0, 1)}
        self.valid_agr = {did for did, y in self.y_agr_by_id.items() if y in (0, 1)}
        self.y_resp_by_id = {did: 1 for did in self.X_m}
        self.y_resp_by_id.update({did: 0 for did in self.X_u})

    def rows_for(self, cell: str):
        """Returns (list of (y, a_vec, v_vec, docket), list of doc ids) for the
        cell, matching aggregate_nc_multiy.py's full_pool construction exactly.

        DETERMINISM FIX (registry 2026-08-10 landmine, permanent): ids are
        ALWAYS produced by sorting a Python set/dict, never by iterating one
        directly. Raw set iteration order is randomized per-process (Python's
        string-hash randomization, PYTHONHASHSEED), which made the pre-fix
        nc_{responded,outcome,agree}_va_nl_oof_*.npy row order permanently
        unrecoverable -- five candidate reorderings all gated at ~.50 AUC vs
        the published .581-.609, i.e. the saved arrays were paired with the
        wrong row on every read. This mirrors the fix already carried by
        closure/{maps_batch1,peer_curation_ext,peer_revealed}/cells.py::_load_nc
        ("Row order is `sorted(...)` so it is reproducible across processes --
        NCData's valid_out / valid_agr are Python sets -- never iterate them
        directly."). The caller must save the returned ids alongside any
        per-row array -- never assume a saved OOF array's row order without it.
        """
        if cell == "outcome":
            ids = sorted(self.valid_out)
            return [build_row(d, self.y_out_by_id[d], self.X_m, self.docket_m, self.text_m) for d in ids], ids
        if cell == "agree":
            ids = sorted(self.valid_agr)
            return [build_row(d, self.y_agr_by_id[d], self.X_m, self.docket_m, self.text_m) for d in ids], ids
        if cell == "responded":
            ids_m = sorted(self.X_m)
            ids_u = sorted(self.X_u)
            rows = [build_row(d, self.y_resp_by_id[d], self.X_m, self.docket_m, self.text_m) for d in ids_m]
            rows += [build_row(d, self.y_resp_by_id[d], self.X_u, self.docket_u, self.text_u) for d in ids_u]
            return rows, ids_m + ids_u
        raise ValueError(cell)


def load_cell(cell: str, data: NCData):
    items, ids = data.rows_for(cell)
    y = np.array([it[0] for it in items])
    A = np.array([it[1] for it in items], dtype=float)
    V = np.array([it[2] for it in items], dtype=float)
    groups = np.array([it[3] for it in items])

    Ac, a_keep = clean_cols(A)
    Vc, v_keep = clean_cols(V)
    VA = np.column_stack([Vc, Ac]) if Vc.shape[1] and Ac.shape[1] else (Vc if Vc.shape[1] else Ac)
    names = {
        "V": [V_NAMES[j] for j in v_keep],
        "A": [data.a_names[j] for j in a_keep],
    }
    names["VA"] = names["V"] + names["A"]
    mats = {"V": Vc, "A": Ac, "VA": VA}
    return mats, names, y, groups, ids


# --------------------------------------------------------------- models ----
def outer_folds(n, groups):
    gkf = GroupKFold(n_splits=min(N_OUTER, len(np.unique(groups))))
    return list(gkf.split(np.zeros(n), groups=groups))


def linear_oof(Xf, y, groups, folds):
    if Xf.shape[1] == 0:
        return float("nan"), None
    oof = np.zeros(len(y))
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(Xf[tr], y[tr])
        oof[te] = clf.predict_proba(Xf[te])[:, 1]
    return float(roc_auc_score(y, oof)), oof


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
    """Nested: inner GroupKFold(3) grid pick inside each train fold only."""
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


def bootstrap_delta_interact(oof_lin, oof_nl_mean, y, n_boot=2000, seed=0):
    """Paired bootstrap over OOF ROWS (pilot's original method,
    notes/2026-08-05__layer1_peer_verdict_pilot.md S3: "paired 2,000x bootstrap
    over the ... OOF rows"). nl = mean of the 3 seeds' OOF probabilities.

    Reported as a SECONDARY diagnostic only -- rows are not exchangeable here
    (docket-grouped OOF folds), so this treats within-docket correlation as if
    absent and understates the true CI width. The primary CI is the
    docket-level cluster bootstrap below."""
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


def bootstrap_delta_interact_docket(oof_lin, oof_nl_mean, y, groups, n_boot=2000, seed=0):
    """PRIMARY bootstrap CI on Delta_interact: cluster (docket-level) resample.
    Groups (dockets) are the grouping unit of the GroupKFold OOF folds -- rows
    within a docket are not independent (same rulemaking, correlated V/A
    scores), so a row-level bootstrap understates the CI width. Resample
    dockets with replacement (same count as observed), pool all rows belonging
    to each drawn docket (duplicated when a docket is drawn more than once),
    recompute pooled AUC for lin and nl-mean on that resampled row set."""
    rng = np.random.default_rng(seed)
    uniq_groups = np.unique(groups)
    n_groups = len(uniq_groups)
    # doc-index lookup per group, built once
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
def shap_interactions(Xf, y, names, seed=0, n_sub=300, top_k=15):
    """Top-10 SHAP interaction pairs (descriptive). Screened per pilot method:
    fit frozen-grid model on all rows, rank by mean|SHAP|, refit top_k, exact
    TreeSHAP interactions on a subsample."""
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
    top = np.argsort(-imp)[:top_k]
    top_names = [names[j] for j in top]

    m2 = _fit_gbm(GRID[1], seed)
    m2.fit(Xf[:, top], y)
    ex2 = shap.TreeExplainer(m2)
    iv = ex2.shap_interaction_values(Xf[sub][:, top])
    if isinstance(iv, list):
        iv = iv[-1]
    if iv.ndim == 4:
        iv = iv[:, :, :, -1]
    M = np.abs(iv).mean(0)
    pairs = []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            pairs.append((top_names[i], top_names[j], float(M[i, j] + M[j, i])))
    pairs.sort(key=lambda t: -t[2])
    diag = [(top_names[i], float(M[i, i])) for i in range(len(top))]
    off_frac = float((M.sum() - np.trace(M)) / M.sum())
    return {
        "method": "TreeSHAP exact interaction values on top-15-feature refit (screened by mean|SHAP| on full model)",
        "top_features": [{"name": n, "mean_abs_shap": float(imp[j])} for n, j in zip(top_names, top)],
        "top_pairs": [{"a": a, "b": b, "mean_abs_interaction": v} for a, b, v in pairs[:10]],
        "main_effects": [{"name": n, "mean_abs_main": v} for n, v in diag],
        "offdiagonal_mass_fraction": off_frac,
        "n_subsample": int(len(sub)),
    }


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=["responded", "outcome", "agree"])
    ap.add_argument("--shap", action="store_true",
                     help="also compute SHAP interactions for this cell (only for the "
                          "cell with the largest |Delta_interact| per protocol item 4)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or str(RESULTS_DIR / f"nc_{args.cell}_layer1.json")

    t0 = time.time()
    data = NCData()
    mats, names, y, groups, ids = load_cell(args.cell, data)
    folds = outer_folds(len(y), groups)
    n_groups = len(np.unique(groups))
    print(f"cell={args.cell} n={len(y)} pos={y.mean():.4f} groups={n_groups} "
          f"V={mats['V'].shape[1]}c A={mats['A'].shape[1]}c VA={mats['VA'].shape[1]}c "
          f"(matched_unmatched_overlap_dropped={data.n_overlap_dropped})")

    assert len(ids) == len(y), f"ids/y length mismatch: {len(ids)} vs {len(y)}"
    if args.cell != "responded":
        # outcome/agree: ids = sorted(valid_out)/sorted(valid_agr) verbatim
        assert list(ids) == sorted(ids), (
            "ids are not sorted -- rows_for() determinism fix regressed")
    else:
        # responded: sorted(X_m) then sorted(X_u), concatenated (matched block
        # is not required to precede the unmatched block alphabetically)
        n_m = len(data.X_m)
        assert list(ids[:n_m]) == sorted(ids[:n_m]) and list(ids[n_m:]) == sorted(ids[n_m:]), (
            "ids blocks are not sorted -- rows_for() determinism fix regressed")

    res = {
        "cell": f"nc {args.cell}",
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "n_groups": int(n_groups),
        "group_column": "docket",
        "matrix": str(NC),
        "a_bank": "pre-GEPA, 198 rubrics, Gemma-4-31B (nc_scores_shard0..4.npz), FROZEN GKF design",
        "n_features": {k: int(v.shape[1]) for k, v in mats.items()},
        "T_dense": T_VALUES[args.cell],
        "row_order_v2": {
            "policy": "sorted(ids) -- see NCData.rows_for docstring "
                      "(registry 2026-08-10 landmine fix)",
            "ids_npy": f"nc_{args.cell}_oof_ids_v2.npy",
            "n_ids": int(len(ids)),
            "ids_first3": [str(x) for x in ids[:3]],
            "ids_last3": [str(x) for x in ids[-3:]],
        },
        "linear": {},
        "nonlinear": {},
    }

    lin_oof = {}
    for k in ["V", "A", "VA"]:
        auc, oof = linear_oof(mats[k], y, groups, folds)
        res["linear"][k] = auc
        lin_oof[k] = oof
        print(f"  linear  {k:2s}: {auc:.4f}")

    # nonlinear, seeds 0,1,2 for V, A, VA (task protocol item 2: "VA_nl and V_nl
    # per frozen spec ... seeds 0,1,2"; A_nl run the same way for consistency)
    nl_seed_runs = {"V": {}, "A": {}, "VA": {}}
    for k in ["V", "A", "VA"]:
        for s in (0, 1, 2):
            print(f"  gbm {k} seed {s} ...")
            r = gbm_oof(mats[k], y, groups, folds, seed=s, verbose=(s == 0))
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
            # v2 regeneration (registry 2026-08-10 landmine fix): row order is
            # now sorted(ids) (see NCData.rows_for), deterministic across
            # processes, and every saved OOF array ships its aligned ids
            # vector alongside. The ORIGINAL nc_{cell}_va_nl_oof_{seed0,mean3}.npy
            # files are intentionally left untouched -- never overwrite (see
            # results/nc_{cell}_va_nl_oof_DEPRECATED_ORDER_UNRECOVERABLE.md).
            np.save(Path(out_path).with_name(f"nc_{args.cell}_oof_ids_v2.npy"),
                    np.array(ids, dtype=object))
            for s in (0, 1, 2):
                np.save(Path(out_path).with_name(f"nc_{args.cell}_va_nl_oof_seed{s}_v2.npy"),
                        nl_seed_runs["VA"][s]["oof"])
            np.save(Path(out_path).with_name(f"nc_{args.cell}_va_nl_oof_mean3_v2.npy"), oof_mean)

    L, N = res["linear"], res["nonlinear"]
    VA_nl_mean = N["VA"]["mean_auc"]
    T = T_VALUES[args.cell]
    ledger = {
        "V_lin": L["V"], "V_nl_mean": N["V"]["mean_auc"], "V_nl_spread": N["V"]["spread"],
        "A_lin": L["A"], "A_nl_mean": N["A"]["mean_auc"], "A_nl_spread": N["A"]["spread"],
        "VA_lin": L["VA"], "VA_nl_mean": VA_nl_mean, "VA_nl_spread": N["VA"]["spread"],
        "Delta_interact": VA_nl_mean - L["VA"],
        "V_interact": N["V"]["mean_auc"] - L["V"],
    }
    if args.cell == "agree":
        ledger["T_eval"] = T["eval"]
        ledger["T_test"] = T["test"]
        ledger["Delta_total_eval"] = T["eval"] - L["VA"]
        ledger["Delta_total_test"] = T["test"] - L["VA"]
        ledger["Delta_beyond_eval"] = T["eval"] - VA_nl_mean
        ledger["Delta_beyond_test"] = T["test"] - VA_nl_mean
    else:
        Tval = T["eval"]
        ledger["T"] = Tval
        ledger["Delta_total"] = Tval - L["VA"]
        ledger["Delta_beyond"] = Tval - VA_nl_mean
    res["ledger"] = ledger
    res["overfit_gap"] = {k: N[k]["train_auc_mean_seed0"] - N[k]["seed_aucs"]["0"] for k in ["V", "A", "VA"]}

    # bootstrap CI on Delta_interact (nl = mean-of-3-seeds OOF probs).
    # PRIMARY = docket-level cluster bootstrap (rows within a docket are not
    # independent -- same rulemaking -- so a row-level resample understates
    # the CI); row-level kept as a secondary diagnostic per the pilot's method.
    oof_nl_mean_va = np.mean([nl_seed_runs["VA"][s]["oof"] for s in (0, 1, 2)], axis=0)
    res["delta_interact_bootstrap_docket_PRIMARY"] = bootstrap_delta_interact_docket(
        lin_oof["VA"], oof_nl_mean_va, y, groups)
    res["delta_interact_bootstrap_row_secondary"] = bootstrap_delta_interact(lin_oof["VA"], oof_nl_mean_va, y)

    # gate against published full_pool numbers
    published_all = json.loads(PUBLISHED_JSON.read_text())["full_pool"]
    published = published_all[CELL_TO_JSON_KEY[args.cell]]
    res["gate"] = {
        k: {"published": published[k], "reproduced": L[k], "abs_diff": abs(L[k] - published[k]),
            "pass": abs(L[k] - published[k]) <= 0.005}
        for k in ("V", "A", "VA")
    }
    res["gate"]["n_check"] = {"published": published["n"], "reproduced": int(len(y)),
                               "pass": published["n"] == int(len(y))}

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
