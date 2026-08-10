#!/usr/bin/env python3
"""Layer-1 "nonlinear stack" on the Gemma-4-31B-scored CW + humor cells.

Design: notes/2026-08-05__taste-decomposition-design.md (S0 ledger, S1 protocol).
Pilot / protocol notes carried forward: notes/2026-08-05__layer1_peer_verdict_pilot.md
(7 numbered protocol notes), notably:
  (2) degeneracy/impute guard must be IDENTICAL for linear and nonlinear models,
  (3) VA_nl / V_nl := mean over seeds {0,1,2}, spread reported (FREEZE CHANGE 1),
  (6) SHAP is a descriptive screen only, the OOF ledger adjudicates.

Five cells, two "families" of existing linear pipeline that must be mirrored
EXACTLY for the gate:

  family1 (va_gemma_banks readout, datasets/va_gemma_banks/readout_va_gemma.py):
    SimpleImputer(median, add_indicator=True) + StandardScaler +
    LogisticRegression(C=1, solver='liblinear', max_iter=2000, random_state=20260728),
    GroupKFold(5). Cells: cw_community, hashtagwars_verdict, style_inv_toptier.

  family2 (caption multi-y, datasets/humor/caption_multiy/aggregate_captions_multiy.py):
    clean_cols (median-impute + degeneracy drop) + StandardScaler +
    LogisticRegression(C=1, max_iter=2000), GroupKFold(5) group=contest.
    Cells: cap_crowd, cap_finalist.

Both existing modules are imported directly (not re-typed) so the gate calls the
literal published code. GBM (VA_nl / V_nl) reuses the frozen grid from
methods/taste_decomposition/layer1_stack.py (also imported directly, for the
SHAP screen too): HistGradientBoostingClassifier, max_leaf_nodes in {15,31},
learning_rate .06, max_iter 400 + early stopping, grid picked by inner
GroupKFold(3) inside each outer train fold only, same outer folds as linear.

CPU only. No GPU. No new judging. Usage:
  python layer1_gemma_cells.py --cell cw_community
  python layer1_gemma_cells.py --all
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("NR_REPO", str(REPO))
os.environ.setdefault("VA_OUT", str(REPO / "outputs" / "va_gemma_banks"))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_module(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


rvg = load_module(REPO / "datasets/va_gemma_banks/readout_va_gemma.py", "rvg_taste_decomp")
capagg = load_module(REPO / "datasets/humor/caption_multiy/aggregate_captions_multiy.py", "capagg_taste_decomp")
pilot = load_module(REPO / "methods/taste_decomposition/layer1_stack.py", "layer1_pilot")

# frozen grid (design section 1), identical to the pilot
GRID = [
    {"max_leaf_nodes": 15, "learning_rate": 0.06, "max_iter": 400},
    {"max_leaf_nodes": 31, "learning_rate": 0.06, "max_iter": 400},
]
N_INNER = 3
GBM_SEEDS = (0, 1, 2)
LINEAR_SEED_FAMILY1 = rvg.SEED  # 20260728, matches published readout


# ---------------------------------------------------------------- folds ----
def outer_folds(n, groups, n_splits=5):
    ns = min(n_splits, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=ns)
    return list(gkf.split(np.zeros(n), groups=groups))


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


def gbm_oof_raw(Xf, y, groups, folds, seed):
    """GBM OOF on an ALREADY-preprocessed matrix identical for linear + GBM
    (family2: clean_cols matrix computed once, shared)."""
    oof = np.zeros(len(y))
    picks, train_aucs = [], []
    for tr, te in folds:
        gtr = groups[tr]
        inner = list(GroupKFold(n_splits=min(N_INNER, len(np.unique(gtr))))
                     .split(np.zeros(len(tr)), groups=gtr))
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
        m = _fit_gbm(GRID[best], seed)
        m.fit(Xf[tr], y[tr])
        oof[te] = m.predict_proba(Xf[te])[:, 1]
        train_aucs.append(float(roc_auc_score(y[tr], m.predict_proba(Xf[tr])[:, 1])))
    return {"auc": float(roc_auc_score(y, oof)), "picks": picks,
            "train_auc_mean": float(np.mean(train_aucs)), "oof": oof}


def gbm_oof_family1(Xraw, y, groups, folds, seed):
    """GBM OOF where the linear pipeline's preprocessing (SimpleImputer median +
    add_indicator) is fit PER TRAIN FOLD (no leakage) and the SAME per-fold
    imputed+indicator matrix is fed to the GBM -- 'same matrix' invariant,
    mirroring the linear pipeline's own leak-free fold structure."""
    oof = np.zeros(len(y))
    picks, train_aucs = [], []
    for tr, te in folds:
        imp = SimpleImputer(strategy="median", add_indicator=True)
        Xtr = imp.fit_transform(Xraw[tr])
        Xte = imp.transform(Xraw[te])
        gtr = groups[tr]
        inner = list(GroupKFold(n_splits=min(N_INNER, len(np.unique(gtr))))
                     .split(np.zeros(len(tr)), groups=gtr))
        scores = []
        for params in GRID:
            aucs = []
            for itr, ite in inner:
                m = _fit_gbm(params, seed)
                m.fit(Xtr[itr], y[tr][itr])
                aucs.append(roc_auc_score(y[tr][ite], m.predict_proba(Xtr[ite])[:, 1]))
            scores.append(float(np.mean(aucs)))
        best = int(np.argmax(scores))
        picks.append(GRID[best]["max_leaf_nodes"])
        m = _fit_gbm(GRID[best], seed)
        m.fit(Xtr, y[tr])
        oof[te] = m.predict_proba(Xte)[:, 1]
        train_aucs.append(float(roc_auc_score(y[tr], m.predict_proba(Xtr)[:, 1])))
    return {"auc": float(roc_auc_score(y, oof)), "picks": picks,
            "train_auc_mean": float(np.mean(train_aucs)), "oof": oof}


# --------------------------------------------------------- linear (gate) ---
def linear_oof_family1(Xraw, y, groups, folds):
    """Re-derivation of rvg.oof_auc's exact pipeline, but returning the OOF
    prediction array (rvg.oof_auc only returns the scalar + per-fold audit)."""
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        pipe = make_pipeline_family1()
        pipe.fit(Xraw[tr], y[tr])
        oof[te] = pipe.predict_proba(Xraw[te])[:, 1]
    ok = np.isfinite(oof)
    return float(roc_auc_score(y[ok], oof[ok])), oof


def make_pipeline_family1():
    from sklearn.pipeline import make_pipeline
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(C=1.0, solver="liblinear", max_iter=2000,
                            random_state=LINEAR_SEED_FAMILY1))


def linear_oof_family2(Xf, y, groups, folds):
    from sklearn.pipeline import make_pipeline
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        clf.fit(Xf[tr], y[tr])
        oof[te] = clf.predict_proba(Xf[te])[:, 1]
    ok = np.isfinite(oof)
    return float(roc_auc_score(y[ok], oof[ok])), oof


# --------------------------------------------------------------- bootstrap -
def bootstrap_delta_interact(y, lin_pred, nl_pred, n_boot=2000, seed=12345):
    point = float(roc_auc_score(y, nl_pred) - roc_auc_score(y, lin_pred))
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    tries = 0
    while len(deltas) < n_boot and tries < n_boot * 4:
        tries += 1
        idx = rng.integers(0, n, size=n)
        ys = y[idx]
        if ys.min() == ys.max():
            continue
        deltas.append(float(roc_auc_score(ys, nl_pred[idx]) - roc_auc_score(ys, lin_pred[idx])))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"estimate": point, "ci95": [float(lo), float(hi)],
            "p_gt_0": float((deltas > 0).mean()), "n_boot_used": int(len(deltas)),
            "note": "row-level paired bootstrap over OOF rows (linear vs GBM seed 0), "
                    "NOT group-level -- for coarse-group cells (hashtagwars 40 groups, "
                    "style_inv 316 weeks, captions ~227 contests) this likely understates "
                    "true CI width; CW is near-singleton-grouped so row-level is close to exact."}


# ------------------------------------------------------------ cell loaders -
def load_cw_community():
    meta, A, V, groups, _ = rvg.load_bank("creative_writing")
    y = np.array(meta["ys"]["score_ge10"])
    names = {"V": list(meta["v_names"]), "A": list(meta["a_names"])}
    names["VA"] = names["V"] + names["A"]
    mats = {"V": V, "A": A, "VA": np.column_stack([V, A])}
    published = {"V": 0.6039379189714367, "A": 0.6053040296263997, "VA": 0.6265657518258979}
    return dict(family="family1", mats=mats, names=names, y=y, groups=groups,
                group_column="prompt_id", published=published,
                matrix="outputs/va_gemma_banks/creative_writing_shard*.npz")


def load_hashtagwars_verdict():
    meta, A, V, groups, _ = rvg.load_bank("hashtagwars")
    y = np.array(meta["ys"]["top10_or_winner"])
    names = {"V": list(meta["v_names"]), "A": list(meta["a_names"])}
    names["VA"] = names["V"] + names["A"]
    mats = {"V": V, "A": A, "VA": np.column_stack([V, A])}
    published = {"V": 0.5592008584351312, "A": 0.6349579560091446, "VA": 0.6477522951764967}
    return dict(family="family1", mats=mats, names=names, y=y, groups=groups,
                group_column="hashtag contest", published=published,
                matrix="outputs/va_gemma_banks/hashtagwars_shard*.npz")


def load_style_inv_toptier():
    meta, A, V, groups, _ = rvg.load_bank("style_invitational")
    y = np.array(meta["ys"]["top_tier"])
    names = {"V": list(meta["v_names"]), "A": list(meta["a_names"])}
    names["VA"] = names["V"] + names["A"]
    mats = {"V": V, "A": A, "VA": np.column_stack([V, A])}
    published = {"V": 0.6227366863149641, "A": 0.608951549099352, "VA": 0.6161209906011281}
    return dict(family="family1", mats=mats, names=names, y=y, groups=groups,
                group_column="week_id", published=published,
                matrix="outputs/va_gemma_banks/style_invitational_shard*.npz")


_CAPTION_CACHE = {}


def _caption_pools():
    if "loaded" not in _CAPTION_CACHE:
        X_by_id, V_by_id, contest_by_id, role_by_id, a_names, v_names, anchors = capagg.load_scores()
        meta = capagg.load_pool()
        y_fin, y_crowd = {}, {}
        by_contest_crowd = defaultdict(list)
        for did in X_by_id:
            m = meta.get(did)
            if m is None:
                continue
            role = role_by_id[did]
            if role in ("finalist", "neg_hard", "neg_random"):
                y_fin[did] = 1 if role == "finalist" else 0
            if m.get("crowd_mean") is not None and (m.get("crowd_votes") or 0) >= capagg.MIN_VOTES:
                by_contest_crowd[m["contest"]].append((did, m["crowd_mean"]))
        for c, items in by_contest_crowd.items():
            if len(items) < 6:
                continue
            med = float(np.median([v for _, v in items]))
            for did, v in items:
                if v == med:
                    continue
                y_crowd[did] = int(v > med)
        hardneg_ids = {d for d in y_fin if role_by_id[d] in ("finalist", "neg_hard")}
        crowd_ids = set(y_crowd)
        _CAPTION_CACHE.update(dict(
            loaded=True, X_by_id=X_by_id, V_by_id=V_by_id, contest_by_id=contest_by_id,
            role_by_id=role_by_id, a_names=a_names, v_names=v_names,
            y_fin=y_fin, y_crowd=y_crowd, hardneg_ids=hardneg_ids, crowd_ids=crowd_ids))
    return _CAPTION_CACHE


def _caption_cell(id_set_key, y_key):
    c = _caption_pools()
    ids = sorted(d for d in c[id_set_key] if d in c["X_by_id"])  # deterministic order
    y_by_id = c[y_key]
    y = np.array([y_by_id[d] for d in ids])
    A = np.array([c["X_by_id"][d] for d in ids], dtype=float)
    V = np.array([c["V_by_id"][d] for d in ids], dtype=float)
    groups = np.array([c["contest_by_id"][d] for d in ids])
    Ac, a_keep = capagg.clean_cols(A)
    Vc, v_keep = capagg.clean_cols(V)
    VA = np.column_stack([Vc, Ac]) if Vc.shape[1] and Ac.shape[1] else (Vc if Vc.shape[1] else Ac)
    names = {"V": [c["v_names"][j] for j in v_keep], "A": [c["a_names"][j] for j in a_keep]}
    names["VA"] = names["V"] + names["A"]
    mats = {"V": Vc, "A": Ac, "VA": VA}
    return mats, names, y, groups


def load_cap_crowd():
    mats, names, y, groups = _caption_cell("crowd_ids", "y_crowd")
    published = {"V": 0.5273704876828627, "A": 0.64777156830621, "VA": 0.6485275259604455}
    return dict(family="family2", mats=mats, names=names, y=y, groups=groups,
                group_column="contest", published=published,
                matrix="datasets/humor/caption_multiy/cap_scores_shard*.npz "
                       "(y=crowd-C median split, votes>=100)")


def load_cap_finalist():
    mats, names, y, groups = _caption_cell("hardneg_ids", "y_fin")
    published = {"V": 0.6246951061037257, "A": 0.6298987693787117, "VA": 0.6507764479617428}
    return dict(family="family2", mats=mats, names=names, y=y, groups=groups,
                group_column="contest", published=published,
                matrix="datasets/humor/caption_multiy/cap_scores_shard*.npz "
                       "(y=finalist-B hardneg pool: finalist vs neg_hard)")


CELLS = {
    "cw_community": {"loader": load_cw_community, "T": 0.7801,
                      "title": "CW community (writingprompts, score>=10)"},
    "hashtagwars_verdict": {"loader": load_hashtagwars_verdict, "T": None,
                              "title": "HashtagWars humor verdict (full)"},
    "style_inv_toptier": {"loader": load_style_inv_toptier, "T": None,
                            "title": "Style Invitational top-tier"},
    "cap_crowd": {"loader": load_cap_crowd, "T": 0.5631,
                   "title": "humor caption crowd-C (median split, votes>=100)"},
    "cap_finalist": {"loader": load_cap_finalist, "T": 0.6252,
                       "title": "humor caption finalist-B (hardneg pool)"},
}

GATE_TOL = 0.006


# ------------------------------------------------------------------- main --
def run_cell(slug, verbose=True):
    t0 = time.time()
    cfg = CELLS[slug]
    d = cfg["loader"]()
    mats, names, y, groups, family = d["mats"], d["names"], d["y"], d["groups"], d["family"]
    n = len(y)
    folds = outer_folds(n, groups, n_splits=5)
    if verbose:
        print(f"=== {slug} ({cfg['title']}) === n={n} pos={y.mean():.4f} "
              f"groups={len(np.unique(groups))} family={family} "
              f"V={mats['V'].shape[1]}c A={mats['A'].shape[1]}c VA={mats['VA'].shape[1]}c")

    res = {
        "cell": slug, "title": cfg["title"], "n": int(n), "pos_rate": float(y.mean()),
        "n_groups": int(len(np.unique(groups))), "group_column": d["group_column"],
        "matrix": d["matrix"], "n_features": {k: int(v.shape[1]) for k, v in mats.items()},
        "T_dense": cfg["T"], "sklearn_version": sklearn.__version__,
        "linear": {}, "linear_oof": {}, "nonlinear": {},
    }

    # ---- linear gate (reproduces published V / A / V+A) --------------------
    linfn = linear_oof_family1 if family == "family1" else linear_oof_family2
    for k in ["V", "A", "VA"]:
        auc, oof = linfn(mats[k], y, groups, folds)
        res["linear"][k] = auc
        res["linear_oof"][k] = oof
        if verbose:
            print(f"  linear  {k:2s}: {auc:.4f}  (published {d['published'][k]:.4f})")

    res["gate"] = {
        k: {"published": d["published"][k], "reproduced": res["linear"][k],
            "abs_diff": abs(res["linear"][k] - d["published"][k]),
            "pass": abs(res["linear"][k] - d["published"][k]) <= GATE_TOL}
        for k in d["published"]
    }
    gate_pass = all(g["pass"] for g in res["gate"].values())
    res["gate_pass"] = gate_pass
    if verbose:
        print(f"  gate: {'PASS' if gate_pass else 'FAIL'} " +
              json.dumps({k: round(g["abs_diff"], 5) for k, g in res["gate"].items()}))

    if not gate_pass:
        res["runtime_sec"] = time.time() - t0
        (RESULTS_DIR / f"{slug}_layer1.json").write_text(
            json.dumps({k: v for k, v in res.items() if k != "linear_oof"}, indent=2, default=str))
        print(f"  GATE FAILED for {slug} -- STOPPING this cell, no nonlinear stack run.")
        return res

    # ---- nonlinear: V_nl and VA_nl, seeds 0/1/2 -----------------------------
    gbmfn = gbm_oof_family1 if family == "family1" else gbm_oof_raw
    nl_oof_by_key_seed = {}
    for k in ["V", "VA"]:
        res["nonlinear"][k] = {}
        for seed in GBM_SEEDS:
            if verbose:
                print(f"  gbm {k} seed {seed} ...")
            r = gbmfn(mats[k], y, groups, folds, seed)
            oof = r.pop("oof")
            nl_oof_by_key_seed[(k, seed)] = oof
            res["nonlinear"][k][str(seed)] = r
            if verbose:
                print(f"    -> auc {r['auc']:.4f}  train {r['train_auc_mean']:.4f}  picks {r['picks']}")

    va_seed_aucs = [res["nonlinear"]["VA"][str(s)]["auc"] for s in GBM_SEEDS]
    v_seed_aucs = [res["nonlinear"]["V"][str(s)]["auc"] for s in GBM_SEEDS]
    res["seed_spread"] = {"VA": {str(s): a for s, a in zip(GBM_SEEDS, va_seed_aucs)},
                            "V": {str(s): a for s, a in zip(GBM_SEEDS, v_seed_aucs)}}
    res["seed_spread_range"] = {"VA": float(max(va_seed_aucs) - min(va_seed_aucs)),
                                  "V": float(max(v_seed_aucs) - min(v_seed_aucs))}

    VA_nl_mean = float(np.mean(va_seed_aucs))
    V_nl_mean = float(np.mean(v_seed_aucs))
    VA_lin, V_lin, A_lin = res["linear"]["VA"], res["linear"]["V"], res["linear"]["A"]

    ledger = {
        "V_lin": V_lin, "V_nl_mean": V_nl_mean, "V_nl_seeds": v_seed_aucs,
        "A_lin": A_lin,
        "VA_lin": VA_lin, "VA_nl_mean": VA_nl_mean, "VA_nl_seeds": va_seed_aucs,
        "T": cfg["T"],
        "Delta_interact": VA_nl_mean - VA_lin,
        "V_interact": V_nl_mean - V_lin,
    }
    if cfg["T"] is not None:
        ledger["Delta_total"] = cfg["T"] - VA_lin
        ledger["Delta_beyond"] = cfg["T"] - VA_nl_mean
    res["ledger"] = ledger
    res["overfit_gap"] = {
        k: {str(s): res["nonlinear"][k][str(s)]["train_auc_mean"] - res["nonlinear"][k][str(s)]["auc"]
            for s in GBM_SEEDS}
        for k in ["V", "VA"]
    }

    # ---- bootstrap CI on Delta_interact (seed 0 pairing) --------------------
    res["bootstrap_delta_interact"] = bootstrap_delta_interact(
        y, res["linear_oof"]["VA"], nl_oof_by_key_seed[("VA", 0)])
    if verbose:
        b = res["bootstrap_delta_interact"]
        print(f"  Delta_interact (seed0 point) = {b['estimate']:.4f}  "
              f"95% CI [{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}]  P(>0)={b['p_gt_0']:.2f}")

    res["protocol_notes"] = [
        f"Family = {family}. Gate reproduces {d['group_column']}-grouped published V/A/V+A "
        f"within tolerance {GATE_TOL}.",
        "VA_nl / V_nl reported as mean over seeds {0,1,2} per FREEZE CHANGE 1 "
        "(notes/2026-08-05__taste-decomposition-design.md S6); per-seed spread reported "
        "alongside as the load-bearing caveat on any Delta_interact smaller than the spread.",
        "Delta_interact point estimate and its bootstrap CI use the seed-0 VA_nl OOF array "
        "(not the seed-mean), consistent with the pilot; compare against seed_spread_range "
        "before reading Delta_interact's sign as reliable.",
        "Bootstrap is row-level, not group-level -- see bootstrap_delta_interact.note.",
    ]

    # cache OOF arrays for downstream SHAP-pair-selection step
    res["_nl_oof_seed0_VA"] = nl_oof_by_key_seed[("VA", 0)]
    res["_names"] = names
    res["_mats_VA"] = mats["VA"]
    res["_y"] = y

    res["runtime_sec"] = time.time() - t0
    if verbose:
        print(f"  [{slug}] done in {res['runtime_sec']:.1f}s")
    return res


def save_result(res, with_shap=None):
    slug = res["cell"]
    out = {k: v for k, v in res.items() if not k.startswith("_") and k != "linear_oof"}
    if with_shap is not None:
        out["shap"] = with_shap
    (RESULTS_DIR / f"{slug}_layer1.json").write_text(json.dumps(out, indent=2, default=str))
    print("wrote", RESULTS_DIR / f"{slug}_layer1.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=None, choices=list(CELLS.keys()))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--shap-top", type=int, default=2,
                     help="run SHAP interaction screen on the N cells with largest |Delta_interact|")
    args = ap.parse_args()

    slugs = list(CELLS.keys()) if args.all or args.cell is None else [args.cell]

    results = {}
    for slug in slugs:
        r = run_cell(slug)
        results[slug] = r
        if not r.get("gate_pass", False):
            continue

    # rank by |Delta_interact| among cells that passed the gate and have a ledger
    ranked = sorted(
        (s for s in results if "ledger" in results[s]),
        key=lambda s: -abs(results[s]["ledger"]["Delta_interact"]))
    shap_cells = ranked[:args.shap_top]
    print("SHAP screen scheduled for:", shap_cells)

    for slug in slugs:
        r = results[slug]
        shap_out = None
        if slug in shap_cells and "ledger" in r:
            print(f"  shap for {slug} ...")
            try:
                shap_out = pilot.shap_interactions(r["_mats_VA"], r["_y"], r["_names"]["VA"], seed=0)
            except Exception as e:  # pragma: no cover
                shap_out = {"error": repr(e)}
                print("  shap FAILED:", e)
        save_result(r, with_shap=shap_out)

    print("\n=== summary ===")
    for slug in slugs:
        r = results[slug]
        if not r.get("gate_pass", False):
            print(f"{slug}: GATE FAILED")
            continue
        L = r["ledger"]
        db = f"{L['Delta_beyond']:+.4f}" if "Delta_beyond" in L else "n/a (no T)"
        print(f"{slug}: VA_lin={L['VA_lin']:.4f} VA_nl={L['VA_nl_mean']:.4f} "
              f"Delta_interact={L['Delta_interact']:+.4f} Delta_beyond={db}")


if __name__ == "__main__":
    main()
