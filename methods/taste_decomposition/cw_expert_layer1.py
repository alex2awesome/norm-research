#!/usr/bin/env python3
"""Full V / A / VA_lin / VA_nl / T Layer-1 ledger for the two CW EXPERT cells
rebuilt with the mature instrument: RoyalRoad market VERDICT and Wigleaf
editorial CURATION (notes/2026-08-08__cw_nullbank_reaudit.md).

FIRST-FIT cells: no earlier V+A stack of this construction exists for either,
so the linear leg IS the first fit and there is no external reproduction gate
(the press-verdict / scale-up-C precedent, design note S4b wave 3). The old
2026-07-05/06 craft-bank numbers (.505 RoyalRoad / .578 Wigleaf) came from a
different instrument -- k-medoid NON-GEPA bank, likely Llama-3.3-70B judge, no
anchor battery -- and are carried as `prior_instrument` CONTEXT ONLY, never
differenced as if same-instrument.

Protocol is the frozen Layer-1 one, machinery imported (never re-typed) from
layer1_gemma_cells.py / scaleupC_layer1.py:
  * linear      family1: SimpleImputer(median, add_indicator) + StandardScaler +
                LogisticRegression(C=1, liblinear, max_iter 2000, rs 20260728),
                GroupKFold(5) on the cell's grouping unit.
  * nonlinear   HistGradientBoosting, frozen grid {15,31} leaves, lr .06,
                max_iter 400 + early stopping; grid picked by inner GroupKFold(3)
                INSIDE each outer train fold only; per-fold imputation identical
                to the linear leg.
  * FREEZE CHANGE 1  VA_nl / V_nl := mean over seeds {0,1,2}, spread reported.
  * FREEZE CHANGE 2  T is SAME-ROWS: the dense arm trains on the identical
                frozen population/split, so its eval rows are a subset of the
                A/V-scored rows.
  * FREEZE CHANGE 3  Delta_interact CI = GROUP-level bootstrap.

Two things this file adds over the scale-up-C precedent, both required by the
rebuild charge:
  * OOF arrays are saved WITH their ids vector (npz: oof / ids / y / groups),
    not as bare positional .npy;
  * an ASSEMBLED-ORDER GATE: the headline AUC is recomputed from the
    id-keyed OOF after an independent re-assembly of the sharded matrices, and
    must agree to < 1e-9.

CPU only. No GPU, no new judging.

  python methods/taste_decomposition/cw_expert_layer1.py --cell cw_royalroad_verdict
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layer1_gemma_cells as L      # noqa: E402
import scaleupC_layer1 as SC        # noqa: E402  (load_*, dense_T, bootstraps)

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CWX_OUT = REPO / "outputs/va_gemma_banks_cw_expert"
ORDER_GATE_TOL = 1e-9


def load_cw_expert_bank(name):
    return SC.load_scaleupC_bank(name, out=CWX_OUT)


# ------------------------------------------------ enforced collapse gate ------
# RULING 2026-08-10: the collapse gate is ENFORCED inside clean_fit, not merely
# reported. Any criterion whose modal value covers > 98% of its finite entries
# carries no usable rank information and is DROPPED before fitting.
#
# Two properties that make this a gate rather than a cosmetic filter:
#   * the mask is computed on the TRAIN fold ONLY, inside each outer fold, so it
#     never leaks held-out information;
#   * the SAME mask is applied to the linear and the nonlinear leg (design note
#     protocol point 2: the degeneracy/impute guard must be IDENTICAL for both),
#     so Delta_interact stays a like-for-like contrast.
COLLAPSE_MODAL_MAX = 0.98


def collapse_mask(Xtr, modal_max=COLLAPSE_MODAL_MAX):
    keep = np.ones(Xtr.shape[1], dtype=bool)
    modal = np.ones(Xtr.shape[1], dtype=float)
    for j in range(Xtr.shape[1]):
        fin = Xtr[:, j][np.isfinite(Xtr[:, j])]
        if fin.size == 0:
            keep[j] = False
            continue
        _, cnts = np.unique(fin, return_counts=True)
        modal[j] = cnts.max() / fin.size
        if modal[j] > modal_max:
            keep[j] = False
    return keep, modal


def linear_oof_gated(Xraw, y, groups, folds):
    """family1 linear pipeline behind the enforced collapse gate."""
    oof = np.full(len(y), np.nan)
    dropped = []
    for tr, te in folds:
        keep, _ = collapse_mask(Xraw[tr])
        if keep.sum() == 0:
            keep[:] = True                      # never fit an empty matrix
        dropped.append(int((~keep).sum()))
        pipe = L.make_pipeline_family1()
        pipe.fit(Xraw[tr][:, keep], y[tr])
        oof[te] = pipe.predict_proba(Xraw[te][:, keep])[:, 1]
    ok = np.isfinite(oof)
    return float(roc_auc_score(y[ok], oof[ok])), oof, dropped


def gbm_oof_gated(Xraw, y, groups, folds, seed):
    """family1 GBM leg behind the IDENTICAL enforced collapse gate + the same
    per-train-fold SimpleImputer(median, add_indicator) the linear leg uses."""
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GroupKFold
    oof = np.zeros(len(y))
    picks, train_aucs, dropped = [], [], []
    for tr, te in folds:
        keep, _ = collapse_mask(Xraw[tr])
        if keep.sum() == 0:
            keep[:] = True
        dropped.append(int((~keep).sum()))
        imp = SimpleImputer(strategy="median", add_indicator=True)
        Xtr = imp.fit_transform(Xraw[tr][:, keep])
        Xte = imp.transform(Xraw[te][:, keep])
        gtr = groups[tr]
        inner = list(GroupKFold(n_splits=min(L.N_INNER, len(np.unique(gtr))))
                     .split(np.zeros(len(tr)), groups=gtr))
        scores = []
        for params in L.GRID:
            aucs = []
            for itr, ite in inner:
                m = L._fit_gbm(params, seed)
                m.fit(Xtr[itr], y[tr][itr])
                aucs.append(roc_auc_score(y[tr][ite], m.predict_proba(Xtr[ite])[:, 1]))
            scores.append(float(np.mean(aucs)))
        best = int(np.argmax(scores))
        picks.append(L.GRID[best]["max_leaf_nodes"])
        m = L._fit_gbm(L.GRID[best], seed)
        m.fit(Xtr, y[tr])
        oof[te] = m.predict_proba(Xte)[:, 1]
        train_aucs.append(float(roc_auc_score(y[tr], m.predict_proba(Xtr)[:, 1])))
    return {"auc": float(roc_auc_score(y, oof)), "picks": picks,
            "train_auc_mean": float(np.mean(train_aucs)), "oof": oof,
            "collapse_dropped_per_fold": dropped}


# ------------------------------------------------------------ cell registry --
def cell_cw_royalroad_verdict():
    meta, A, V, groups, shard, ids = load_cw_expert_bank("cw_royalroad_verdict")
    y = np.array(meta["ys"]["judgement"])
    dense = REPO / "datasets/creative-writing/royalroad_stubs/dense_standard"
    T, Tinfo = SC.dense_T(dense)
    sec = np.array(meta["meta"]["secondary_group_of_item"], dtype=object)
    return dict(
        title="RoyalRoad market VERDICT (opening chapter -> KU/Amazon pickup)",
        slug="cw_royalroad_verdict", A=A, V=V, y=y, groups=groups, ids=ids, meta=meta,
        shard_of=shard, group_column="fiction_id", T=T, T_info=Tinfo,
        secondary_groups=sec, secondary_group_column="topic_cluster",
        matrix="outputs/va_gemma_banks_cw_expert/cw_royalroad_verdict_shard*.npz",
        dense_dir=str(dense.relative_to(REPO)),
        class_weighting="not required (balanced 637/637)",
        prior_instrument=meta["meta"]["prior_instrument"])


def cell_cw_wigleaf_curation():
    meta, A, V, groups, shard, ids = load_cw_expert_bank("cw_wigleaf_curation")
    y = np.array(meta["ys"]["judgement"])
    dense = REPO / "datasets/creative-writing/wigleaf/dense_standard"
    T, Tinfo = SC.dense_T(dense)
    sec = np.array(meta["meta"]["secondary_group_of_item"], dtype=object)
    return dict(
        title="Wigleaf editorial CURATION (flash fiction -> Top-50 editor's cut)",
        slug="cw_wigleaf_curation", A=A, V=V, y=y, groups=groups, ids=ids, meta=meta,
        shard_of=shard, group_column="story id", T=T, T_info=Tinfo,
        secondary_groups=sec, secondary_group_column="magazine",
        matrix="outputs/va_gemma_banks_cw_expert/cw_wigleaf_curation_shard*.npz",
        dense_dir=str(dense.relative_to(REPO)),
        class_weighting="REQUIRED and applied: dense arm trained with "
                        "--class_weight_auto (404 positives / 1,164 negatives)",
        power_caveat="404 ABSOLUTE positives (train 313 / eval 43 / test 48). Same "
                     "order of magnitude as the mathlib false-null case (~360 minority "
                     "train rows) that motivated the pre-kill checklist: read every "
                     "number from this cell against that power limit, and treat "
                     "eval/test dense AUCs (43 and 48 positives) as wide.",
        prior_instrument=meta["meta"]["prior_instrument"])


CELLS = {"cw_royalroad_verdict": cell_cw_royalroad_verdict,
         "cw_wigleaf_curation": cell_cw_wigleaf_curation}


# ------------------------------------------------------- assembled-order gate -
def assembled_order_gate(slug, d, headline, oof_by_key):
    """Independently re-assemble the sharded matrices, re-key the saved OOF
    vectors by item id, and re-derive every headline AUC. Must match < 1e-9.

    This is the gate that a positional .npy cannot support: it proves the
    published numbers do not depend on shard concatenation order."""
    meta2, A2, V2, groups2, _, ids2 = load_cw_expert_bank(slug)
    checks = {"matrix_A_identical": bool(np.array_equal(np.nan_to_num(A2, nan=-9e9),
                                                        np.nan_to_num(d["A"], nan=-9e9))),
              "matrix_V_identical": bool(np.array_equal(np.nan_to_num(V2, nan=-9e9),
                                                        np.nan_to_num(d["V"], nan=-9e9))),
              "ids_identical": bool(np.array_equal(ids2, d["ids"])),
              "groups_identical": bool(np.array_equal(groups2, d["groups"]))}
    y2 = np.array(meta2["ys"]["judgement"])
    # shuffle the row order, look OOF up BY ID, and recompute
    rng = np.random.default_rng(20260808)
    perm = rng.permutation(len(ids2))
    recomputed, diffs = {}, {}
    for key, val in headline.items():
        vec = np.array([oof_by_key[key][i] for i in ids2[perm]])
        recomputed[key] = float(roc_auc_score(y2[perm], vec))
        diffs[key] = abs(recomputed[key] - val)
    checks["recomputed"] = recomputed
    checks["abs_diff"] = diffs
    checks["max_abs_diff"] = float(max(diffs.values()))
    checks["tolerance"] = ORDER_GATE_TOL
    checks["PASS"] = bool(all(checks[k] for k in
                              ("matrix_A_identical", "matrix_V_identical",
                               "ids_identical", "groups_identical"))
                          and checks["max_abs_diff"] < ORDER_GATE_TOL)
    checks["note"] = ("Headline AUCs recomputed from id-keyed OOF vectors after an "
                      "independent shard re-assembly and a random row permutation; "
                      "agreement < 1e-9 certifies the published numbers are "
                      "assembly-order invariant. FIRST-FIT cells have no external "
                      "published number to reproduce, so this self-consistency gate "
                      "replaces the usual reproduction gate.")
    print(f"  [order gate] PASS={checks['PASS']} max|diff|={checks['max_abs_diff']:.3e}")
    return checks


# ------------------------------------------------------------------- runner ---
def run_cell(slug):
    t0 = time.time()
    d = CELLS[slug]()
    A, V, y, groups, ids = d["A"], d["V"], d["y"], d["groups"], d["ids"]
    VA = np.column_stack([V, A])
    mats = {"V": V, "A": A, "VA": VA}
    n = len(y)
    folds = L.outer_folds(n, groups, n_splits=5)
    names = {"V": list(d["meta"]["v_names"]), "A": list(d["meta"]["a_names"])}
    names["VA"] = names["V"] + names["A"]
    print(f"=== {slug} === n={n} pos={y.mean():.4f} (abs {int(y.sum())}) "
          f"groups={len(np.unique(groups))} V={V.shape[1]}c A={A.shape[1]}c")

    res = {"cell": slug, "title": d["title"], "n": int(n), "pos_rate": float(y.mean()),
           "n_pos_absolute": int(y.sum()), "n_neg_absolute": int((1 - y).sum()),
           "n_groups": int(len(np.unique(groups))), "group_column": d["group_column"],
           "matrix": d["matrix"], "dense_dir": d["dense_dir"],
           "n_features": {k: int(v.shape[1]) for k, v in mats.items()},
           "T_dense": d["T"], "T_info": d["T_info"],
           "sklearn_version": sklearn.__version__, "first_fit": True,
           "class_weighting": d["class_weighting"],
           "prior_instrument": d["prior_instrument"],
           "linear": {}, "nonlinear": {}}
    if d.get("power_caveat"):
        res["power_caveat"] = d["power_caveat"]

    lin_oof, collapse_log = {}, {}
    for k in ["V", "A", "VA"]:
        auc, oof, dropped = linear_oof_gated(mats[k], y, groups, folds)
        res["linear"][k] = auc
        lin_oof[k] = oof
        collapse_log[k] = dropped
        print(f"  linear  {k:2s}: {auc:.4f}  (collapse-gate dropped/fold {dropped})")
    res["collapse_gate"] = {
        "modal_max": COLLAPSE_MODAL_MAX,
        "enforced_inside_clean_fit": True,
        "computed_on": "TRAIN fold only, inside each outer fold (no leakage)",
        "applied_identically_to": ["linear", "nonlinear"],
        "n_dropped_per_fold": collapse_log,
        "n_features_before": {k: int(v.shape[1]) for k, v in mats.items()},
        "ruling": "2026-08-10: collapse gate is ENFORCED, not merely reported"}

    nl_oof = {}
    for k in ["V", "VA"]:
        res["nonlinear"][k] = {}
        for seed in L.GBM_SEEDS:
            r = gbm_oof_gated(mats[k], y, groups, folds, seed)
            nl_oof[(k, seed)] = r.pop("oof")
            res["nonlinear"][k][str(seed)] = r
            print(f"  gbm {k:2s} seed {seed}: {r['auc']:.4f} "
                  f"(train {r['train_auc_mean']:.4f}, picks {r['picks']})")

    va_seeds = [res["nonlinear"]["VA"][str(s)]["auc"] for s in L.GBM_SEEDS]
    v_seeds = [res["nonlinear"]["V"][str(s)]["auc"] for s in L.GBM_SEEDS]
    res["seed_spread"] = {"VA": va_seeds, "V": v_seeds}
    res["seed_spread_range"] = {"VA": float(max(va_seeds) - min(va_seeds)),
                                "V": float(max(v_seeds) - min(v_seeds))}
    VA_nl, V_nl = float(np.mean(va_seeds)), float(np.mean(v_seeds))
    ledger = {"V_lin": res["linear"]["V"], "V_nl_mean": V_nl, "V_nl_seeds": v_seeds,
              "A_lin": res["linear"]["A"], "VA_lin": res["linear"]["VA"],
              "VA_nl_mean": VA_nl, "VA_nl_seeds": va_seeds, "T": d["T"],
              "Delta_interact": VA_nl - res["linear"]["VA"],
              "V_interact": V_nl - res["linear"]["V"]}
    if d["T"] is not None:
        ledger["Delta_total"] = d["T"] - res["linear"]["VA"]
        ledger["Delta_beyond"] = d["T"] - VA_nl
    res["ledger"] = ledger
    res["overfit_gap"] = {
        k: {str(s): res["nonlinear"][k][str(s)]["train_auc_mean"]
            - res["nonlinear"][k][str(s)]["auc"] for s in L.GBM_SEEDS}
        for k in ["V", "VA"]}

    res["group_bootstrap_delta_interact"] = SC.group_bootstrap_delta(
        y, groups, lin_oof["VA"], nl_oof[("VA", 0)])
    res["group_bootstrap_ci95"] = {
        "V_lin": SC.group_bootstrap_auc(y, groups, lin_oof["V"]),
        "A_lin": SC.group_bootstrap_auc(y, groups, lin_oof["A"]),
        "VA_lin": SC.group_bootstrap_auc(y, groups, lin_oof["VA"]),
        "VA_nl_seed0": SC.group_bootstrap_auc(y, groups, nl_oof[("VA", 0)]),
    }
    b = res["group_bootstrap_delta_interact"]
    print(f"  Delta_interact seed0 = {b['estimate']:+.4f} "
          f"95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] P(>0)={b['p_gt_0']:.2f}")

    # ---- OOF arrays WITH ids ------------------------------------------------
    va_nl_mean = np.mean([nl_oof[("VA", s)] for s in L.GBM_SEEDS], axis=0)
    oof_store = {"V_lin": lin_oof["V"], "A_lin": lin_oof["A"], "VA_lin": lin_oof["VA"],
                 "V_nl_seed0": nl_oof[("V", 0)], "VA_nl_seed0": nl_oof[("VA", 0)],
                 "VA_nl_seed1": nl_oof[("VA", 1)], "VA_nl_seed2": nl_oof[("VA", 2)],
                 "VA_nl_mean3": va_nl_mean}
    np.savez_compressed(RESULTS_DIR / f"{slug}_oof.npz",
                        ids=np.asarray(ids, dtype=object),
                        groups=np.asarray(groups, dtype=object), y=y,
                        secondary_groups=np.asarray(d["secondary_groups"], dtype=object),
                        **oof_store)
    # legacy positional twins, for tools that already glob *_va_nl_oof_*.npy
    np.save(RESULTS_DIR / f"{slug}_va_nl_oof_seed0.npy", nl_oof[("VA", 0)])
    np.save(RESULTS_DIR / f"{slug}_va_nl_oof_mean3.npy", va_nl_mean)
    res["oof_arrays"] = {
        "path": f"methods/taste_decomposition/results/{slug}_oof.npz",
        "keys": sorted(oof_store) + ["ids", "groups", "y", "secondary_groups"],
        "note": "every OOF vector is aligned to the `ids` vector in the same file."}

    # ---- assembled-order gate ----------------------------------------------
    headline = {"V_lin": res["linear"]["V"], "A_lin": res["linear"]["A"],
                "VA_lin": res["linear"]["VA"],
                "VA_nl_seed0": res["nonlinear"]["VA"]["0"]["auc"]}
    oof_by_key = {k: dict(zip(ids, oof_store[k])) for k in headline}
    res["assembled_order_gate"] = assembled_order_gate(slug, d, headline, oof_by_key)

    # ---- descriptive screens ------------------------------------------------
    sys.path.insert(0, str(REPO / "datasets/va_gemma_banks"))
    import readout_va_gemma as rvg
    res["univariate_A"] = rvg.univariate(A, y, names["A"])[:15]
    res["na_rate_overall"] = float(np.isnan(A).mean())
    res["collapsed_criteria"] = [r["criterion"] for r in rvg.univariate(A, y, names["A"])
                                 if r["near_constant"]]
    for fn, key in ((CWX_OUT / "anchor_battery.json", "anchor_battery"),
                    (CWX_OUT / "distribution_check.json", "judge_distribution_check")):
        res[key] = json.loads(fn.read_text()).get(slug) if fn.exists() else None
    res["anchor_reports"] = d["meta"].get("anchor_reports")

    bad = [r["shard"] for r in (d["meta"].get("anchor_reports") or []) if not r["valid"]]
    if bad and d.get("shard_of") is not None:
        sh = np.asarray(d["shard_of"])
        keep = np.flatnonzero(~np.isin(sh, bad))
        if len(keep) > 200 and len(np.unique(y[keep])) > 1:
            f3 = L.outer_folds(len(keep), groups[keep], n_splits=5)
            sens = {"dropped_shards": bad, "n": int(len(keep))}
            for k in ["V", "A", "VA"]:
                sens[f"{k}_lin"] = linear_oof_gated(
                    mats[k][keep], y[keep], groups[keep], f3)[0]
            sens["VA_nl_seed0"] = gbm_oof_gated(
                mats["VA"][keep], y[keep], groups[keep], f3, 0)["auc"]
            res["invalid_shard_sensitivity"] = sens
            print(f"  [invalid-shard sensitivity, dropped {bad}] n={sens['n']} "
                  f"V {sens['V_lin']:.4f} A {sens['A_lin']:.4f} VA {sens['VA_lin']:.4f}")
    res["invalid_shards"] = bad

    # ---- secondary grouping readout (descriptive) ---------------------------
    g2 = np.asarray(d["secondary_groups"])
    if len(np.unique(g2)) >= 5:
        f2 = L.outer_folds(n, g2, n_splits=5)
        sec = {"group_column": d["secondary_group_column"],
               "n_groups": int(len(np.unique(g2)))}
        for k in ["V", "A", "VA"]:
            sec[f"{k}_lin"] = linear_oof_gated(mats[k], y, g2, f2)[0]
        sec["VA_nl_seed0"] = gbm_oof_gated(mats["VA"], y, g2, f2, 0)["auc"]
        res["secondary_grouping"] = sec
        print(f"  [secondary grouping {sec['group_column']}, {sec['n_groups']} groups] "
              f"V {sec['V_lin']:.4f} A {sec['A_lin']:.4f} VA {sec['VA_lin']:.4f} "
              f"VA_nl {sec['VA_nl_seed0']:.4f}")

    res["protocol_notes"] = [
        "FIRST-FIT cell: no prior V+A stack of this construction exists, so the linear "
        "leg is the first fit and no external reproduction gate applies; the "
        "assembled-order gate (< 1e-9) stands in its place.",
        "prior_instrument (.505 RoyalRoad / .578 Wigleaf, 2026-07-05/06) is a DIFFERENT "
        "instrument -- k-medoid NON-GEPA bank, likely Llama-3.3-70B judge, no anchor "
        "battery, no T. Context only; never differenced as if same-instrument.",
        "VA_nl / V_nl = mean over GBM seeds {0,1,2} (FREEZE CHANGE 1); read "
        "Delta_interact only against seed_spread_range.",
        "T is same-rows by construction: the dense arm trains on the identical frozen "
        "population and the identical stable-hash split (FREEZE CHANGE 2).",
        "T is the EVAL-split seed-mean, matching every other cell in this program; eval "
        "was also the checkpoint-selection split, so T is mildly optimistic. Per-seed "
        "test-split AUCs are recorded in T_info.raw.",
        "Delta_interact CI is a GROUP-level bootstrap (FREEZE CHANGE 3).",
        "RULING 2026-08-10: the collapse gate (modal > .98) is ENFORCED inside "
        "clean_fit -- computed on the TRAIN fold only inside each outer fold, and "
        "applied identically to the linear and nonlinear legs.",
        "RULING 2026-08-10: judge context truncation is measured in TOKENS "
        "(gemma-4-31b tokenizer, 1600 source / 960 head / 640 tail), not characters.",
    ]
    res["runtime_sec"] = time.time() - t0
    (RESULTS_DIR / f"{slug}_ledger.json").write_text(json.dumps(res, indent=2, default=str))
    print("wrote", RESULTS_DIR / f"{slug}_ledger.json")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=sorted(CELLS) + ["all"])
    a = ap.parse_args()
    slugs = sorted(CELLS) if a.cell == "all" else [a.cell]
    for slug in slugs:
        r = run_cell(slug)
        Lg = r["ledger"]
        print(f"\n=== LEDGER {slug} ===")
        for k in ("V_lin", "V_nl_mean", "A_lin", "VA_lin", "VA_nl_mean", "T",
                  "Delta_interact", "Delta_total", "Delta_beyond"):
            if Lg.get(k) is not None:
                print(f"  {k:16s} {Lg[k]:+.4f}")
        print(f"  seed spread VA_nl {r['seed_spread_range']['VA']:.4f}")
        print()
    print("CW_EXPERT_LAYER1_DONE")


if __name__ == "__main__":
    main()
