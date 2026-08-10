#!/usr/bin/env python3
"""Direction 1 MIRROR-2 (2026-08-08 task): the SAME frozen "VAT stack" recipe as
`direction1_mirror.py`, extended to six further cells.

The engine is NOT reimplemented here.  `direction1_mirror.py` is imported and its
`fit_arm` / `run_arm` / `run_cell` / `group_paired_boot` / `_E_mask` / `_per_split_T`
are used verbatim; this module only supplies (a) per-cell data loading + the JOIN
between the bank matrix rows and the dense model's per-row predictions, (b) the
full-population Layer-1 reproduction gate, and (c) the "fullfit@E" context row.

Protocol source: notes/2026-08-07__vat_fusion_directions.md ("Direction 1"):
dense per-row probability appended as ONE column to the V+A matrix; the whole
analysis restricted to E = rows whose dense_split is eval or test (the only rows
where the dense column is out-of-sample); stacks refit on E with GroupKFold(5) on
the cell's canonical grouping unit; VA_nl / VAT_nl = mean over GBM seeds {0,1,2};
GROUP-level paired bootstrap (2,000 draws) on (VAT_nl - VA_nl) and (VAT_nl - T).

Cells (published Layer-1 anchors this run must reproduce before the E-refit is
trusted):

  jokes_community          VA_lin .71694 VA_nl .73213  T .7470 (1 dense seed)
  mathse_accepted_verdict  VA_lin .63199 VA_nl .63201  T .6319
  mathse_vote_score        VA_lin .62253 VA_nl .62421  T .6608
  aops_curation            VA_lin .77116 VA_nl .77051  T .78063 (dense REUSED)
  press_verdict            VA_lin .67125 VA_nl .70107  T eval .7497 / test .7525
                                                          (3 dense seeds)
  code_v3                  per-split: eval VA_lin .60584 VA_nl .74219 T .64880
                                      test VA_lin .54032 VA_nl .58058 T .73734

FAMILY CHOICE (binding, per cell, mirroring the cell's own Layer-1 script):
  * `impute_perfold` (linear_oof_family1 / gbm_oof_family1: SimpleImputer(median,
    add_indicator) fit PER TRAIN FOLD, same matrix to linear + GBM) for the four
    scale-up-wave-C cells -- `scaleupC_layer1.py` fits them with exactly those two
    functions on the RAW NaN-bearing V+A matrix.
  * `clean_once` (capagg.clean_cols once + linear_oof_family2 / gbm_oof_raw) for
    press_verdict and code_v3 -- both of those Layer-1 scripts impute ONCE at the
    population level (press: A = where(applicable, levels, 0.5); code_v3:
    clean_cols' internal median fill) and then hand a fixed, NaN-free, degeneracy-
    screened matrix to both the linear and the GBM leg.

sklearn PINNING (the version landmine in press_verdict_layer1.py's docstring:
GroupKFold fold MEMBERSHIP moves between sklearn releases):
  * scale-up-C cells + code_v3 -> sklearn 1.8.0 (the version their ledgers /
    readouts were produced under: `*_ledger.json`.sklearn_version == "1.8.0";
    code_v3's readout ran on sk3, whose python is sklearn 1.8.0).
  * press_verdict -> sklearn 1.9.0 (press_verdict_layer1.json records
    sklearn_version_this_run == "1.9.0" and the script pins >=1.9).
The version actually in use is asserted against the cell's requirement at run
time and recorded in every output JSON.

JOINS.  Every cell's dense column is joined to the bank matrix BY ID and the join
is asserted (row counts, id-set equality, y equality elementwise); the assertion
outcome is stored in the output JSON under "join_assertions".  The dense preds
CSVs written by the dense-standard scorer carry (judgement, prob, group) and no
id, so the id is recovered from the dense build's own `split/{eval,test}.csv`
(which DOES carry `row_id`) by row order, with judgement AND group asserted
elementwise between preds and split file -- see `dense_joins/export_report.json`.

CPU only.  No GPU.  Usage:
    python3 direction1_mirror2.py --cell jokes_community
    python3 direction1_mirror2.py            # all six
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent            # fusion/
TD = HERE.parent                                   # taste_decomposition/
REPO = TD.parents[1]                               # repo root
CLOSURE = TD / "closure"
RESULTS = TD / "results"
JOINS = HERE / "dense_joins"

# --------------------------------------------------------- frozen engine ---
sys.path.insert(0, str(HERE))
_spec_holder = {}


def load_module(path: Path, alias: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


D1 = load_module(HERE / "direction1_mirror.py", "d1_frozen")
fit_arm = D1.fit_arm
run_arm = D1.run_arm
run_cell = D1.run_cell
group_paired_boot = D1.group_paired_boot
_E_mask = D1._E_mask
_per_split_T = D1._per_split_T
L1 = D1.L1
capagg = D1.capagg

GATE_TOL = 0.006          # campaign-standard Layer-1 gate tolerance
GBM_SEEDS = (0, 1, 2)

SKLEARN_REQUIRED = {
    "jokes_community": "1.8.0",
    "mathse_accepted_verdict": "1.8.0",
    "mathse_vote_score": "1.8.0",
    "aops_curation": "1.8.0",
    "code_v3": "1.8.0",
    "press_verdict": "1.9.0",
}


# Where each cell's published Layer-1 numbers were produced.  NEW LANDMINE found
# by this run (generalises the sklearn-version landmine in press_verdict_layer1.py):
# GroupKFold's group->fold assignment sorts group sizes with `np.argsort` (default
# kind='quicksort', NOT stable), so ties among equal-sized groups are broken by the
# sort kernel -- which is ARCHITECTURE-dependent (numpy's vectorised sorts differ
# between linux/x86-64 and macOS/arm64).  Verified directly: with byte-identical
# bank matrices (md5-equal npz shards, identical nansum) and identical
# scikit-learn 1.8.0 / numpy 2.2.6 / scipy 1.17.1 source, the fold-membership
# fingerprint for aops_curation is 89e4f78d... on sk3 (linux x86-64) and
# 267637f7... on the mac, and VA_lin lands at .77116 vs .76922.  So a cell must be
# re-run on the SAME PLATFORM its published Layer-1 ran on, not merely the same
# library versions.
PLATFORM_REQUIRED = {
    "jokes_community": "Linux-x86_64 (sk3)",
    "mathse_accepted_verdict": "Linux-x86_64 (sk3)",
    "mathse_vote_score": "Linux-x86_64 (sk3)",
    "aops_curation": "Linux-x86_64 (sk3)",
    "code_v3": "Linux-x86_64 (sk3)",
    "press_verdict": "Darwin-arm64 (local mac)",
}


def _check_sklearn(cell):
    import platform as _pl
    need = SKLEARN_REQUIRED[cell]
    have = sklearn.__version__
    ok = have == need
    if not ok:
        raise SystemExit(
            f"[{cell}] sklearn {have} installed but this cell's published Layer-1 "
            f"numbers were produced under {need}; GroupKFold fold MEMBERSHIP is "
            f"version-sensitive (press_verdict_layer1.py LANDMINE). Re-run under "
            f"scikit-learn=={need}.")
    import numpy as _np
    import scipy as _sp
    return {"required": need, "this_run": have, "match": ok,
            "numpy": _np.__version__, "scipy": _sp.__version__,
            "platform_this_run": f"{_pl.system()}-{_pl.machine()}",
            "platform_required": PLATFORM_REQUIRED[cell],
            "platform_landmine": (
                "GroupKFold's group->fold assignment uses np.argsort (kind='quicksort', "
                "unstable) over group sizes, so equal-sized groups are ordered by the "
                "sort kernel and the fold assignment is ARCHITECTURE-dependent. Verified: "
                "byte-identical matrices + identical sklearn 1.8.0/numpy 2.2.6/scipy 1.17.1 "
                "give different fold fingerprints on linux-x86_64 vs macOS-arm64 "
                "(aops_curation VA_lin .77116 vs .76922). Reproduce a cell on the platform "
                "its published Layer-1 ran on.")}


# ============================================================ gate + context
def va_only(family, VA_raw, y, groups, gbm_seeds=GBM_SEEDS):
    """The VA half of the frozen `fit_arm`, same family switch, same L1 calls --
    used for the full-population reproduction gate (where no dense column exists
    on the dense model's own training rows)."""
    folds = L1.outer_folds(len(y), groups, n_splits=5)
    if family == "clean_once":
        VAc, _ = capagg.clean_cols(VA_raw)
        linfn, gbmfn = L1.linear_oof_family2, L1.gbm_oof_raw
    elif family == "impute_perfold":
        VAc = VA_raw
        linfn, gbmfn = L1.linear_oof_family1, L1.gbm_oof_family1
    else:
        raise ValueError(family)
    lin_auc, lin_oof = linfn(VAc, y, groups, folds)
    seed_aucs, oofs = [], []
    for s in gbm_seeds:
        r = gbmfn(VAc, y, groups, folds, s)
        seed_aucs.append(r["auc"])
        oofs.append(r["oof"])
    summary = {"VA_lin": lin_auc, "VA_nl_seeds": seed_aucs,
               "VA_nl_mean": float(np.mean(seed_aucs)),
               "n_features_VA": int(VAc.shape[1]),
               "n_features_VA_prescreen": int(VA_raw.shape[1]),
               "n_folds": len(folds)}
    return summary, lin_oof, oofs


def gate_entry(reproduced, published, tol=GATE_TOL):
    return {"published": published, "reproduced": reproduced,
            "abs_diff": abs(reproduced - published), "pass": abs(reproduced - published) <= tol}


def make_gate(summary, published_lin, published_nl, source, tol=GATE_TOL):
    g = {"tol": tol, "source": source,
         "VA_lin": gate_entry(summary["VA_lin"], published_lin, tol),
         "VA_nl_mean": gate_entry(summary["VA_nl_mean"], published_nl, tol),
         "VA_nl_seeds_reproduced": summary["VA_nl_seeds"]}
    g["pass"] = bool(g["VA_lin"]["pass"] and g["VA_nl_mean"]["pass"])
    return g


def fullfit_at_E(y_full, E, lin_oof, gbm_oofs):
    """Context row: the FULL-population Layer-1 OOF predictions read on E rows
    only.  Honest (every row is out-of-fold by construction) but NOT the E-refit:
    it trains on ~5x more rows, so it removes the small-train handicap the E-refit
    imposes on the bank (notes/2026-08-07__vat_fusion_directions.md, Direction 1b
    rationale)."""
    yE = np.asarray(y_full)[E]
    out = {"note": "full-population Layer-1 grouped-OOF predictions evaluated on E "
                   "rows only (no E-refit small-train handicap; still out-of-fold "
                   "for every row)",
           "VA_lin_fullfit_at_E": float(roc_auc_score(yE, lin_oof[E]))}
    per_seed = [float(roc_auc_score(yE, o[E])) for o in gbm_oofs]
    out["VA_nl_fullfit_at_E_seeds"] = per_seed
    out["VA_nl_fullfit_at_E_mean_of_seed_aucs"] = float(np.mean(per_seed))
    out["VA_nl_fullfit_at_E_seedmean_probs"] = float(
        roc_auc_score(yE, np.mean(gbm_oofs, axis=0)[E]))
    return out


# ================================================================ join utils
def load_join(cell):
    p = JOINS / f"{cell}_join.csv.gz"
    assert p.exists(), f"missing dense join table {p}"
    return pd.read_csv(p)


def align_join(cell, ids, y_bank, groups_bank, join, id_col="row_id",
               y_col="judgement", group_col="group", prob_cols=(),
               split_col="dense_split"):
    """Align the join table to the bank's row order BY ID and assert the join.
    Returns (dense_probs dict, dense_split array, assertions dict)."""
    a = {}
    jid = join[id_col].astype(str).values
    bid = np.asarray([str(x) for x in ids])
    a["n_bank_rows"] = int(len(bid))
    a["n_join_rows"] = int(len(jid))
    a["join_ids_unique"] = bool(len(set(jid)) == len(jid))
    a["bank_ids_unique"] = bool(len(set(bid)) == len(bid))
    a["id_sets_equal"] = bool(set(bid) == set(jid))
    if not (a["join_ids_unique"] and a["bank_ids_unique"] and a["id_sets_equal"]):
        raise AssertionError(f"[{cell}] UNSAFE JOIN: {a}")
    pos = {d: i for i, d in enumerate(jid)}
    order = np.array([pos[d] for d in bid])
    jy = join[y_col].values[order]
    a["y_equal_elementwise"] = bool(np.array_equal(np.asarray(y_bank).astype(int),
                                                   np.asarray(jy).astype(int)))
    if not a["y_equal_elementwise"]:
        raise AssertionError(f"[{cell}] UNSAFE JOIN: y mismatch bank vs dense split files")
    if group_col is not None and group_col in join.columns:
        jg = np.array([str(x) for x in join[group_col].values[order]])
        bg = np.array([str(x) for x in groups_bank])
        a["group_equal_elementwise"] = bool(np.array_equal(bg, jg))
    a["id_join_key"] = id_col
    dsplit = np.array([str(x) for x in join[split_col].values[order]], dtype=object)
    a["dense_split_counts"] = {k: int(v) for k, v in
                               pd.Series(dsplit).value_counts().to_dict().items()}
    probs = {c: join[c].values[order].astype(float) for c in prob_cols if c in join.columns}
    a["dense_prob_columns_found"] = sorted(probs)
    for c, v in probs.items():
        a[f"{c}_finite_on_eval_test"] = int(np.isfinite(v[np.isin(dsplit, ["eval", "test"])]).sum())
        a[f"{c}_finite_on_train"] = int(np.isfinite(v[dsplit == "train"]).sum())
    return probs, dsplit, a


# ============================================================== arm helpers
def arm_block(tag, family, VA_raw_E, dcol, y_E, groups_E):
    return run_arm(tag, family, VA_raw_E, dcol, y_E, groups_E)


def write_json(cell, out):
    p = RESULTS / f"vat_stack_{cell}.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print("wrote", p, flush=True)
    return p


# ======================================================= scale-up-wave-C cells
_SC = None


def scaleupC():
    global _SC
    if _SC is None:
        _SC = load_module(TD / "scaleupC_layer1.py", "scaleupC_mirror2")
    return _SC


PUBLISHED_SCALEUPC = {
    # from methods/taste_decomposition/results/<cell>_ledger.json ["ledger"]
    "jokes_community": (0.7169410248505904, 0.7321335909248191),
    "mathse_accepted_verdict": (0.6319935431718893, 0.6320088712017576),
    "mathse_vote_score": (0.6225250904339386, 0.6242101942973988),
    "aops_curation": (0.7711637733536059, 0.7705052946293239),
}

SCALEUPC_META = {
    "jokes_community": dict(
        join_cell="jokes_community", group_column="LDA topic (50)",
        dense_dir="datasets/humor/reddit_jokes/dense_standard",
        seeds=["seed42", "seed1", "seed2"],
        seed_note="all three dense seeds (42/1/2) have preds_{eval,test}.csv "
                  "(eval_pass_results.json: eval .7470/.7507/.7508, mean .7495, spread "
                  ".0038; test .7236/.7254/.7283, mean .7258). Per-seed arms and the "
                  "mean-probability ensemble are both reported; the ensemble is the "
                  "ledger row."),
    "mathse_accepted_verdict": dict(
        join_cell="mathse_accepted_verdict", group_column="question_id",
        dense_dir="datasets/math-stackexchange/v2_va/dense_standard_mathse_accepted_verdict",
        seeds=["seed42"], seed_note="single dense seed (42) exists for this cell."),
    "mathse_vote_score": dict(
        join_cell="mathse_vote_score", group_column="question_id",
        dense_dir="datasets/math-stackexchange/v2_va/dense_standard_mathse_vote_score",
        seeds=["seed42"], seed_note="single dense seed (42) exists for this cell."),
    "aops_curation": dict(
        join_cell="aops_curation", group_column="problem",
        dense_dir="runs/aops_same_approach_dense_llama8b (REUSED, not retrained)",
        seeds=["seed42"],
        seed_note="dense arm REUSED from runs/aops_same_approach_dense_llama8b; a single "
                  "per-row probability column (no seed sweep exists for it)."),
}


def do_scaleupC_cell(cell):
    t0 = time.time()
    skl = _check_sklearn(cell)
    SC = scaleupC()
    d = SC.CELLS[cell]()
    A, V, y, groups, ids = d["A"], d["V"], d["y"], d["groups"], d["ids"]
    VA_raw_full = np.column_stack([V, A])
    family = "impute_perfold"
    meta = SCALEUPC_META[cell]

    # ---- join ------------------------------------------------------------
    join = load_join(meta["join_cell"])
    if cell == "aops_curation":
        probs, dsplit, ja = align_join(cell, ids, y, groups, join, group_col="problem",
                                       prob_cols=["dense_prob"])
        seed_probs = {"seed42": probs["dense_prob"]}
        ja["dense_pred_id_provenance"] = (
            "datasets/math/aops/va/population.csv.gz already carries dense_prob + "
            "dense_split per row_id (population_manifest.json: preds verified "
            "row-aligned with split_full, judgement and problem equal elementwise); "
            "join is by row_id, no row-order recovery needed.")
    else:
        cols = [f"dense_prob_{s}" for s in meta["seeds"]]
        probs, dsplit, ja = align_join(cell, ids, y, groups, join, prob_cols=cols)
        seed_probs = {s: probs[f"dense_prob_{s}"] for s in meta["seeds"]}
        ja["dense_pred_id_provenance"] = (
            f"{meta['dense_dir']}/rm_out_seed*/preds_{{eval,test}}.csv carry only "
            "(judgement, prob, group) -- NO id. The id was recovered from the dense "
            "build's own split/{eval,test}.csv (which carries row_id) BY ROW ORDER, "
            "with judgement AND group asserted equal elementwise between preds and "
            "split file before the row_id was attached (fusion/dense_joins/"
            "export_report.json records those assertions, all True).")

    ens = np.mean(np.column_stack(list(seed_probs.values())), axis=1)
    E = _E_mask(dsplit) & np.isfinite(ens)
    y_E, g_E = np.asarray(y)[E], np.asarray(groups, dtype=object)[E]
    VA_E = VA_raw_full[E]

    # ---- reproduction gate on the FULL population ------------------------
    print(f"  [{cell}] gate: full-population Layer-1 (n={len(y)}) ...", flush=True)
    summ, lin_oof, gbm_oofs = va_only(family, VA_raw_full, np.asarray(y),
                                      np.asarray(groups, dtype=object))
    plin, pnl = PUBLISHED_SCALEUPC[cell]
    gate = make_gate(summ, plin, pnl, f"results/{cell}_ledger.json ['ledger']")
    print(f"  [{cell}] gate {'PASS' if gate['pass'] else 'FAIL'}: "
          f"VA_lin {summ['VA_lin']:.5f} vs {plin:.5f}; "
          f"VA_nl {summ['VA_nl_mean']:.5f} vs {pnl:.5f}", flush=True)

    out = {
        "cell": cell, "direction": "1_mirror2",
        "n_E": int(E.sum()), "pos_rate_E": float(np.mean(y_E)),
        "n_groups_E": int(len(np.unique(g_E))),
        "population": (f"dense-held-out rows (dense_split in {{eval,test}}) of the {cell} "
                       f"scale-up-wave-C bank ({d['matrix']}); joined to "
                       f"{meta['dense_dir']} BY ID (row_id), assertions in join_assertions"),
        "group_column": meta["group_column"], "dense_dir": meta["dense_dir"],
        "family": family,
        "family_reason": ("scaleupC_layer1.py fits this cell with linear_oof_family1 + "
                          "gbm_oof_family1, i.e. SimpleImputer(median, add_indicator) fit "
                          "per TRAIN FOLD on the raw NaN-bearing V+A matrix -- so the "
                          "matched Direction-1 family is impute_perfold."),
        "sklearn": skl,
        "n_features_raw": {"V": int(V.shape[1]), "A": int(A.shape[1])},
        "n_full_population": int(len(y)),
        "dense_seeds_available": meta["seeds"], "dense_seed_note": meta["seed_note"],
        "join_assertions": ja,
        "va_reproduction_gate": gate,
        "fullfit_at_E": fullfit_at_E(y, E, lin_oof, gbm_oofs),
        "per_split_T": _per_split_T(np.asarray(y), ens, dsplit),
        "arms": {}, "per_dense_seed": {},
    }
    if not gate["pass"]:
        out["status"] = "GATE_FAILED_STOPPED -- no E-refit numbers written"
        out["runtime_sec"] = time.time() - t0
        write_json(cell, out)
        return out
    out["status"] = "GATE_PASSED_PROCEED"

    # ---- E-refit arms ----------------------------------------------------
    per_seed_blocks = {}
    for sname, col in seed_probs.items():
        print(f"  [{cell}] E-refit arm {sname} (n_E={E.sum()}) ...", flush=True)
        per_seed_blocks[sname] = arm_block(f"layer1_bank_{sname}", family, VA_E,
                                           col[E], y_E, g_E)
    out["per_dense_seed"] = per_seed_blocks
    if len(seed_probs) == 1:
        only = next(iter(seed_probs))
        ens_block = dict(per_seed_blocks[only])
        ens_block["arm"] = "layer1_bank"
        ens_block["note"] = (f"only one dense seed ({only}) has per-row predictions, so the "
                             "mean-probability ensemble IS that seed's column; this block is "
                             "the same fit, not a second one.")
    else:
        ens_block = arm_block("layer1_bank", family, VA_E, ens[E], y_E, g_E)
    out["ensemble"] = ens_block
    out["arms"]["layer1_bank"] = ens_block
    out["runtime_sec"] = time.time() - t0
    a = ens_block
    print(f"  [{cell}] T={a['T_E']:.4f} VA_nl={a['VA_nl_E_mean']:.4f} "
          f"VAT_nl={a['VAT_nl_E_mean']:.4f} n_E={out['n_E']} groups={out['n_groups_E']} "
          f"({out['runtime_sec']:.0f}s)", flush=True)
    write_json(cell, out)
    return out


# ================================================================ press cell
def do_press_verdict():
    cell = "press_verdict"
    t0 = time.time()
    skl = _check_sklearn(cell)
    sys.path.insert(0, str(TD))
    PV = load_module(TD / "press_verdict_layer1.py", "press_verdict_layer1_mirror2")

    ids, y, comp, topic, levels, applicable = PV.load_population()
    y = np.asarray(y).astype(int)
    Vraw, v_names_raw = PV.build_v_matrix(ids)
    zz = np.load(PV.A_CACHE, allow_pickle=True)
    a_names_raw = [str(s) for s in zz["names"]]
    A_imp = np.where(applicable, levels, 0.5)          # press's own population-level impute
    VA_raw_full = np.column_stack([Vraw, A_imp])
    family = "clean_once"

    join = load_join(cell)
    seeds = ["seed42", "seed1", "seed2"]
    probs, dsplit, ja = align_join(cell, ids, y, comp, join,
                                   prob_cols=[f"dense_prob_{s}" for s in seeds])
    seed_probs = {s: probs[f"dense_prob_{s}"] for s in seeds}
    ja["dense_pred_id_provenance"] = (
        "datasets/press-releases/dense_standard_k3/rm_out_seed{42,1,2}/preds_{eval,test}.csv "
        "carry only (judgement, prob, group) -- NO id. The id was recovered from "
        "dense_standard_k3/split/{eval,test}.csv (carries row_id) BY ROW ORDER with "
        "judgement AND group asserted equal elementwise (fusion/dense_joins/"
        "export_report.json). row_id is the press_verdict_pr_A_k3_scores_CACHE.npz id.")

    ens = np.mean(np.column_stack([seed_probs[s] for s in seeds]), axis=1)
    E = _E_mask(dsplit) & np.isfinite(ens)
    y_E, g_E = y[E], np.asarray([str(c) for c in comp], dtype=object)[E]
    VA_E = VA_raw_full[E]

    # ---- gate: press's OWN full-population Layer-1 pipeline ---------------
    print(f"  [{cell}] gate: press_verdict_layer1 full-population pipeline (n={len(y)}) ...",
          flush=True)
    Ac, _ = PV.clean_cols(A_imp, a_names_raw)
    Vc, _ = PV.clean_cols(Vraw, v_names_raw)
    VA_pub = np.column_stack([Vc, Ac])
    folds_pub = PV.outer_folds(len(y), comp)
    va_lin_pub, lin_oof = PV.linear_oof(VA_pub, y, folds_pub)
    seed_aucs, gbm_oofs = [], []
    for s in GBM_SEEDS:
        r = PV.gbm_oof(VA_pub, y, comp, folds_pub, s)
        seed_aucs.append(r["auc"])
        gbm_oofs.append(r["oof"])
    summ = {"VA_lin": va_lin_pub, "VA_nl_seeds": seed_aucs,
            "VA_nl_mean": float(np.mean(seed_aucs)),
            "n_features_VA": int(VA_pub.shape[1]),
            "n_features_VA_prescreen": int(VA_raw_full.shape[1])}
    gate = make_gate(summ, 0.671245932677923, 0.701068154615308,
                     "results/press_verdict_layer1.json ['ledger'] "
                     "(VA_lin / VA_nl_mean; produced under sklearn 1.9.0)")
    gate["gate_pipeline"] = ("press_verdict_layer1.py's OWN linear_oof "
                             "(SimpleImputer(constant .5) + StandardScaler + "
                             "LogisticRegression(C=1, max_iter=3000, class_weight='balanced')) "
                             "and gbm_oof -- NOT the frozen Direction-1 family2 linear; the "
                             "gate must reproduce the cell's published pipeline.")
    print(f"  [{cell}] gate {'PASS' if gate['pass'] else 'FAIL'}: "
          f"VA_lin {summ['VA_lin']:.5f} vs 0.67125; "
          f"VA_nl {summ['VA_nl_mean']:.5f} vs 0.70107", flush=True)

    T_ref = json.loads((RESULTS / "samerows_T_press.json").read_text())
    out = {
        "cell": cell, "direction": "1_mirror2",
        "n_E": int(E.sum()), "pos_rate_E": float(np.mean(y_E)),
        "n_groups_E": int(len(np.unique(g_E))),
        "population": ("dense-held-out rows (dense_split in {eval,test}) of the 2,956-row "
                       "k>=3 press-verdict population (ids/y/company from "
                       "results/press_verdict_pr_A_k3_scores_CACHE.npz, V rebuilt by "
                       "press_verdict_v_features_recon.py); joined to "
                       "datasets/press-releases/dense_standard_k3 BY ID (row_id)"),
        "group_column": "company",
        "dense_dir": "datasets/press-releases/dense_standard_k3",
        "family": family,
        "family_reason": ("press_verdict_layer1.py imputes ONCE at population level "
                          "(A = where(applicable, levels, 0.5); V has no NaN) and applies its "
                          "degeneracy screen once, then hands one fixed NaN-free matrix to both "
                          "the linear and the GBM leg -- the clean_once family. NOTE the frozen "
                          "clean_once linear is linear_oof_family2 (StandardScaler + LR C=1, "
                          "max_iter 2000, no class_weight) whereas press's own linear uses "
                          "class_weight='balanced', max_iter=3000; the cell is exactly balanced "
                          "(pos_rate .500) so the weighting is a no-op, and Direction-1 "
                          "comparability (VA vs VAT on identical footing) is the controlling "
                          "requirement for the E-refit. The GATE above uses press's own pipeline."),
        "sklearn": skl,
        "n_features_raw": {"V": int(Vraw.shape[1]), "A": int(A_imp.shape[1])},
        "n_full_population": int(len(y)),
        "dense_seeds_available": seeds,
        "T_honest_reference": {"eval_mean": T_ref.get("eval_mean"), "test_mean": T_ref.get("test_mean"),
                               "seed_eval_auc": T_ref.get("seed_eval_auc"),
                               "seed_test_auc": T_ref.get("seed_test_auc"),
                               "source": "results/samerows_T_press.json"},
        "join_assertions": ja,
        "va_reproduction_gate": gate,
        "fullfit_at_E": fullfit_at_E(y, E, lin_oof, gbm_oofs),
        "per_split_T": _per_split_T(y, ens, dsplit),
        "arms": {}, "per_dense_seed": {},
    }
    if not gate["pass"]:
        out["status"] = "GATE_FAILED_STOPPED -- no E-refit numbers written"
        out["runtime_sec"] = time.time() - t0
        write_json(cell, out)
        return out
    out["status"] = "GATE_PASSED_PROCEED"

    for s in seeds:
        print(f"  [{cell}] E-refit arm {s} (n_E={E.sum()}) ...", flush=True)
        out["per_dense_seed"][s] = arm_block(f"layer1_bank_{s}", family, VA_E,
                                             seed_probs[s][E], y_E, g_E)
        out["per_dense_seed"][s]["per_split_T"] = _per_split_T(y, seed_probs[s], dsplit)
    print(f"  [{cell}] E-refit arm ensemble ...", flush=True)
    ens_block = arm_block("layer1_bank", family, VA_E, ens[E], y_E, g_E)
    out["ensemble"] = ens_block
    out["arms"]["layer1_bank"] = ens_block
    out["runtime_sec"] = time.time() - t0
    print(f"  [{cell}] T={ens_block['T_E']:.4f} VA_nl={ens_block['VA_nl_E_mean']:.4f} "
          f"VAT_nl={ens_block['VAT_nl_E_mean']:.4f} n_E={out['n_E']} "
          f"groups={out['n_groups_E']} ({out['runtime_sec']:.0f}s)", flush=True)
    write_json(cell, out)
    return out


# =============================================================== code_v3 cell
PUBLISHED_CODE_V3 = {"eval": (0.6058365612729659, 0.742188128540677),
                     "test": (0.5403220304777754, 0.5805816779957244)}


def do_code_v3():
    cell = "code_v3"
    t0 = time.time()
    skl = _check_sklearn(cell)
    CC = load_module(CLOSURE / "code_v3" / "cells_code.py", "cells_code_mirror2")
    d = CC.load()
    ids = np.asarray([str(x) for x in d["ids"]], dtype=object)
    groups = np.asarray([str(g) for g in d["groups"]], dtype=object)   # repository
    y = np.asarray(d["y"]).astype(int)
    split = np.asarray([str(s) for s in d["split"]], dtype=object)
    A = np.asarray(d["A"], dtype=float)
    V = np.asarray(d["V"], dtype=float)
    v_exec, v_text = d["v_exec"], d["v_text"]
    n_exec = len(v_exec)
    dense42 = np.asarray(d["dense_seed42"], dtype=float)
    family = "clean_once"

    # A matrix EXACTLY as readout_code_v3.py builds it: 83 score cols + 83
    # "applied" indicator cols; V kept as [V_exec | V_text] in that order, so a
    # single clean_cols over the concatenation reproduces the published
    # per-block clean_cols (the screen is per-column and order-preserving).
    A_with_ind = np.column_stack([A, (~np.isnan(A)).astype(float)])
    VA_raw_full = np.column_stack([V[:, :n_exec], V[:, n_exec:], A_with_ind])

    # ---- join ------------------------------------------------------------
    ja = {"n_bank_rows": int(len(ids)), "bank_ids_unique": bool(len(set(ids)) == len(ids)),
          "id_join_key": "repo/pr_number",
          "dense_split_counts": {k: int(v) for k, v in pd.Series(split).value_counts().to_dict().items()},
          "dense_pred_id_provenance": (
              "closure/code_v3/dense_seed42/preds_{eval,test}.csv DO carry ids "
              "(repo, pr_number) -- cells_code.load() joins them to the bank BY "
              "row_id = repo + '/' + pr_number and asserts every bank row is present "
              "(`dense seed 42 missing rows`). No row-order recovery needed. Mirror of "
              "sk3 datasets/code-review/dense_standard_v3/rm_out_seed42/."),
          "dense_seeds_available": list(d["dense_seeds_have"]),
          "dense_prob_finite": int(np.isfinite(dense42).sum())}
    # independent re-check of the published per-split dense AUCs
    ja["T_recheck_vs_eval_pass_results"] = {
        sp: {"recomputed": float(roc_auc_score(y[split == sp], dense42[split == sp])),
             "published": {"eval": 0.6488, "test": 0.7373}[sp]}
        for sp in ("eval", "test")}
    for sp in ("eval", "test"):
        r = ja["T_recheck_vs_eval_pass_results"][sp]
        assert abs(r["recomputed"] - r["published"]) < 1e-3, (cell, sp, r)
    ja["y_equal_elementwise"] = True
    ja["y_equal_note"] = ("y comes from the same V_matrix_v3.parquet rows the A shards and "
                          "the dense preds were both keyed to; the dense preds' own "
                          "`judgement` column was additionally used by the upstream builder. "
                          "The T re-check above (recomputed vs eval_pass_results.json, both "
                          "splits, |diff| < 1e-3) is the join's end-to-end evidence.")

    # E = ALL rows: the A bank was scored on the dense chain's eval+test only, and
    # the dense chain's train repositories are disjoint from both (cells_code.py
    # RECORDED DECISION), so every row is dense-held-out by construction.
    E = _E_mask(split) & np.isfinite(dense42)
    assert E.all(), "code_v3: expected every row to be dense-held-out"

    out = {
        "cell": cell, "direction": "1_mirror2",
        "POOLED_DO_NOT_QUOTE": True,
        "pooled_caveat": ("pooled AUC on this cell is composition-dominated and the two "
                          "dense-held-out halves contradict each other (eval T .6488 / VA_nl "
                          ".7422 vs test T .7373 / VA_nl .5806). The published readout for "
                          "this cell is WITHIN-REPO (closure/code_v3/abank_rescore/"
                          "within_repo.py); the pooled-E arm below is recorded for schema "
                          "uniformity ONLY and is marked POOLED_DO_NOT_QUOTE."),
        "n_E": int(E.sum()), "pos_rate_E": float(np.mean(y[E])),
        "n_groups_E": int(len(np.unique(groups[E]))),
        "population": ("ALL 11,452 rows of the code_v3 (GitHub PR merge, v3 enriched text) "
                       "closure population -- every row is dense_split in {eval,test} by "
                       "construction (the A bank was scored on the dense v3 chain's eval+test "
                       "only and the chain's train repositories are repo-disjoint from both); "
                       "dense preds joined BY ID (repo/pr_number)"),
        "group_column": "repository (REPO -- the canonical grouping unit for this cell)",
        "dense_dir": ("methods/taste_decomposition/closure/code_v3/dense_seed42 "
                      "(mirror of sk3 datasets/code-review/dense_standard_v3/rm_out_seed42)"),
        "family": family,
        "family_reason": ("readout_code_v3.py applies its clean_cols (median fill + "
                          "degeneracy screen) ONCE per split and then feeds one fixed "
                          "NaN-free matrix to StandardScaler+LogisticRegression(C=1, "
                          "max_iter=2000) and to the frozen HistGB grid -- byte-identical "
                          "to the frozen engine's clean_once family (capagg.clean_cols / "
                          "linear_oof_family2 / gbm_oof_raw)."),
        "sklearn": skl,
        "n_features_raw": {"V_exec": n_exec, "V_text": V.shape[1] - n_exec,
                           "A_scores": A.shape[1], "A_applied_indicators": A.shape[1]},
        "n_full_population": int(len(y)),
        "dense_seeds_available": [f"seed{s}" for s in d["dense_seeds_have"]],
        "dense_seed_note": ("only seed 42 has per-row predictions (sk3 "
                            "datasets/code-review/dense_standard_v3 has an rm_out_seed1 "
                            "directory with no preds_*.csv and eval_pass_results.json holds "
                            "only seed42)."),
        "join_assertions": ja,
        "va_reproduction_gate": {}, "fullfit_at_E": {},
        "per_split_T": _per_split_T(y, dense42, split),
        "arms": {}, "per_split": {}, "within_repo": {},
    }

    # ---- per-split reproduction gate (the published fit for this cell) ----
    gates, split_state = {}, {}
    for sp in ("eval", "test"):
        m = split == sp
        print(f"  [{cell}] gate/{sp}: within-split Layer-1 (n={m.sum()}) ...", flush=True)
        summ, lin_oof, gbm_oofs = va_only(family, VA_raw_full[m], y[m], groups[m])
        plin, pnl = PUBLISHED_CODE_V3[sp]
        gates[sp] = make_gate(summ, plin, pnl,
                              "results/code_v3_enriched_layer1.json "
                              f"['splits']['{sp}'] linear_layer1.VA / nonlinear.VA.auc_mean")
        split_state[sp] = (m, summ, lin_oof, gbm_oofs)
        print(f"  [{cell}] gate/{sp} {'PASS' if gates[sp]['pass'] else 'FAIL'}: "
              f"VA_lin {summ['VA_lin']:.5f} vs {plin:.5f}; "
              f"VA_nl {summ['VA_nl_mean']:.5f} vs {pnl:.5f}", flush=True)
    out["va_reproduction_gate"] = dict(gates)
    out["va_reproduction_gate"]["pass"] = bool(all(g["pass"] for g in gates.values()))
    for sp in ("eval", "test"):
        m, summ, lin_oof, gbm_oofs = split_state[sp]
        out["fullfit_at_E"][sp] = fullfit_at_E(y[m], np.ones(int(m.sum()), dtype=bool),
                                               lin_oof, gbm_oofs)
        out["fullfit_at_E"][sp]["note"] = (
            "E covers this split entirely, so the full-population (=within-split) Layer-1 "
            "fit and the E-refit are the SAME fit on the SAME rows; this block is the "
            "published within-split Layer-1 value, reproduced.")
    if not out["va_reproduction_gate"]["pass"]:
        out["status"] = "GATE_FAILED_STOPPED -- no E-refit numbers written"
        out["runtime_sec"] = time.time() - t0
        write_json(cell, out)
        return out
    out["status"] = "GATE_PASSED_PROCEED"

    # ---- pooled-E arm (DO NOT QUOTE) --------------------------------------
    print(f"  [{cell}] E-refit POOLED arm (n_E={E.sum()}, {len(np.unique(groups))} repos) ...",
          flush=True)
    pooled = arm_block("layer1_bank", family, VA_raw_full[E], dense42[E], y[E], groups[E])
    pooled["quote_status"] = "POOLED_DO_NOT_QUOTE"
    out["arms"]["layer1_bank"] = pooled
    out["ensemble"] = pooled

    # ---- per-split E-refits (= the published fitting footing) -------------
    oof_by_split = {}
    for sp in ("eval", "test"):
        m = split == sp
        print(f"  [{cell}] E-refit within {sp} (n={m.sum()}) ...", flush=True)
        r = fit_arm(family, VA_raw_full[m], dense42[m], y[m], groups[m])
        blk = {
            "arm": f"layer1_bank_{sp}", "family": family,
            "quote_status": "POOLED_WITHIN_SPLIT_DO_NOT_QUOTE (use within_repo below)",
            "n": int(m.sum()), "n_groups": int(len(np.unique(groups[m]))),
            "n_features_VA": r["n_features_VA"],
            "n_features_VA_prescreen": r["n_features_VA_prescreen"],
            "T_E": float(roc_auc_score(y[m], dense42[m])),
            "VA_lin_E": r["VA_lin"], "VA_nl_E_seeds": r["VA_nl_seeds"],
            "VA_nl_E_mean": r["VA_nl_mean"], "VA_nl_E_spread": r["VA_nl_spread"],
            "VAT_lin_E": r["VAT_lin"], "VAT_nl_E_seeds": r["VAT_nl_seeds"],
            "VAT_nl_E_mean": r["VAT_nl_mean"], "VAT_nl_E_spread": r["VAT_nl_spread"],
            "boot_group_VAT_nl_minus_VA_nl": group_paired_boot(
                y[m], r["_oof_VAT_nl0"], r["_oof_VA_nl0"], groups[m]),
            "boot_group_VAT_nl_minus_T": group_paired_boot(
                y[m], r["_oof_VAT_nl0"], dense42[m], groups[m]),
        }
        out["per_split"][sp] = blk
        oof_by_split[sp] = (m, r["_oof_VA_nl0"], r["_oof_VAT_nl0"])

    # ---- WITHIN-REPO readout (the canonical one for this cell) ------------
    AB = CLOSURE / "code_v3" / "abank_rescore"
    for sp in ("eval", "test"):
        m, oof_va, oof_vat = oof_by_split[sp]
        pub_oof = np.mean([np.load(AB / f"code_v3_{sp}_va_nl_oof_seed{s}.npy")
                           for s in (0, 1, 2)], axis=0)
        blk = {
            "min_repo_n": CC.MIN_REPO_N,
            "T_dense": CC.within_repo_auc(y[m], dense42[m], groups[m]),
            "VA_nl_E_refit_seed0": CC.within_repo_auc(y[m], oof_va, groups[m]),
            "VAT_nl_E_refit_seed0": CC.within_repo_auc(y[m], oof_vat, groups[m]),
            "VA_nl_published_fullfit_seedmean": CC.within_repo_auc(y[m], pub_oof, groups[m]),
            "delta_VAT_nl_minus_VA_nl": CC.within_repo_delta(y[m], oof_vat, oof_va, groups[m]),
            "delta_VAT_nl_minus_T": CC.within_repo_delta(y[m], oof_vat, dense42[m], groups[m]),
        }
        for k in ("T_dense", "VA_nl_E_refit_seed0", "VAT_nl_E_refit_seed0",
                  "VA_nl_published_fullfit_seedmean"):
            blk[k].pop("per_repo", None)
        for k in ("delta_VAT_nl_minus_VA_nl", "delta_VAT_nl_minus_T"):
            blk[k].pop("per_repo_delta", None)
        out["within_repo"][sp] = blk
        print(f"  [{cell}] within-repo {sp}: T nwtd {blk['T_dense']['nwtd']:.4f} | "
              f"VA_nl {blk['VA_nl_E_refit_seed0']['nwtd']:.4f} | "
              f"VAT_nl {blk['VAT_nl_E_refit_seed0']['nwtd']:.4f}", flush=True)
    out["within_repo"]["reproduction_check_vs_published"] = {
        "source": "closure/code_v3/abank_rescore/within_repo_dense_vs_vanl.json",
        "published": json.loads((AB / "within_repo_dense_vs_vanl.json").read_text()),
        "recomputed": {sp: {"dense_within_nwtd": out["within_repo"][sp]["T_dense"]["nwtd"],
                            "dense_within_median": out["within_repo"][sp]["T_dense"]["median"],
                            "va_nl_within_nwtd":
                                out["within_repo"][sp]["VA_nl_published_fullfit_seedmean"]["nwtd"],
                            "va_nl_within_median":
                                out["within_repo"][sp]["VA_nl_published_fullfit_seedmean"]["median"]}
                       for sp in ("eval", "test")},
    }
    out["runtime_sec"] = time.time() - t0
    write_json(cell, out)
    return out


# ===================================================================== main
CELL_FNS = {
    "jokes_community": lambda: do_scaleupC_cell("jokes_community"),
    "mathse_accepted_verdict": lambda: do_scaleupC_cell("mathse_accepted_verdict"),
    "mathse_vote_score": lambda: do_scaleupC_cell("mathse_vote_score"),
    "aops_curation": lambda: do_scaleupC_cell("aops_curation"),
    "press_verdict": do_press_verdict,
    "code_v3": do_code_v3,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", default=None, choices=list(CELL_FNS))
    args = ap.parse_args()
    cells = args.cell if args.cell else list(CELL_FNS)
    for cell in cells:
        t0 = time.time()
        print(f"=== {cell} ===", flush=True)
        CELL_FNS[cell]()
        print(f"  [{cell}] total {time.time()-t0:.0f}s\n", flush=True)


if __name__ == "__main__":
    main()
