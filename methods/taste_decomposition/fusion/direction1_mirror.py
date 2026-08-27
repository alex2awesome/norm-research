#!/usr/bin/env python3
"""Direction 1 MIRROR (2026-08-09 task): dense per-row score appended as ONE
column to the V+A bank matrix, then the frozen Layer-1 linear + HistGB stacks
refit -- for the cells where the DENSE side WINS OR TIES the articulated bank
(the mirror of notes/2026-08-07__vat_fusion_directions.md Direction 1, which
ran the identical recipe on the three cells where the bank beat dense).

Protocol source: notes/2026-08-07__vat_fusion_directions.md ("Direction 1")
and fusion/direction1_stack.py (reused directly: L1 = layer1_gemma_cells.py's
outer_folds / linear_oof_family1&2 / gbm_oof_family1 / gbm_oof_raw, and
capagg.clean_cols for the population-level degeneracy screen).

Cells (T = registry dense AUC, VA_nl = population-level Layer-1 nonlinear
bank AUC; methods/taste_decomposition/results/*_layer1.json):
  peer_verdict          T .753        VA_nl .688   (round0 vs round4-mined, 56 crit)
  cw_community           T .780        VA_nl .621   (round0 vs round8[=round7], gate FAILED r8)
  nc_responded           T .808        VA_nl .724   (round0 vs round5-mined)
  nc_outcome             T .622        VA_nl .610   (Layer-1 matrix only)
  nc_agree               T_eval .566 / T_test .639  VA_nl .584 (Layer-1 matrix; split FLIPS)
  hashtagwars_verdict    T .664        VA_nl .630   (Layer-1 matrix; 3 dense seeds)
  peer_curation          T .593/.588   VA_nl .559   (Layer-1 matrix)
  peer_revealed          T .871/.896   VA_nl .767   (Layer-1 matrix)

LEAKAGE RULES (binding):
  * dense column only on rows OUTSIDE that dense model's own training split ->
    entire analysis restricted to E = dense_split in {eval, test}
  * stacks fit with grouped OOF (GroupKFold(5)) on E ONLY, grouped by the
    cell's own canonical group column (never re-using any mining-phase
    FIT+MINE/MONITOR split from the closure campaigns -- those existed to
    mine NEW criteria; this mirror only asks whether a FIXED bank + T beats
    the bank alone, so E is refit fresh exactly as Direction 1 did for the
    three original cells).
  * "clean_once" family (peer_verdict, peer_curation, peer_revealed,
    nc_outcome, nc_agree, nc_responded): population-level clean_cols
    (median-impute + degeneracy screen, UNSUPERVISED, no y) applied ONCE on
    the E submatrix -- identical treatment to how Direction 1 handled the two
    family2 caption cells; then StandardScaler+LR (no imputer) / HistGB fed
    the already-clean matrix (linear_oof_family2 / gbm_oof_raw).
  * "impute_perfold" family (cw_community, hashtagwars_verdict): raw
    NaN-bearing matrix, SimpleImputer(median, add_indicator) fit PER TRAIN
    FOLD, fed to both the linear pipeline and the GBM (family1:
    linear_oof_family1 / gbm_oof_family1).
  * VA_nl / VAT_nl = mean over seeds {0,1,2}, spread reported.
  * group-level (not row-level) paired bootstrap CIs on (VAT_nl-T) and
    (VAT_nl-VA_nl) -- deviation from Direction 1's original row-level
    bootstrap, per this task's explicit instruction.

CPU only. No GPU. Usage: python3 direction1_mirror.py [--cell peer_verdict ...]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent          # fusion/
TD = HERE.parent                                  # taste_decomposition/
REPO = TD.parents[1]                               # repo root
CLOSURE = TD / "closure"
RESULTS = TD / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def load_module(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- frozen L1 ---
L1 = load_module(TD / "layer1_gemma_cells.py", "l1_mirror")
capagg = sys.modules["capagg_taste_decomp"]  # loaded as a side effect of L1


# ================================================================= engine ==
def fit_arm(family, VA_raw, dense_col, y, groups, gbm_seeds=(0, 1, 2)):
    """One bank arm, restricted to E.  Returns AUCs + OOF arrays (seed0) for
    bootstrap pairing."""
    n = len(y)
    folds = L1.outer_folds(n, groups, n_splits=5)
    if family == "clean_once":
        VAc, keep = capagg.clean_cols(VA_raw)
        linfn = L1.linear_oof_family2
        gbmfn = L1.gbm_oof_raw
    elif family == "impute_perfold":
        VAc = VA_raw
        keep = list(range(VA_raw.shape[1]))
        linfn = L1.linear_oof_family1
        gbmfn = L1.gbm_oof_family1
    else:
        raise ValueError(family)
    VAT = np.column_stack([VAc, dense_col])

    res = {"n_features_VA": int(VAc.shape[1]), "n_features_VA_prescreen": int(VA_raw.shape[1])}
    for key, M in (("VA", VAc), ("VAT", VAT)):
        auc, oof = linfn(M, y, groups, folds)
        res[f"{key}_lin"] = auc
        seed_aucs, oof0 = [], None
        for s in gbm_seeds:
            r = gbmfn(M, y, groups, folds, s)
            seed_aucs.append(r["auc"])
            if s == gbm_seeds[0]:
                oof0 = r["oof"]
        res[f"{key}_nl_seeds"] = seed_aucs
        res[f"{key}_nl_mean"] = float(np.mean(seed_aucs))
        res[f"{key}_nl_spread"] = float(max(seed_aucs) - min(seed_aucs))
        res[f"_oof_{key}_nl0"] = oof0
    res["n_folds"] = len(folds)
    return res


def group_paired_boot(y, pred_a, pred_b, groups, n_boot=2000, seed=12345):
    """Group-level (cluster) paired bootstrap of AUC(pred_a) - AUC(pred_b)."""
    y = np.asarray(y)
    point = float(roc_auc_score(y, pred_a) - roc_auc_score(y, pred_b))
    uq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uq}
    rng = np.random.default_rng(seed)
    deltas = []
    tries = 0
    while len(deltas) < n_boot and tries < n_boot * 4:
        tries += 1
        gs = rng.choice(uq, size=len(uq), replace=True)
        idx = np.concatenate([idx_by[g] for g in gs])
        ys = y[idx]
        if ys.min() == ys.max():
            continue
        deltas.append(float(roc_auc_score(ys, pred_a[idx]) - roc_auc_score(ys, pred_b[idx])))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"estimate": point, "ci95": [float(lo), float(hi)],
            "p_gt_0": float((deltas > 0).mean()), "n_boot_used": int(len(deltas)),
            "n_groups": int(len(uq))}


def run_arm(tag, family, VA_raw, dense_col, y, groups):
    r = fit_arm(family, VA_raw, dense_col, y, groups)
    T_E = float(roc_auc_score(y, dense_col))
    out = {
        "arm": tag, "family": family,
        "n_features_VA": r["n_features_VA"], "n_features_VA_prescreen": r["n_features_VA_prescreen"],
        "T_E": T_E,
        "VA_lin_E": r["VA_lin"], "VA_nl_E_seeds": r["VA_nl_seeds"],
        "VA_nl_E_mean": r["VA_nl_mean"], "VA_nl_E_spread": r["VA_nl_spread"],
        "VAT_lin_E": r["VAT_lin"], "VAT_nl_E_seeds": r["VAT_nl_seeds"],
        "VAT_nl_E_mean": r["VAT_nl_mean"], "VAT_nl_E_spread": r["VAT_nl_spread"],
        "boot_group_VAT_nl_minus_VA_nl": group_paired_boot(y, r["_oof_VAT_nl0"], r["_oof_VA_nl0"], groups),
        "boot_group_VAT_nl_minus_T": group_paired_boot(y, r["_oof_VAT_nl0"], dense_col, groups),
    }
    return out


def run_cell(cell, y, groups, arms, dense_col, extra=None, verbose=True, write=True, tag=""):
    """arms: dict[str, (family, VA_raw_matrix)] already restricted to E (or a split of E)."""
    t0 = time.time()
    label = f"{cell}{('/' + tag) if tag else ''}"
    out = {
        "cell": cell, "direction": "1_mirror",
        "n_E": int(len(y)), "pos_rate_E": float(np.mean(y)),
        "n_groups_E": int(len(np.unique(groups))),
        "arms": {},
    }
    if extra:
        out.update(extra)
    for atag, (family, VA_raw) in arms.items():
        if verbose:
            print(f"  [{label}] arm={atag} family={family} VA_raw={VA_raw.shape} ...", flush=True)
        out["arms"][atag] = run_arm(atag, family, VA_raw, dense_col, y, groups)
    out["runtime_sec"] = time.time() - t0
    if verbose:
        for atag, a in out["arms"].items():
            print(f"  [{label}] {atag}: T={a['T_E']:.4f} VA_nl={a['VA_nl_E_mean']:.4f} "
                  f"VAT_nl={a['VAT_nl_E_mean']:.4f} n={out['n_E']} groups={out['n_groups_E']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    if write:
        p = RESULTS / f"vat_stack_{cell}.json"
        p.write_text(json.dumps(out, indent=2))
        print("wrote", p, flush=True)
    return out


# ============================================================== cell I/O ==
CELLS_BATCH1 = load_module(CLOSURE / "maps_batch1" / "cells.py", "cells_batch1_mirror")
CELLS_HWSI = load_module(CLOSURE / "maps_hw_si" / "cells.py", "cells_hwsi_mirror")


def _E_mask(dense_split):
    return np.isin(np.asarray(dense_split, dtype=object), ["eval", "test"])


# ---- simple Layer-1-matrix-only cells (maps_batch1 adapter) --------------
def do_maps_batch1_cell(cell, family="clean_once"):
    d = CELLS_BATCH1.load(cell)
    dense = np.asarray(d["dense"], dtype=float)
    dsplit = np.asarray(d["dense_split"], dtype=object)
    E = _E_mask(dsplit) & np.isfinite(dense)
    y = np.asarray(d["y"])[E]
    groups = np.asarray(d["groups"], dtype=object)[E]
    V = np.asarray(d["V"], dtype=float)[E]
    A = np.asarray(d["A"], dtype=float)[E]
    VA = np.column_stack([V, A]) if V.shape[1] and A.shape[1] else (V if V.shape[1] else A)
    dcol = dense[E]
    extra = {
        "population": f"dense-held-out rows of the {cell} Layer-1 matrix "
                       "(dense_split in {eval,test}); maps_batch1/cells.py adapter",
        "n_features_raw": {"V": int(V.shape[1]), "A": int(A.shape[1])},
        "per_split_T": _per_split_T(np.asarray(d["y"]), dense, dsplit),
    }
    return run_cell(cell, y, groups, {"layer1_bank": (family, VA)}, dcol, extra=extra)


def _per_split_T(y_full, dense_full, dsplit_full):
    out = {}
    for sp in ("eval", "test"):
        m = (dsplit_full == sp) & np.isfinite(dense_full)
        if m.sum() > 20 and len(set(y_full[m])) == 2:
            out[sp] = {"n": int(m.sum()), "T": float(roc_auc_score(y_full[m], dense_full[m]))}
    return out


# ---- hashtagwars_verdict (maps_hw_si adapter, 3 dense seeds) --------------
def do_hashtagwars_verdict():
    d = CELLS_HWSI.load("hashtagwars_verdict")
    dsplit = np.asarray(d["dense_split"], dtype=object)
    E = _E_mask(dsplit)
    y = np.asarray(d["y"])[E]
    groups = np.asarray(d["groups"], dtype=object)[E]
    V = np.asarray(d["V"], dtype=float)[E]
    A = np.asarray(d["A"], dtype=float)[E]
    VA_raw = np.column_stack([V, A]) if V.shape[1] and A.shape[1] else (V if V.shape[1] else A)
    seeds_E = np.asarray(d["dense_seeds"], dtype=float)[E]  # (n_E, 3): p_seed1,p_seed2,p_seed42
    ens_E = np.asarray(d["dense"], dtype=float)[E]
    seed_names = ["seed1", "seed2", "seed42"]

    t0 = time.time()
    out = {
        "cell": "hashtagwars_verdict", "direction": "1_mirror",
        "population": "dense-held-out rows (dense_split in {eval,test}); maps_hw_si/cells.py adapter",
        "n_E": int(len(y)), "pos_rate_E": float(y.mean()),
        "n_groups_E": int(len(np.unique(groups))),
        "n_features_raw": {"V": int(V.shape[1]), "A": int(A.shape[1])},
        "per_dense_seed": {}, "ensemble": {},
    }
    for k, sname in enumerate(seed_names):
        dcol = seeds_E[:, k]
        r = fit_arm("impute_perfold", VA_raw, dcol, y, groups)
        T_E = float(roc_auc_score(y, dcol))
        out["per_dense_seed"][sname] = {
            "T_E": T_E,
            "VA_lin_E": r["VA_lin"], "VA_nl_E_seeds": r["VA_nl_seeds"], "VA_nl_E_mean": r["VA_nl_mean"],
            "VAT_lin_E": r["VAT_lin"], "VAT_nl_E_seeds": r["VAT_nl_seeds"], "VAT_nl_E_mean": r["VAT_nl_mean"],
            "boot_group_VAT_nl_minus_VA_nl": group_paired_boot(y, r["_oof_VAT_nl0"], r["_oof_VA_nl0"], groups),
            "boot_group_VAT_nl_minus_T": group_paired_boot(y, r["_oof_VAT_nl0"], dcol, groups),
        }
        print(f"  [hashtagwars_verdict] {sname}: T={T_E:.4f} VA_nl={r['VA_nl_mean']:.4f} "
              f"VAT_nl={r['VAT_nl_mean']:.4f} ({time.time()-t0:.0f}s)", flush=True)

    r = fit_arm("impute_perfold", VA_raw, ens_E, y, groups)
    T_ens = float(roc_auc_score(y, ens_E))
    out["ensemble"] = {
        "T_E": T_ens,
        "VA_lin_E": r["VA_lin"], "VA_nl_E_seeds": r["VA_nl_seeds"], "VA_nl_E_mean": r["VA_nl_mean"],
        "VAT_lin_E": r["VAT_lin"], "VAT_nl_E_seeds": r["VAT_nl_seeds"], "VAT_nl_E_mean": r["VAT_nl_mean"],
        "boot_group_VAT_nl_minus_VA_nl": group_paired_boot(y, r["_oof_VAT_nl0"], r["_oof_VA_nl0"], groups),
        "boot_group_VAT_nl_minus_T": group_paired_boot(y, r["_oof_VAT_nl0"], ens_E, groups),
    }
    out["runtime_sec"] = time.time() - t0
    p = RESULTS / "vat_stack_hashtagwars_verdict.json"
    p.write_text(json.dumps(out, indent=2))
    print("wrote", p, flush=True)
    return out


# ---- cw_community: round0 vs round8(=round7, round8 GATE FAILED) ---------
def do_cw_community():
    cw = CLOSURE / "cw_community"
    z0 = np.load(cw / "round0_state.npz", allow_pickle=True)
    z7 = np.load(cw / "round7_state.npz", allow_pickle=True)
    y = z0["y"].astype(int)
    groups = np.array([str(g) for g in z0["groups"]], dtype=object)
    dense = z0["T"].astype(float)
    assert (z0["y"] == z7["y"]).all() and (z0["ids"] == z7["ids"]).all(), "round0/round7 row mismatch"
    assert np.allclose(z0["T"], z7["T"]), "round0/round7 T mismatch"
    # entire 7,008-row honest population is dense-held-out by construction
    # (cw_honest_dense_preds.report.json: n=7008, all dense_split in {eval,test})
    VA0 = z0["VA"].astype(float)
    VA8 = z7["VA"].astype(float)  # round8 gate FAILED (anchor battery), added 0 A cols -> == round7
    extra = {
        "population": "cw_honest_population (n=7008), ALL rows dense_split in {eval,test} "
                       "by construction (cw_honest_dense_preds.report.json)",
        "round8_note": "round8 anchor-battery gate FAILED (coherent-vs-scrambled AUC .3205 < .5); "
                        "0 criteria admitted to A -> round8 bank IS round7's 144-col bank verbatim "
                        "(CAMPAIGN_TABLES.md; round8_results_GATE_FAILED.json)",
        "bank_sizes": {"round0": int(VA0.shape[1]), "round8": int(VA8.shape[1])},
    }
    return run_cell("cw_community", y, groups,
                     {"round0": ("impute_perfold", VA0), "round8": ("impute_perfold", VA8)},
                     dense, extra=extra)


# ---- peer_verdict: round0 vs round4 (56 mined criteria across rounds1-4) --
def do_peer_verdict():
    sys.path.insert(0, str(CLOSURE))
    import stage4_readout as SR4          # noqa: E402  (unique module name repo-wide)
    import stage4_round4 as S4R4          # noqa: E402

    pop, split, dsplit, XA1, XB1, a1_ids, b1_ids, summary = SR4.build_blocks()
    XA2, _, a2_ids, _, _ = S4R4.load_round_blocks(2)
    XA3, _, a3_ids, _, _ = S4R4.load_round_blocks(3)
    XA4, _, a4_ids, _, _ = S4R4.load_round_blocks(4)
    y_full, nt_full = pop["y"], pop["ntitle"]

    dp = pd.read_csv(CLOSURE / "peer_verdict_dense_preds.csv")
    assert len(dp) == len(y_full), (len(dp), len(y_full))
    assert (dp["dense_split"].values == dsplit).all(), "dense_split order mismatch vs build_blocks()"
    assert (dp["judgement"].astype(int).values == y_full).all(), "y order mismatch vs build_blocks()"
    dense_full = dp["dense_prob"].astype(float).values

    E = np.isin(dsplit, ["eval", "test"])
    y = y_full[E]
    groups = np.array([str(s) for s in nt_full[E]], dtype=object)
    dcol = dense_full[E]
    V, A = pop["V"][E], pop["A"][E]
    VA0 = np.column_stack([V, A])
    VA4 = np.column_stack([V, A, XA1[E], XA2[E], XA3[E], XA4[E]])
    n_mined = sum(x.shape[1] for x in (XA1, XA2, XA3, XA4))
    extra = {
        "population": "dense-held-out rows (dense_split in {eval,test}); "
                      "closure/peer_verdict_dense_preds.csv join by row order (asserted)",
        "n_mined_criteria_rounds1_4": int(n_mined),
        "bank_sizes": {"round0": int(VA0.shape[1]), "round4": int(VA4.shape[1])},
    }
    return run_cell("peer_verdict", y, groups,
                     {"round0": ("clean_once", VA0), "round4": ("clean_once", VA4)},
                     dcol, extra=extra)


# ---- nc_responded: round0 vs round5 (mined); eval-half/test-half separate -
def do_nc_responded():
    ncr_dir = CLOSURE / "nc_responded"
    sys.path.insert(0, str(ncr_dir))
    import nc_closure_lib as NCL          # noqa: E402  (unique module name repo-wide)
    import readout as NCR                 # noqa: E402  (only "readout" import in this process)

    pop = NCL.load_population()
    summary, split, dsplit, mining, monitor_full = NCL.load_splits()
    y_full = pop["y"]
    docket_full = np.array([str(s) for s in pop["docket"]], dtype=object)

    dp = pd.read_csv(ncr_dir / "nc_responded_dense_preds_aligned.csv")
    pr = dict(zip(dp["doc_id"].astype(str), dp["dense_prob"].astype(float)))
    ds = dict(zip(dp["doc_id"].astype(str), dp["dense_split"].astype(str)))
    doc_id_full = pop["doc_id"]
    dense_full = np.array([pr.get(str(d), np.nan) for d in doc_id_full])
    dsplit_from_preds = np.array([ds.get(str(d), "unmapped") for d in doc_id_full])
    assert (dsplit_from_preds == dsplit).all(), "dense_split mismatch: load_splits() vs preds csv"

    Xr, names_r = NCR.load_round_scores([1, 2, 3, 4, 5])
    n_mined = 0 if Xr is None else Xr.shape[1]

    def bank_for(mask):
        V, A = pop["V"][mask], pop["A"][mask]
        VA0 = np.column_stack([V, A])
        VA5 = np.column_stack([V, A, Xr[mask]]) if Xr is not None else VA0
        return VA0, VA5

    E = np.isin(dsplit, ["eval", "test"])
    y_E = y_full[E]
    groups_E = docket_full[E]
    dcol_E = dense_full[E]
    VA0_E, VA5_E = bank_for(E)
    extra = {
        "population": "dense-held-out rows (dense_split in {eval,test}); "
                      "nc_responded_dense_preds_aligned.csv join by doc_id (asserted vs load_splits())",
        "n_mined_criteria_rounds1_5": int(n_mined),
        "bank_sizes": {"round0": int(VA0_E.shape[1]), "round5": int(VA5_E.shape[1])},
        "selection_caveat": "the dense checkpoint was selected on a split inside E (registry "
                            "convention); eval-half and test-half are reported separately below "
                            "(per_split) so either convention can be read cleanly, in addition to "
                            "the pooled E=eval+test readout in `arms`.",
    }
    pooled = run_cell("nc_responded", y_E, groups_E,
                       {"round0": ("clean_once", VA0_E), "round5": ("clean_once", VA5_E)},
                       dcol_E, extra=extra, write=False)

    per_split = {}
    for sp in ("eval", "test"):
        m = dsplit == sp
        y_s, g_s, d_s = y_full[m], docket_full[m], dense_full[m]
        VA0_s, VA5_s = bank_for(m)
        per_split[sp] = run_cell("nc_responded", y_s, g_s,
                                  {"round0": ("clean_once", VA0_s), "round5": ("clean_once", VA5_s)},
                                  d_s, write=False, tag=sp)
    pooled["per_split"] = per_split
    p = RESULTS / "vat_stack_nc_responded.json"
    p.write_text(json.dumps(pooled, indent=2))
    print("wrote", p, flush=True)
    return pooled


# ===================================================================== main
CELL_FNS = {
    "peer_verdict": do_peer_verdict,
    "cw_community": do_cw_community,
    "nc_responded": do_nc_responded,
    "nc_outcome": lambda: do_maps_batch1_cell("nc_outcome"),
    "nc_agree": lambda: do_maps_batch1_cell("nc_agree"),
    "hashtagwars_verdict": do_hashtagwars_verdict,
    "peer_curation": lambda: do_maps_batch1_cell("peer_curation"),
    "peer_revealed": lambda: do_maps_batch1_cell("peer_revealed"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", default=None, choices=list(CELL_FNS))
    args = ap.parse_args()
    cells = args.cell if args.cell else list(CELL_FNS)
    for cell in cells:
        t0 = time.time()
        print(f"=== {cell} ===", flush=True)
        try:
            CELL_FNS[cell]()
        except Exception as e:
            print(f"  [{cell}] FAILED: {type(e).__name__}: {e}", flush=True)
            raise
        print(f"  [{cell}] total {time.time()-t0:.0f}s\n", flush=True)


if __name__ == "__main__":
    main()
