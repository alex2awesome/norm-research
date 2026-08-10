#!/usr/bin/env python3
"""Harvest one V3 full-grid cell into results/v3_grid_<slug>.json.

Emits, on E = the cell's evaluation-valid rows (eval + test pooled):
    auc_E, auc_eval, auc_test, n_E, n_groups_E
    the manifest's k and block placement (APPEND/PREPEND), truncation rate
    group-level paired bootstraps (2,000 draws, resampling the cell's canonical
    GROUPING UNIT) of (V3 - T) and (V3 - VA_nl)

ROW-ORDER SAFETY (the whole point of this file)
-----------------------------------------------
The per-row bank vector `results/<slug>_va_nl_oof_mean3.npy` is stored in the
LAYER-1 MATRIX row order, which is NOT the population/join order for most
cells.  Differencing V3 against a misaligned bank vector silently fabricates a
CI, so every bank vector must pass a REPRODUCTION GATE before it is used:

    AUC(y_full, <slug>_va_nl_oof_seed0.npy)  ==  the cell's published
    nonlinear.VA.seed_aucs["0"]   (exact identity, tol 1e-6)

seed0 is used rather than mean3 because mean3's AUC is not the mean of the
per-seed AUCs, so mean3 can only ever be checked loosely.  If the gate fails,
this script REFUSES to emit boot_v3_minus_va_nl and records why.  A missing CI
is fine; a CI against a misaligned vector is a fabricated number.

The same discipline applies to T: its per-row column is joined by key and
asserted judgement-equal elementwise before differencing.

Usage:
    python3 harvest_v3_grid_cell.py --slug peer_curation
    python3 harvest_v3_grid_cell.py --slug peer_curation --gate-only
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
FUS = HERE.parent
TD = FUS.parent
REPO = TD.parent.parent
RESULTS = TD / "results"
DATA = FUS / "dense_data"
CLOSURE = TD / "closure"
JOINS = FUS / "dense_joins"

NBOOT = 2000
GATE_TOL = 1e-6

K_CAVEAT = (
    "k=20 is the V3 audit's production recipe but it FAILED its confirmation "
    "cell: cap_finalist k20 .6431 vs k10 .6707, delta -.0276 [-.058,+.003] "
    "P(>0)=.043 (2026-08-08). k in {10,20} is a wash with cell-specific sign. "
    "Do NOT read a V3-vs-bank gap smaller than ~.03 as real."
)


def load_module(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- bootstrap --
def group_paired_boot(y, pred_a, pred_b, groups, n_boot=NBOOT, seed=12345):
    """AUC(pred_a) - AUC(pred_b), resampling GROUPS with replacement.

    Same estimator as fusion/direction1_mirror.py: draw |G| groups with
    replacement, concatenate their member rows, recompute both AUCs on that
    resampled row set, difference.
    """
    y = np.asarray(y)
    pa, pb = np.asarray(pred_a, dtype=float), np.asarray(pred_b, dtype=float)
    groups = np.asarray(groups, dtype=object)
    base = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    uq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uq}
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        pick = rng.choice(uq, size=len(uq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        ds.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
    ds = np.asarray(ds)
    if not len(ds):
        return None
    return {
        "estimate": float(base),
        "ci95": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
        "p_gt_0": float((ds > 0).mean()),
        "n_boot_used": int(len(ds)),
        "n_groups": int(len(uq)),
        "resampled_unit": "group",
    }


# ------------------------------------------------------- bank vector + gate --
NC_V2_CELLS = {"nc_responded": "responded", "nc_outcome": "outcome", "nc_agree": "agree"}


def published_seed0(slug):
    """The cell's published nonlinear VA seed-0 AUC, whatever file/schema holds it.

    Three schemas are in the wild:
      layer1 (peer/press/nc):  nonlinear.VA.seed_aucs["0"]        -> float
      ledger (scale-up wave C): nonlinear.VA["0"]["auc"]          -> float
      either:                   ledger.VA_nl_seeds[0]             -> float
    """
    # v2 regeneration (N&C cells, registry 2026-08-10 landmine fix) carries the
    # new gateable reference seed AUCs -- check it before the original file.
    candidates = [RESULTS / f"{slug}_layer1_v2.json"] if slug in NC_V2_CELLS else []
    candidates += [RESULTS / f"{slug}_layer1.json", RESULTS / f"{slug}_ledger.json"]
    for path in candidates:
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        nl = (d.get("nonlinear") or {}).get("VA") or {}
        sa = nl.get("seed_aucs")
        if isinstance(sa, dict) and "0" in sa:
            v = sa["0"]
            return (float(v["auc"]) if isinstance(v, dict) else float(v)), \
                f"{path.name}:nonlinear.VA.seed_aucs['0']"
        if isinstance(nl.get("0"), dict) and "auc" in nl["0"]:
            return float(nl["0"]["auc"]), f"{path.name}:nonlinear.VA['0'].auc"
        seeds = (d.get("ledger") or {}).get("VA_nl_seeds")
        if isinstance(seeds, list) and seeds:
            return float(seeds[0]), f"{path.name}:ledger.VA_nl_seeds[0]"
        # SINGLE-FIT cells (peer_verdict): the Layer-1 run recorded one VA_nl
        # fit, not a seed sweep, so the seed-0 array IS that fit. Both keys are
        # the same number; the exact-identity gate still does the real work --
        # a wrong reference fails loudly rather than passing quietly.
        va_auc = nl.get("auc")
        if isinstance(va_auc, (int, float)):
            return float(va_auc), f"{path.name}:nonlinear.VA.auc (single-fit cell)"
        va_nl = (d.get("ledger") or {}).get("VA_nl")
        if isinstance(va_nl, (int, float)):
            return float(va_nl), f"{path.name}:ledger.VA_nl (single-fit cell)"
    return None, None


def bank_index(slug):
    """(list of bank row ids in NPY ORDER, y in that order, provenance str).

    Returns (None, None, reason) when the cell has no recoverable index.
    """
    # --- N&C v2 regeneration: index = the ids vector saved alongside the ----
    # v2 OOF arrays themselves (registry 2026-08-10 landmine fix -- see
    # nc_layer1_stack.py::NCData.rows_for and results/nc_oof_regeneration.json).
    # This covers nc_responded too, which the maps_batch1 adapter below never
    # indexed. Falls through to the adapter path (still correct, same sorted
    # order) if the v2 ids file is missing.
    if slug in NC_V2_CELLS:
        ids_path = RESULTS / f"{slug}_oof_ids_v2.npy"
        if ids_path.exists():
            ids = [str(x) for x in np.load(ids_path, allow_pickle=True)]
            m = load_module(TD / "nc_layer1_stack.py", "nc_v2_bank")
            data = m.NCData()
            y_by = {"responded": data.y_resp_by_id,
                     "outcome": data.y_out_by_id,
                     "agree": data.y_agr_by_id}[NC_V2_CELLS[slug]]
            y = np.array([y_by[d] for d in ids], dtype=int)
            return ids, y, f"results/{slug}_oof_ids_v2.npy (regenerated, sorted, ids-carried)"

    # --- scale-up wave C cells: index = bank meta item_ids -------------------
    scaleupC = {
        "jokes_community": "jokes_community",
        "aops_curation": "aops_curation",
        "homepage_curation": "homepage_curation",
        "mathse_accepted_verdict": "mathse_multiy",
        "mathse_vote_score": "mathse_multiy",
    }
    if slug in scaleupC:
        bank = scaleupC[slug]
        meta = json.loads((REPO / "outputs/va_gemma_banks_scaleupC" /
                           f"{bank}_meta.json").read_text())
        ids = [str(x) for x in meta["item_ids"]]
        jf = JOINS / f"{slug}_join.csv.gz"
        if not jf.exists():
            return None, None, f"no dense_joins file for {slug}"
        j = pd.read_csv(jf)
        j["row_id"] = j.row_id.astype(str)
        keep = set(j.row_id)
        # the two mathse y's share ONE matrix; the npy is the kept subset in
        # item_ids order
        ids = [k for k in ids if k in keep]
        y = j.set_index("row_id").loc[ids, "judgement"].to_numpy()
        return ids, y, (f"outputs/va_gemma_banks_scaleupC/{bank}_meta.json item_ids"
                        f" (subset to the {slug} population, item_ids order)")

    # --- press_verdict: index = A-cache ids ---------------------------------
    if slug == "press_verdict":
        z = np.load(RESULTS / "press_verdict_pr_A_k3_scores_CACHE.npz",
                    allow_pickle=True)
        return ([str(x) for x in z["ids"]], np.asarray(z["y"], dtype=int),
                "results/press_verdict_pr_A_k3_scores_CACHE.npz ids")

    # --- maps_batch1 adapter cells: index = adapter ids ---------------------
    b1 = {"peer_curation", "peer_revealed", "nc_outcome", "nc_agree",
          "cap_crowd", "cap_finalist"}
    if slug in b1:
        m = load_module(CLOSURE / "maps_batch1" / "cells.py", "v3h_b1")
        d = m.load(slug)
        return ([str(x) for x in d["ids"]], np.asarray(d["y"], dtype=int),
                f"closure/maps_batch1/cells.py load('{slug}') ids")

    if slug in {"hashtagwars_verdict", "style_inv_toptier"}:
        m = load_module(CLOSURE / "maps_hw_si" / "cells.py", "v3h_hw")
        d = m.load(slug)
        return ([str(x) for x in d["ids"]], np.asarray(d["y"], dtype=int),
                f"closure/maps_hw_si/cells.py load('{slug}') ids")

    # --- peer_verdict: index = stage4_readout pop ntitle order --------------
    # Same adapter fusion/v3_grid/build_v3_cell.py::_peer_verdict() uses for
    # bank_ids and fusion/direction1_mirror.py::do_peer_verdict() uses for the
    # Layer-1 matrix, so the OOF npy is in exactly this order. The seed-0
    # identity gate below is what actually certifies that.
    if slug == "peer_verdict":
        sys.path.insert(0, str(CLOSURE))
        SR4 = load_module(CLOSURE / "stage4_readout.py", "v3h_sr4")
        pop, _split, _dsplit, _XA1, _XB1, _a1, _b1, _summary = SR4.build_blocks()
        return ([str(x) for x in pop["ntitle"]], np.asarray(pop["y"], dtype=int),
                "closure/stage4_readout.py build_blocks() pop['ntitle'] order")

    return None, None, f"no bank index rule registered for {slug}"


def gated_bank_vector(slug):
    """Per-row VA_nl bank prediction as a Series keyed by bank id, or None.

    Second element of the tuple is a dict describing the gate outcome.
    """
    # N&C v2 regeneration (registry 2026-08-10 landmine fix): prefer the
    # row-order-safe v2 arrays when they exist; fall back to the original
    # filenames otherwise (which will correctly fail the gate below for any
    # N&C cell not yet regenerated).
    if slug in NC_V2_CELLS and (RESULTS / f"{slug}_va_nl_oof_seed0_v2.npy").exists():
        npy_mean = RESULTS / f"{slug}_va_nl_oof_mean3_v2.npy"
        npy_seed0 = RESULTS / f"{slug}_va_nl_oof_seed0_v2.npy"
    else:
        npy_mean = RESULTS / f"{slug}_va_nl_oof_mean3.npy"
        npy_seed0 = RESULTS / f"{slug}_va_nl_oof_seed0.npy"
    gate = {"attempted": True}
    # Some cells (peer_verdict) never had a mean3 array written -- only seed0.
    # Coordinator relay 5(3): for those, difference against the SEED0 vector
    # (gated first), never mean3. Falling through to "no bank vector" would
    # throw away a CI that the exact-identity gate can fully certify.
    vec_path, vec_kind = npy_mean, "mean3"
    if not npy_mean.exists():
        if npy_seed0.exists():
            vec_path, vec_kind = npy_seed0, "seed0"
        else:
            gate.update(passed=False,
                        reason=f"no per-row bank vector {npy_mean.name} (and no seed0)")
            return None, gate
    gate["vector_used"] = vec_kind
    idx, y_full, prov = bank_index(slug)
    if idx is None:
        gate.update(passed=False, reason=prov)
        return None, gate
    vec = np.load(vec_path)
    if len(vec) != len(idx):
        gate.update(passed=False,
                    reason=f"length mismatch: npy {len(vec)} vs index {len(idx)}")
        return None, gate

    ref, ref_src = published_seed0(slug)
    if npy_seed0.exists() and ref is not None:
        obs = float(roc_auc_score(y_full, np.load(npy_seed0)))
        ok = abs(obs - ref) < GATE_TOL
        gate.update(passed=bool(ok), observed_seed0_auc=obs,
                    published_seed0_auc=ref, published_source=ref_src,
                    tol=GATE_TOL, index_provenance=prov)
        if not ok:
            gate["reason"] = ("bank vector failed the row-order reproduction "
                              "gate: AUC(y, oof_seed0) does not reproduce the "
                              "published seed-0 VA_nl")
            return None, gate
    else:
        # cannot run the exact identity; fall back to a loose check on mean3
        gate.update(passed=False, index_provenance=prov,
                    reason="no seed0 npy and/or no published seed_aucs['0'] to "
                           "gate against; refusing to difference ungated")
        return None, gate

    s = pd.Series(vec, index=idx)
    gate[f"auc_{vec_kind}_full_population"] = float(roc_auc_score(y_full, vec))
    return s, gate


# ------------------------------------------------------------- T per row -----
def t_vector(slug, dids):
    """Per-row dense T prediction aligned to `dids`, plus provenance."""
    jf = JOINS / f"{slug}_join.csv.gz"
    if jf.exists():
        j = pd.read_csv(jf)
        j["row_id"] = j.row_id.astype(str)
        cols = [c for c in j.columns if c.startswith("dense_prob")]
        j = j.set_index("row_id")
        miss = [d for d in dids if d not in j.index]
        if not miss:
            sub = j.loc[list(dids)]
            vals = sub[cols].to_numpy(dtype=float)
            ens = np.nanmean(vals, axis=1)
            if not np.isnan(ens).any():
                return ens, sub["judgement"].to_numpy(), (
                    f"fusion/dense_joins/{slug}_join.csv.gz "
                    f"{'mean of ' + ','.join(cols) if len(cols) > 1 else cols[0]}")
    # adapter path
    for mod, alias, cells in (
        (CLOSURE / "maps_batch1" / "cells.py", "v3t_b1",
         {"peer_curation", "peer_revealed", "nc_outcome", "nc_agree",
          "cap_crowd", "cap_finalist"}),
        (CLOSURE / "maps_hw_si" / "cells.py", "v3t_hw",
         {"hashtagwars_verdict", "style_inv_toptier"}),
    ):
        if slug in cells:
            m = load_module(mod, alias)
            d = m.load(slug)
            ids = [str(x) for x in d["ids"]]
            dense = d.get("dense")
            if dense is None:
                return None, None, f"adapter {slug} carries no dense column"
            dense = np.asarray(dense, dtype=float)
            if dense.ndim > 1:          # several dense seeds -> ensemble
                dense = np.nanmean(dense, axis=1)
            s = pd.Series(dense, index=ids)
            yy = pd.Series(np.asarray(d["y"], dtype=int), index=ids)
            miss = [x for x in dids if x not in s.index]
            if miss:
                return None, None, (f"{len(miss)} E rows absent from the adapter "
                                    f"dense column")
            vals = s.loc[list(dids)].to_numpy()
            if np.isnan(vals).any():
                # the adapter leaves NaN on dense-TRAIN rows; a NaN here means an
                # E row of this V3 arm was a training row for the dense T arm
                return None, None, (
                    f"{int(np.isnan(vals).sum())} of {len(vals)} E rows have no "
                    f"dense probability (dense-train rows) -- T not comparable "
                    f"on this row set")
            return (vals, yy.loc[list(dids)].to_numpy(),
                    f"closure adapter load('{slug}') dense seed-ensemble column")

    # ---- cells whose dense T lives in a dedicated, bank-row-aligned CSV ----
    # (the same files fusion/direction1_mirror.py differences against; each is
    # asserted row-aligned to the cell's bank id vector before use)
    DEDICATED = {
        "peer_verdict": (CLOSURE / "peer_verdict_dense_preds.csv", "ntitle"),
        "nc_responded": (CLOSURE / "nc_responded" / "nc_responded_dense_preds_aligned.csv",
                         "doc_id"),
    }
    if slug in DEDICATED:
        path, keycol = DEDICATED[slug]
        if not path.exists():
            return None, None, f"{path.name} not found for {slug}"
        dp = pd.read_csv(path)
        if keycol not in dp.columns:
            return None, None, f"{path.name} has no `{keycol}` column"
        dp[keycol] = dp[keycol].astype(str)
        # a few keys repeat (peer_verdict: ~30 duplicate ntitles); pairing on an
        # ambiguous key would fabricate the join, so drop them from the pairing
        dp = dp.drop_duplicates(keycol, keep=False).set_index(keycol)
        miss = [x for x in dids if x not in dp.index]
        if len(miss) > 0.02 * len(dids):
            return None, None, (f"{len(miss)} of {len(dids)} E rows have no "
                                f"unambiguous row in {path.name} (>2%) -- "
                                f"refusing to difference on a thinned row set")
        # A handful of keys repeat (peer_verdict: ~30 duplicate ntitles out of
        # 6,030). Pairing those rows would mean guessing which dense row belongs
        # to which V3 row, so emit NaN and let the caller drop exactly them.
        sub = dp.reindex(list(dids))
        vals = sub["dense_prob"].to_numpy(dtype=float)
        yy = sub["judgement"].to_numpy(dtype=float)
        # keep y aligned & comparable on the surviving rows; NaN y only where
        # the key was ambiguous, which is precisely where vals is NaN too
        yy = np.where(np.isnan(vals), np.nan, yy)
        return (vals, yy,
                f"closure/{path.name} dense_prob, joined on {keycol}; "
                f"{len(miss)} ambiguous/duplicate-key row(s) dropped from the pairing")

    return None, None, f"no T source registered for {slug}"


# ------------------------------------------------------------------ main -----
def harvest(slug, write=True):
    d = DATA / f"v3grid_{slug}"
    man = json.loads((d / "manifest.json").read_text())
    run = d / "rm_out_seed42"

    frames = []
    for sp in ("eval", "test"):
        pr = pd.read_csv(run / f"preds_{sp}.csv")
        sf = pd.read_csv(d / "split" / f"{sp}.csv")
        assert len(pr) == len(sf), f"{slug}/{sp}: preds {len(pr)} vs split {len(sf)}"
        assert (pr.judgement.values == sf.judgement.values).all(), \
            f"{slug}/{sp}: preds/split judgement mismatch -- refusing to join"
        frames.append(pd.DataFrame({
            "did": sf["did"].astype(str).values,
            "group": sf["group"].astype(str).values,
            "y": sf["judgement"].astype(int).values,
            "v3": pr["prob"].astype(float).values,
            "split": sp,
        }))
    E = pd.concat(frames, ignore_index=True)

    out = {
        "cell": slug,
        "arm": man.get("arm", f"v3grid_{slug}"),
        "k": man.get("k"),
        "block_placement": man.get("block_placement"),
        "block_placement_reason": man.get("block_placement_reason"),
        "selection_split": man.get("selection_split"),
        "group_column": man.get("group_column"),
        "n_E": int(len(E)),
        "n_eval": int((E.split == "eval").sum()),
        "n_test": int((E.split == "test").sum()),
        "n_groups_E": int(E.group.nunique()),
        "pos_rate_E": float(E.y.mean()),
        "auc_E": float(roc_auc_score(E.y, E.v3)),
        "auc_eval": float(roc_auc_score(E[E.split == "eval"].y, E[E.split == "eval"].v3)),
        "auc_test": float(roc_auc_score(E[E.split == "test"].y, E[E.split == "test"].v3)),
        "truncation": man.get("truncation"),
        "truncation_rate_with_block": (man.get("truncation") or {}).get(
            "with_block", {}).get("rate_over_max_len"),
        "truncation_rate_raw_text_only": (man.get("truncation") or {}).get(
            "raw_text_only", {}).get("rate_over_max_len"),
        "eval_test_row_sets": man.get("eval_test_row_sets", {}).get("status"),
        "join_overall_coverage": (man.get("join") or {}).get("overall_coverage"),
        "k_caveat": K_CAVEAT,
        "recipe": man.get("recipe"),
        "trained_by": "v3chain-gpu2 (sk3 GPU 2), frozen dense-standard recipe, seed 42",
    }

    # -------- the block is not always free: truncation confound -------------
    raw_r = out["truncation_rate_raw_text_only"]
    blk_r = out["truncation_rate_with_block"]
    if raw_r is not None and blk_r is not None and (blk_r - raw_r) > 0.05:
        out["truncation_confound"] = {
            "present": True,
            "raw_text_only_rate": raw_r,
            "with_block_rate": blk_r,
            "delta": round(blk_r - raw_r, 4),
            "block_tokens": (man.get("truncation") or {}).get("block_tokens"),
            "caveat": (
                "The k=20 criteria block DISPLACES document text on this cell: "
                f"truncation rises from {raw_r:.3f} (raw text) to {blk_r:.3f} "
                "(with block) at max_len. Placement is PREPEND so the block "
                "always survives, but on those rows the tail of the document is "
                "cut that the raw/T arm kept. The V3 and T arms therefore did "
                "NOT see the same text. A V3 <= T result on this cell is NOT "
                "interpretable as 'the criteria block does not help' without a "
                "matched-n raw-text control arm; the confound runs in the "
                "direction of making V3 look worse."),
        }
    else:
        out["truncation_confound"] = {"present": False, "raw_text_only_rate": raw_r,
                                      "with_block_rate": blk_r}

    # -------- reference scalars from the bank stack -------------------------
    vs = RESULTS / f"vat_stack_{slug}.json"
    T_ref = VA_ref = None
    if vs.exists():
        st = json.loads(vs.read_text())
        arms = st.get("arms") or {}
        arm = st.get("ensemble") or (arms[list(arms)[0]] if arms else {})
        T_ref, VA_ref = arm.get("T_E"), arm.get("VA_nl_E_mean")
        out.update({
            "T_E_reference": T_ref, "VA_nl_E_mean_reference": VA_ref,
            "vat_stack_n_E": st.get("n_E"), "vat_stack_n_groups_E": st.get("n_groups_E"),
            "vat_stack_source": str(vs),
            "row_set_matches_vat_stack": (st.get("n_E") == len(E)),
        })
    else:
        out["vat_stack_source"] = None
        out["note_no_vat_stack"] = (
            f"no results/vat_stack_{slug}.json; T reference taken from the "
            f"cell's own dense join / ledger instead")

    # -------- T per row + (V3 - T) bootstrap --------------------------------
    tvec, tY, tprov = t_vector(slug, E.did.tolist())
    out["T_source"] = tprov
    ET = E
    if tvec is not None and np.isnan(np.asarray(tvec, dtype=float)).any():
        # only the dedicated-CSV path emits NaN, and only on ambiguous keys
        keep = ~np.isnan(np.asarray(tvec, dtype=float))
        out["n_dropped_ambiguous_T_rows"] = int((~keep).sum())
        tvec = np.asarray(tvec, dtype=float)[keep]
        tY = np.asarray(tY, dtype=float)[keep]
        ET = E[keep].reset_index(drop=True)
    if tvec is None:
        out["boot_v3_minus_T"] = None
        out["boot_v3_minus_T_reason"] = tprov
    elif not (np.asarray(tY).astype(int) == ET.y.values).all():
        out["boot_v3_minus_T"] = None
        out["boot_v3_minus_T_reason"] = (
            "T join UNSAFE: judgement disagrees elementwise between the V3 "
            "split rows and the T source")
    else:
        # T_same_rows: the dense arm's per-row predictions restricted to EXACTLY
        # the rows this V3 arm was evaluated on. On SUBSET_OF_ORIGINAL cells the
        # bank does not cover every dense row, so the dense arm's PUBLISHED
        # full-split AUC is computed on a strictly larger row set and must never
        # be differenced against auc_E. The two are recorded separately.
        out["T_same_rows_at_E"] = float(roc_auc_score(ET.y, tvec))
        out["T_at_E_observed"] = out["T_same_rows_at_E"]   # back-compat alias
        out["T_published_full_split"] = T_ref
        out["T_row_set_note"] = (
            "T_same_rows_at_E is computed on this V3 arm's own E rows. "
            + ("This cell is SUBSET_OF_ORIGINAL (join coverage "
               f"{out.get('join_overall_coverage')}): the dense arm's published "
               "full-split AUC covers MORE rows and is NOT the right comparand -- "
               "difference against T_same_rows_at_E only."
               if out.get("eval_test_row_sets") != "IDENTICAL" else
               "Row set is IDENTICAL to the original dense split, so the "
               "same-rows and published T refer to the same rows."))
        out["boot_v3_minus_T"] = group_paired_boot(ET.y.values, ET.v3.values,
                                                   tvec, ET.group.values)

    # -------- VA_nl per row + (V3 - VA_nl) bootstrap ------------------------
    bank, gate = gated_bank_vector(slug)
    out["bank_gate"] = gate
    if bank is None:
        out["boot_v3_minus_VA_nl"] = None
        out["boot_v3_minus_VA_nl_reason"] = gate.get("reason")
    else:
        # A few cells key the bank by a non-unique id (peer_verdict: ~30
        # duplicate ntitles). `.loc` on a duplicated index silently EXPANDS the
        # row set, so drop ambiguous keys and pair only on the unambiguous ones.
        n_dup_keys = int(bank.index.duplicated().sum())
        if n_dup_keys:
            bank = bank[~bank.index.duplicated(keep=False)]
            out["n_ambiguous_bank_keys_dropped"] = n_dup_keys
        miss = [x for x in E.did if x not in bank.index]
        EB = E[~E.did.isin(set(miss))].reset_index(drop=True) if miss else E
        if len(miss) > 0.02 * len(E):
            out["boot_v3_minus_VA_nl"] = None
            out["boot_v3_minus_VA_nl_reason"] = (
                f"{len(miss)} of {len(E)} E rows absent/ambiguous in the gated "
                f"bank vector (>2%) -- refusing to difference on a thinned row set")
        else:
            if miss:
                out["n_dropped_ambiguous_bank_rows"] = len(miss)
            bv = bank.loc[EB.did.tolist()].to_numpy(dtype=float)
            out["VA_nl_oof_at_E_observed"] = float(roc_auc_score(EB.y, bv))
            out["boot_v3_minus_VA_nl"] = group_paired_boot(
                EB.y.values, EB.v3.values, bv, EB.group.values)
            out["boot_v3_minus_VA_nl_note"] = (
                "differenced against the per-row OOF bank vector restricted to "
                "E (VA_nl_oof_at_E_observed). The scalar VA_nl_E_mean_reference "
                "comes from a stack REFIT ON E, so the two differ slightly; the "
                "CI is internally consistent with the vector actually used.")

    if write:
        p = RESULTS / f"v3_grid_{slug}.json"
        p.write_text(json.dumps(out, indent=2, default=str))
        print("wrote", p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    if a.gate_only:
        bank, gate = gated_bank_vector(a.slug)
        print(json.dumps({a.slug: gate}, indent=2, default=str))
        return
    o = harvest(a.slug, write=not a.no_write)
    keys = ["cell", "k", "block_placement", "n_E", "n_groups_E", "auc_E",
            "auc_eval", "auc_test", "T_E_reference", "T_at_E_observed",
            "VA_nl_E_mean_reference", "VA_nl_oof_at_E_observed"]
    print(json.dumps({k: o.get(k) for k in keys}, indent=2, default=str))
    for b in ("boot_v3_minus_T", "boot_v3_minus_VA_nl"):
        print(b, "=", json.dumps(o.get(b), default=str),
              "" if o.get(b) else "REASON: " + str(o.get(b + "_reason")))


if __name__ == "__main__":
    main()
