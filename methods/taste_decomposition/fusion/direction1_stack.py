#!/usr/bin/env python3
"""Direction 1 of the V+A+T fusion directions (2026-08-07 task): add the dense
model's per-row OUT-OF-SAMPLE prediction as ONE additional column to the V+A
score matrix and refit the frozen Layer-1 linear + HistGB stacks.

Cells: the three where the dense standard FAILS to beat the articulated bank —
  cap_crowd          (same-rows T .5554 vs VA_nl .6656)
  cap_finalist       (same-rows T .6124 vs VA_nl .6800)
  style_inv_toptier  (T .6343 vs VA_nl .6651)

LEAKAGE RULE (critical): the dense column must be out-of-sample everywhere it
is used, so the ENTIRE analysis is restricted to the evaluation-valid rows E =
rows outside the dense model's own training split (dense_split in {eval,test}).
On E the dense per-row probability was produced by a model that never saw any
E row in training. (Selection caveat: the dense checkpoint was selected on its
eval split, which is inside E — the same caveat already attached to the
registry T / same-rows T; matched across arms, noted, not fixable here.)

Per cell, ALL on the same rows E, same GroupKFold(5) folds:
  T_E        = AUC(y_E, dense_prob)
  VA_lin_E   = frozen family linear protocol refit on E (grouped OOF)
  VA_nl_E    = frozen HistGB protocol refit on E (grouped OOF, seeds 0-2 mean)
  VAT_lin_E  = same linear, matrix = [V, A, dense_prob]
  VAT_nl_E   = same HistGB, matrix = [V, A, dense_prob], seeds 0-2 mean
Secondary (context): the FULL-population OOF predictions (linear + GBM) of the
Layer-1 protocol evaluated on E only ("VA@E, full-fit") — no retraining
handicap, but not apples-to-apples with the E-refit VAT (more train rows).

style_inv has 3 dense seeds (42,1,2); everything dense-column-dependent is run
once per dense seed and reported per-seed + mean. Captions have one dense model
each (dense_llama/{crowd,finalist}/rm_out).

CPU only. Usage: python3 direction1_stack.py [--cell cap_crowd] [--skip-fullfit]
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
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
TD = HERE.parent
REPO = TD.parents[1]
RESULTS = HERE
SLIM = TD / "closure" / "samerows_preds"
SI_DS = REPO / "datasets/humor/style_invitational/dense_standard"


def load_module(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


L1 = load_module(TD / "layer1_gemma_cells.py", "l1_fusion")
capagg = sys.modules["capagg_taste_decomp"]  # loaded by layer1_gemma_cells


# ------------------------------------------------------------- dense preds --
def caption_dense(cell_slug: str):
    """Per-row dense preds for cap cells from the same-rows rescore batch
    (methods/taste_decomposition/closure/samerows_preds/*_slim.csv)."""
    name = {"cap_crowd": "cap_crowd", "cap_finalist": "cap_finalist"}[cell_slug]
    df = pd.read_csv(SLIM / f"{name}_dense_preds_slim.csv")
    df["id"] = df["id"].astype(str)
    return {r.id: (float(r.dense_prob), str(r.dense_split), bool(r.in_dense_train))
            for r in df.itertuples()}


def si_dense():
    """Per-row dense preds for style_inv_toptier, per dense seed.

    preds_{eval,test}.csv (judgement, prob, group) are index-aligned with
    split/{eval,test}.csv (text, judgement, group). Join split rows back to the
    jsonl population (which carries the npz row id = sha1(week\\0i\\0entry)[:20],
    i = jsonl line index) by (group, ctx-text, occurrence rank) — both the
    data.csv build and the split csvs preserve jsonl order, so occurrence rank
    within (group, text) disambiguates duplicate entries.
    Returns {seed: {id20: prob}}, plus the split of each id.
    """
    import hashlib
    jl = REPO / "datasets/humor/style_invitational/style_invitational.jsonl"
    rows = [json.loads(l) for l in open(jl) if l.strip()]
    keyed = {}  # (group, text, k) -> id20
    seen = {}
    for i, r in enumerate(rows):
        g = str(r["week_id"])
        text = f'CONTEST PROMPT: {r["contest_prompt"]}\n\nENTRY: "{r["entry_text"]}"'
        id20 = hashlib.sha1(f"{r['week_id']}\0{i}\0{r['entry_text']}".encode()).hexdigest()[:20]
        k = seen.get((g, text), 0)
        seen[(g, text)] = k + 1
        keyed[(g, text, k)] = (id20, 1 if r["tier"] in ("winner", "runnerup") else 0)

    out = {}
    split_of = {}
    for seed in ("42", "1", "2"):
        probs_by_id = {}
        for sp in ("eval", "test"):
            sdf = pd.read_csv(SI_DS / "split" / f"{sp}.csv")
            pdf = pd.read_csv(HERE / "preds" / f"si_seed{seed}_preds_{sp}.csv")
            assert len(sdf) == len(pdf), (sp, len(sdf), len(pdf))
            assert (sdf.judgement.values == pdf.judgement.values).all(), f"{sp}: y misalignment"
            occ = {}
            for (text, yj, g), prob in zip(
                    sdf[["text", "judgement", "group"]].itertuples(index=False),
                    pdf["prob"].values):
                g = str(g)
                k = occ.get((g, text), 0)
                occ[(g, text)] = k + 1
                id20, y_pop = keyed[(g, text, k)]
                assert int(yj) == y_pop, "label mismatch on join"
                probs_by_id[id20] = float(prob)
                split_of[id20] = sp
        out[seed] = probs_by_id
    ns = {s: len(v) for s, v in out.items()}
    assert len(set(ns.values())) == 1, ns
    return out, split_of


# ----------------------------------------------------------------- fitting --
def refit_cell_on_E(family, Vc, Ac, dense_col, y, groups, gbm_seeds=(0, 1, 2)):
    """Frozen Layer-1 protocol on the restricted population E.

    family2 (captions): clean_cols was already applied (to the E submatrix,
    mirroring the protocol's population-level guard); linear = family2 linear,
    GBM = gbm_oof_raw. family1 (SI): raw matrices, per-fold imputer paths.
    Returns dict of AUCs + OOF arrays.
    """
    n = len(y)
    folds = L1.outer_folds(n, groups, n_splits=5)
    VA = np.column_stack([Vc, Ac]) if Vc.shape[1] and Ac.shape[1] else (Vc if Vc.shape[1] else Ac)
    VAT = np.column_stack([VA, dense_col])
    linfn = L1.linear_oof_family1 if family == "family1" else L1.linear_oof_family2
    gbmfn = L1.gbm_oof_family1 if family == "family1" else L1.gbm_oof_raw

    res = {}
    for key, M in (("VA", VA), ("VAT", VAT)):
        auc, oof = linfn(M, y, groups, folds)
        res[f"{key}_lin"] = auc
        res[f"_oof_{key}_lin"] = oof
        seed_aucs, oof0 = [], None
        for s in gbm_seeds:
            r = gbmfn(M, y, groups, folds, s)
            seed_aucs.append(r["auc"])
            if s == 0:
                oof0 = r["oof"]
        res[f"{key}_nl_seeds"] = seed_aucs
        res[f"{key}_nl_mean"] = float(np.mean(seed_aucs))
        res[f"_oof_{key}_nl0"] = oof0
    res["n_folds"] = len(folds)
    return res


def paired_boot(y, pred_a, pred_b, n_boot=2000, seed=12345):
    """AUC(pred_a) - AUC(pred_b), row-level paired bootstrap."""
    point = float(roc_auc_score(y, pred_a) - roc_auc_score(y, pred_b))
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
        deltas.append(float(roc_auc_score(ys, pred_a[idx]) - roc_auc_score(ys, pred_b[idx])))
    deltas = np.array(deltas)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"estimate": point, "ci95": [float(lo), float(hi)],
            "p_gt_0": float((deltas > 0).mean()), "n_boot_used": int(len(deltas))}


def fullfit_at_E(cell, mask_E):
    """Secondary: full-population Layer-1 OOF predictions evaluated on E only."""
    d = L1.CELLS[cell]["loader"]()
    mats, y, groups, family = d["mats"], d["y"], d["groups"], d["family"]
    folds = L1.outer_folds(len(y), groups, n_splits=5)
    linfn = L1.linear_oof_family1 if family == "family1" else L1.linear_oof_family2
    gbmfn = L1.gbm_oof_family1 if family == "family1" else L1.gbm_oof_raw
    _, lin_oof = linfn(mats["VA"], y, groups, folds)
    nl_aucs_E = []
    for s in (0, 1, 2):
        r = gbmfn(mats["VA"], y, groups, folds, s)
        nl_aucs_E.append(float(roc_auc_score(y[mask_E], r["oof"][mask_E])))
    return {
        "VA_lin_fullfit_at_E": float(roc_auc_score(y[mask_E], lin_oof[mask_E])),
        "VA_nl_fullfit_at_E_seeds": nl_aucs_E,
        "VA_nl_fullfit_at_E_mean": float(np.mean(nl_aucs_E)),
    }


# ------------------------------------------------------------------- cells --
def run_caption(cell, skip_fullfit=False):
    dense = caption_dense(cell)
    c = L1._caption_pools()
    id_key = "crowd_ids" if cell == "cap_crowd" else "hardneg_ids"
    y_key = "y_crowd" if cell == "cap_crowd" else "y_fin"
    ids_all = sorted(d for d in c[id_key] if d in c["X_by_id"])
    miss = [d for d in ids_all if d not in dense]
    assert not miss, f"{cell}: {len(miss)} rows missing dense preds"
    ids_E = [d for d in ids_all if not dense[d][2]]  # in_dense_train == False
    y = np.array([c[y_key][d] for d in ids_E])
    A = np.array([c["X_by_id"][d] for d in ids_E], dtype=float)
    V = np.array([c["V_by_id"][d] for d in ids_E], dtype=float)
    groups = np.array([c["contest_by_id"][d] for d in ids_E])
    dcol = np.array([dense[d][0] for d in ids_E])
    Ac, _ = capagg.clean_cols(A)
    Vc, _ = capagg.clean_cols(V)

    res = refit_cell_on_E("family2", Vc, Ac, dcol, y, groups)
    T_E = float(roc_auc_score(y, dcol))
    out = {
        "cell": cell, "direction": 1,
        "population": "evaluation-valid rows E = dense_split in {eval,test} (in_dense_train == False)",
        "n_E": int(len(y)), "pos_rate_E": float(y.mean()),
        "n_groups_E": int(len(np.unique(groups))),
        "n_features": {"V": int(Vc.shape[1]), "A": int(Ac.shape[1]), "VAT_extra": 1},
        "T_E": T_E,
        "VA_lin_E": res["VA_lin"], "VA_nl_E_seeds": res["VA_nl_seeds"], "VA_nl_E_mean": res["VA_nl_mean"],
        "VAT_lin_E": res["VAT_lin"], "VAT_nl_E_seeds": res["VAT_nl_seeds"], "VAT_nl_E_mean": res["VAT_nl_mean"],
        "boot_VAT_nl_minus_VA_nl": paired_boot(y, res["_oof_VAT_nl0"], res["_oof_VA_nl0"]),
        "boot_VAT_nl_minus_T": paired_boot(y, res["_oof_VAT_nl0"], dcol),
        "dense_preds_source": f"closure/samerows_preds/{cell}_dense_preds_slim.csv",
    }
    if not skip_fullfit:
        mask_E = np.array([not dense[d][2] for d in ids_all])
        out.update(fullfit_at_E(cell, mask_E))
    return out


def run_style_inv(skip_fullfit=False):
    probs_by_seed, split_of = si_dense()
    meta, A, V, groups, _ = L1.rvg.load_bank("style_invitational")
    y_all = np.array(meta["ys"]["top_tier"])
    ids = list(meta["item_ids"])
    in_E = np.array([d in probs_by_seed["42"] for d in ids])
    idx_E = np.flatnonzero(in_E)
    y = y_all[idx_E]
    Ae, Ve, ge = A[idx_E], V[idx_E], groups[idx_E]

    per_seed = {}
    for seed in ("42", "1", "2"):
        dcol = np.array([probs_by_seed[seed][ids[i]] for i in idx_E])
        res = refit_cell_on_E("family1", Ve, Ae, dcol, y, ge)
        per_seed[seed] = {
            "T_E": float(roc_auc_score(y, dcol)),
            "VA_lin_E": res["VA_lin"], "VA_nl_E_seeds": res["VA_nl_seeds"], "VA_nl_E_mean": res["VA_nl_mean"],
            "VAT_lin_E": res["VAT_lin"], "VAT_nl_E_seeds": res["VAT_nl_seeds"], "VAT_nl_E_mean": res["VAT_nl_mean"],
            "boot_VAT_nl_minus_VA_nl": paired_boot(y, res["_oof_VAT_nl0"], res["_oof_VA_nl0"]),
            "boot_VAT_nl_minus_T": paired_boot(y, res["_oof_VAT_nl0"], dcol),
        }
    ens = np.array([np.mean([probs_by_seed[s][ids[i]] for s in ("42", "1", "2")]) for i in idx_E])
    out = {
        "cell": "style_inv_toptier", "direction": 1,
        "population": "evaluation-valid rows E = dense eval+test splits",
        "n_E": int(len(y)), "pos_rate_E": float(y.mean()),
        "n_groups_E": int(len(np.unique(ge))),
        "n_features": {"V": int(Ve.shape[1]), "A": int(Ae.shape[1]), "VAT_extra": 1},
        "per_dense_seed": per_seed,
        "T_E_mean_over_dense_seeds": float(np.mean([per_seed[s]["T_E"] for s in per_seed])),
        "T_E_ensemble_meanprob": float(roc_auc_score(y, ens)),
        "VA_lin_E": float(np.mean([per_seed[s]["VA_lin_E"] for s in per_seed])),  # identical across seeds
        "VA_nl_E_mean": float(np.mean([per_seed[s]["VA_nl_E_mean"] for s in per_seed])),
        "VAT_lin_E_mean_over_dense_seeds": float(np.mean([per_seed[s]["VAT_lin_E"] for s in per_seed])),
        "VAT_nl_E_mean_over_dense_seeds": float(np.mean([per_seed[s]["VAT_nl_E_mean"] for s in per_seed])),
        "dense_preds_source": "datasets/humor/style_invitational/dense_standard/rm_out_seed{42,1,2}/preds_{eval,test}.csv",
    }
    if not skip_fullfit:
        out.update(fullfit_at_E("style_inv_toptier", in_E))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default=None,
                    choices=["cap_crowd", "cap_finalist", "style_inv_toptier"])
    ap.add_argument("--skip-fullfit", action="store_true")
    args = ap.parse_args()
    cells = [args.cell] if args.cell else ["cap_crowd", "cap_finalist", "style_inv_toptier"]
    for cell in cells:
        t0 = time.time()
        print(f"=== {cell} ===", flush=True)
        if cell == "style_inv_toptier":
            out = run_style_inv(skip_fullfit=args.skip_fullfit)
        else:
            out = run_caption(cell, skip_fullfit=args.skip_fullfit)
        out["runtime_sec"] = time.time() - t0
        p = RESULTS / f"{cell}_direction1.json"
        p.write_text(json.dumps(out, indent=2))
        keys = ["n_E", "T_E", "VA_lin_E", "VA_nl_E_mean", "VAT_lin_E", "VAT_nl_E_mean"]
        print(json.dumps({k: out.get(k) for k in keys if k in out}, indent=1))
        if "per_dense_seed" in out:
            print(json.dumps({s: {k: v for k, v in d.items() if not k.startswith("boot")}
                              for s, d in out["per_dense_seed"].items()}, indent=1))
        print("wrote", p, f"({out['runtime_sec']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
