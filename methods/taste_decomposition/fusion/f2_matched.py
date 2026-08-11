#!/usr/bin/env python3
"""F2 addendum: the MATCHED-STRENGTH companion (D1b-style two-stage).

Why (coordinator, 2026-08-11): the primary arms (a)-(e) refit the enriched bank on E
only, which starves it. On cells where the bank's full-strength fit is materially
better than its E-refit, the primary increment (d)-(c) is NOT comparable to the
closure campaign's own same-rows verdict, which was taken against the FULL-STRENGTH
terminal bank. press_verdict is the grid's worst case (original-bank E-refit .681 vs
fullfit@E .744).

Design (mirrors fusion/direction1b_twostage.py's logic, using the frozen Layer-1
machinery so nothing else changes):

  stage 1  the ENRICHED bank fit on the FULL population with grouped OOF
           (direction1_mirror2.va_only -- honest out-of-fold for every row), its
           prediction read on E  ->  ONE column `bank_full_oof`
  stage 2  on E, the frozen stack over
             (c*) [bank_full_oof + nuisance]
             (d*) [bank_full_oof + nuisance + T]
             (e*) [bank_full_oof + nuisance + T0]

The companion increment is (d*)-(c*). It removes the BANK's small-train handicap
while leaving everything else (folds, family, seeds, nuisance block) identical to
the primary.

Cells whose adapter returns E == the whole population (cw_community, peer_verdict)
have no larger population to fit on: the companion is identical to the primary and
is recorded as such rather than recomputed.

Writes a schema-versioned `matched_strength_companion/v1` block into
results/f2_deconf_<cell>.json. Never recomputes an existing arm.

CPU only.  Usage: python3 f2_matched.py --cell press_verdict
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
TD = HERE.parent
RESULTS = TD / "results"
ROWS = HERE / "t0_rows"
SCORES = HERE / "t0_scores"

SCHEMA = "matched_strength_companion/v1"
GAP_TRIGGER = 0.02


def _mod(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m
    spec.loader.exec_module(m)
    return m


D1 = _mod(HERE / "direction1_mirror.py", "d1_ms")
D2 = _mod(HERE / "direction1_mirror2.py", "d2_ms")
F2C = _mod(HERE / "f2_cells.py", "f2_cells_ms")
fit_arm, gboot, va_only = D1.fit_arm, D1.group_paired_boot, D2.va_only


def run(cell, n_boot=2000):
    t0 = time.time()
    p = RESULTS / f"f2_deconf_{cell}.json"
    res = json.loads(p.read_text())
    meta = json.loads((ROWS / f"{cell}.meta.json").read_text())
    family = meta["family"]

    z = np.load(ROWS / f"{cell}.npz", allow_pickle=True)
    ids_E = [str(i) for i in z["ids"]]
    y_E = z["y"].astype(int)
    groups_E = np.array([str(g) for g in z["groups"]], dtype=object)
    T_E = z["dense"].astype(float)
    uids = [str(u) for u in z["uids"]]
    pm = {}
    with gzip.open(SCORES / f"{cell}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            pm[r["uid"]] = float(r["p_yes"])
    T0_E = np.array([pm[u] for u in uids], dtype=float)

    a = F2C.ADAPTERS[cell]()
    y_pop = np.asarray(a["y"]).astype(int)
    groups_pop = np.array([str(g) for g in a["groups"]], dtype=object)
    Emask = np.asarray(a["E"], dtype=bool)

    blk = {"schema": SCHEMA, "gap_trigger": GAP_TRIGGER,
           "rationale": ("the primary (a)-(e) refit the enriched bank on E only; where the "
                         "bank's FULL-STRENGTH fit is materially better, the primary "
                         "increment is not comparable to the closure campaign's same-rows "
                         "verdict against the full-strength terminal bank"),
           "design": ("stage 1 = enriched bank, FULL-population grouped OOF "
                      "(direction1_mirror2.va_only), read on E as one column; stage 2 = the "
                      "frozen stack on E over [bank_full_oof + nuisance (+T / +T0])")}

    if int(Emask.sum()) == len(y_pop):
        blk.update({
            "applicable": False,
            "reason": ("E is the whole population for this cell, so there is no larger "
                       "training set for stage 1: the matched-strength companion is "
                       "IDENTICAL to the primary by construction"),
            "companion_increment": res["PRIMARY_stacked_increment_d_minus_c"],
        })
        res["matched_strength_companion"] = blk
        p.write_text(json.dumps(res, indent=2, default=str))
        print(f"  [{cell}] E == population -> companion identical to primary", flush=True)
        return blk

    pos = {str(i): k for k, i in enumerate(a["ids"])}
    idx = np.array([pos[i] for i in ids_E])
    assert np.array_equal(y_pop[idx], y_E), f"{cell}: y mismatch on the E join"
    bank_pop, nuis_pop = a["bank"], a["nuis"]

    # ---- stage 1: enriched bank at FULL training strength -------------------
    print(f"  [{cell}] stage1: full-population enriched bank "
          f"(n={len(y_pop)}, {bank_pop.shape[1]} cols) ...", flush=True)
    summ, lin_oof, gbm_oofs = va_only(family, bank_pop, y_pop, groups_pop)
    bank_oof_pop = np.mean(gbm_oofs, axis=0)
    bank_oof_E = bank_oof_pop[idx]
    a_full_on_E = float(roc_auc_score(y_E, bank_oof_E))
    a_erefit = res["arms"]["a_VA_enr_nl"]
    gap = a_full_on_E - a_erefit

    blk.update({
        "applicable": True,
        "stage1": {"n_population": int(len(y_pop)), "n_E": int(len(y_E)),
                   "bank_cols": int(bank_pop.shape[1]),
                   "VA_enr_full_lin_on_E": float(roc_auc_score(y_E, lin_oof[idx])),
                   "VA_enr_full_nl_on_E_seedmean_probs": a_full_on_E,
                   "VA_enr_full_nl_on_E_per_seed":
                       [float(roc_auc_score(y_E, o[idx])) for o in gbm_oofs],
                   "VA_enr_full_nl_whole_population": summ["VA_nl_mean"]},
        "enriched_bank_gap_fullfit_minus_Erefit": gap,
        "gap_exceeds_trigger": bool(gap > GAP_TRIGGER),
    })

    # ---- stage 2: combiner on E --------------------------------------------
    M = np.column_stack([bank_oof_E.reshape(-1, 1), nuis_pop[idx]])
    print(f"  [{cell}] stage2: combiner on E ({M.shape[1]} cols + T) ...", flush=True)
    r_T = fit_arm(family, M, T_E, y_E, groups_E)
    r_0 = fit_arm(family, M, T0_E, y_E, groups_E)
    assert np.allclose(r_T["_oof_VA_nl0"], r_0["_oof_VA_nl0"]), f"{cell}: (c*) differs"

    prim = gboot(y_E, r_T["_oof_VAT_nl0"], r_T["_oof_VA_nl0"], groups_E, n_boot=n_boot)
    sec = gboot(y_E, r_0["_oof_VAT_nl0"], r_0["_oof_VA_nl0"], groups_E, n_boot=n_boot)
    blk.update({
        "arms_matched": {
            "c_star_bankfull_plus_NUIS_nl": r_T["VA_nl_mean"],
            "d_star_plus_T_nl": r_T["VAT_nl_mean"],
            "e_star_plus_T0_nl": r_0["VAT_nl_mean"],
            "c_star_lin": r_T["VA_lin"], "d_star_lin": r_T["VAT_lin"],
        },
        "COMPANION_increment_dstar_minus_cstar": prim,
        "companion_untrained_increment_estar_minus_cstar": sec,
        "primary_Erefit_increment_for_contrast":
            res["PRIMARY_stacked_increment_d_minus_c"]["estimate"],
        "quoting_rule": ("where gap_exceeds_trigger is true, the COMPANION increment is the "
                         "quotable number against any full-strength bank comparison "
                         "(including the closure campaign's same-rows verdict); the E-refit "
                         "primary is a matched-footing readout and is NOT comparable to it"),
    })
    res["matched_strength_companion"] = blk
    p.write_text(json.dumps(res, indent=2, default=str))
    print(f"  [{cell}] bank E-refit {a_erefit:.4f} -> full-strength on E {a_full_on_E:.4f} "
          f"(gap {gap:+.4f}) | (c*) {r_T['VA_nl_mean']:.4f} (d*) {r_T['VAT_nl_mean']:.4f} "
          f"| COMPANION {prim['estimate']:+.4f} [{prim['ci95'][0]:+.4f},{prim['ci95'][1]:+.4f}] "
          f"P={prim['p_gt_0']:.3f} (E-refit primary was "
          f"{blk['primary_Erefit_increment_for_contrast']:+.4f}) | {time.time()-t0:.0f}s",
          flush=True)
    return blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    for c in args.cell:
        print(f"=== matched-strength {c} ===", flush=True)
        try:
            run(c, args.n_boot)
        except Exception as e:
            print(f"  [{c}] FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
