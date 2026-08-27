#!/usr/bin/env python3
"""Layer-1 "nonlinear stack" of the taste-residual decomposition, peer-review
CURATION (oral/spotlight) and REVEALED (citation-percentile) cells.

Design: notes/2026-08-05__taste-decomposition-design.md (S0 ledger, S1 protocol,
S6 freeze changes). Pilot precedent: notes/2026-08-05__layer1_peer_verdict_pilot.md
+ methods/taste_decomposition/layer1_stack.py (peer-review VERDICT pilot). This
script is Wave 2 of the rollout ("same-matrix extensions"): the SAME
datasets/peer-review/vat_3y/union_scores.npz (154 A + 17 V, union n=14,307) and
the SAME aggregate_3y.py::rung_row mirrored pipeline, just switched to the
`curation` and `revealed` cell jsonls (identical schema to `verdict.jsonl`:
ntitle/judgement/...). Grouping unit = ntitle, identical to the pilot and to
aggregate_3y.py.

Rather than re-typing the pipeline, this script IMPORTS the pilot's data
loader, linear/GBM OOF fitters, and SHAP screen directly from layer1_stack.py
(load_cell, outer_folds, linear_oof, gbm_oof, shap_interactions, _fit_gbm) --
those already generalize to any cell name because load_cell() takes `cell` and
opens `{cell}.jsonl`. What is NEW here is the orchestration required by the two
freeze changes recorded after the pilot (design S6):

  FREEZE CHANGE 1 -- VA_nl (and V_nl) reported as the MEAN over seeds {0,1,2}
    with spread, not a single seed (single-seed spread on the pilot, .0099, was
    5x |Delta_interact|).
  FREEZE CHANGE 3 -- Delta_interact's 95% CI uses a GROUP-LEVEL (ntitle-level)
    cluster bootstrap, resampling groups (papers) with replacement rather than
    rows, mirroring nc_layer1_stack.py::bootstrap_delta_interact_docket (same
    idea, ntitle instead of docket). A row-level bootstrap is also reported as a
    secondary diagnostic for continuity with the pilot's original method.

A_nl is run at seed 0 only (optional / informational, same effort level as the
original pilot before the seed-mean freeze change existed -- not part of the
frozen mean+spread protocol, which per design S1 applies to "every V and V+A
calculation").

SHAP interaction screen (descriptive only) is run ONLY for whichever of the two
cells has the larger |Delta_interact| (protocol item 4), after both cells' OOF
ledgers are computed.

CPU only. No GPU. No new judging. Usage:
  python layer1_peer_curation_revealed.py            # both cells
  python layer1_peer_curation_revealed.py --cell curation
  python layer1_peer_curation_revealed.py --cell revealed
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import layer1_stack as pilot  # same directory; reuses the pilot's pipeline verbatim

REPO = Path(__file__).resolve().parents[2]
VAT = REPO / "datasets" / "peer-review" / "vat_3y"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
PUBLISHED_JSON = VAT / "vat_3y_results.json"

GBM_SEEDS = (0, 1, 2)
GATE_TOL = 0.005

# T (dense clean-eval AUC), registry notes/2026-07-27__vat-run-registry.md
# "DENSE CHAIN -- CLEAN-EVAL FINAL (2026-07-28)" table (title-grouped eval split,
# Llama-3.1-8B LoRA). peer revealed's T rides a SMALL eval split (n_eval=223,
# flagged "optimistic" in the registry) and the whole revealed cell "rides a
# topic floor" (notes/2026-07-22__vat-paper-plan.md: citation percentile is
# substantially topic-predictable, so V/A/T all sit high in part because the
# construct itself leans on topic popularity) -- carried as a caveat below,
# not reinterpreted.
T_VALUES = {
    "curation": {"eval": 0.593, "test": 0.588},
    "revealed": {"eval": 0.871, "test": 0.896},
}
T_PROVENANCE = {
    "curation": "notes/2026-07-27__vat-run-registry.md 'DENSE CHAIN - CLEAN-EVAL FINAL "
                "(2026-07-28)': peer curation T eval .593 / test .588, title-grouped, "
                "Llama-3.1-8B LoRA; sk3 methods/dense/eval_pass_results.json.",
    "revealed": "notes/2026-07-27__vat-run-registry.md 'DENSE CHAIN - CLEAN-EVAL FINAL "
                "(2026-07-28)': peer revealed T eval .871 / test .896 (test flagged "
                "'optimistic, n_eval 223' in the registry -- SMALL eval split, report "
                "with that caveat); sk3 methods/dense/eval_pass_results.json. "
                "notes/2026-07-22__vat-paper-plan.md: revealed 'rides a topic floor' -- "
                "citation percentile is substantially topic-predictable, so high V/A/T "
                "here partly reflect topic popularity rather than paper quality per se. "
                "Carried as a caveat, not reinterpreted.",
}

CAVEATS = {
    "revealed": "peer revealed is the IMPACT cell riding a topic floor (citation "
                "percentile is substantially topic-predictable); the whole ledger -- "
                "V, A, VA_lin, VA_nl, T -- sits high in part for that reason, and T's "
                "test-split number is additionally flagged optimistic on a small "
                "n_eval=223. Noted, not reinterpreted.",
}


# ------------------------------------------------------------- bootstrap ----
def bootstrap_delta_interact_rows(oof_lin, oof_nl, y, n_boot=2000, seed=0):
    """SECONDARY diagnostic: paired bootstrap over OOF rows (pilot's original
    method). Rows are not exchangeable under GroupKFold(ntitle), so this likely
    understates the true CI width; kept only for continuity with the pilot."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        deltas.append(float(roc_auc_score(yb, oof_nl[idx]) - roc_auc_score(yb, oof_lin[idx])))
    deltas = np.array(deltas)
    return {
        "n_boot_used": int(len(deltas)),
        "mean": float(deltas.mean()),
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "p_gt_0": float((deltas > 0).mean()),
        "note": "row-level paired bootstrap, secondary only -- see the group-level "
                "(ntitle-resampled) bootstrap for the primary CI.",
    }


def bootstrap_delta_interact_groups(oof_lin, oof_nl, y, groups, n_boot=2000, seed=0):
    """PRIMARY: cluster (ntitle-level) bootstrap. Resample papers (ntitle) with
    replacement (same count as observed), pool all rows belonging to each drawn
    group (duplicated if drawn more than once), recompute pooled AUC for lin and
    nl on that resampled row set. Mirrors
    nc_layer1_stack.py::bootstrap_delta_interact_docket (FREEZE CHANGE 3)."""
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
        deltas.append(float(roc_auc_score(yb, oof_nl[idx]) - roc_auc_score(yb, oof_lin[idx])))
    deltas = np.array(deltas)
    return {
        "n_boot_used": int(len(deltas)),
        "n_groups_resampled": int(n_groups),
        "mean": float(deltas.mean()),
        "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "p_gt_0": float((deltas > 0).mean()),
    }


# -------------------------------------------------------------- per-cell ----
def run_cell(cell: str, verbose=True):
    t0 = time.time()
    mats, names, y, groups = pilot.load_cell(cell)
    folds = pilot.outer_folds(len(y), groups)
    n_groups = len(np.unique(groups))
    if verbose:
        print(f"=== peer {cell} === n={len(y)} pos={y.mean():.4f} groups={n_groups} "
              f"V={mats['V'].shape[1]}c A={mats['A'].shape[1]}c VA={mats['VA'].shape[1]}c")

    res = {
        "cell": f"peer-review {cell}",
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "n_groups": int(n_groups),
        "group_column": "ntitle",
        "matrix": str(pilot.NPZ),
        "n_features": {k: int(v.shape[1]) for k, v in mats.items()},
        "T_dense": T_VALUES[cell],
        "sklearn_version": sklearn.__version__,
        "linear": {},
        "nonlinear": {},
    }

    # ---- linear gate (mirrors aggregate_3y.py::rung_row exactly, via pilot.linear_oof) ----
    lin_oof = {}
    for k in ["V", "A", "VA"]:
        auc, oof = pilot.linear_oof(mats[k], y, groups, folds)
        res["linear"][k] = auc
        lin_oof[k] = oof
        if verbose:
            print(f"  linear  {k:2s}: {auc:.4f}")

    published_all = json.loads(PUBLISHED_JSON.read_text())["rungs"][cell]
    res["gate"] = {
        k: {"published": published_all[k], "reproduced": res["linear"][k],
            "abs_diff": abs(res["linear"][k] - published_all[k]),
            "pass": abs(res["linear"][k] - published_all[k]) <= GATE_TOL}
        for k in ("V", "A", "VA")
    }
    res["gate"]["n_check"] = {"published": published_all["n"], "reproduced": int(len(y)),
                               "pass": published_all["n"] == int(len(y))}
    gate_pass = all(g["pass"] for k, g in res["gate"].items() if isinstance(g, dict) and "pass" in g)
    res["gate_pass"] = gate_pass
    if verbose:
        print(f"  gate: {'PASS' if gate_pass else 'FAIL'} " +
              json.dumps({k: round(g["abs_diff"], 5) for k, g in res["gate"].items() if "abs_diff" in g}))

    if not gate_pass:
        res["runtime_sec"] = time.time() - t0
        res["diagnosis"] = "linear reproduction did not match vat_3y_results.json rungs." + cell + \
                            " within GATE_TOL=" + str(GATE_TOL) + " -- STOPPED, no nonlinear stack run."
        print(f"  GATE FAILED for {cell} -- STOPPING this cell, no nonlinear stack run.")
        return res

    # ---- nonlinear: V and VA at seeds 0/1/2 (FREEZE CHANGE 1: mean + spread); A at seed 0 only ----
    nl_seed_runs = {"V": {}, "VA": {}}
    for k in ["V", "VA"]:
        for s in GBM_SEEDS:
            if verbose:
                print(f"  gbm {k} seed {s} ...")
            r = pilot.gbm_oof(mats[k], y, groups, folds, seed=s, verbose=(s == 0))
            nl_seed_runs[k][s] = r
            if verbose:
                print(f"    -> auc {r['auc']:.4f}  train {r['train_auc_mean']:.4f}  picks {r['picks']}")

    for k in ["V", "VA"]:
        aucs = [nl_seed_runs[k][s]["auc"] for s in GBM_SEEDS]
        res["nonlinear"][k] = {
            "seed_aucs": {str(s): nl_seed_runs[k][s]["auc"] for s in GBM_SEEDS},
            "mean_auc": float(np.mean(aucs)),
            "spread": float(max(aucs) - min(aucs)),
            "train_auc_mean_seed0": nl_seed_runs[k][0]["train_auc_mean"],
            "picks_seed0": nl_seed_runs[k][0]["picks"],
        }

    # A_nl: seed 0 only, optional/informational (design S1 mean+spread protocol
    # is not required for A; skipped here like the pilot did before the
    # seed-mean freeze change existed for VA/V).
    if verbose:
        print("  gbm A seed 0 (optional, single seed) ...")
    a_r = pilot.gbm_oof(mats["A"], y, groups, folds, seed=0)
    res["nonlinear"]["A"] = {
        "auc": a_r["auc"], "picks": a_r["picks"], "train_auc_mean": a_r["train_auc_mean"],
        "note": "seed 0 only -- optional, not part of the frozen seed-mean protocol (design S6 "
                "applies mean+spread to V_nl/VA_nl).",
    }

    oof_nl_mean_va = np.mean([nl_seed_runs["VA"][s]["oof"] for s in GBM_SEEDS], axis=0)
    oof_nl_seed0_va = nl_seed_runs["VA"][0]["oof"]
    np.save(RESULTS_DIR / f"peer_{cell}_va_nl_oof_seed0.npy", oof_nl_seed0_va)
    np.save(RESULTS_DIR / f"peer_{cell}_va_nl_oof_mean3.npy", oof_nl_mean_va)

    L, N = res["linear"], res["nonlinear"]
    V_nl_mean, VA_nl_mean = N["V"]["mean_auc"], N["VA"]["mean_auc"]
    T = T_VALUES[cell]["eval"]
    ledger = {
        "V_lin": L["V"], "V_nl": V_nl_mean, "V_nl_spread": N["V"]["spread"],
        "A_lin": L["A"], "A_nl": N["A"]["auc"],
        "VA_lin": L["VA"], "VA_nl": VA_nl_mean, "VA_nl_spread": N["VA"]["spread"],
        "T": T, "T_test": T_VALUES[cell]["test"],
        "Delta_total": T - L["VA"],
        "Delta_interact": VA_nl_mean - L["VA"],
        "Delta_beyond": T - VA_nl_mean,
        "V_interact": V_nl_mean - L["V"],
    }
    res["ledger"] = ledger
    res["overfit_gap"] = {
        "V": N["V"]["train_auc_mean_seed0"] - N["V"]["seed_aucs"]["0"],
        "A": N["A"]["train_auc_mean"] - N["A"]["auc"],
        "VA": N["VA"]["train_auc_mean_seed0"] - N["VA"]["seed_aucs"]["0"],
    }

    # ---- Delta_interact CI: group-level (ntitle) PRIMARY, row-level secondary ----
    res["bootstrap_delta_interact_group_PRIMARY"] = bootstrap_delta_interact_groups(
        lin_oof["VA"], oof_nl_mean_va, y, groups)
    res["bootstrap_delta_interact_row_secondary"] = bootstrap_delta_interact_rows(
        lin_oof["VA"], oof_nl_mean_va, y)
    if verbose:
        b = res["bootstrap_delta_interact_group_PRIMARY"]
        print(f"  Delta_interact (mean-of-3-seeds) = {ledger['Delta_interact']:+.4f}  "
              f"group-level 95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  P(>0)={b['p_gt_0']:.2f}")

    res["T_provenance"] = T_PROVENANCE[cell]
    protocol_notes = [
        f"Group column = 'ntitle' (normalised paper title), identical to "
        f"aggregate_3y.py rung_row(). {n_groups} unique groups over {len(y)} rows.",
        f"Degeneracy guard: V {mats['V'].shape[1]} of 17 cols kept, A {mats['A'].shape[1]} of "
        f"154 cols kept (<5 off-modal values or zero variance dropped); linear and GBM see the "
        f"identical post-guard matrix.",
        "VA_nl and V_nl reported as the mean over seeds {0,1,2} per FREEZE CHANGE 1 "
        "(notes/2026-08-05__taste-decomposition-design.md S6); per-seed spread reported "
        "alongside as the load-bearing caveat on any Delta_interact smaller than the spread.",
        "Delta_interact point estimate and both its bootstrap CIs use the mean-of-3-seeds VA_nl "
        "OOF probability array (probability-space average), consistent with VA_nl being reported "
        "as the seed mean.",
        "PRIMARY Delta_interact CI is the ntitle-level cluster bootstrap (FREEZE CHANGE 3, "
        "notes/2026-08-05__taste-decomposition-design.md S6) -- resamples papers, not rows. "
        "Row-level bootstrap kept as a secondary diagnostic for continuity with the pilot.",
        "T is measured on the dense model's own title-grouped eval split, not on the A/V-scored "
        "rows used for VA_lin/VA_nl -- population mismatch inherited from the existing VAT "
        "registry convention (same as the verdict pilot); Delta_beyond carries it.",
    ]
    if cell in CAVEATS:
        protocol_notes.append(CAVEATS[cell])
    res["protocol_notes"] = protocol_notes

    # stash for cross-cell SHAP selection
    res["_lin_oof_VA"] = lin_oof["VA"]
    res["_nl_oof_mean3_VA"] = oof_nl_mean_va
    res["_mats_VA"] = mats["VA"]
    res["_names_VA"] = names["VA"]
    res["_y"] = y

    res["runtime_sec"] = time.time() - t0
    if verbose:
        print(f"  [{cell}] done in {res['runtime_sec']:.1f}s")
    return res


def save_result(res, shap=None):
    cell_slug = res["cell"].split()[-1]  # "peer-review curation" -> "curation"
    out = {k: v for k, v in res.items() if not k.startswith("_")}
    if shap is not None:
        out["shap"] = shap
    out_path = RESULTS_DIR / f"peer_{cell_slug}_layer1.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("wrote", out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["curation", "revealed"], default=None)
    args = ap.parse_args()

    cells = [args.cell] if args.cell else ["curation", "revealed"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = {c: run_cell(c) for c in cells}

    # SHAP screen (descriptive only) for whichever gate-passed cell has the
    # larger |Delta_interact| among those run this invocation (protocol item 4).
    ranked = sorted(
        (c for c in results if results[c].get("gate_pass") and "ledger" in results[c]),
        key=lambda c: -abs(results[c]["ledger"]["Delta_interact"]))
    shap_cell = ranked[0] if ranked else None
    if shap_cell:
        di_str = ", ".join(f"{c}={results[c]['ledger']['Delta_interact']:+.4f}" for c in ranked)
        print(f"\nSHAP screen: {shap_cell} has larger |Delta_interact| ({di_str})")

    for c in cells:
        r = results[c]
        shap_out = None
        if c == shap_cell:
            print(f"  shap for {c} ...")
            try:
                shap_out = pilot.shap_interactions(r["_mats_VA"], r["_y"], r["_names_VA"], seed=0)
            except Exception as e:  # pragma: no cover
                shap_out = {"error": repr(e)}
                print("  shap FAILED:", e)
        elif r.get("gate_pass"):
            shap_out = {"skipped": f"SHAP screen run only for the larger-|Delta_interact| cell "
                                    f"({shap_cell}) per protocol item 4."}
        save_result(r, shap=shap_out)

    print("\n=== summary ===")
    for c in cells:
        r = results[c]
        if not r.get("gate_pass"):
            print(f"peer {c}: GATE FAILED")
            continue
        L = r["ledger"]
        print(f"peer {c}: VA_lin={L['VA_lin']:.4f} VA_nl={L['VA_nl']:.4f} "
              f"Delta_interact={L['Delta_interact']:+.4f} Delta_beyond={L['Delta_beyond']:+.4f}")


if __name__ == "__main__":
    main()
