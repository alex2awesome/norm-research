#!/usr/bin/env python3
"""Seed-count-agnostic gate readout for the code_v3 cell.

Reads whatever dense seeds are present in dense_seed42/ dense_seed1/ dense_seed2/ and
emits the full within-repo gate table:

  * per-seed within-repo T, and the across-seed mean / SD / range  (= "3-seed within-repo
    T +/- spread", the round-0 item-1 deliverable);
  * per-seed and ensemble Δ = T − VA_nl under BOTH protocols:
      - LAYER-1  (VA_nl grouped-OOF fit WITHIN each split) -- the ruler the published
        +.0576 / +.0390 gate numbers were measured on;
      - CLOSURE  (VA_nl fit on FIT+MINE only, honest-full vector from round0_state.npz)
        -- no refit needed, the vectors are on disk;
  * three tiers: eval, test, and BOTH-splits-combined-at-the-repository-level (the
    best-powered honest statement; legitimate because the readout is repo-centred and
    the two dense splits contribute DISJOINT repositories -- this is adding repos to a
    within-repo average, not pooling rows across repos);
  * repo-cluster bootstrap CI + leave-one-repo-out jackknife + paired Wilcoxon, and the
    equal-repo (unweighted) Δ alongside the n-weighted one, because the n-weighted mean
    is dominated by a few large repositories;
  * the GATE verdict against the frozen threshold of .02.

Runs the MANDATORY OOF alignment gate first and REFUSES to report if it fails.

CPU only.  Usage: python gate_readout.py [--label INTERIM]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "maps_hw_si"))

import cells_code as C                                       # noqa: E402

THRESHOLD = 0.02
N_BOOT = 4000


def alignment_gate():
    r = subprocess.run([sys.executable, str(HERE / "oof_alignment_gate.py")],
                       capture_output=True, text=True, cwd=str(HERE))
    g = json.loads((HERE / "oof_alignment_gate.json").read_text())
    assert g["GATE_PASS"], f"OOF ALIGNMENT GATE FAILED -- refusing to report\n{r.stdout}"
    return g


def layer1_va(d):
    """VA_nl in the LAYER-1 protocol: mean of the three stack-seed grouped-OOF vectors,
    fit within each split."""
    va = np.full(len(d["y"]), np.nan)
    for sp in ("eval", "test"):
        m = d["split"] == sp
        va[m] = np.mean([np.load(HERE / "abank_rescore" /
                                 f"code_v3_{sp}_va_nl_oof_seed{s}.npy") for s in (0, 1, 2)],
                        axis=0)
    return va


def closure_va(d):
    p = HERE / "round0_state.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    ids = [str(x) for x in z["ids"]]
    assert ids == list(d["ids"]), "round0_state.npz row order does not match the cell loader"
    return z["VA_nl"]


def delta_block(d, dense, va, mask, rng):
    y, g = d["y"], d["groups"]
    w = C.within_repo_delta(y, dense, va, g, mask)
    per = w["per_repo_delta"]
    ns = np.array([x["n"] for x in per], float)
    dd = np.array([x["d"] for x in per])
    bs = np.array([float((ns[k] * dd[k]).sum() / ns[k].sum())
                   for k in (rng.integers(0, len(ns), (N_BOOT, len(ns))))])
    ub = np.array([float(dd[k].mean()) for k in rng.integers(0, len(dd), (N_BOOT, len(dd)))])
    return {
        "n_repos": w["n_repos"], "n_rows": w["n_rows"],
        "T_within": w["a_nwtd"], "VA_nl_within": w["b_nwtd"],
        "delta_nwtd": w["delta_nwtd"], "delta_unweighted": float(dd.mean()),
        "delta_median_repo": float(np.median(dd)),
        "dense_wins_repos": w["a_wins_repos"],
        "wilcoxon_p": float(wilcoxon([x["a"] for x in per], [x["b"] for x in per]).pvalue),
        "nwtd_boot_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
        "nwtd_boot_p_gt0": float((bs > 0).mean()),
        "unwtd_boot_ci95": [float(np.percentile(ub, 2.5)), float(np.percentile(ub, 97.5))],
        "unwtd_boot_p_gt0": float((ub > 0).mean()),
        "jackknife_se": w.get("jackknife_se"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="INTERIM",
                    help="INTERIM (<3 seeds) or FINAL (3 seeds)")
    a = ap.parse_args()

    g = alignment_gate()
    d = C.load()
    seeds = d["dense_seeds_have"]
    rng = np.random.default_rng(0)
    y, gr = d["y"], d["groups"]

    label = "FINAL" if len(seeds) >= 3 else a.label
    out = {"cell": "code_v3", "label": label,
           "n_dense_seeds": len(seeds), "dense_seeds": seeds,
           "oof_alignment_gate": {"pass": g["GATE_PASS"],
                                  "shuffled_counterfactual":
                                      {k: v["shuffled_counterfactual_auc"]
                                       for k, v in g["splits"].items()}},
           "threshold": THRESHOLD,
           "readout": "WITHIN-REPO n-weighted (repos with n>=20 and both classes); "
                      "pooled AUC is never a residual on this cell",
           "tiers": {}}

    tiers = {"eval": d["split"] == "eval", "test": d["split"] == "test",
             "both_combined_at_repo_level": np.ones(len(y), bool)}
    va_l1 = layer1_va(d)
    va_cl = closure_va(d)

    for tname, m in tiers.items():
        rec = {"per_seed": {}, "protocols": {}}
        Ts = []
        for k, s in enumerate(seeds):
            p = d["dense_seed_probs"][:, k]
            wr = C.within_repo_auc(y, p, gr, m)
            rec["per_seed"][f"seed{s}"] = {
                "T_within": wr["nwtd"], "T_within_median_repo": wr["median"],
                "T_pooled_NOT_A_READOUT": float(roc_auc_score(y[m], p[m])),
                "delta_layer1": C.within_repo_delta(y, p, va_l1, gr, m)["delta_nwtd"]}
            Ts.append(wr["nwtd"])
        rec["across_seed_T_within"] = {
            "mean": float(np.mean(Ts)), "values": Ts,
            "sd": float(np.std(Ts, ddof=1)) if len(Ts) > 1 else None,
            "range": [float(min(Ts)), float(max(Ts))],
            "spread": float(max(Ts) - min(Ts))}
        rec["protocols"]["layer1_ensemble"] = delta_block(d, d["dense"], va_l1, m, rng)
        if va_cl is not None:
            rec["protocols"]["closure_ensemble"] = delta_block(d, d["dense"], va_cl, m, rng)
        # TWO defensible definitions of "the N-seed residual", both reported (the same
        # ambiguity maps_hw_si/cells.py recorded as T_ensemble vs T_registry_mean_of_seed_AUC):
        #   ensemble      -- Delta of the seed-ensemble score; the row-level score every
        #                    fit and stratification in this campaign actually uses;
        #   mean-per-seed -- mean of the per-seed Deltas; strictly more conservative,
        #                    because ensembling two seeds is itself a better ranker.
        dl1 = [rec["per_seed"][f"seed{s}"]["delta_layer1"] for s in seeds]
        rec["delta_layer1_mean_of_per_seed"] = float(np.mean(dl1))
        rec["delta_layer1_per_seed_range"] = [float(min(dl1)), float(max(dl1))]
        rec["delta_layer1_per_seed_min_clears_threshold"] = bool(min(dl1) > THRESHOLD)
        out["tiers"][tname] = rec

    ht = out["tiers"]["both_combined_at_repo_level"]
    hd = ht["protocols"]["layer1_ensemble"]
    conservative = min(hd["delta_nwtd"], ht["delta_layer1_mean_of_per_seed"],
                       out["tiers"]["test"]["protocols"]["layer1_ensemble"]["delta_nwtd"],
                       out["tiers"]["test"]["delta_layer1_mean_of_per_seed"])
    out["GATE"] = {
        "statistic": "within-repo n-weighted Delta, both splits combined at the repo level, "
                     "LAYER-1 protocol, dense = seed ensemble",
        "value": hd["delta_nwtd"], "threshold": THRESHOLD,
        "ci95": hd["nwtd_boot_ci95"], "wilcoxon_p": hd["wilcoxon_p"],
        "n_seeds": len(seeds),
        "headline_delta_mean_of_per_seed": ht["delta_layer1_mean_of_per_seed"],
        "most_conservative_reading": conservative,
        "most_conservative_clears_threshold": bool(conservative > THRESHOLD),
        "verdict": ("PASS -> rounds 1..5" if hd["delta_nwtd"] > THRESHOLD
                    else "FAIL -> STOP at round 0; the seed verdict is terminal"),
        "BINDING": len(seeds) >= 3,
        "note": ("3 seeds present: this IS the frozen gate." if len(seeds) >= 3 else
                 f"ONLY {len(seeds)} seed(s) present -- this is an INTERIM reading and is "
                 f"NOT the frozen gate, which requires 3. Rounds stay held."),
    }
    (HERE / f"gate_readout_{len(seeds)}seed.json").write_text(json.dumps(out, indent=1,
                                                                        default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
