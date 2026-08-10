#!/usr/bin/env python3
"""Robustness of the residual LEVEL, N&C RESPONDED.

Motivated by two things at once:
  * this cell's Layer-2 profile -- docket-identity-alone AUC .916, within-docket AUC
    .675 vs pooled .726 -- so a POOLED Delta could in principle be a between-docket
    artifact;
  * the CW campaign's Stage-0 lesson (coordinator, 2026-08-07): when a readout
    population is thin, prefer ENLARGEMENT over a small-n reading. This campaign
    already applies that rule by construction (the saturation statistic is read on
    MONITOR_FULL n=1,892 rather than the T-honest MONITOR n=377, and the level is
    read on all 1,904 dense-held-out rows). This script measures what that choice
    is worth, and whether Delta survives the cell's own docket structure.

Readouts per bank state, all on saved predictions (no refits):
  1. Delta on the honest population (n=1,904) with a DOCKET-cluster bootstrap CI;
  2. Delta on eval-only and test-only halves separately (the dense chain's own split;
     recall this cell's dense chain selected on TEST -- flagged everywhere);
  3. WITHIN-DOCKET Delta: n-weighted mean of within-docket AUC for T and for VA_nl
     over dockets with >= min_n rows and both classes -- the readout that answers
     "is the residual a between-docket effect?";
  4. the same three for the thin MONITOR set, so the enlargement gain is explicit.

Usage: python delta_robustness.py --upto 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import nc_closure_lib as L
from readout import load_dense

HERE = Path(__file__).resolve().parent


def boot_delta(y, dense, va, docket, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(docket)
    idx_by = {g: np.where(docket == g)[0] for g in uniq}
    out = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in draw])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        out.append(roc_auc_score(yb, dense[idx]) - roc_auc_score(yb, va[idx]))
    out = np.array(out)
    return {"mean": float(out.mean()),
            "ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
            "p_gt_0": float((out > 0).mean())}


def within_group_pairs(y, dense, va, groups):
    """WITHIN-DOCKET pair concordance -- the enlargement that makes the within-group
    question answerable on a thin honest population.

    Per-docket AUC needs dockets with many rows AND both classes; on the 1,904
    dense-held-out rows only 6 dockets clear even a 5-row bar. But every docket that
    contains at least one responded and one unresponded comment contributes at least
    one ORDERED PAIR, and AUC on a binary label is exactly pair concordance. So pool
    all (positive, negative) pairs that lie inside a single docket and read both
    instruments on exactly those pairs. This uses ~30x more information than the
    per-docket-AUC route and answers the same question: is the residual a
    between-docket effect?
    """
    npairs = 0
    cT = cV = 0.0
    for g in np.unique(groups):
        m = groups == g
        yy = y[m]
        P = np.where(yy == 1)[0]
        N = np.where(yy == 0)[0]
        if len(P) == 0 or len(N) == 0:
            continue
        dd, vv = dense[m], va[m]
        dT = np.sign(dd[P][:, None] - dd[N][None, :])
        dV = np.sign(vv[P][:, None] - vv[N][None, :])
        cT += float((dT > 0).sum() + 0.5 * (dT == 0).sum())
        cV += float((dV > 0).sum() + 0.5 * (dV == 0).sum())
        npairs += dT.size
    if npairs == 0:
        return {"n_pairs": 0}
    return {"n_pairs": int(npairs), "T": cT / npairs, "VA_nl": cV / npairs,
            "Delta": (cT - cV) / npairs,
            "n_dockets_contributing": int(sum(
                1 for g in np.unique(groups)
                if len(set(y[groups == g])) == 2))}


def within_group(y, p, groups, min_n):
    tot, num, used = 0, 0.0, 0
    for g in np.unique(groups):
        m = groups == g
        n = int(m.sum())
        if n < min_n or len(set(y[m])) < 2:
            continue
        num += n * roc_auc_score(y[m], p[m])
        tot += n
        used += 1
    if tot == 0:
        return float("nan"), 0, 0
    return float(num / tot), used, int(tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upto", type=int, required=True)
    ap.add_argument("--min-docket", type=int, default=10)
    a = ap.parse_args()

    pop = L.load_population()
    summary, split, dsplit, mining, monitor_full = L.load_splits()
    y, docket = pop["y"], pop["docket"]
    dense = load_dense()
    heldout = np.isin(dsplit, ["eval", "test"])
    mon = split == "monitor"

    out = {"cell": "nc_responded", "min_docket_rows": a.min_docket, "states": {}}
    for r in range(0, a.upto + 1):
        p = HERE / f"state{r}_preds.npz"
        if not p.exists():
            continue
        va = np.load(p, allow_pickle=True)["nl_mean"]
        rec = {}
        for label, mask in (("honest_n1904", heldout),
                            ("eval_only", dsplit == "eval"),
                            ("test_only", dsplit == "test"),
                            ("monitor_n377", mon)):
            yy, dd, vv, gg = y[mask], dense[mask], va[mask], docket[mask]
            T = float(roc_auc_score(yy, dd))
            V = float(roc_auc_score(yy, vv))
            d = {"n": int(mask.sum()), "T": T, "VA_nl": V, "Delta": T - V}
            if label in ("honest_n1904", "monitor_n377"):
                d["docket_bootstrap"] = boot_delta(yy, dd, vv, gg)
            wt, nd, nr = within_group(yy, dd, gg, a.min_docket)
            wv, _, _ = within_group(yy, vv, gg, a.min_docket)
            d["within_docket"] = {"T": wt, "VA_nl": wv, "Delta": wt - wv,
                                  "n_dockets_used": nd, "n_rows_used": nr}
            d["within_docket_pairs"] = within_group_pairs(yy, dd, vv, gg)
            rec[label] = d
        out["states"][str(r)] = rec
        h = rec["honest_n1904"]
        print(f"[state {r}] honest Delta={h['Delta']:+.4f} "
              f"CI=[{h['docket_bootstrap']['ci95'][0]:+.4f},{h['docket_bootstrap']['ci95'][1]:+.4f}] "
              f"P={h['docket_bootstrap']['p_gt_0']:.3f} | within-docket Delta="
              f"{h['within_docket']['Delta']:+.4f} ({h['within_docket']['n_dockets_used']} dockets, "
              f"{h['within_docket']['n_rows_used']} rows) | eval {rec['eval_only']['Delta']:+.4f} "
              f"test {rec['test_only']['Delta']:+.4f} | monitor377 {rec['monitor_n377']['Delta']:+.4f}\n"
              f"          within-docket PAIRS n={h['within_docket_pairs']['n_pairs']} "
              f"({h['within_docket_pairs'].get('n_dockets_contributing')} dockets): "
              f"T={h['within_docket_pairs'].get('T', float('nan')):.4f} "
              f"VA={h['within_docket_pairs'].get('VA_nl', float('nan')):.4f} "
              f"Delta={h['within_docket_pairs'].get('Delta', float('nan')):+.4f}",
              flush=True)

    (HERE / f"round{a.upto}_delta_robustness.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
