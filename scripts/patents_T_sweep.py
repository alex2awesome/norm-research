#!/usr/bin/env python3
"""Offline T / aggregator sweep on cached element scores (NO GPU).

Arm C (delta unit) degenerated to T=1.0 (the top of score_gold_mixture's T_GRID), so it may be
leaving signal on the table. This loads the cached *_elscores.json, reproduces the pipeline's
own split-half softmin AUC to validate (should match the logged arm number), then sweeps a WIDER
T grid plus alternative aggregators (mean / min / max / median / logsumexp), all with the same
split-half discipline (choose the aggregator param on the OTHER half, apply, pool).

Run ON sk3 (CPU): python3 scripts/patents_T_sweep.py <arm_elscores.json> [condition]
"""
import json, sys
import numpy as np
sys.path.insert(0, "/lfs/skampere3/0/alexspan/norm-research/scripts")
from score_gold_mixture import softmin, half_of  # faithful to the scoring pipeline
from sklearn.metrics import roc_auc_score

WIDE_T = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]


def agg(es, name, p):
    es = np.asarray(es, dtype=np.float64)
    if name == "softmin":
        return softmin(es, p)
    if name == "mean":
        return es.mean()
    if name == "min":
        return es.min()
    if name == "max":
        return es.max()
    if name == "median":
        return float(np.median(es))
    if name == "logsumexp":  # smooth-max at temperature p
        return float(p * np.log(np.exp(es / p).mean()))
    raise ValueError(name)


def oriented_auc(y, s):
    """Report the direction-agnostic separation (max of the two orientations) so aggregator
    comparisons aren't confounded by sign; softmin-low vs -high is handled per-pipeline anyway."""
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def split_half_auc(items, name, grid):
    """items: [(key, y, es)]. Choose param on the other half, apply, pool. Returns (auc, param_by_half)."""
    halves = {0: [], 1: []}
    for k, y, es in items:
        halves[half_of(k)].append((k, y, es))
    best = {}
    for h in (0, 1):
        other = halves[1 - h]
        ys = np.array([y for _, y, _ in other])
        if len(set(ys.tolist())) < 2 or len(ys) < 4:
            best[h] = grid[len(grid) // 2]
            continue
        bp, ba = grid[0], -1
        for p in grid:
            ss = np.array([agg(es, name, p) for _, _, es in other])
            try:
                a = oriented_auc(ys, ss)
            except ValueError:
                continue
            if a > ba:
                ba, bp = a, p
        best[h] = bp
    ys, ss = [], []
    for h in (0, 1):
        for k, y, es in halves[h]:
            ys.append(y); ss.append(agg(es, name, best[h]))
    return oriented_auc(np.array(ys), np.array(ss)), best


def main():
    path = sys.argv[1]
    cond = sys.argv[2] if len(sys.argv) > 2 else "retrieved_only"
    d = json.load(open(path))
    if cond not in d:
        print(f"conditions available: {list(d)}"); return
    rows = d[cond]  # {key: [label, [el_scores], doc_key, half]}
    items = [(k, v[0], v[1]) for k, v in rows.items() if v[1]]
    print(f"{path.split('/')[-1]} [{cond}]: {len(items)} claims", flush=True)

    # per rejection type split too, using doc_key isn't enough; just report ALL here
    print(f"{'aggregator':14s} {'grid/param':22s} {'AUC':>7s}")
    # 1. reproduce the pipeline's own softmin over its narrow grid (validation)
    a_narrow, bt = split_half_auc(items, "softmin", [0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    print(f"{'softmin(narrow)':14s} {'T∈[.05..1.0] '+str(bt):22s} {a_narrow:7.4f}   <- reproduces logged arm number")
    # 2. wide softmin
    a_wide, bt = split_half_auc(items, "softmin", WIDE_T)
    print(f"{'softmin(wide)':14s} {'T∈[.02..10] '+str(bt):22s} {a_wide:7.4f}")
    # 3. alternative aggregators
    for name, grid in [("mean", [0]), ("min", [0]), ("max", [0]), ("median", [0]),
                       ("logsumexp", WIDE_T)]:
        a, bp = split_half_auc(items, name, grid)
        tag = str(bp) if name == "logsumexp" else "-"
        print(f"{name:14s} {tag:22s} {a:7.4f}")
    print("T_SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
