"""Powered L-axis test across many metrics. Reports per-task per-bucket mean I(L), the nested
running-max, and UNBIASED paired tests on fixed buckets (I(100)-I(40), I(600)-I(40)) so the
max-selection bias of the running-max doesn't contaminate the significance test."""
import json
import sys
import collections
import numpy as np

OUT = "/lfs/skampere3/0/alexspan/tmp_vinfo"
fname = sys.argv[1] if len(sys.argv) > 1 else "recon_lsweep_free_big.json"
rows = json.load(open(f"{OUT}/{fname}"))


def nz(x):
    return float("nan") if x is None else x


Ls = sorted({r.get("l_cap") for r in rows})
by = collections.defaultdict(dict)
for r in rows:
    by[(r["task"], r["metric_id"])][r.get("l_cap")] = nz(r.get("iv_transmission"))


def sign_p(deltas):
    """two-sided sign test vs 0 (binomial), ignoring exact ties."""
    from math import comb
    d = [x for x in deltas if np.isfinite(x) and abs(x) > 1e-9]
    n = len(d); k = sum(1 for x in d if x > 0)
    if n == 0:
        return float("nan"), 0, 0
    kk = max(k, n - k)
    p = 2 * sum(comb(n, i) for i in range(kk, n + 1)) / (2 ** n)
    return min(p, 1.0), k, n


print(f"file={fname}  L grid={Ls}\n")
for task in sorted({t for t, _ in by}):
    mets = [m for (t, m) in by if t == task]
    M = np.array([[by[(task, m)].get(L, np.nan) for L in Ls] for m in mets])
    mean = np.nanmean(M, axis=0)
    rmax = np.nanmean(np.maximum.accumulate(np.where(np.isfinite(M), M, -1), axis=1), axis=0)
    print(f"=== {task}  (n_metrics={len(mets)}) ===")
    print("  per-bucket mean I(L): " + "  ".join(f"L{L}={mean[i]:.3f}" for i, L in enumerate(Ls)))
    print("  nested running-max  : " + "  ".join(f"L{L}={rmax[i]:.3f}" for i, L in enumerate(Ls)))
    i40 = Ls.index(40) if 40 in Ls else 0
    for Ltarget in Ls[1:]:
        j = Ls.index(Ltarget)
        d = M[:, j] - M[:, i40]
        p, k, n = sign_p(d)
        print(f"  Δ(L{Ltarget}-L{Ls[i40]}): mean={np.nanmean(d):+.3f}  median={np.nanmedian(d):+.3f}  "
              f"frac>0={k}/{n}  sign-p={p:.3f}")
    # where does each metric peak?
    peak = collections.Counter(Ls[int(np.nanargmax(row))] for row in M if np.isfinite(row).any())
    print(f"  peak-L histogram: {dict(sorted(peak.items()))}\n")
