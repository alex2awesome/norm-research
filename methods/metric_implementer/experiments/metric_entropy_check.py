"""Check whether low-H_M metrics are low because the probe set doesn't exercise them
(near-constant verdict = probe-coverage artifact) vs genuinely un-articulable.

Per metric: H_M (binary entropy of the orbit-M_i verdict), base rate (fraction YES at >0.5),
n_minority (count of the less-frequent class out of 300). Low H_M + extreme base rate +
tiny n_minority => the 300 probes barely fire the metric => low ceiling is a probe artifact.
"""
import glob
import os
import sys

import numpy as np


def h_bits(p):
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def main(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*_sigs.npz"))):
        z = np.load(f, allow_pickle=True)
        if "M_i" not in z.files:
            continue
        M = (np.asarray(z["M_i"], float) > 0.5).astype(int)
        n = len(M)
        yes = int(M.sum())
        p = yes / n
        rows.append((str(z.get("name", os.path.basename(f)))[:34], h_bits(p), p, min(yes, n - yes), n))
    rows.sort(key=lambda x: x[1])
    print(f"=== {os.path.basename(d)} (n={len(rows)}), sorted by H_M (low => near-constant verdict) ===")
    print(f"  {'metric':34s} H_M   base   n_min  n")
    for nm, H, p, nmin, n in rows:
        flag = "  <- near-constant (probe artifact?)" if (H < 0.50 or nmin < 20) else ""
        print(f"  {nm:34s} {H:.2f}  {p:.2f}   {nmin:3d}/{n}{flag}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/lfs/skampere3/0/alexspan/outputs/r3_cw/aligned_8b_orbit_v2")
