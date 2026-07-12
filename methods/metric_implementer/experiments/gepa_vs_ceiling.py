"""Structural proxy for 'how well does the best single prompt meet the OPT_Ω ceiling?'

The CW Ω pool has no GEPA criteria (run_alpha_probe was run without --gepa-registry), so the
GEPA-optimal prompt is NOT in the pool. But the certificate's greedy gains answer the structural
question: g1 (best SINGLE criterion's gain) vs OPT_Ω (best SUBSET, Σ gains). The GEPA-optimal
prompt is >= the best single freegen criterion (GEPA optimizes for recovery), so g1/OPT_Ω is a
LOWER BOUND on (GEPA-prompt recovery)/OPT_Ω. The complement 1 - g1/OPT_Ω = the combiner gap
(value of adding criteria beyond the best single).

Usage: python -m methods.metric_implementer.experiments.gepa_vs_ceiling [dir] [dir...]
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

from .value_certificate import greedy_head
from .alpha_probe import conditional_crc_report


def run(d: str) -> None:
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*_sigs.npz"))):
        z = np.load(f, allow_pickle=True)
        if "M_i" not in z.files:
            continue
        S = np.asarray(z["sigs"], float)
        B = (np.nan_to_num(S, nan=0.5) > 0.5).astype(int)
        M = (np.asarray(z["M_i"], float) > 0.5).astype(int)
        tags = [str(t) for t in z["tags"]]
        h = greedy_head(B, M)
        g = h["gains"]
        opt = h["opt_omega_bits"]
        g1 = g[0] if g else 0.0
        g3 = sum(g[:3])
        ncrit = len(B)                       # |Ω| raw candidate pool
        cr = conditional_crc_report(S, tags)  # D_obs = behaviorally distinct species
        rows.append({
            "name": str(z.get("name", os.path.basename(f)))[:34], "opt": opt, "g1": g1, "g3": g3,
            "k": len(g), "H_M": h["H_M"], "frac_H": h["frac_H"], "ncrit": ncrit,
            "D_obs": cr["D_obs_lower"], "B_E": cr["B_E_upper"],
        })
    if not rows:
        print(f"  {d}: no checkpoints with M_i"); return
    opts = np.array([r["opt"] for r in rows]); g1s = np.array([r["g1"] for r in rows])
    r1 = g1s / opts
    print(f"\n=== {os.path.basename(d)} (n={len(rows)}) ===")
    print(f"  OPT_Ω mean={opts.mean():.3f} | g1/OPT_Ω mean={r1.mean():.3f} median={np.median(r1):.3f} | "
          f"|Ω|raw mean={np.mean([r['ncrit'] for r in rows]):.0f} D_obs mean={np.mean([r['D_obs'] for r in rows]):.0f} "
          f"| H_M mean={np.mean([r['H_M'] for r in rows]):.2f} %H(OPT/H_M) mean={np.mean([r['frac_H'] for r in rows]):.2f}")
    print(f"\n  {'metric':34s} H_M  OPT   %H   g1   g1/OPT  |Ω|raw D_obs head")
    for r in sorted(rows, key=lambda x: -x["opt"]):
        ratio = r['g1'] / r['opt'] if r['opt'] > 1e-9 else 0.0
        print(f"  {r['name']:34s} {r['H_M']:.2f} {r['opt']:.3f} {r['frac_H'] if np.isfinite(r['frac_H']) else 0:.2f} "
              f"{r['g1']:.3f} {ratio:.2f}   {int(r['ncrit']):5d} {int(r['D_obs']):5d} {int(r['k']):3d}")


if __name__ == "__main__":
    dirs = sys.argv[1:] or ["/lfs/skampere3/0/alexspan/outputs/r3_cw/aligned_8b_orbit_v2"]
    for d in dirs:
        run(d)
