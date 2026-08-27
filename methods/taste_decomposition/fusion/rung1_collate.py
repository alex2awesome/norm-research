#!/usr/bin/env python3
"""RUNG 1 collate: merge per-cell jsons (mac + sk3) and compute the
preregistered cross-cell readout (design doc §1.4 + Addendum A: correlation
reported pooled AND grouped-only; pairwise cells marked).

Usage: python3 rung1_collate.py   (after scp'ing the sk3 cell jsons into results/)
"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RESULTS = Path(__file__).resolve().parent.parent / "results"

rows = []
for f in sorted(glob.glob(str(RESULTS / "rung1_selection_regret_*.json"))):
    d = json.load(open(f))
    p = d["point"]
    d["win_ratio"] = p["dense_wins_on_disagree"] / max(p["bank_wins_on_disagree"], 1)
    d["mode"] = p.get("mode", "grouped")
    rows.append(d)

print(f"{'cell':26s} {'mode':9s} {'gap':>5s} {'regret':>7s} {'CI95':>17s} {'d:b':>13s} ratio")
for d in sorted(rows, key=lambda r: (r["rel_gap_fig3"] is None, r["rel_gap_fig3"] or 0)):
    p = d["point"]
    g = f"{d['rel_gap_fig3']:.1f}" if d["rel_gap_fig3"] is not None else "excl"
    print(f"{d['cell']:26s} {d['mode'][:9]:9s} {g:>5s} {p['regret']:+.3f} "
          f"[{d['regret_ci95'][0]:+.3f},{d['regret_ci95'][1]:+.3f}] "
          f"{p['dense_wins_on_disagree']:>6d}:{p['bank_wins_on_disagree']:<6d} "
          f"{d['win_ratio']:.2f}")

for label, sel in (("ALL in-Fig3", [r for r in rows if r["rel_gap_fig3"] is not None]),
                   ("grouped-only", [r for r in rows if r["rel_gap_fig3"] is not None
                                     and r["mode"] == "grouped"])):
    if len(sel) < 4:
        continue
    x = [r["rel_gap_fig3"] for r in sel]
    for key in ("regret", "win_ratio"):
        yv = [r["point"]["regret"] if key == "regret" else r["win_ratio"] for r in sel]
        rho, pv = spearmanr(x, yv)
        print(f"cross-cell [{label}] {key} vs rel_gap: rho={rho:+.3f} p={pv:.3f} n={len(sel)}")

(RESULTS / "rung1_collated.json").write_text(json.dumps(rows, indent=2))
print("wrote rung1_collated.json")
