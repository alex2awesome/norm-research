#!/usr/bin/env python3
"""GEPA Stage 2b for N&C: pick winners from the probe-scored variants.

Fair comparison: recomputes the INCUMBENT's fidelity on the EXACT SAME probe
subset the variants were scored on (the current, already-house-style-rewritten
text -- this is the test of whether GEPA can beat N&C's own substitute pass).

python gepa_select_nc.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MARGIN, SCALE_HALF = 0.02, 5.0


def fidelity(col):
    fin = np.isfinite(col)
    if fin.sum() < 20:
        return 0.0, {"modal_share": 1.0, "na_rate": 1.0, "spread": 0.0}
    vals, cnts = np.unique(col[fin], return_counts=True)
    modal = float(cnts.max() / fin.sum())
    na = float((~fin).mean())
    spread = float(np.std(col[fin]) / SCALE_HALF)
    f = 0.5 * (1 - modal) + 0.3 * (1 - na) + 0.2 * min(spread, 1.0)
    return f, {"modal_share": modal, "na_rate": na, "spread": spread}


def main():
    targets = {t["tag"]: t for t in
               json.loads((HERE / "gepa_targets_nc.json").read_text())}
    z = np.load(HERE / "gepa_probe_scores_nc.npz", allow_pickle=True)
    probe_ids = [str(x) for x in z["probe_ids"]]

    pop = pd.read_csv(HERE / "nc_responded_population.csv")
    doc_to_row = {str(d): k for k, d in enumerate(pop["doc_id"].values)}
    probe_rows = [doc_to_row[d] for d in probe_ids]

    round_cache = {}

    def incumbent_probe_col(tag):
        rnd, cid = tag.split(":")
        r = int(rnd[1:])
        if r not in round_cache:
            round_cache[r] = np.load(HERE / f"round{r}_scores.npz", allow_pickle=True)
        zr = round_cache[r]
        cids = [str(s) for s in zr["crit_ids"]]
        j = cids.index(cid)
        return zr["X"][probe_rows, j]

    by_parent = {}
    for j, vid in enumerate(z["cids"]):
        vid = str(vid)
        parent = str(z["parent_tags"][j])
        f, d = fidelity(z["X"][:, j])
        by_parent.setdefault(parent, []).append({"cid": vid, "fidelity": f, **d})

    winners = []
    for parent, cands in by_parent.items():
        inc_col = incumbent_probe_col(parent)
        inc_f, inc_d = fidelity(inc_col)
        pop_f = targets[parent]["incumbent_fidelity"]
        best = max(cands, key=lambda c: c["fidelity"])
        keep = best["fidelity"] > inc_f + MARGIN
        winners.append({"parent_tag": parent, "incumbent_fidelity_on_probe": inc_f,
                        "incumbent_fidelity_on_population": pop_f,
                        "best_variant": best["cid"], "best_fidelity": best["fidelity"],
                        "ACCEPTED": keep, "margin": MARGIN, "all_candidates": cands})

    (HERE / "gepa_winners_nc.json").write_text(json.dumps(winners, indent=2))
    acc = sum(w["ACCEPTED"] for w in winners)
    print(f"{acc}/{len(winners)} targeted criteria improved beyond the {MARGIN} margin "
          f"(probe-vs-probe; incumbent is N&C's ALREADY house-style-rewritten text)")
    for w in sorted(winners, key=lambda w: -w["best_fidelity"]):
        print(f"  {w['parent_tag']:10s} incumbent(probe)={w['incumbent_fidelity_on_probe']:.4f} "
              f"best={w['best_fidelity']:.4f} ({w['best_variant']}) ACCEPTED={w['ACCEPTED']}")


if __name__ == "__main__":
    main()
