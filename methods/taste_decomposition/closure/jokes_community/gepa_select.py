#!/usr/bin/env python3
"""GEPA STAGE 4 — accept or reject each variant, probe-vs-probe, label-blind.

RULE (inherited from ../gepa_select_peer.py): a variant is ACCEPTED only if its fidelity
on the probe rows beats its INCUMBENT's fidelity on the SAME probe rows by at least
MARGIN = .02. Ties and near-ties go to the incumbent, so the pass can only fire when the
rephrasing demonstrably fixed the degeneracy it was written to fix.

    fidelity = .5*(1 - modal_share) + .3*(1 - na_rate) + .2*min(spread/5.0, 1)

Nothing here reads y, any AUC, or MONITOR. Selection is purely an instrument-quality
decision on the score distribution, which is what keeps the phrasing pass from becoming a
label-fitting loop.

Where more than one variant of a criterion clears the margin, the best-fidelity one wins;
ties break on stable sha256 of the variant id, so no position or authoring order is
favoured.

CPU only.  Usage: python3 gepa_select.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CELL = "jokes_community"
MARGIN, SCALE_HALF = 0.02, 5.0


def fidelity(col):
    fin = np.isfinite(col)
    if fin.sum() < 20:
        return 0.0, {"modal_share": 1.0, "na_rate": 1.0, "spread": 0.0, "n": int(fin.sum())}
    _, cnts = np.unique(col[fin], return_counts=True)
    modal = float(cnts.max() / fin.sum())
    na = float((~fin).mean())
    spread = float(np.std(col[fin]) / SCALE_HALF)
    return (0.5 * (1 - modal) + 0.3 * (1 - na) + 0.2 * min(spread, 1.0),
            {"modal_share": modal, "na_rate": na, "spread": spread, "n": int(fin.sum())})


def main():
    z = np.load(HERE / f"{CELL}_gepa_probe_scores.npz", allow_pickle=True)
    X = z["X"]
    uid = [str(s) for s in z["uid"]]
    kind = [str(s) for s in z["kind"]]
    tid = [str(s) for s in z["target_id"]]
    var = {v["variant_id"]: v for v in
           json.loads((HERE / f"{CELL}_gepa_variants.json").read_text())["variants"]}

    inc = {}
    for k, u in enumerate(uid):
        if kind[k] == "incumbent":
            f, info = fidelity(X[:, k])
            inc[tid[k]] = {"fidelity": f, **info}

    rows = []
    for k, u in enumerate(uid):
        if kind[k] != "variant":
            continue
        f, info = fidelity(X[:, k])
        base = inc.get(tid[k])
        gain = f - base["fidelity"] if base else None
        rows.append({"variant_id": u, "target_id": tid[k],
                     "name": var.get(u, {}).get("name"),
                     "fidelity_probe": f, **{f"variant_{a}": b for a, b in info.items()},
                     "incumbent_fidelity_probe": base["fidelity"] if base else None,
                     "incumbent_modal_share": base["modal_share"] if base else None,
                     "fidelity_gain": gain,
                     "clears_margin": bool(gain is not None and gain >= MARGIN),
                     "sort_hash": hashlib.sha256(u.encode()).hexdigest()})

    winners = {}
    for r in sorted(rows, key=lambda r: (-(r["fidelity_gain"] or -9), r["sort_hash"])):
        if r["clears_margin"] and r["target_id"] not in winners:
            winners[r["target_id"]] = r

    out = {"cell": CELL, "margin": MARGIN, "n_probe_rows": int(X.shape[0]),
           "rule": "accept a variant only if probe fidelity beats the incumbent's probe "
                   "fidelity by >= .02; label-blind, MONITOR never read",
           "n_targets": len(inc), "n_variants": len(rows),
           "n_targets_with_winner": len(winners),
           "incumbents": inc, "variants": rows,
           "winners": {k: v["variant_id"] for k, v in winners.items()},
           "winner_detail": list(winners.values())}
    (HERE / f"{CELL}_gepa_selection.json").write_text(json.dumps(out, indent=1))

    print(f"probe rows {X.shape[0]} | margin {MARGIN}")
    print(f"{'variant':22s} {'fid':>6s} {'inc':>6s} {'gain':>7s} {'modal':>6s} "
          f"{'incmodal':>8s}  accept")
    for r in sorted(rows, key=lambda r: (r["target_id"], -(r["fidelity_gain"] or -9))):
        print(f"{r['variant_id']:22s} {r['fidelity_probe']:6.3f} "
              f"{(r['incumbent_fidelity_probe'] or 0):6.3f} "
              f"{(r['fidelity_gain'] or 0):+7.3f} {r['variant_modal_share']:6.3f} "
              f"{(r['incumbent_modal_share'] or 0):8.3f}  "
              f"{'ACCEPT' if r['variant_id'] in out['winners'].values() else ''}")
    print(f"\n{len(winners)}/{len(inc)} targets have an accepted rephrasing")


if __name__ == "__main__":
    main()
