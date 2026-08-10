#!/usr/bin/env python3
"""Bounded GEPA phrasing pass on the SURVIVING mined A criteria.

The freeze requires a GEPA-iterated phrasing pass before any confirmatory number is
quoted (the pilot was explicitly pre-GEPA).  A full per-criterion GEPA against a
frontier reference judge is unaffordable at this bank size, so this is a bounded,
label-blind variant that targets the failure mode phrasing actually causes:

  OBJECTIVE (label-blind, Gemma-only, never sees y):
      fidelity = 0.5 * (1 - modal_share)          # does the criterion discriminate?
               + 0.3 * (1 - na_rate)              # is it answerable from the text?
               + 0.2 * spread                     # normalised sd of the score column
  Nothing in the objective references the outcome variable, so a GEPA-improved
  criterion cannot become label-aware.

  LOOP: for each targeted criterion, a frontier proposer writes K rephrasings of the
  DESCRIPTION ONLY (the criterion's identity is held fixed and asserted by name), the
  variants are scored by the same Gemma judge on a fixed probe subset of the FIT+MINE
  rows, and the best variant is kept if it beats the incumbent by MARGIN.

  Targeting: criteria whose incumbent modal_share > .75 or na_rate > .20 -- i.e. the
  ones where the phrasing is plausibly the binding constraint.  Untargeted criteria
  keep their original phrasing and are reported as such.

Stage 1 (this file, CPU): pick targets, write the proposer brief.
Stage 2: variants scored by score_round_gemma.py --criteria gepa_variants.json.
Stage 3: pick winners, rescore corpus-wide, report pre/post Delta.

Usage:
  python gepa_phrasing.py targets --rounds 1,2,3
  python gepa_phrasing.py select --probe-scores gepa_probe_scores.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODAL_TRIGGER, NA_TRIGGER, MARGIN, K_VARIANTS = 0.75, 0.20, 0.02, 3


def fidelity(col):
    fin = np.isfinite(col)
    if fin.sum() < 20:
        return 0.0, {"modal_share": 1.0, "na_rate": 1.0, "spread": 0.0}
    vals, cnts = np.unique(col[fin], return_counts=True)
    modal = float(cnts.max() / fin.sum())
    na = float((~fin).mean())
    spread = float(np.std(col[fin]) / 0.5)      # scores live on {0,.5,1}; sd<=.5
    f = 0.5 * (1 - modal) + 0.3 * (1 - na) + 0.2 * min(spread, 1.0)
    return f, {"modal_share": modal, "na_rate": na, "spread": spread}


def cmd_targets(a):
    targets = []
    for r in [int(x) for x in a.rounds.split(",") if x]:
        sc = HERE / f"round{r}_scores.npz"
        rt = HERE / f"round{r}_routing.json"
        if not (sc.exists() and rt.exists()):
            continue
        z = np.load(sc, allow_pickle=True)
        final = {d["cid"]: d["final_track"] for d in
                 json.loads(rt.read_text())["decisions"]}
        crits = {c["cid"]: c for c in json.loads(
            (HERE / f"round{r}_criteria.json").read_text())}
        for j, cid in enumerate([str(s) for s in z["cids"]]):
            if final.get(cid) != "A":
                continue
            f, d = fidelity(z["X"][:, j])
            rec = {"round": r, "cid": cid, "name": crits[cid]["name"],
                   "instruction": crits[cid]["instruction"],
                   "incumbent_fidelity": f, **d,
                   "targeted": bool(d["modal_share"] > MODAL_TRIGGER
                                    or d["na_rate"] > NA_TRIGGER)}
            targets.append(rec)
    (HERE / "gepa_targets.json").write_text(json.dumps(targets, indent=1))
    n = sum(t["targeted"] for t in targets)
    print(f"{len(targets)} surviving A criteria; {n} targeted for rephrasing "
          f"(modal_share>{MODAL_TRIGGER} or na_rate>{NA_TRIGGER})")
    for t in targets:
        if t["targeted"]:
            print(f"  {t['cid']} modal={t['modal_share']:.2f} na={t['na_rate']:.2f} "
                  f"fid={t['incumbent_fidelity']:.3f}  {t['name']}")


def cmd_select(a):
    """Pick winners from the probe-scored variants."""
    targets = {t["cid"]: t for t in json.loads((HERE / "gepa_targets.json").read_text())}
    variants = json.loads((HERE / "gepa_variants.json").read_text())
    z = np.load(HERE / a.probe_scores, allow_pickle=True)
    cids = [str(s) for s in z["cids"]]
    X = z["X"]
    by_parent = {}
    for j, vid in enumerate(cids):
        parent = next(v["parent_cid"] for v in variants if v["cid"] == vid)
        f, d = fidelity(X[:, j])
        by_parent.setdefault(parent, []).append({"cid": vid, "fidelity": f, **d})
    winners = []
    for parent, cands in by_parent.items():
        inc = targets[parent]["incumbent_fidelity"]
        best = max(cands, key=lambda c: c["fidelity"])
        keep = best["fidelity"] > inc + MARGIN
        winners.append({"parent_cid": parent, "incumbent_fidelity": inc,
                        "best_variant": best["cid"], "best_fidelity": best["fidelity"],
                        "ACCEPTED": keep, "margin": MARGIN,
                        "all_candidates": cands})
    (HERE / "gepa_winners.json").write_text(json.dumps(winners, indent=1))
    acc = sum(w["ACCEPTED"] for w in winners)
    print(f"{acc}/{len(winners)} targeted criteria improved beyond the {MARGIN} margin")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("targets")
    t.add_argument("--rounds", required=True)
    t.set_defaults(fn=cmd_targets)
    s = sub.add_parser("select")
    s.add_argument("--probe-scores", default="gepa_probe_scores.npz")
    s.set_defaults(fn=cmd_select)
    a = ap.parse_args()
    a.fn(a)
