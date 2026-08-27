#!/usr/bin/env python3
"""GEPA Stage 1 for N&C RESPONDED (0-10 scale bank), triggered because the
campaign's own label-blind house-style pass + fidelity(concept)-gate substitute
(phrasing_pass.py) was found NOT to satisfy "GEPA-iterated phrasing" in substance
(single-shot rewrite + identity gate, no fidelity objective, no candidate
iteration) -- see notes/2026-08-09__gepa_requotes.md Part 1 for the verdict.

Computes label-blind fidelity = 0.5*(1-modal_share) + 0.3*(1-na_rate)
+ 0.2*min(spread,1), spread normalised by scale_half=5.0 (0-10 scale), for the
67 surviving mined A criteria (rounds 1-5), and flags modal_share>.75 or
na_rate>.20 for rephrasing.

CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODAL_TRIGGER, NA_TRIGGER, SCALE_HALF = 0.75, 0.20, 5.0


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
    targets = []
    n_total = 0
    for r in (1, 2, 3, 4, 5):
        z = np.load(HERE / f"round{r}_scores.npz", allow_pickle=True)
        cids = [str(s) for s in z["crit_ids"]]
        routing = json.loads((HERE / f"round{r}_routing_final.json").read_text())
        src_of = {c["id"]: c["src_id"] for c in routing["A"]}   # blind P0x -> bank A0x
        gate = json.loads((HERE / f"round{r}_score_report.json").read_text())
        crit_final = {c["id"]: c for c in
                      json.loads((HERE / f"round{r}_criteria_final.json").read_text())["A"]}
        for j, cid in enumerate(cids):
            if cid not in src_of:
                continue
            if gate["per_criterion"].get(cid, {}).get("collapsed"):
                continue
            n_total += 1
            c = crit_final.get(src_of[cid])
            if c is None:
                continue  # decomposition-only ids etc.
            f, d = fidelity(z["X"][:, j])
            tag = f"r{r}:{cid}"
            targeted = bool(d["modal_share"] > MODAL_TRIGGER or d["na_rate"] > NA_TRIGGER)
            targets.append({"tag": tag, "round": r, "cid": cid, "name": c["name"],
                            "instruction": c["instruction"],
                            "phrasing_pass_already_applied": c.get("phrasing"),
                            "incumbent_fidelity": f, **d, "targeted": targeted})

    print(f"{n_total} surviving A criteria found across rounds 1-5 (expect 67)")
    (HERE / "gepa_targets_nc.json").write_text(json.dumps(targets, indent=2))
    nt = sum(t["targeted"] for t in targets)
    print(f"{len(targets)} criteria with name+instruction resolved; "
          f"{nt} targeted for rephrasing (modal_share>{MODAL_TRIGGER} or na_rate>{NA_TRIGGER})")
    for t in targets:
        if t["targeted"]:
            print(f"  {t['tag']} modal={t['modal_share']:.2f} na={t['na_rate']:.2f} "
                  f"fid={t['incumbent_fidelity']:.3f} (house-style pass: "
                  f"{t['phrasing_pass_already_applied']})  {t['name']}")


if __name__ == "__main__":
    main()
