#!/usr/bin/env python3
"""CROSS-CELL TRANSPLANT (TIER D) -- the controlled form of the registered P1/P2 test.

WHY.  notes/2026-08-09__closure_cap_crowd.md section 4 registered a prediction about
cap_finalist's sign-contradicting craft criteria flipping positive on cap_crowd.  The
first attempt tested it by joining criterion NAMES across the two cells' rounds 1-2 and
found ZERO shared names -- those rounds were proposed independently on each cell, before
the coordinator's same-names rule existed, so a name join cannot test anything.  Recorded
rather than quietly dropped.

This is the controlled version and it is strictly better: take cap_finalist's criterion
TEXT verbatim and score it on cap_crowd, same judge, same bank, same cartoon+caption item
view.  Then the only thing that differs between the two alone-AUCs is THE LABEL --
editor selection versus crowd vote -- which is exactly the curation-vs-community contrast
the two campaigns exist to draw.

TIER D (directed).  These criteria were authored against a different cell and transplanted
here, not proposed from this cell's slice, so by the two-tier rule they are excluded from
every Good-Turing / missing-mass quantity and they do NOT join this cell's bank or its
closure curve.  They are a measurement, and the species file carries `tier: D` so nothing
downstream can count them.

SELECTION, fixed here before the scores exist: all 15 criteria named in cap_finalist
section 10.7 as sign-contradicting, plus the 10 highest-alone-AUC cap_finalist criteria
(its positive-signed end).  Two signed ends give P2's rank correlation something to
correlate over.

CPU only.  Usage: python build_transplant.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIN = HERE.parent / "cap_finalist"

from sign_flip_test import SIGN_TRIGGERED_ON_FINALIST  # the registered list

N_TOP = 10


def main():
    # criterion text, by name, from every cap_finalist round's species file
    text = {}
    for r in ("1", "2", "4", "5"):
        p = FIN / f"cap_finalist_r{r}_species.json"
        if not p.exists():
            continue
        for c in json.loads(p.read_text())["selected"]:
            text.setdefault(c["name"], {"instruction": c["instruction"], "round": r,
                                        "blind_id": c["blind_id"], "track": c["track"]})

    gp = json.loads((FIN / "cap_finalist_gepa_targets.json").read_text())
    auc = {c["name"]: c["alone_AUC_fitmine"] for c in gp["criteria"]
           if c.get("alone_AUC_fitmine") is not None}

    chosen, why = [], {}
    for nm in SIGN_TRIGGERED_ON_FINALIST:
        if nm in text:
            chosen.append(nm)
            why[nm] = "sign_contradicting_on_cap_finalist"
    for nm, a in sorted(auc.items(), key=lambda kv: -kv[1]):
        if len(why) >= len(SIGN_TRIGGERED_ON_FINALIST) + N_TOP:
            break
        if nm in text and nm not in why:
            chosen.append(nm)
            why[nm] = "top_alone_AUC_on_cap_finalist"

    selected, prov = [], []
    for j, nm in enumerate(chosen):
        bid = f"X{j+1:02d}"
        selected.append({"blind_id": bid, "name": nm,
                         "instruction": text[nm]["instruction"],
                         "track": text[nm]["track"]})
        prov.append({"blind_id": bid, "name": nm,
                     "origin_cell": "cap_finalist",
                     "origin_round": text[nm]["round"],
                     "origin_blind_id": text[nm]["blind_id"],
                     "origin_track": text[nm]["track"],
                     "selection_reason": why[nm],
                     "alone_AUC_on_cap_finalist_fitmine": auc.get(nm)})

    out = {
        "tag": "cap_crowd_x1", "cell": "cap_crowd", "round": "x1", "tier": "D",
        "kind": "CROSS-CELL TRANSPLANT of cap_finalist criterion TEXT, scored on cap_crowd "
                "with the same judge, bank and cartoon+caption item view. Tests the "
                "registered P1/P2 predictions with the label as the only difference.",
        "two_tier_rule": "TIER D: directed, not proposed from this cell's slice. Excluded "
                         "from every Good-Turing / missing-mass quantity; does NOT join "
                         "this cell's bank and does NOT enter the closure curve.",
        "composition": {
            "n": len(selected),
            "sign_contradicting_on_cap_finalist": sum(
                1 for v in why.values() if v.startswith("sign")),
            "top_alone_AUC_on_cap_finalist": sum(
                1 for v in why.values() if v.startswith("top")),
        },
        "provenance": prov, "selected": selected,
    }
    (HERE / "cap_crowd_x1_species.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["composition"], indent=1))
    missing = [n for n in SIGN_TRIGGERED_ON_FINALIST if n not in text]
    if missing:
        print("NOT FOUND in cap_finalist species files (recorded):", missing)
    print(f"wrote cap_crowd_x1_species.json ({len(selected)} criteria, TIER D)")


if __name__ == "__main__":
    main()
