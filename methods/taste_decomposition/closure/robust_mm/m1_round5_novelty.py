#!/usr/bin/env python3
"""M1 live probe -- what a 5th round with a SEALED FLEET would have proposed.

The pilot stopped at round 4 with two consecutive sub-epsilon rounds.  This asks the
question the stopping rule cannot: if six independent proposers, none of whom can see
the bank, read the round-4 disagreement slice, how much of what they name is ALREADY
in the round-4 bank (54 original concepts + the 50 mined species), and how much is new?

  * recapture of the EXISTING bank by sealed proposers  -> the detector's sensitivity
    on concepts that are genuinely present and genuinely findable from this slice;
  * novel species                                       -> the articulable signal the
    4-round stop actually left on the table, in concept units.

No scoring: novel species are counted and banked, not judged (design Sec 8 cost note).
CPU only.  Usage: python m1_round5_novelty.py [--tau 0.79]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLOSURE = HERE.parent
sys.path.insert(0, str(HERE))
import embed_lib as E  # noqa: E402


def mined_criteria():
    out = []
    for r in (1, 2, 3, 4):
        prop = {c["id"]: c for c in json.loads(
            (CLOSURE / f"round{r}_proposals_blinded.json").read_text())["criteria"]}
        routing = json.loads((CLOSURE / f"round{r}_routing_final.json").read_text())
        rep = json.loads((CLOSURE / f"round{r}_score_report.json").read_text())
        collapsed = {k for k, v in rep["per_criterion"].items() if v.get("collapsed")}
        for x in routing["final"]:
            if x["final_route"] == "A" and x["blind_id"] not in collapsed:
                c = prop[x["blind_id"]]
                out.append({"src": f"mined_r{r}", "name": c["name"],
                            "text": E.crit_text(c["name"], c["instruction"])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=E.TAU)
    ap.add_argument("--tag", default="round5")
    a = ap.parse_args()

    bankmap = E.bank_concept_texts()
    cfg = json.loads((HERE / "m3_concepts.json").read_text())
    surviving = list(cfg["concept_footprints"].keys())          # the 54 effective concepts
    existing = ([{"src": "bank54", "name": c, "text": E.crit_text(c, bankmap[c])} for c in surviving]
                + mined_criteria())

    props = json.loads((HERE / f"proposals_{a.tag}.json").read_text())["proposals"]
    Ee = E.embed([e["text"] for e in existing])
    Ep = E.embed([E.crit_text(p["name"], p["instruction"]) for p in props])
    S = Ep @ Ee.T                                               # N_prop x N_existing

    rows = []
    for j, p in enumerate(props):
        b = int(np.argmax(S[j]))
        rows.append({"pid": p["pid"], "proposer": p["proposer"], "family": p["family"],
                     "name": p["name"], "max_cos_to_existing": float(S[j, b]),
                     "nearest_existing": existing[b]["name"], "nearest_src": existing[b]["src"]})

    # species structure among the fleet's own proposals
    Pp = Ep @ Ep.T
    np.fill_diagonal(Pp, -1.0)
    lab = E.single_linkage(Pp, a.tau)
    sizes = np.bincount(lab)

    out = {"tag": a.tag, "tau": a.tau,
           "n_existing_concepts": len(existing),
           "n_existing_bank54": len(surviving), "n_existing_mined": len(existing) - len(surviving),
           "n_proposals": len(props), "n_proposers": len({p["proposer"] for p in props}),
           "proposals": rows}

    for t in (0.77, a.tau, 0.81):
        recap = np.array([r["max_cos_to_existing"] >= t for r in rows])
        # species-level: a species is NOVEL if none of its members recaptures anything
        novel_sp = sum(1 for s in range(len(sizes))
                       if not recap[np.where(lab == s)[0]].any())
        out[f"tau{t}"] = {
            "proposal_level_recapture_rate": float(recap.mean()),
            "n_proposals_recapturing_existing": int(recap.sum()),
            "n_species_total": int(len(sizes)),
            "n_species_novel": int(novel_sp),
            "n_species_recapturing": int(len(sizes) - novel_sp),
            "fraction_species_novel": float(novel_sp / len(sizes)),
            "n_existing_concepts_recaptured": int(((S >= t).any(axis=0)).sum()),
            "existing_recapture_rate": float(((S >= t).any(axis=0)).mean()),
            "bank54_recapture_rate": float(((S[:, :len(surviving)] >= t).any(axis=0)).mean()),
            "mined_recapture_rate": float(((S[:, len(surviving):] >= t).any(axis=0)).mean()),
        }

    out["max_cos_summary"] = {
        "median": float(np.median([r["max_cos_to_existing"] for r in rows])),
        "p90": float(np.percentile([r["max_cos_to_existing"] for r in rows], 90)),
        "max": float(max(r["max_cos_to_existing"] for r in rows)),
    }
    out["top_recaptures"] = sorted(rows, key=lambda r: -r["max_cos_to_existing"])[:10]

    (HERE / f"m1_novelty_{a.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("proposals", "top_recaptures")}, indent=1))
    print("\nwrote", HERE / f"m1_novelty_{a.tag}.json")


if __name__ == "__main__":
    main()
