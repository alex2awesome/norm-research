#!/usr/bin/env python3
"""DECOMPOSITION-ONLY round assembly (FREEZE ADDENDUM 3), no fresh proposer fleet.

WHY A CELL-SPECIFIC SCRIPT, recorded rather than silent.  `decompose_round.py merge`
composes a round's scored set as 12 fleet-A + 3 candidate-real + 7 fleet-B + 3 surface,
i.e. it assumes the round ALSO ran a sealed fleet.  On this cell round 3 is a
decomposition pass and NOTHING ELSE, following the mathse_vote precedent
(`decompose_r2.py`: "round 2 runs the decomposition pass ONLY -- no fresh proposer
fleet").  Keeping the fleet out is what makes the round unambiguously exempt from the
stopping rule: the registered 2026-08-08 rule is that a sub-epsilon round counts toward
stopping only if it was a PROPOSING round, and a round with no proposers cannot be one.

TWO-TIER RULE, applied here.  Decomposition components are DIRECTED: they are authored
against four named parents rather than independently proposed from the slice, so they are
TIER D.  They may be scored and may join the bank through the ordinary blind audit, but
they are excluded from every Good-Turing / missing-mass quantity.  This script therefore
writes a species file whose `tracks` block carries NO good_turing estimate and an explicit
`tier: D` marker, so nothing downstream can accidentally count them as sealed-fleet
species.

FREEZE ADDENDUM 3 also requires that each component be routed through the blind audit
INDEPENDENTLY -- that happens next, because `audit.py build` reads the `selected` list
this script writes and is blind to `origin`.

CPU only.  Usage: python3 decompose_only_merge.py --cell jokes_community --round 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad") / HERE.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--round", required=True)
    a = ap.parse_args()
    tag = f"{a.cell}_r{a.round}"

    import harness_maps as H
    txt = (SCRATCH / tag / "out_decomposer.txt").read_text().strip()
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[1].rsplit("```", 1)[0]
    obj = json.loads(H.JSON_RE.search(txt).group(0))
    comps = obj["components"]

    parents = json.loads((HERE / f"{tag}_parents_used.json").read_text())["parents"]
    pnames = [p["criterion"] for p in parents]

    merged, na, nb = [], 0, 0
    for c in comps:
        real = c["kind"] == "candidate_real"
        if real:
            na += 1; bid = f"A{na:02d}"
        else:
            nb += 1; bid = f"B{nb:02d}"
        rec = {"blind_id": bid, "track": "A" if real else "B",
               "name": c["name"].strip(), "instruction": c["instruction"].strip(),
               "rationale": c.get("rationale", ""), "pid": c["id"],
               "proposer": "decomposer_opus", "family": "claude", "tier": "D",
               "origin": "addendum3_decomposition", "parent": c["parent"],
               "component_kind": c["kind"], "n_proposers_naming": 1, "n_members": 1,
               "member_names": [c["name"].strip()]}
        if not real:
            rec["upstream_parent"] = f"surface carrier of parent channel: {c['parent']}"
            rec["mixed_proposed"] = False
            rec["mixed_any_member"] = False
        merged.append(rec)

    unknown = sorted({c["parent"] for c in comps} - set(pnames))
    spec = {
        "tag": tag, "cell": a.cell, "round": a.round,
        "round_kind": "ADDENDUM-3 DECOMPOSITION ONLY (no sealed fleet)",
        "stopping_rule_note": "this round has no proposers, so it cannot be a PROPOSING "
                              "round and never counts toward the sub-epsilon stopping "
                              "rule (registered 2026-08-08)",
        "tier": "D",
        "two_tier_note": "directed components: scored and auditable, but excluded from "
                         "every Good-Turing / missing-mass quantity",
        "parents_decomposed": pnames,
        "parents_unmatched_in_output": unknown,
        "n_components": len(comps),
        "tracks": {},          # deliberately empty: TIER D contributes no species mass
        "selected": merged,
    }
    (HERE / f"{tag}_species.json").write_text(json.dumps(spec, indent=1))
    print(f"{tag}: {len(merged)} decomposition components "
          f"({na} candidate-real / {nb} surface) from {len(pnames)} parents; TIER D, "
          f"no Good-Turing contribution")
    if unknown:
        print("  WARNING parents in output not in the parent list:", unknown)
    for m in merged:
        print(f"  {m['blind_id']} [{m['component_kind']:14s}] {m['name'][:52]:52s} "
              f"<- {m['parent'][:40]}")


if __name__ == "__main__":
    main()
