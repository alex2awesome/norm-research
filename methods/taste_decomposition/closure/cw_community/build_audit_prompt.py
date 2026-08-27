#!/usr/bin/env python3
"""Emit the blind ROUTING-AUDIT prompt (and, with --disputes, the arbiter prompt).

Usage:
  python build_audit_prompt.py --round 1                > /tmp/audit_r1.txt
  python build_audit_prompt.py --round 1 --disputes     > /tmp/arbiter_r1.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

AUDIT = """You are auditing a pool of {n} candidate scoring criteria for a study of
short stories posted to an online creative-writing community. Authorship and
provenance have been stripped; some were written to capture genuine story quality and
some were written to capture predictive-but-spurious channels, and you are NOT told
which is which.

For each item decide:
  quality_relevant -- a careful editor would treat this property as evidence about how
    good the story IS.
  incidental -- the property might well predict how a story is received, but a careful
    editor would not treat it as evidence of the story's quality (surface form,
    platform furniture, fashionable topic, a fingerprint of who wrote it or when it was
    posted).

Judge the criterion AS WRITTEN -- what its scoring instruction actually measures -- not
the concept it gestures at. Do NOT read any file; work only from this message.

Output exactly one JSON object and nothing else:
{{"verdicts": [{{"aid": "...", "verdict": "quality_relevant"|"incidental",
                "confidence": 0.0-1.0, "reason": "<= 25 words"}}, ...]}}
Every aid below must appear exactly once.

--- POOL ---
"""

ARBITER = """Two independent processes disagree about how to route these scoring
criteria for a study of short stories posted to an online creative-writing community.
For each item you are given the criterion, the track it was PROPOSED under (A = its
author intended it as quality-relevant, B = its author intended it as a
suspected-spurious channel) and a blind auditor's verdict with its reason. You are the
arbiter; your call is final.

Rule: a criterion belongs on the A side only if a careful editor would treat the
property as evidence about how good the story IS. If the property mainly tracks who
wrote it, when it was posted, what platform conventions it follows, or how it looks
rather than how it reads, it belongs on the B side. When a property genuinely does
both, say so and pick the side its SCORING INSTRUCTION actually measures.

Output exactly one JSON object and nothing else:
{{"decisions": [{{"aid": "...", "final": "A"|"B", "reason": "<= 30 words"}}, ...]}}

--- DISPUTED ITEMS ---
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--disputes", action="store_true")
    a = ap.parse_args()
    r = a.round
    blind = json.loads((HERE / f"round{r}_audit_blind.json").read_text())
    if not a.disputes:
        print(AUDIT.format(n=len(blind)))
        for b in blind:
            print(f"aid: {b['aid']}\n  name: {b['name']}\n  instruction: {b['instruction']}\n")
        return

    key = json.loads((HERE / f"round{r}_audit_key.json").read_text())
    prov = {c["cid"]: c for c in json.loads(
        (HERE / f"round{r}_proposals_provenance.json").read_text())["criteria"]}
    verd = {d["aid"]: d for d in json.loads(
        (HERE / f"round{r}_audit_verdicts.json").read_text())}
    by_aid = {b["aid"]: b for b in blind}
    n = 0
    lines = []
    for aid, cid in key.items():
        v = verd[aid]
        proposed = prov[cid]["proposed_track"]
        audit = "A" if v["verdict"] == "quality_relevant" else "B"
        if audit == proposed:
            continue
        n += 1
        b = by_aid[aid]
        lines.append(f"aid: {aid}\n  name: {b['name']}\n  instruction: {b['instruction']}\n"
                     f"  proposed_track: {proposed}\n  auditor_verdict: {audit} "
                     f"(confidence {v.get('confidence')}) -- {v.get('reason')}\n")
    print(ARBITER)
    print(f"({n} disputed items)\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
