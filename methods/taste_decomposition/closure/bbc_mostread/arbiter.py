#!/usr/bin/env python3
"""Frontier-arbiter prompt builder + ruling applier for routing disputes.

The freeze requires an arbiter on every audit/proposer disagreement.  Unlike the
auditor, the arbiter SEES provenance by design -- it is adjudicating between the
proposing track and the blind audit, so it has to know which is which.

  python arbiter.py build --tag cap_crowd_r2 [--tag ...]   -> <tags>_arbiter_prompt.txt
  python arbiter.py apply --raw <file>                     -> <cell>_r<r>_arbiter.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cells as C

HERE = Path(__file__).resolve().parent

HEAD = """You are the ARBITER for a preregistered criterion-routing protocol.

Two independent parties disagree about where a scoring criterion belongs:

  TRACK A = the criterion measures genuine evidence about the construct named below
            (substance, reasoning, evidence, craft, content). Track-A criteria JOIN the
            articulated-criterion bank and are allowed to raise the bank's measured AUC.
  TRACK B = the criterion measures something predictive but NOT the construct: length,
            format, boilerplate, community style, topic markers, temporal tells, or a
            textual fingerprint of some upstream circumstance of the item's production
            (who made it, with what resources, when, with what help). Track-B channels
            become DECLARED NUISANCES and are used only to discount the other models.

For each dispute below you see: the criterion, the track that proposed it, that
proposer's rationale, and an independent blind auditor's contrary verdict. Rule on each.

Decide on the criterion's TEXT and what it would actually make a judge measure -- not on
who proposed it and not on which answer is more generous. The decisive question is:
if a judge scored exactly this instruction, would a high score be evidence ABOUT the
construct, or evidence about something that merely travels with it?

Also return `mixed`: true if you route it B but its upstream cause plausibly produces
genuine quality as well (these are reported in both discounted and undiscounted readouts).

DISPUTES:

"""

TAIL = """

OUTPUT. Emit exactly one JSON object and nothing else:

{"rulings": [
  {"cell": "<cell>", "round": <round>, "blind_id": "<id>", "route": "A" or "B",
   "mixed": true or false, "reason": "<two sentences max>"},
  ... one per dispute ...
]}
"""


def cmd_build(a):
    body, n = [], 0
    for tag in a.tags.split(","):
        tag = tag.strip()
        cell, rnd = tag.rsplit("_r", 1)
        fin = json.loads((HERE / f"{tag}_routing_final.json").read_text())
        sel = {x["blind_id"]: x for x in
               json.loads((HERE / f"{tag}_species.json").read_text())["selected"]}
        m = C.CELL_META[cell]
        for f in fin["final"]:
            if f["agree"]:
                continue
            s = sel[f["blind_id"]]
            n += 1
            body.append(
                f"--- DISPUTE cell={cell} round={rnd} id={f['blind_id']} ---\n"
                f"CORPUS: {m['corpus']}   ITEM: {m['item']}   CONSTRUCT: {m['construct']}\n"
                f"CRITERION NAME: {f['name']}\n"
                f"SCORING INSTRUCTION: {s['instruction']}\n"
                f"PROPOSED BY TRACK: {f['proposed_track']}\n"
                f"PROPOSER'S UPSTREAM PARENT TAG: {s.get('upstream_parent')}\n"
                f"PROPOSER'S RATIONALE: {(s.get('rationale') or '')[:600]}\n"
                f"BLIND AUDITOR SAID: {f['audit_label']} -- {f['audit_justification']}")
    out = HERE / f"{a.out}_arbiter_prompt.txt"
    out.write_text(HEAD + "\n\n".join(body) + TAIL)
    print(f"{n} disputes -> {out.name} ({out.stat().st_size} bytes)")


def cmd_apply(a):
    rul = json.loads(Path(a.raw).read_text())["rulings"]
    by = {}
    for r in rul:
        key = f"{r['cell']}_r{r.get('round', a.default_round)}"
        by.setdefault(key, []).append({"blind_id": r["blind_id"], "route": r["route"],
                                       "mixed": bool(r.get("mixed", False)),
                                       "reason": r["reason"]})
        print(f"{key:18s} {r['blind_id']} -> {r['route']} mixed={r.get('mixed')} | "
              f"{r['reason'][:100]}")
    for key, rs in by.items():
        (HERE / f"{key}_arbiter.json").write_text(json.dumps(
            {"arbiter": "claude-opus (frontier; provenance visible by design)",
             "rulings": rs}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--tags", required=True)
    b.add_argument("--out", required=True)
    p = sub.add_parser("apply")
    p.add_argument("--raw", required=True)
    p.add_argument("--default-round", type=int, default=2)
    a = ap.parse_args()
    {"build": cmd_build, "apply": cmd_apply}[a.cmd](a)
