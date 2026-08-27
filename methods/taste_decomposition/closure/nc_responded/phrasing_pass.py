#!/usr/bin/env python3
"""LABEL-BLIND house-style phrasing pass ("GEPA-style") + concept-fidelity gate.

Why this exists and what it is NOT.  The freeze requires "GEPA-iterated phrasing per
the A-bank standard" before any confirmatory Delta is quoted.  The repo's GEPA
machinery optimises rubric wording against a LABELLED signal; using it here would
break this campaign's hard label-blindness rule (proposers and criteria never see y).
So the pass implemented here is the label-blind half of that standard:

  step 1  a frontier rewriter puts every selected criterion into the A-bank house
          style -- one scorable property, explicit 0 and 10 anchors, judgeable from
          the comment text alone, no reference to any outcome, and specifically no
          instruction to score DOCUMENT SHAPE in place of merit (the pilot's
          twice-caught authoring failure mode);
  step 2  a blind FIDELITY GATE: two sealed judges see the original and rewritten
          instruction as an unlabelled X/Y pair (interleaved with the same authored
          anchor battery used elsewhere) and answer "same underlying concept?".  A
          rewrite the judges call DIFFERENT is REJECTED and the original phrasing is
          scored instead -- so the pass can only improve phrasing, never silently
          swap the concept.

Every quoted number in this campaign is post-phrasing-pass, and the rejection count
is reported per round.

Usage:
  python phrasing_pass.py build    --round 1     # writes the rewriter prompt
  python phrasing_pass.py gate     --round 1     # writes the fidelity-check prompt
  python phrasing_pass.py finalize --round 1     # applies verdicts -> roundN_criteria_final.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

REWRITE_INSTRUCTION = """You are an editor for a bank of scoring criteria used to judge PUBLIC
COMMENTS submitted to United States federal agencies on proposed rules. Each criterion
is scored 0-10 by an independent judge who sees ONLY one comment's text and ONE
criterion.

Rewrite EVERY criterion below into the bank's house style. House style is:

  * ONE scorable property per criterion, stated in the first sentence;
  * an explicit description of what a 10 looks like and what a 0 looks like;
  * judgeable from the comment text alone, by a reader with no outside information;
  * plain declarative sentences, 2-4 sentences, no more than ~80 words;
  * NO reference to any outcome (agency response, rule change, acceptance, approval);
  * NO instruction that tells the judge to score the SHAPE of the document in place of
    its substance (e.g. "score the question-and-answer structure, not the importance of
    the question"). If the underlying property is substantive, the instruction must
    score the substance; if the underlying property is a surface feature, say so
    plainly and score the surface feature, without dressing it as merit.

HARD CONSTRAINT: preserve the CONCEPT exactly. You may sharpen, disambiguate and
anchor the wording. You may NOT broaden it, narrow it, or replace it with a nearby
idea. An independent judge must agree that the rewritten criterion measures the same
thing as the original.

Emit exactly one JSON object and nothing else:

{"rewrites": [
  {"id": "<the id given below, unchanged>",
   "name": "<short name, <= 12 words; keep the original unless it is unclear>",
   "instruction": "<the rewritten 0-10 scoring instruction>"},
  ... one entry for EVERY criterion below ...
]}
"""

GATE_INSTRUCTION = """You are auditing edits to a bank of scoring criteria for PUBLIC COMMENTS
submitted to United States federal agencies on proposed rules.

For each PAIR below, decide ONE question:

  Would an independent judge, scoring public comments against criterion X and against
  criterion Y, be measuring THE SAME UNDERLYING CONCEPT?

Answer SAME only if the two would produce essentially interchangeable scores. Answer
DIFFERENT if a comment could plausibly score high on one and low on the other --
including when one is a broadened, narrowed, or shifted version of the other. A
sharper wording of the same property is SAME; a different property is DIFFERENT.

Emit exactly one JSON object and nothing else:

{"verdicts": [
  {"pair_id": "GP001", "verdict": "SAME" or "DIFFERENT", "confidence": "high"/"medium"/"low",
   "reason": "<one sentence>"},
  ... one entry for EVERY pair ...
]}
"""

# same authored anchor battery used by the species instrument (missing-mass PART-4 fix 5)
from fleet_species import ANCHORS  # noqa: E402


def cmd_build(a):
    sel = json.loads((HERE / f"round{a.round}_selection.json").read_text())
    items = [{"id": c["id"], "name": c["name"], "instruction": c["instruction"]}
             for t in ("A", "B") for c in sel[t]]
    body = REWRITE_INSTRUCTION + "\n\n" + "\n\n".join(
        f"--- CRITERION {c['id']} ---\nNAME: {c['name']}\nINSTRUCTION: {c['instruction']}"
        for c in items) + "\n"
    (HERE / f"round{a.round}_phrasing_prompt.txt").write_text(body)
    print(f"r{a.round}: {len(items)} criteria -> round{a.round}_phrasing_prompt.txt ({len(body)} chars)")


def cmd_gate(a):
    sel = json.loads((HERE / f"round{a.round}_selection.json").read_text())
    orig = {c["id"]: c for t in ("A", "B") for c in sel[t]}
    rw = {r["id"]: r for r in json.loads(
        (HERE / f"round{a.round}_phrasing_rewrites.json").read_text())["rewrites"]}

    items = []
    for cid in orig:
        if cid not in rw:
            continue
        flip = int(hashlib.sha256(f"gp|{a.round}|{cid}".encode()).hexdigest(), 16) % 2
        X = {"name": orig[cid]["name"], "instruction": orig[cid]["instruction"]}
        Y = {"name": rw[cid]["name"], "instruction": rw[cid]["instruction"]}
        if flip:
            X, Y = Y, X
        items.append({"kind": "real", "cid": cid, "X": X, "Y": Y})
    for n, anc in enumerate(ANCHORS):
        flip = int(hashlib.sha256(f"ganchor|{a.round}|{n}".encode()).hexdigest(), 16) % 2
        X, Y = (anc["X"], anc["Y"]) if not flip else (anc["Y"], anc["X"])
        items.append({"kind": "anchor", "anchor_label": anc["label"], "X": X, "Y": Y})

    items.sort(key=lambda p: hashlib.sha256(
        f"gs|{a.round}|{p['X']['name']}|{p['Y']['name']}".encode()).hexdigest())
    for k, it in enumerate(items):
        it["shown_id"] = f"GP{k + 1:03d}"

    body = GATE_INSTRUCTION + "\n\n" + "\n\n".join(
        f"--- PAIR {it['shown_id']} ---\n"
        f"X NAME: {it['X']['name']}\nX INSTRUCTION: {it['X']['instruction']}\n"
        f"Y NAME: {it['Y']['name']}\nY INSTRUCTION: {it['Y']['instruction']}"
        for it in items) + "\n"
    (HERE / f"round{a.round}_phrasing_gate_prompt.txt").write_text(body)
    (HERE / f"round{a.round}_phrasing_gate_key.json").write_text(json.dumps(
        {"items": [{k: v for k, v in it.items() if k not in ("X", "Y")} for it in items]}, indent=1))
    print(f"r{a.round}: {len(items)} gate pairs -> round{a.round}_phrasing_gate_prompt.txt")


def cmd_finalize(a):
    sel = json.loads((HERE / f"round{a.round}_selection.json").read_text())
    rw = {r["id"]: r for r in json.loads(
        (HERE / f"round{a.round}_phrasing_rewrites.json").read_text())["rewrites"]}
    key = {it["shown_id"]: it for it in json.loads(
        (HERE / f"round{a.round}_phrasing_gate_key.json").read_text())["items"]}

    def load(p):
        return {v["pair_id"]: v for v in json.loads((HERE / p).read_text())["verdicts"]}

    j1 = load(f"round{a.round}_phrasing_gate_judge1.json")
    j2 = load(f"round{a.round}_phrasing_gate_judge2.json")

    verdict, anchors = {}, []
    for sid, it in key.items():
        v1, v2 = j1.get(sid), j2.get(sid)
        if not v1 or not v2:
            continue
        same = v1["verdict"] == "SAME" and v2["verdict"] == "SAME"
        if it["kind"] == "anchor":
            anchors.append({"label": it["anchor_label"], "j1": v1["verdict"], "j2": v2["verdict"],
                            "j1_correct": v1["verdict"] == it["anchor_label"],
                            "j2_correct": v2["verdict"] == it["anchor_label"]})
        else:
            verdict[it["cid"]] = {"same": same, "j1": v1["verdict"], "j2": v2["verdict"]}

    out = {"round": a.round, "A": [], "B": [], "rejected": [], "anchor_battery": anchors}
    for t in ("A", "B"):
        for c in sel[t]:
            cid = c["id"]
            v = verdict.get(cid)
            use_rw = bool(v and v["same"] and cid in rw)
            rec = dict(c)
            rec["phrasing"] = "rewritten" if use_rw else "original"
            if use_rw:
                rec["original_name"], rec["original_instruction"] = c["name"], c["instruction"]
                rec["name"], rec["instruction"] = rw[cid]["name"], rw[cid]["instruction"]
            else:
                rec["gate_verdict"] = v
                if cid in rw:
                    out["rejected"].append({"id": cid, "name": c["name"], "verdict": v})
            out[t].append(rec)
    out["n_rewritten"] = sum(c["phrasing"] == "rewritten" for t in ("A", "B") for c in out[t])
    out["n_rejected"] = len(out["rejected"])
    out["anchor_scores"] = {
        "j1": f"{sum(x['j1_correct'] for x in anchors)}/{len(anchors)}",
        "j2": f"{sum(x['j2_correct'] for x in anchors)}/{len(anchors)}"}
    (HERE / f"round{a.round}_criteria_final.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("A", "B", "anchor_battery")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("build", "gate", "finalize"):
        s = sub.add_parser(c)
        s.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    {"build": cmd_build, "gate": cmd_gate, "finalize": cmd_finalize}[a.cmd](a)
