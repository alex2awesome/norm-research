#!/usr/bin/env python3
"""FREEZE ADDENDUM 3 -- the DECOMPOSITION-FIRST pass for the code_v3 cell, run BEFORE
round 1, exactly as the campaign brief directs ("decomposition-first pass on MIXED
channels first").

WHY THIS CELL DECOMPOSES COMMUNICATION QUALITY.  The enriched rescore found that the
single most predictive articulated criterion in the code cell is *"is this change
communicated well?"* -- a78 "change/commit/PR communication quality" tops BOTH splits
univariately (.550 eval / .552 test), with a105 "change description clarity,
completeness and rationale" second on eval (.549) and a123 "contribution readiness and
submission norms" second on test (.547).  The round-0 concept census then showed that
a78 and a105 are ONE concept: the bank's #1 and #2 criteria on eval are a single concept
measured twice.  That concept is MIXED in the Addendum-2 sense -- "well communicated"
plausibly mixes (i) a real property of the contribution (the author actually explains
what the change is for and what a reviewer needs to know) with (ii) submission-hygiene
surface that a judge can see and be swayed by (a filled-in PR template, a checklist, a
conventional-commit subject, an issue reference).  The brief's own decomposition is
adopted verbatim: description completeness, motivation clarity, reviewer-oriented
framing (candidate-real) versus checklist/boilerplate compliance (surface).

The pass also carries FREEZE ADDENDUM 4 position-in-container channels.  The container
here is the REPOSITORY and the position is where the PR falls in that repository's own
PR-number timeline -- the known repo-local recency channel measured in §4 of the note.

  build   -> writes the sealed DECOMPOSER prompt (label-blind: parent name, parent
             rubric text, the surface features it interacts with, corpus, construct)
  collect -> parses the decomposer output into code_v3_rd_species.json

CPU only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cells as C                                            # noqa: E402

SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/code_v3")
N_POSITION = 5
CRIT = HERE / "abank_rescore" / "criteria_code_abank.jsonl"

HEAD = """You are the DECOMPOSER in a preregistered measurement protocol. You are working
label-blind: you have not been shown any outcome, decision, merge status, vote or score
for any item, and nothing you write may refer to one.

CORPUS: {corpus}
ITEM: a {item}
CONSTRUCT the scorecard is trying to measure: {construct}

BACKGROUND. A scorecard of explicitly written criteria is judged 0-10 per criterion by
an LLM judge and aggregated. Two screens have flagged the criteria below as MIXED
channels. A MIXED channel is one whose judged score plausibly mixes (i) a real property
of the contribution that a careful maintainer would count as substance, with (ii) a
submission-hygiene or document-shape habit that merely travels with it and that a judge
can see and be swayed by -- a filled-in pull-request template, a checklist, a
conventional-commit subject line, an issue reference, tidy section headings, a long
description that says little.

YOUR TASK, PART 1 -- DECOMPOSE. For each parent criterion below, author EXACTLY TWO
replacement criteria:

  * a CANDIDATE-REAL component: the substantive property the parent was trying to name,
    rewritten so that a judge scoring it is forced to judge the SUBSTANCE and is
    explicitly told NOT to reward the hygiene carrier (template sections present,
    checklist ticked, headings, length, issue links, commit-message convention). It must
    be possible for a three-sentence plainly-typed description with no template to score
    10, and for a long, immaculately templated, fully checklisted description to score 0.

  * a SURFACE component: the hygiene or shape habit itself, phrased as a pure EXTENT
    question that a judge can count or estimate from the text, with an explicit
    instruction that the judge must NOT decide whether the habit is good.

The two components must be SEPARABLE: it must be easy to describe a pull request that
scores high on one and low on the other, in BOTH directions. State that dissociation in
your rationale for each pair.

PARENTS TO DECOMPOSE:

{parents}

YOUR TASK, PART 2 -- POSITION FINGERPRINTS. Author EXACTLY {n_pos} further channels of a
different kind. Every pull request sits inside a CONTAINER -- one repository -- alongside
all the other pull requests that repository has ever received, and it has a POSITION in
that container: how early or late it arrived in the repository's own history, how busy
the repository was around it, and how mature the codebase and its conventions were by
then. You cannot see the position. You CAN sometimes see a TEXTUAL FINGERPRINT of it --
for example a change that reads as arriving into a young, convention-less codebase
versus one that reads as arriving into a mature project with settled idioms and an
established review culture; a description that presupposes shared project history or an
ongoing migration; tooling, dependency or language-version vocabulary that pins the
change to a moment in the project's life; review comments whose register implies a
long-standing relationship between author and maintainers; or the signature of an
automated dependency-bump or bot-generated submission. Author {n_pos} such fingerprint
channels, each a pure extent question scorable from the pull-request text alone, each
explicitly not a quality judgement.

HARD CONSTRAINTS FOR EVERYTHING YOU WRITE.
1. LABEL-BLIND: no reference to any outcome, merge, close, approval, rejection, review
   verdict, rank or score.
2. Every criterion is scored 0-10 by an independent judge reading only the pull-request
   text (title, description, review comments, diff). Say what a 10 looks like and what a
   0 looks like.
3. Do not write a criterion that scores DOCUMENT SHAPE in place of substance. A criterion
   whose high end can be reached by filling in a template alone belongs on the surface
   side, and you must put it there.
4. Names <= 12 words.

OUTPUT. Emit exactly one JSON object and nothing else:

{{"components": [
  {{"id": "D01", "parent": "<parent criterion name>", "kind": "candidate_real",
    "name": "<name>", "instruction": "<0-10 scoring instruction>",
    "rationale": "<why this isolates the substance; state the two-way dissociation>"}},
  {{"id": "D02", "parent": "<same parent>", "kind": "surface",
    "name": "<name>", "instruction": "<0-10 extent instruction, explicitly not a quality judgement>",
    "rationale": "<why this is the carrier; state the two-way dissociation>"}},
  ... two entries per parent, in the order the parents are listed ...
 ],
 "position_channels": [
  {{"id": "Q01", "name": "<name>",
    "instruction": "<0-10 extent instruction, explicitly not a quality judgement>",
    "upstream_parent": "<the position fact you conjecture is upstream>",
    "mixed": true or false,
    "rationale": "<what position this fingerprints and why it is not the construct>"}},
  ... exactly {n_pos} entries ...
 ]}}
"""


def cmd_build(args):
    meta = C.CELL_META["code_v3"]
    par = json.loads((HERE / "parents_code.json").read_text())
    ordered, seen = [], set()
    for n in par["selected_parents"]:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    rub = {}
    for line in open(CRIT):
        r = json.loads(line)
        rub[r["aspect_id"]] = r
    partner = {x["criterion"]: x["top_surface_partners"] for x in par["candidates"]}

    blocks = []
    for j, n in enumerate(ordered):
        aid = n.split("|")[0]
        ps = partner.get(n, [])
        pl = ", ".join(p["feature"] for p in ps) or \
            "vt_len_desc, vt_desc_n_bullets, vt_desc_has_issue_ref"
        blocks.append(f"[P{j+1:02d}] PARENT: {rub[aid]['name']}\n"
                      f"      RUBRIC AS WRITTEN: {rub[aid]['description']}\n"
                      f"      SURFACE FEATURES IT INTERACTS WITH: {pl}")
    prompt = HEAD.format(corpus=meta["corpus"], item=meta["item"],
                         construct=meta["construct"], parents="\n\n".join(blocks),
                         n_pos=N_POSITION)
    d = SCRATCH / "code_v3_rd"
    d.mkdir(parents=True, exist_ok=True)
    (d / "prompt_decomposer.txt").write_text(prompt)
    (HERE / "code_v3_rd_parents_used.json").write_text(json.dumps(
        {"cell": "code_v3", "parents": ordered, "n_parents": len(ordered),
         "parent_names": [rub[n.split("|")[0]]["name"] for n in ordered],
         "n_position_channels": N_POSITION,
         "n_scored_total": 2 * len(ordered) + N_POSITION,
         "surface_partners": {n: partner.get(n, []) for n in ordered},
         "brief_parents_added": par["selected_parents_brief"]}, indent=1))
    print(f"code_v3: {len(ordered)} parents -> {2*len(ordered)} components + "
          f"{N_POSITION} position channels = {2*len(ordered)+N_POSITION} scored; "
          f"prompt {len(prompt)} chars -> {d/'prompt_decomposer.txt'}")


def cmd_collect(args):
    import re
    d = SCRATCH / "code_v3_rd"
    txt = (d / "out_decomposer.txt").read_text().strip()
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[1].rsplit("```", 1)[0]
    obj = json.loads(re.search(r"\{[\s\S]*\}", txt).group(0))
    used = json.loads((HERE / "code_v3_rd_parents_used.json").read_text())

    selected, na, nb = [], 0, 0
    for c in obj["components"]:
        real = c["kind"] == "candidate_real"
        na, nb = (na + 1, nb) if real else (na, nb + 1)
        rec = {"blind_id": f"A{na:02d}" if real else f"B{nb:02d}",
               "track": "A" if real else "B",
               "name": c["name"].strip(), "instruction": c["instruction"].strip(),
               "rationale": c.get("rationale", ""), "pid": c["id"],
               "proposer": "decomposer_opus", "family": "claude",
               "origin": "addendum3_decomposition", "parent": c["parent"],
               "component_kind": c["kind"], "n_proposers_naming": 1, "n_members": 1,
               "member_names": [c["name"].strip()]}
        if not real:
            rec["upstream_parent"] = f"surface carrier of bank criterion: {c['parent']}"
            rec["mixed_proposed"] = False
            rec["mixed_any_member"] = False
        selected.append(rec)
    for c in obj["position_channels"]:
        nb += 1
        selected.append({"blind_id": f"B{nb:02d}", "track": "B",
                         "name": c["name"].strip(), "instruction": c["instruction"].strip(),
                         "rationale": c.get("rationale", ""), "pid": c["id"],
                         "proposer": "decomposer_opus", "family": "claude",
                         "origin": "addendum4_position_fingerprint",
                         "upstream_parent": c.get("upstream_parent",
                                                  "position in the repository's PR timeline"),
                         "mixed_proposed": bool(c.get("mixed", False)),
                         "mixed_any_member": bool(c.get("mixed", False)),
                         "n_proposers_naming": 1, "n_members": 1,
                         "member_names": [c["name"].strip()]})

    out = {"tag": "code_v3_rd", "cell": "code_v3", "round": "d",
           "selection_rule": "FREEZE ADDENDUM 3 decomposition pass: 2 components per MIXED "
                             "parent (candidate-real + surface) authored by a sealed frontier "
                             "decomposer, plus FREEZE ADDENDUM 4 position-fingerprint channels; "
                             "every component is routed INDEPENDENTLY by the blind audit",
           "parents": used["parents"], "parent_names": used["parent_names"],
           "tracks": {}, "selected": selected}
    (HERE / "code_v3_rd_species.json").write_text(json.dumps(out, indent=1))
    print(f"code_v3_rd: {len(selected)} scored criteria "
          f"({sum(1 for s in selected if s['track']=='A')} proposed A / "
          f"{sum(1 for s in selected if s['track']=='B')} proposed B)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("build", "collect"):
        sub.add_parser(n)
    a = ap.parse_args()
    {"build": cmd_build, "collect": cmd_collect}[a.cmd](a)
