#!/usr/bin/env python3
"""FREEZE ADDENDUM 3 -- the DECOMPOSITION-FIRST pass, run BEFORE round 1.

Addendum 3 (2026-08-07): "MIXED channels ... get a DECOMPOSITION PASS: author >=2
refined criteria isolating the components (e.g. staccato lineation -> 'deliberate
one-line beats serving tension/pacing, judged in context' [candidate-real] vs
'editor-default blank-line/whitespace habits' [surface]), score each separately,
route each through the blind audit independently.  The parent channel is retired
from readouts once its components are scored."

Here the MIXED parents are the Layer-1 bank criteria whose SHAP contribution is
mediated by a programmatic length/format feature (`select_parents.py`), which is
what the Layer-1 SHAP screens on these two cells actually found.

The pass also carries the FREEZE ADDENDUM 4 position-fingerprint channels, exactly
as the N&C round-5 cap round did (16 decomposition components + 5 position
fingerprints + probe pairs).

  build   -> writes the sealed DECOMPOSER prompt (label-blind: parent name,
             parent rubric text, the surface features it interacts with, the
             corpus and the construct -- nothing else, no AUCs, no labels)
  collect -> parses the decomposer output into <cell>_rd_species.json, the
             format audit.py / score_gemma_maps.py / readout.py already consume

CPU only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cells as C

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/maps_hw_si")
RUBRICS = {
    "hashtagwars_verdict": Path("datasets/humor/hashtagwars/va/rubrics.jsonl"),
    "style_inv_toptier": Path("datasets/humor/style_invitational/va/rubrics.jsonl"),
}
# Parents: the FIT+MINE SHAP selection, PLUS (recorded, not silent) the A-bank
# criteria named in the PUBLISHED Layer-1 SHAP top-pairs for that cell, which is
# the pointer the campaign brief actually names.  For Style Invitational that adds
# "Linguistic polish" (published pair with v_punctuation_density, mass .109) and
# "Reference or target recognizability" (published pair with v_char_count, .047);
# neither reaches the FIT+MINE top-15, and that discrepancy is itself reported.
EXTRA_PARENTS = {
    "hashtagwars_verdict": [],
    "style_inv_toptier": ["Linguistic polish", "Reference or target recognizability"],
}
N_POSITION = {"hashtagwars_verdict": 5, "style_inv_toptier": 7}

HEAD = """You are the DECOMPOSER in a preregistered measurement protocol. You are working
label-blind: you have not been shown any outcome, score, vote or selection for any item,
and nothing you write may refer to one.

CORPUS: {corpus}
ITEM: a {item}
CONSTRUCT the scorecard is trying to measure: {construct}

BACKGROUND. A scorecard of explicitly written criteria is judged 0-10 per criterion by
an LLM judge and aggregated. An interaction screen found that several of the scorecard's
criteria carry their predictive weight ONLY in combination with a purely mechanical
surface feature of the text (character count, word count, punctuation density,
capitalisation ratio, readability index). That makes each such criterion a MIXED
channel: its judged score plausibly mixes (i) a real property of the writing that a
careful expert would count as craft, with (ii) a surface habit that merely travels with
it and that a judge can see and be swayed by.

YOUR TASK, PART 1 -- DECOMPOSE. For each parent criterion below, author EXACTLY TWO
replacement criteria:

  * a CANDIDATE-REAL component: the craft property the parent was trying to name,
    rewritten so that a judge scoring it is forced to judge the property IN CONTEXT and
    is explicitly told NOT to reward the surface carrier (length, punctuation volume,
    capitalisation, tidy typography). It must be possible for a very short, plainly
    typed item to score 10 and for a long, heavily punctuated, immaculately typed item
    to score 0.

  * a SURFACE component: the mechanical habit itself, phrased as a pure EXTENT question
    that a judge can count or estimate from the text, with an explicit instruction that
    the judge must NOT decide whether the habit is good.

The two components must be SEPARABLE: it must be easy to describe an item that scores
high on one and low on the other, in both directions. State that dissociation in your
rationale for each pair.

PARENTS TO DECOMPOSE:

{parents}

YOUR TASK, PART 2 -- POSITION FINGERPRINTS. Author EXACTLY {n_pos} further channels of a
different kind. Every item here sits inside a CONTAINER (one contest / one week's
contest) alongside many sibling entries, and it has a POSITION in that container: how
early or late it was written relative to the others, how crowded the container is, and
where that contest falls in the long run of contests. You cannot see the position. You
CAN sometimes see a TEXTUAL FINGERPRINT of it -- for example an entry that reads as one
of the first, most obvious readings of the prompt, versus one that reads as a
late-arriving riff by someone who has already seen the obvious takes taken; or a
formulation that presupposes a shared running joke; or vocabulary that pins the entry to
a particular moment in time. Author {n_pos} such fingerprint channels, each a pure
extent question scorable from the item text alone, each explicitly not a quality
judgement.

HARD CONSTRAINTS FOR EVERYTHING YOU WRITE.
1. LABEL-BLIND: no reference to any outcome, winner, selection, vote, rank or score.
2. Every criterion is scored 0-10 by an independent judge reading only the item text
   (plus the contest prompt shown with it). Say what a 10 looks like and what a 0 looks
   like.
3. Do not write a criterion that scores DOCUMENT SHAPE in place of merit. A criterion
   whose high end can be reached by formatting alone belongs on the surface side, and
   you must put it there.
4. Names <= 12 words.

OUTPUT. Emit exactly one JSON object and nothing else:

{{"components": [
  {{"id": "D01", "parent": "<parent criterion name>", "kind": "candidate_real",
    "name": "<name>", "instruction": "<0-10 scoring instruction>",
    "rationale": "<why this isolates the craft; state the two-way dissociation>"}},
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
    cell = args.cell
    meta = C.CELL_META[cell]
    par = json.loads((HERE / f"{cell}_parents.json").read_text())
    names = [x["criterion"] for x in par["candidates"]] + EXTRA_PARENTS[cell]
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    rub = {}
    for line in open(Path(__file__).resolve().parents[4] / RUBRICS[cell]):
        if line.strip():
            r = json.loads(line)
            rub[r["name"]] = r["description"]
    partner = {x["criterion"]: x["top_surface_partners"] for x in par["candidates"]}
    blocks = []
    for j, n in enumerate(ordered):
        ps = partner.get(n, [])
        pl = ", ".join(p["feature"] for p in ps) or ("v_punctuation_density, v_char_count"
                                                     if cell == "style_inv_toptier"
                                                     else "v_char_count, v_uppercase_letter_ratio")
        blocks.append(f"[P{j+1:02d}] PARENT: {n}\n"
                      f"      RUBRIC AS WRITTEN: {rub[n]}\n"
                      f"      SURFACE FEATURES IT INTERACTS WITH: {pl}")
    prompt = HEAD.format(corpus=meta["corpus"], item=meta["item"],
                         construct=meta["construct"], parents="\n\n".join(blocks),
                         n_pos=N_POSITION[cell])
    d = SCRATCH / f"{cell}_rd"
    d.mkdir(parents=True, exist_ok=True)
    (d / "prompt_decomposer.txt").write_text(prompt)
    (HERE / f"{cell}_rd_parents_used.json").write_text(json.dumps(
        {"cell": cell, "parents": ordered, "n_parents": len(ordered),
         "n_position_channels": N_POSITION[cell],
         "n_scored_total": 2 * len(ordered) + N_POSITION[cell],
         "surface_partners": {n: partner.get(n, []) for n in ordered},
         "extra_parents_from_published_layer1_shap": EXTRA_PARENTS[cell]}, indent=1))
    print(f"{cell}: {len(ordered)} parents -> {2*len(ordered)} components + "
          f"{N_POSITION[cell]} position channels = {2*len(ordered)+N_POSITION[cell]} scored; "
          f"prompt {len(prompt)} chars -> {d/'prompt_decomposer.txt'}")


def cmd_collect(args):
    cell = args.cell
    d = SCRATCH / f"{cell}_rd"
    raw = (d / "out_decomposer.txt").read_text()
    import harness_maps as H
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[1].rsplit("```", 1)[0]
    obj = json.loads(H.JSON_RE.search(txt).group(0))
    used = json.loads((HERE / f"{cell}_rd_parents_used.json").read_text())

    selected = []
    na, nb = 0, 0
    for c in obj["components"]:
        real = c["kind"] == "candidate_real"
        if real:
            na += 1
            bid = f"A{na:02d}"
        else:
            nb += 1
            bid = f"B{nb:02d}"
        rec = {"blind_id": bid, "track": "A" if real else "B",
               "name": c["name"].strip(), "instruction": c["instruction"].strip(),
               "rationale": c.get("rationale", ""), "pid": c["id"],
               "proposer": "decomposer_opus", "family": "claude",
               "origin": "addendum3_decomposition", "parent": c["parent"],
               "component_kind": c["kind"],
               "n_proposers_naming": 1, "n_members": 1,
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
                         "upstream_parent": c.get("upstream_parent", "position in container"),
                         "mixed_proposed": bool(c.get("mixed", False)),
                         "mixed_any_member": bool(c.get("mixed", False)),
                         "n_proposers_naming": 1, "n_members": 1,
                         "member_names": [c["name"].strip()]})

    out = {"tag": f"{cell}_rd", "cell": cell, "round": "d",
           "selection_rule": "FREEZE ADDENDUM 3 decomposition pass: 2 components per "
                             "MIXED parent (candidate-real + surface) authored by a "
                             "sealed frontier decomposer, plus FREEZE ADDENDUM 4 "
                             "position-fingerprint channels; every component is routed "
                             "INDEPENDENTLY by the blind audit",
           "parents": used["parents"], "tracks": {}, "selected": selected}
    (HERE / f"{cell}_rd_species.json").write_text(json.dumps(out, indent=1))
    print(f"{cell}_rd: {len(selected)} scored criteria "
          f"({sum(1 for s in selected if s['track']=='A')} proposed A / "
          f"{sum(1 for s in selected if s['track']=='B')} proposed B)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "collect"):
        s = sub.add_parser(name)
        s.add_argument("--cell", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "collect": cmd_collect}[a.cmd](a)
