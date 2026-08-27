#!/usr/bin/env python3
"""Per-round decomposition-first pass for rounds >= 3 (FREEZE ADDENDUM 3, continued).

Round 0's pass decomposed frozen-bank rubrics; from round 3 the parents can also be
MINED criteria (new_parents.py picks them off a FIT+MINE SHAP screen over the whole
current bank).  Addendum 3 says decomposed components count toward the round's k
budgets, so the round's scored set is composed as

    k_A = 15 = 12 fleet species + 3 decomposition candidate-real components
    k_B = 10 =  7 fleet species + 3 decomposition surface components

  build   -> sealed decomposer prompt for the round's new parents
  merge   -> rewrites <cell>_r<r>_species.json so it holds the merged 25

CPU only.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import cells as C

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/mathse_vote")
N_FLEET_A, N_FLEET_B = 12, 7

HEAD = """You are the DECOMPOSER in a preregistered measurement protocol, working label-blind:
you have not been shown any outcome, score, vote or selection for any item, and nothing you
write may refer to one.

CORPUS: {corpus}
ITEM: a {item}
CONSTRUCT the scorecard is trying to measure: {construct}

BACKGROUND. A scorecard of explicitly written criteria is judged 0-10 per criterion by an
LLM judge and aggregated. An interaction screen over the CURRENT scorecard found that the
criteria below carry their predictive weight only in combination with a purely mechanical
surface feature of the text (character count, capitalisation ratio, handle/hashtag count).
That makes each a MIXED channel: its judged score plausibly mixes (i) a real property of
the writing that a careful expert would count as craft, with (ii) a surface habit that
merely travels with it and that a judge can see and be swayed by.

YOUR TASK. For each parent below author EXACTLY TWO replacement criteria:

  * a CANDIDATE-REAL component: the craft property the parent was trying to name, rewritten
    so a judge is forced to judge it IN CONTEXT and is explicitly told NOT to reward the
    surface carrier (length, capitalisation, tag furniture, typography). It must be
    possible for a very short, plainly typed item to score 10 and for a long, shouty,
    tag-heavy item to score 0.
  * a SURFACE component: the mechanical habit itself, as a pure EXTENT question a judge can
    count or estimate, with an explicit instruction NOT to decide whether it is good.

The two must be SEPARABLE: describe, in each rationale, an item that scores high on one and
low on the other, in both directions.

PARENTS TO DECOMPOSE:

{parents}

HARD CONSTRAINTS. (1) Label-blind: no reference to any outcome, winner, selection, vote,
rank or score. (2) Each criterion is scored 0-10 by an independent judge reading only the
item text plus the contest prompt shown with it; say what a 10 and a 0 look like. (3) Do
not write a criterion whose high end can be reached by formatting alone unless you are
putting it on the surface side. (4) Names <= 12 words.

OUTPUT. Emit exactly one JSON object and nothing else:

{{"components": [
  {{"id": "D01", "parent": "<parent name>", "kind": "candidate_real",
    "name": "<name>", "instruction": "<0-10 scoring instruction>",
    "rationale": "<why this isolates the craft; state the two-way dissociation>"}},
  {{"id": "D02", "parent": "<same parent>", "kind": "surface",
    "name": "<name>", "instruction": "<0-10 extent instruction, explicitly not a quality judgement>",
    "rationale": "<why this is the carrier; state the two-way dissociation>"}},
  ... two entries per parent, in the order the parents are listed ...
]}}
"""


def instruction_of(cell, name):
    for tag in ["d"] + [str(r) for r in range(1, 9)]:
        f = HERE / f"{cell}_r{tag}_species.json"
        if not f.exists():
            continue
        for s in json.loads(f.read_text())["selected"]:
            if s["name"] == name:
                return s["instruction"]
    return None


def cmd_build(a):
    meta = C.CELL_META[a.cell]
    par = json.loads((HERE / f"{a.cell}_r{a.round}_newparents.json").read_text())
    sel = par["selected_parents"]
    partner = {x["criterion"]: x["top_surface_partners"] for x in par["candidates"]}
    blocks = []
    for j, n in enumerate(sel):
        ins = instruction_of(a.cell, n) or "(frozen bank rubric)"
        pl = ", ".join(p["feature"] for p in partner.get(n, []))
        blocks.append(f"[P{j+1:02d}] PARENT: {n}\n      INSTRUCTION AS WRITTEN: {ins}\n"
                      f"      SURFACE FEATURES IT INTERACTS WITH: {pl}")
    prompt = HEAD.format(corpus=meta["corpus"], item=meta["item"],
                         construct=meta["construct"], parents="\n\n".join(blocks))
    d = SCRATCH / f"{a.cell}_r{a.round}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "prompt_decomposer.txt").write_text(prompt)
    (HERE / f"{a.cell}_r{a.round}_parents_used.json").write_text(json.dumps(
        {"cell": a.cell, "round": a.round, "parents": sel,
         "surface_partners": {n: partner.get(n, []) for n in sel},
         "composition": {"fleet_A": N_FLEET_A, "fleet_B": N_FLEET_B,
                         "decomposition_candidate_real": len(sel),
                         "decomposition_surface": len(sel)}}, indent=1))
    print(f"{a.cell} r{a.round}: {len(sel)} new MIXED parents -> {2*len(sel)} components; "
          f"prompt {len(prompt)} chars -> {d/'prompt_decomposer.txt'}")


def cmd_merge(a):
    import harness_maps as H
    d = SCRATCH / f"{a.cell}_r{a.round}"
    txt = (d / "out_decomposer.txt").read_text().strip()
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[1].rsplit("```", 1)[0]
    obj = json.loads(H.JSON_RE.search(txt).group(0))

    spf = HERE / f"{a.cell}_r{a.round}_species.json"
    spec = json.loads(spf.read_text())
    fleet = spec["selected"]
    keptA = [s for s in fleet if s["track"] == "A"][:N_FLEET_A]
    keptB = [s for s in fleet if s["track"] == "B"][:N_FLEET_B]
    merged, na, nb = [], 0, 0
    for s in keptA:
        na += 1
        s = dict(s); s["blind_id"] = f"A{na:02d}"; s["origin"] = "fleet"
        merged.append(s)
    for s in keptB:
        nb += 1
        s = dict(s); s["blind_id"] = f"B{nb:02d}"; s["origin"] = "fleet"
        merged.append(s)
    for c in obj["components"]:
        real = c["kind"] == "candidate_real"
        if real:
            na += 1; bid = f"A{na:02d}"
        else:
            nb += 1; bid = f"B{nb:02d}"
        rec = {"blind_id": bid, "track": "A" if real else "B",
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
        merged.append(rec)
    spec["selected"] = merged
    spec["composition"] = {"fleet_A": len(keptA), "fleet_B": len(keptB),
                           "decomposition_components": len(obj["components"])}
    spf.write_text(json.dumps(spec, indent=1))
    print(f"{a.cell} r{a.round}: merged species -> {len(merged)} scored "
          f"({sum(1 for m in merged if m['track']=='A')} A / "
          f"{sum(1 for m in merged if m['track']=='B')} B)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("build", "merge"):
        s = sub.add_parser(n); s.add_argument("--cell", required=True); s.add_argument("--round", required=True)
    a = ap.parse_args()
    {"build": cmd_build, "merge": cmd_merge}[a.cmd](a)
