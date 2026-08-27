#!/usr/bin/env python3
"""ROUND 2 = FREEZE ADDENDUM 3 DIRECTED DECOMPOSITION (coordinator GO, this session).

WHY A CELL-SPECIFIC SCRIPT, recorded rather than silent.  `decompose_round.py` builds its
decomposer prompt around a SHAP interaction screen ("this criterion carries its weight only
in combination with character count / capitalisation / handle count").  That is not why
these parents are MIXED here.  On this cell the trigger is the one Addendum 3 actually
names -- a channel whose conjectured upstream parent plausibly causes real quality too --
plus, for the headline parent, round 1's decisive NULL: the fleet's conjectured textual
fingerprint of answer-arrival order scored **.492, chance**, while the OBSERVED arrival-order
covariate reads .614 on the same rows.  Addendum 3's remedy for a MIXED parent that has not
resolved is to SPLIT it, not to re-propose it, and that is what this round does.

PARENT SELECTION, fixed before the decomposer was called:
  * the arrival-order parent (coordinator-directed; the .492 null), allowed THREE
    components rather than two -- one candidate-real and TWO distinct surface fingerprints
    -- precisely because round 1 showed one fingerprint was not enough;
  * the three MIXED channels that carried real alone-AUC AND are already spanned by the V
    block (markup fluency rho=.74, response volume rho=.90, reader coaching rho=.74): the
    decomposition asks whether ANY of their signal is craft rather than lint;
  * the one MIXED channel with real alone-AUC that is NOT already articulated
    (direct engagement with the asker's work, rho=.45).

SCOPE, recorded: round 2 runs the decomposition pass ONLY -- no fresh proposer fleet --
per the coordinator's framing of round 2 as the directed Addendum-3 pass.  The scored set
is the decomposition components plus the ONE channel the strict two-judge merge selected in
round 1 that the loose (scored) selection omitted, which inherits here with its
strict-only provenance carried.  Parents are RETIRED from the readout once their components
are scored (recorded, not deleted), exactly as Addendum 3 requires.

  build  -> sealed decomposer prompt
  merge  -> mathse_vote_r2_species.json (components + inherited channel)

CPU only.
"""
from __future__ import annotations

import argparse, json, re
from pathlib import Path
import cells as C

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/mathse_vote")

HEADLINE = "Presupposes or names sibling answers on this question"

HEAD = """You are the DECOMPOSER in a preregistered measurement protocol, working label-blind:
you have not been shown any outcome, score, vote or selection for any item, and nothing you
write may refer to one.

CORPUS: {corpus}
ITEM: a {item}
CONSTRUCT the scorecard is trying to measure: {construct}

BACKGROUND. A scorecard of explicitly written criteria is judged 0-10 per criterion by an
LLM judge and aggregated. Each parent below is a MIXED channel: the unseen circumstance it
is trying to detect plausibly produces genuinely better work AS WELL AS merely travelling
with success, so its judged score confounds (i) a real property of the writing a careful
expert would count as craft with (ii) a mechanical or circumstantial habit that a judge can
see and be swayed by. Your job is to split each parent into separately judgeable pieces.

YOUR TASK. For each parent below author the number of replacement criteria stated in its
block:

  * a CANDIDATE-REAL component: the craft property the parent was trying to name, rewritten
    so a judge must judge it IN CONTEXT and is explicitly told NOT to reward the surface
    carrier (length, markup density, typography, formatting furniture). It must be possible
    for a very short, plainly typed answer to score 10 and for a long, heavily formatted one
    to score 0.
  * one or more SURFACE components: the mechanical habit itself, as a pure EXTENT question a
    judge can count or estimate, with an explicit instruction NOT to decide whether it is
    good.

Each pair must be SEPARABLE: in every rationale, describe an answer that scores high on one
and low on the other, in BOTH directions.

PARENTS TO DECOMPOSE:

{parents}

HARD CONSTRAINTS. (1) Label-blind: no reference to any outcome, vote, score, acceptance,
rank or selection. (2) Each criterion is scored 0-10 by an independent judge reading only
the answer text plus the question TITLE that is shown with it -- the question body is NOT
available, so do not write a criterion that needs it. (3) Do not write a criterion whose
high end can be reached by formatting alone unless you are putting it on the surface side.
(4) Names <= 12 words. (5) Write for mathematical answers specifically: your surface
components may reference LaTeX/display maths/markup, and your candidate-real components
should be about mathematical exposition, not generic writing quality.

OUTPUT. Emit exactly one JSON object and nothing else:

{{"components": [
  {{"id": "D01", "parent": "<parent name>", "kind": "candidate_real",
    "name": "<name>", "instruction": "<0-10 scoring instruction; say what a 10 and a 0 look like>",
    "rationale": "<why this isolates the craft; state the two-way dissociation>"}},
  {{"id": "D02", "parent": "<same parent>", "kind": "surface",
    "name": "<name>", "instruction": "<0-10 extent instruction, explicitly not a quality judgement>",
    "rationale": "<why this is the carrier; state the two-way dissociation>"}},
  ... the stated number of entries per parent, in the order the parents are listed ...
]}}
"""

EXTRA = {
    HEADLINE: ("THREE components: one candidate-real and TWO DISTINCT surface fingerprints. "
               "A previous round tested a single surface fingerprint for this parent -- "
               "whether the answer explicitly refers to or presupposes other answers -- and "
               "an LLM judge scoring it corpus-wide found essentially nothing. Do not "
               "re-propose that. The unseen circumstance is WHERE THIS ANSWER FALLS IN THE "
               "ORDER OF ANSWERS TO ITS QUESTION: was it written into an empty thread, or "
               "into one that already had answers? Think about what BOTH ends look like in "
               "text, and make your two surface components genuinely different from each "
               "other -- for example, one about how the answer OPENS and orients itself to "
               "the raw question (setting up notation, restating the problem, addressing "
               "the asker's exact wording), and one about SCOPE POSTURE (whether it treats "
               "itself as the whole answer or as a supplement, alternative route, special "
               "case, or refinement). You cannot see the actual position; score the text."),
}


def parents_spec():
    r1 = json.loads((HERE / "mathse_vote_r1_results.json").read_text())
    by = {c["name"]: c for c in r1["spurious_map"]["channels"]}
    sel = {s["name"]: s for s in
           json.loads((HERE / "mathse_vote_r1_species.json").read_text())["selected"]}
    want = [HEADLINE,
            "Site-markup fluency versus plaintext ASCII mathematical orthography",
            "Response Volume",
            "Direct Reader Coaching",
            "Direct engagement with asker's specific work"]
    out = []
    for n in want:
        m = by.get(n) or next((v for k, v in by.items() if k.startswith(n[:28])), None)
        s = sel.get(n) or next((v for k, v in sel.items() if k.startswith(n[:28])), None)
        if m is None or s is None:
            continue
        out.append({"name": s["name"], "instruction": s["instruction"],
                    "alone_AUC_HONEST": m["alone_AUC_HONEST"],
                    "max_abs_rho_with_V_block": m.get("max_abs_rho_with_V_block"),
                    "already_articulated": bool(m.get("ALREADY_ARTICULATED")),
                    "upstream_parent": s.get("upstream_parent"),
                    "n_components": 3 if s["name"] == HEADLINE else 2})
    return out


def cmd_build(a):
    meta = C.CELL_META["mathse_vote"]
    sel = parents_spec()
    blocks = []
    for j, p in enumerate(sel):
        extra = EXTRA.get(p["name"], f"TWO components: one candidate-real and one surface.")
        rho = p["max_abs_rho_with_V_block"]
        note = (f"      NOTE: this channel's judged score already correlates |rho|={rho:.2f} "
                f"with a mechanical lint feature the scorecard ALREADY contains, so its "
                f"surface component is largely known; the question is whether ANY of it is craft."
                if p["already_articulated"] else
                f"      NOTE: |rho|={rho:.2f} with the scorecard's existing lint features -- "
                f"this one is not already spanned.")
        blocks.append(f"[P{j+1:02d}] PARENT: {p['name']}\n"
                      f"      INSTRUCTION AS WRITTEN: {p['instruction']}\n"
                      f"      PROPOSER'S CONJECTURED UPSTREAM CAUSE: {p['upstream_parent']}\n"
                      f"{note}\n      REQUIRED: {extra}")
    prompt = HEAD.format(corpus=meta["corpus"], item=meta["item"],
                         construct=meta["construct"], parents="\n\n".join(blocks))
    d = SCRATCH / "mathse_vote_r2"; d.mkdir(parents=True, exist_ok=True)
    (d / "prompt_decomposer.txt").write_text(prompt)
    (HERE / "mathse_vote_r2_parents_used.json").write_text(json.dumps(
        {"cell": "mathse_vote", "round": 2, "rule": "FREEZE ADDENDUM 3 directed decomposition",
         "parents": sel, "n_components_expected": sum(p["n_components"] for p in sel)}, indent=1))
    print(f"round 2: {len(sel)} MIXED parents -> {sum(p['n_components'] for p in sel)} "
          f"components; prompt {len(prompt)} chars -> {d/'prompt_decomposer.txt'}")
    for p in sel:
        print(f"   P: {p['name'][:52]:52s} aloneAUC={p['alone_AUC_HONEST']:.3f} "
              f"rhoV={p['max_abs_rho_with_V_block']:.2f} n={p['n_components']}")


def cmd_merge(a):
    d = SCRATCH / "mathse_vote_r2"
    txt = (d / "out_decomposer.txt").read_text().strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-zA-Z]*\n", "", txt); txt = re.sub(r"\n```\s*$", "", txt)
    obj = json.loads(re.search(r"\{[\s\S]*\}", txt).group(0))
    comps = obj["components"]
    strict = json.loads((HERE / "mathse_vote_r1_species.STRICT_TWOJUDGE.json").read_text())
    loose = {c["name"] for c in json.loads((HERE / "mathse_vote_r1_species.json").read_text())["selected"]}
    inherited = [c for c in strict["selected"] if c["track"] == "B" and c["name"] not in loose]

    sel, na, nb = [], 0, 0
    for c in comps:
        real = c["kind"] == "candidate_real"
        if real:
            na += 1; bid = f"A{na:02d}"
        else:
            nb += 1; bid = f"B{nb:02d}"
        sel.append({"track": "A" if real else "B", "blind_id": bid, "name": c["name"],
                    "instruction": c["instruction"], "rationale": c.get("rationale", ""),
                    "upstream_parent": c["parent"], "mixed": False,
                    "provenance": f"ADDENDUM-3 decomposition of MIXED parent: {c['parent']}",
                    "kind": c["kind"]})
    for c in inherited:
        nb += 1
        sel.append({**c, "track": "B", "blind_id": f"B{nb:02d}",
                    "provenance": "INHERITED: selected by the round-1 STRICT two-judge merge "
                                  "but omitted from the loose selection that was scored"})
    out = {"tag": "mathse_vote_r2", "cell": "mathse_vote", "round": "2",
           "composition": {"decomposition_candidate_real": na,
                           "decomposition_surface": nb - len(inherited),
                           "inherited_strict_only": len(inherited)},
           "retired_parents": [p["name"] for p in
                               json.loads((HERE / "mathse_vote_r2_parents_used.json").read_text())["parents"]],
           "note": "Addendum 3: parents are RETIRED from readouts once their components are "
                   "scored (recorded, not deleted). No fleet this round -- directed pass.",
           "selected": sel}
    (HERE / "mathse_vote_r2_species.json").write_text(json.dumps(out, indent=1))
    print(f"round 2 scored set: {len(sel)} criteria "
          f"(A={na} candidate-real, B={nb} surface+inherited)")
    for c in sel:
        print(f"   {c['blind_id']} {c['track']} | {c['name'][:56]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build"); sub.add_parser("merge")
    a = ap.parse_args()
    {"build": cmd_build, "merge": cmd_merge}[a.cmd](a)
