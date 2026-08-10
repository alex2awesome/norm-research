#!/usr/bin/env python3
"""Select the round's SCORED criterion set from the fleet species pool, and insert
the planted probe pairs.

PRE-DECLARED SELECTION RULE (recorded in the campaign note BEFORE round 1 ran; the
freeze fixes k_A=15 / k_B=10 as the SCORED budget, and the sealed fleet produces far
more than that, so a label-blind selection rule is required and is stated here):

  * A track: 15 scored = 2 PLANTED substantive probe members + 13 fleet species.
      The 13 = 8 CONSENSUS species (most distinct proposers naming them, ties by
      stable sha256 over the species name) + 5 DIVERSITY species drawn by stable
      sha256 from the singleton species (named by exactly one proposer).
  * B track: 10 scored = 2 PLANTED shape-only probe members + 8 fleet species.
      The 8 = 5 CONSENSUS + 3 DIVERSITY, same rule.
  * If a stratum is short, the shortfall is taken from the other stratum by the same
    ordering, and the substitution is recorded.

Rationale for having BOTH strata: the peer pilot's retrospective found redundancy
saturation (consensus species are the ones the bank already spans) while the species
pool stayed rich; scoring only consensus would understate closure gains, scoring only
singletons would understate what a well-resourced miner actually puts forward. The
strata are recorded per criterion so the round's gain can be attributed to each.

Nothing here reads y, MONITOR, or any score.

Usage: python select_round_set.py --round 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

N_A_CONSENSUS, N_A_DIVERSITY = 8, 5
N_B_CONSENSUS, N_B_DIVERSITY = 5, 3

# Planted probe PAIRS: each pair is (substantive member -> Track A, shape-only
# look-alike -> Track B).  The audit must route them apart; a probe pair that is
# routed together is a failed audit for that round.  Authored fresh per round.
PROBES = {
    1: [
        {"pair": "P1",
         "a": {"name": "Identifies a specific analytical error in the agency's own analysis",
               "instruction": "Score 10 when the comment points to a specific step, assumption, "
                              "figure or table in the agency's analysis and explains concretely "
                              "why it is wrong or incomplete. Score 0 when the comment makes no "
                              "reference to the agency's analysis at all. Judge whether a "
                              "specific analytical defect is identified, not how forcefully."},
         "b": {"name": "Bare mention of the phrase 'cost-benefit analysis'",
               "instruction": "Score 10 when the phrase 'cost-benefit analysis', 'regulatory "
                              "impact analysis' or 'RIA' appears anywhere in the comment. Score 0 "
                              "when none of those phrases appears. Do not consider whether the "
                              "comment engages with any analysis -- score only the presence of "
                              "the phrase.",
               "upstream_parent": "surface-only",
               "mixed": False}},
        {"pair": "P2",
         "a": {"name": "States who would be affected and how, with concrete detail",
               "instruction": "Score 10 when the comment names the specific population, firms or "
                              "operations affected and describes concretely what would change for "
                              "them under the rule. Score 0 when no affected party or effect is "
                              "described. Judge the concreteness of the described effect."},
         "b": {"name": "Numbered or bulleted list formatting",
               "instruction": "Score 10 when the comment is organised into numbered or bulleted "
                              "items. Score 0 when it is continuous prose with no list markers. "
                              "Score only the visual organisation, never the content of the items.",
               "upstream_parent": "professional drafting / template use",
               "mixed": True}},
    ],
    2: [
        {"pair": "P3",
         "a": {"name": "Supplies data the agency does not already have in the record",
               "instruction": "Score 10 when the comment supplies information -- measurements, "
                              "operational records, survey results, field observations -- that the "
                              "agency would not otherwise possess, and says where it comes from. "
                              "Score 0 when the comment supplies no information beyond opinion. "
                              "Judge whether new information enters the record."},
         "b": {"name": "Bare presence of a numeric figure anywhere in the text",
               "instruction": "Score 10 when any digit-bearing quantity (a number, a percentage, "
                              "a dollar amount, a date) appears anywhere in the comment. Score 0 "
                              "when the comment contains no digits at all. Do not consider what "
                              "the number is used for or whether it is correct.",
               "upstream_parent": "surface-only", "mixed": False}},
        {"pair": "P4",
         "a": {"name": "Explains why the proposed approach will not achieve its stated aim",
               "instruction": "Score 10 when the comment gives a reasoned account of a mechanism "
                              "by which the rule as proposed would fail to produce the effect the "
                              "agency intends. Score 0 when no such account is given. Judge the "
                              "reasoning, not the confidence."},
         "b": {"name": "Formal salutation and signature block present",
               "instruction": "Score 10 when the comment opens with a formal salutation "
                              "(\"Dear Administrator\", \"To Whom It May Concern\") and closes "
                              "with a signature block naming a person or organisation. Score 0 "
                              "when neither is present. Score only the letter furniture.",
               "upstream_parent": "professional or organisational correspondence practice",
               "mixed": True}},
    ],
    3: [
        {"pair": "P5",
         "a": {"name": "Identifies an internal inconsistency between parts of the proposal",
               "instruction": "Score 10 when the comment points to two parts of the proposed rule "
                              "or its preamble that cannot both hold, and says why. Score 0 when "
                              "no inconsistency is identified. Judge whether a specific conflict "
                              "is shown."},
         "b": {"name": "Docket number or RIN quoted verbatim",
               "instruction": "Score 10 when a docket identifier or RIN appears verbatim in the "
                              "comment. Score 0 when neither appears. Score only whether the "
                              "identifier string is present, never whether the comment addresses "
                              "the right rule.",
               "upstream_parent": "familiarity with docket conventions", "mixed": True}},
        {"pair": "P6",
         "a": {"name": "States the conditions under which its own recommendation would fail",
               "instruction": "Score 10 when the comment names the circumstances in which its own "
                              "proposed course of action would not work, or what evidence would "
                              "change its recommendation. Score 0 when the comment offers no such "
                              "limits. Judge whether limits on its own claim are stated."},
         "b": {"name": "Text is entirely upper case",
               "instruction": "Score 10 when the comment is written wholly or almost wholly in "
                              "capital letters. Score 0 when it uses ordinary sentence casing. "
                              "Score only the letter casing.",
               "upstream_parent": "surface-only", "mixed": False}},
    ],
    4: [
        {"pair": "P7",
         "a": {"name": "Ties a requested change to a specific paragraph of regulatory text",
               "instruction": "Score 10 when the comment states what it wants changed AND anchors "
                              "that request to an identified paragraph, section or table of the "
                              "proposed text. Score 0 when no change is requested or the request "
                              "floats free of any provision. Judge the anchoring of the request."},
         "b": {"name": "Mentions the number of members or constituents represented",
               "instruction": "Score 10 when the comment states how many members, employees, "
                              "customers or constituents the submitter speaks for. Score 0 when "
                              "no such count appears. Score only the presence of the constituency "
                              "claim, never whether it makes the argument stronger.",
               "upstream_parent": "organisational scale and membership base", "mixed": True}},
        {"pair": "P8",
         "a": {"name": "Distinguishes what the rule requires from what it merely permits",
               "instruction": "Score 10 when the comment correctly separates mandatory from "
                              "discretionary elements of the proposal and reasons about the "
                              "difference. Score 0 when it treats the proposal undifferentiated. "
                              "Judge whether the distinction is drawn and used."},
         "b": {"name": "Repeats the same sentence or phrase multiple times",
               "instruction": "Score 10 when a sentence or phrase is repeated verbatim three or "
                              "more times in the comment. Score 0 when there is no verbatim "
                              "repetition. Score only the repetition.",
               "upstream_parent": "surface-only", "mixed": False}},
    ],
    5: [
        {"pair": "P9",
         "a": {"name": "Reconciles conflicting evidence rather than citing one side",
               "instruction": "Score 10 when the comment acknowledges evidence pointing the other "
                              "way and explains how to reconcile it with its own position. Score "
                              "0 when only supporting evidence is presented. Judge whether "
                              "conflicting evidence is engaged."},
         "b": {"name": "Attachment or exhibit is referenced",
               "instruction": "Score 10 when the comment refers to an attached document, exhibit, "
                              "appendix or enclosure. Score 0 when it refers to none. Score only "
                              "the reference, never the content of the attachment.",
               "upstream_parent": "resources to prepare supporting documents", "mixed": True}},
        {"pair": "P10",
         "a": {"name": "Specifies how compliance would be demonstrated in practice",
               "instruction": "Score 10 when the comment says concretely what a regulated party "
                              "would have to record, measure or show to demonstrate compliance "
                              "under the proposal or under its own alternative. Score 0 when "
                              "compliance demonstration is not addressed. Judge the concreteness "
                              "of the demonstration described."},
         "b": {"name": "Comment is shorter than three sentences",
               "instruction": "Score 10 when the comment consists of fewer than three sentences. "
                              "Score 0 when it is longer. Score only the sentence count.",
               "upstream_parent": "surface-only", "mixed": False}},
    ],
}


def hkey(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def pick(species, n_cons, n_div, r, track):
    cons_pool = sorted([s for s in species if s["n_proposers"] >= 2],
                       key=lambda s: (-s["n_proposers"], -s["n_families"], hkey(r, track, s["name"])))
    div_pool = sorted([s for s in species if s["n_proposers"] == 1],
                      key=lambda s: hkey("div", r, track, s["name"]))
    cons = cons_pool[:n_cons]
    div = div_pool[:n_div]
    subs = []
    short_c = n_cons - len(cons)
    if short_c > 0:
        extra = [s for s in div_pool if s not in div][:short_c]
        div_pool_rest = [s for s in div_pool if s not in div and s not in extra]
        subs.append({"stratum": "consensus", "short": short_c, "filled_from": "diversity"})
        div = div + extra
        del div_pool_rest
    short_d = n_div - len(div)
    if short_d > 0:
        extra = [s for s in cons_pool if s not in cons][:short_d]
        subs.append({"stratum": "diversity", "short": short_d, "filled_from": "consensus"})
        cons = cons + extra
    return cons, div, subs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    a = ap.parse_args()
    r = a.round

    out = {"round": r, "rule": "8 consensus + 5 diversity (A) / 5 + 3 (B) + 2 planted probe pairs",
           "A": [], "B": [], "substitutions": {}, "probes": []}

    for track, (nc, nd) in (("a", (N_A_CONSENSUS, N_A_DIVERSITY)), ("b", (N_B_CONSENSUS, N_B_DIVERSITY))):
        sp = json.loads((HERE / f"round{r}_species_{track}.json").read_text())["species"]
        cons, div, subs = pick(sp, nc, nd, r, track)
        out["substitutions"][track] = subs
        for stratum, group in (("consensus", cons), ("diversity", div)):
            for s in group:
                out[track.upper()].append({
                    "source": "fleet", "stratum": stratum, "species_id": s["species_id"],
                    "name": s["name"], "instruction": s["instruction"],
                    "n_proposers": s["n_proposers"], "n_families": s["n_families"],
                    "families": s["families"], "rep_pid": s["rep_pid"],
                    "upstream_parent": s.get("upstream_parent"), "mixed": s.get("mixed"),
                })

    for p in PROBES[r]:
        out["probes"].append(p["pair"])
        out["A"].append({"source": "planted_probe", "stratum": "probe", "probe_pair": p["pair"],
                         "name": p["a"]["name"], "instruction": p["a"]["instruction"]})
        out["B"].append({"source": "planted_probe", "stratum": "probe", "probe_pair": p["pair"],
                         "name": p["b"]["name"], "instruction": p["b"]["instruction"],
                         "upstream_parent": p["b"].get("upstream_parent"),
                         "mixed": p["b"].get("mixed")})

    for track in ("A", "B"):
        for k, c in enumerate(out[track]):
            c["id"] = f"{track}{k + 1:02d}"

    out["n_A"], out["n_B"] = len(out["A"]), len(out["B"])
    # composite count asserted in code (freeze: "No composite quota; composite count
    # asserted in code")
    comp = sum(1 for c in out["A"] if " together with " in c["instruction"].lower()
               or " TOGETHER WITH " in c["instruction"] or " and also " in c["instruction"].lower())
    out["n_A_composite_lexical"] = comp
    (HERE / f"round{r}_selection.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("A", "B")}, indent=1))
    for t in ("A", "B"):
        print(f"--- {t} ---")
        for c in out[t]:
            print(f"  {c['id']} [{c['stratum']}] {c['name']}")


if __name__ == "__main__":
    main()
