#!/usr/bin/env python3
"""Blind routing audit (prereg step 4) + routing finalisation, humor map batch.

Identical to maps_batch1/audit.py except for ONE fix, which is the carry-forward
that batch recorded: **planted probes are now CORPUS-MATCHED**.  On cap_crowd the
shared corpus-general probes ("explicit statement of the limitation or failure
regime", "reasoning given for why the approach should work") were inapplicable to
a one-line cartoon caption and the auditor scored 2/4 through no fault of its own
(notes/2026-08-06__spurious_maps_batch1.md §1d).  Both cells here are one-liner
humour corpora, so the four probes below are authored for a one-line joke: two
that are unambiguously comic craft, two that are unambiguously typography or
furniture.

CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# CORPUS-MATCHED planted probes (the maps-batch-1 carry-forward, notes/2026-08-06__
# spurious_maps_batch1.md §1d: "planted probes must be corpus-matched").  Four PAIRS
# authored for a machine-learning paper abstract; each pair is one unambiguously
# substantive criterion plus one unambiguously orthographic look-alike.  Two pairs are
# drawn per round by stable sha256 of the round tag, so the fresh auditor each round
# never audits the same planted pair as the previous auditor.  None of the eight
# overlaps a channel any proposer has named on these cells (checked against the r1/r2
# routing files), so a probe failure is an auditor failure, not an ambiguity.
PROBE_PAIRS = [
    [{"probe": "quality_relevant",
      "name": "States a concrete limitation or scope condition of its own claim",
      "instruction": "Score 0-10 for how explicitly the abstract names a condition under "
                     "which its own claim would not hold, or a case it does not cover. "
                     "10 = a specific, checkable limitation or scope condition is stated; "
                     "0 = no limitation or scope condition is acknowledged anywhere."},
     {"probe": "incidental",
      "name": "Use of the Oxford (serial) comma in lists",
      "instruction": "Score 0-10 for how consistently the text places a comma before the "
                     "final conjunction in a list. 10 = used in every list; 0 = never used, "
                     "or there are no lists. Judge the punctuation habit only; this is not "
                     "a judgement of writing quality."}],
    [{"probe": "quality_relevant",
      "name": "Gives a mechanism for why the proposed approach should work",
      "instruction": "Score 0-10 for whether the abstract supplies a reason or mechanism "
                     "for why what it proposes should work, rather than only asserting "
                     "that it does. 10 = an explicit causal or conceptual account; 0 = bare "
                     "assertion with no reasoning."},
     {"probe": "incidental",
      "name": "Proportion of sentences written in the passive voice",
      "instruction": "Score 0-10 for what fraction of the sentences are in the passive "
                     "voice. 10 = essentially every sentence is passive; 0 = every sentence "
                     "is active. Count the grammatical construction only; do not judge "
                     "whether passive voice is good or bad here."}],
    [{"probe": "quality_relevant",
      "name": "Distinguishes what is new here from what was already known",
      "instruction": "Score 0-10 for how clearly the abstract separates its own new "
                     "contribution from prior knowledge it is building on. 10 = the new "
                     "part is stated and delimited against what already existed; 0 = no "
                     "distinction is drawn at all."},
     {"probe": "incidental",
      "name": "Extent of British rather than American spelling",
      "instruction": "Score 0-10 for how much the text uses British rather than American "
                     "spelling (-ise/-isation, -our, -re endings, doubled consonants). "
                     "10 = British spelling throughout; 0 = American spelling throughout or "
                     "no distinguishing words. Count the orthography only; this is not a "
                     "judgement of writing quality."}],
    [{"probe": "quality_relevant",
      "name": "Names the specific prior approach it improves on and how",
      "instruction": "Score 0-10 for whether the abstract names a specific prior approach "
                     "it is positioned against and says in what respect it differs or "
                     "improves. 10 = a named prior approach plus a stated respect; 0 = no "
                     "prior approach is named, only a generic gap."},
     {"probe": "incidental",
      "name": "Density of semicolons and em-dashes",
      "instruction": "Score 0-10 for how many semicolons and em-dashes the text contains "
                     "relative to its length. 10 = they appear constantly; 0 = none at all. "
                     "Count the punctuation only; do not judge whether the writing is good."}],
]


def probes_for(tag):
    """Two of the four pairs, chosen by stable sha256 of the round tag."""
    order = sorted(range(len(PROBE_PAIRS)),
                   key=lambda j: hashlib.sha256(f"{tag}|probepair{j}".encode()).hexdigest())
    out = []
    for j in order[:2]:
        out.extend(PROBE_PAIRS[j])
    return out


AUDIT_PROMPT = """You are auditing a pool of scoring criteria that were written by other
models. Your job is a ROUTING decision, made blind: you are NOT told who wrote each
criterion or what it was written for.

For each criterion, decide:

  "quality_relevant" -- the criterion measures a property that a careful expert would
      count as genuine evidence about {construct}. It is about the comic idea, the
      craft, the wording choices, the content of the item.

  "incidental" -- the criterion measures something that may well PREDICT how the item
      fares but is not itself {construct}: length or verbosity, typography, punctuation
      or capitalisation habits, formatting and trailing furniture, boilerplate or
      template phrasing, community/venue style markers, subject-matter or topic markers,
      temporal tells, orthographic habits, or a textual fingerprint of some upstream
      circumstance of the item's production (who wrote it, with what practice or
      following, when, where in the run of entries).

The corpus is: {corpus}. Each item is a {item}.

SECOND JUDGEMENT, only for criteria you label "incidental". Some incidental channels have
an upstream cause that plausibly ALSO produces genuinely better work (for example, a
practised entrant may both be favoured for irrelevant reasons AND actually write better
entries). Return `mixed`: true when you think the upstream cause plausibly produces real
{construct} too, false when the channel looks like pure nuisance. For criteria you label
"quality_relevant", return mixed=false.

Judge each criterion ON ITS OWN TEXT. Do not try to guess which pool it came from, and
do not let the ORDER of the list influence you.

CRITERIA:

{body}

OUTPUT. Emit exactly one JSON object and nothing else:

{{"verdicts": [
  {{"id": "<the id shown>", "label": "quality_relevant" or "incidental",
    "mixed": true or false, "confidence": "high"/"medium"/"low",
    "justification": "<one sentence>"}},
  ... one entry per criterion, same ids ...
]}}
"""


def _h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def cmd_build(args):
    import cells as C
    tag = f"{args.cell}_r{args.round}"
    sel = json.loads((HERE / f"{tag}_species.json").read_text())["selected"]
    meta = C.CELL_META[args.cell]

    pool = []
    for c in sel:
        pool.append({"true_track": c["track"], "blind_id": c["blind_id"],
                     "name": c["name"], "instruction": c["instruction"], "is_probe": None})
    for j, p in enumerate(probes_for(tag)):
        pool.append({"true_track": "PROBE", "blind_id": f"P{j+1:02d}", "name": p["name"],
                     "instruction": p["instruction"], "is_probe": p["probe"]})

    pool.sort(key=lambda r: _h(f"{tag}|{r['blind_id']}"))
    for j, r in enumerate(pool):
        r["audit_id"] = f"X{j+1:02d}"

    body = "\n\n".join(f"[{r['audit_id']}] {r['name']}\n{r['instruction']}" for r in pool)
    prompt = AUDIT_PROMPT.format(construct=meta["construct"], corpus=meta["corpus"],
                                 item=meta["item"], body=body)
    (HERE / f"{tag}_audit_key.json").write_text(json.dumps(pool, indent=1))
    p = HERE / f"{tag}_audit_prompt.txt"
    p.write_text(prompt)
    print(f"{tag}: audit prompt {len(prompt)} chars, {len(pool)} items "
          f"({sum(1 for r in pool if r['true_track']=='PROBE')} planted) -> {p.name}")


def cmd_finalize(args):
    tag = f"{args.cell}_r{args.round}"
    key = json.loads((HERE / f"{tag}_audit_key.json").read_text())
    ver = json.loads((HERE / f"{tag}_audit_verdicts.json").read_text())
    vmap = {v["id"]: v for v in ver["verdicts"]}
    arb_path = HERE / f"{tag}_arbiter.json"
    arb = json.loads(arb_path.read_text()) if arb_path.exists() else {"rulings": []}
    amap = {a["blind_id"]: a for a in arb.get("rulings", [])}

    sel = {c["blind_id"]: c for c in
           json.loads((HERE / f"{tag}_species.json").read_text())["selected"]}

    probes_ok, probes_n, probe_detail = 0, 0, []
    final, disputes = [], []
    for r in key:
        v = vmap.get(r["audit_id"])
        if r["true_track"] == "PROBE":
            probes_n += 1
            ok = bool(v and v["label"] == r["is_probe"])
            probes_ok += int(ok)
            probe_detail.append({"blind_id": r["blind_id"], "name": r["name"],
                                 "expected": r["is_probe"],
                                 "got": (v or {}).get("label"), "pass": ok})
            continue
        audit_label = v["label"] if v else None
        audit_track = {"quality_relevant": "A", "incidental": "B"}.get(audit_label)
        proposed = r["true_track"]
        c = sel.get(r["blind_id"], {})
        rec = {"blind_id": r["blind_id"], "name": r["name"],
               "proposed_track": proposed, "audit_label": audit_label,
               "audit_track": audit_track,
               "audit_confidence": (v or {}).get("confidence"),
               "audit_justification": (v or {}).get("justification"),
               "audit_mixed": bool((v or {}).get("mixed", False)),
               "origin": c.get("origin", "fleet"),
               "parent": c.get("parent"), "component_kind": c.get("component_kind"),
               "agree": audit_track == proposed}
        if audit_track is None:
            rec["final_route"] = proposed
            rec["route_source"] = "no_verdict_default_proposed"
        elif audit_track == proposed:
            rec["final_route"] = proposed
            rec["route_source"] = "audit_agrees"
        else:
            disputes.append(r["blind_id"])
            if r["blind_id"] in amap:
                rec["final_route"] = amap[r["blind_id"]]["route"]
                rec["route_source"] = "arbiter"
                rec["arbiter_reason"] = amap[r["blind_id"]].get("reason")
            else:
                rec["final_route"] = audit_track
                rec["route_source"] = "audit_reroute_pending_arbiter"
        if proposed == "B" or rec["final_route"] == "B":
            rec["upstream_parent"] = c.get("upstream_parent", "surface-only")
            rec["mixed_proposed"] = bool(c.get("mixed_proposed", False))
            rec["mixed"] = bool(rec["mixed_proposed"] or rec["audit_mixed"])
        final.append(rec)

    n = len(final)
    mis = sum(1 for r in final if not r["agree"])
    out = {"tag": tag, "cell": args.cell, "round": args.round,
           "auditor": ver.get("auditor", "sonnet-blind (fresh session)"),
           "n_criteria": n, "n_misrouted": mis,
           "misrouting_rate": mis / n if n else None,
           "probe_pass": f"{probes_ok}/{probes_n}",
           "probe_pass_rate": probes_ok / probes_n if probes_n else None,
           "probe_detail": probe_detail,
           "disputes": disputes,
           "arbiter_present": bool(amap),
           "n_final_A": sum(1 for r in final if r["final_route"] == "A"),
           "n_final_B": sum(1 for r in final if r["final_route"] == "B"),
           "n_mixed_B": sum(1 for r in final if r["final_route"] == "B" and r.get("mixed")),
           "final": final}
    (HERE / f"{tag}_routing_final.json").write_text(json.dumps(out, indent=1))
    print(f"{tag}: misrouting {mis}/{n} ({out['misrouting_rate']:.2f}), probes "
          f"{out['probe_pass']}, final A={out['n_final_A']} B={out['n_final_B']} "
          f"(mixed {out['n_mixed_B']}), disputes={len(disputes)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "finalize"):
        s = sub.add_parser(name)
        s.add_argument("--cell", required=True)
        s.add_argument("--round", default="1")
    a = ap.parse_args()
    {"build": cmd_build, "finalize": cmd_finalize}[a.cmd](a)
