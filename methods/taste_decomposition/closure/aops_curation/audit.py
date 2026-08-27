#!/usr/bin/env python3
"""Blind routing audit (prereg step 4) + routing finalisation, press/journalism cell.

Identical machinery to peer_revealed/audit.py; what is cell-specific is the planted
probe pool, which the freeze requires to be CORPUS-MATCHED (the maps-batch-1
carry-forward, notes/2026-08-06__spurious_maps_batch1.md 1d: probes authored for the
wrong genre make the auditor fail through no fault of its own).

Four PAIRS authored for a corporate press release; each pair is one unambiguously
newsworthiness-relevant criterion plus one unambiguously incidental look-alike that
would correlate with it. The first two pairs are the ones the campaign brief names
("quantified concrete announcement" vs "contains a number"; "novel information vs
recycled boilerplate" vs "has a boilerplate footer"). Two pairs are drawn per round by
stable sha256 of the round tag, so the fresh auditor each round never audits the same
planted pair as the previous auditor.

CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# CORPUS-MATCHED PLANTED PROBE PAIRS for AoPS competition-forum solutions.
# The math.SE pairs this file inherited spoke of "the asker's approach" and "the
# question that was actually asked"; an AoPS post answers a COMPETITION PROBLEM,
# there is no asker, and every pair below is rewritten to that corpus so the blind
# auditor is not tipped off by register mismatch.  Each pair is one clearly
# QUALITY-RELEVANT criterion beside one clearly INCIDENTAL channel; an auditor who
# routes both correctly has demonstrated it can tell the tracks apart.
PROBE_PAIRS = [
    # PAIR 1
    [{"probe": "quality_relevant",
      "name": "Names the obstacle the solution's central move defeats",
      "instruction": "Score 0-10 for how clearly the post says what makes this problem "
                     "hard and what its central move exists to defeat -- the invariant "
                     "that has to be found, the case that resists, the bound that is "
                     "tight. 10 = the difficulty is named and the key move is presented "
                     "as the answer to it; 0 = the post executes steps without ever "
                     "indicating where the problem's resistance lies."},
     {"probe": "incidental",
      "name": "Density of display-math blocks",
      "instruction": "Score 0-10 for how many set-off display-math blocks the post "
                     "contains relative to its length -- $$...$$, \\[...\\], "
                     "\\begin{align}, \\begin{equation} and the like. 10 = display "
                     "blocks appear constantly; 0 = none at all, everything is inline. "
                     "Count the blocks only; do not judge whether the mathematics inside "
                     "them is correct, needed or well chosen."}],
    # PAIR 2
    [{"probe": "quality_relevant",
      "name": "Says which features of the problem the argument actually used",
      "instruction": "Score 0-10 for how far the post makes its method portable: it "
                     "states which hypotheses or structural features the argument leans "
                     "on, so a reader could run the same argument on a different problem, "
                     "rather than only pushing this problem's particular numbers through "
                     "to an answer. 10 = the general principle and its range of use are "
                     "clear; 0 = a bare computation for these numbers only."},
     {"probe": "incidental",
      "name": "Length of the post",
      "instruction": "Score 0-10 for how long the post is. 10 = very long, many "
                     "paragraphs; 5 = a few paragraphs; 0 = one or two lines. Judge "
                     "extent only -- do not consider whether the length is warranted, "
                     "whether the solution is thorough, or whether anything in it is "
                     "correct."}],
    # PAIR 3
    [{"probe": "quality_relevant",
      "name": "Argues the step a competent reader would stop at",
      "instruction": "Score 0-10 for whether the one move a competent reader would "
                     "question -- an unjustified case reduction, an equality claimed "
                     "without proof, a divisibility or convergence step waved through, an "
                     "appeal to a theorem whose hypotheses are not obviously met -- is "
                     "actually argued rather than asserted. 10 = the load-bearing step is "
                     "supported; 0 = the argument's weight rests on an unsupported "
                     "assertion."},
     {"probe": "incidental",
      "name": "Use of AoPS forum markup furniture",
      "instruction": "Score 0-10 for how much AoPS forum furniture the post uses -- hide "
                     "or spoiler blocks, boxed final answers, quote blocks, bold run-in "
                     "labels, numbered or bulleted step lists -- rather than continuous "
                     "prose. 10 = the post is built out of such furniture; 0 = unbroken "
                     "prose with none of it. Score the layout only; do not judge whether "
                     "the steps are right or whether the structure helps."}],
    # PAIR 4
    [{"probe": "quality_relevant",
      "name": "Delivers the object the problem asked for",
      "instruction": "Score 0-10 for whether the post delivers the thing the problem "
                     "requested -- a proof where a proof was demanded, all solutions "
                     "where all were demanded, a construction where one was demanded, the "
                     "requested closed form -- rather than a related result, a special "
                     "case, or a different question the poster found more interesting. "
                     "10 = exactly the requested deliverable; 0 = a substitute the problem "
                     "did not ask for."},
     {"probe": "incidental",
      "name": "Confident, unhedged register",
      "instruction": "Score 0-10 for how unhedged the prose is: absence of 'I think', "
                     "'probably', 'it seems', 'I might be wrong', 'can someone check', "
                     "question marks addressed to the reader, and apologetic asides. "
                     "10 = flatly declarative throughout; 0 = hedged in almost every "
                     "sentence. Score the register only; do not judge whether the "
                     "confidence is warranted."}],
]


def _draw(tag, banned):
    pool = [j for j in range(len(PROBE_PAIRS)) if j not in banned]
    if len(pool) < 2:
        pool = list(range(len(PROBE_PAIRS)))
    return sorted(pool, key=lambda j: hashlib.sha256(
        f"{tag}|probepair{j}".encode()).hexdigest())[:2]


def _pair_ids(tag):
    return sorted(range(len(PROBE_PAIRS)),
                  key=lambda j: hashlib.sha256(f"{tag}|probepair{j}".encode()).hexdigest())[:2]


def probes_for(tag):
    """Two of the four pairs, chosen by stable sha256 of the round tag.

    FIX (inherited from mathse_accepted round 2): with only 4 pairs and 2 drawn, the plain hash draw can hand
    a round the SAME pair set as the round before -- it did exactly that for r1 -> r2 --
    which defeats the freeze's stated purpose ("a fresh auditor each round never audits the
    same planted pair as the previous auditor"). When the previous round's draw is
    recoverable and the remaining pool is large enough, its pairs are excluded first and the
    hash then orders what is left. Still stable, still unseeded, still provenance-neutral.
    """
    m = re.match(r"^(.*)_r(\d+)$", tag)
    if not m:
        chosen = _draw(tag, set())
    else:
        cell, r = m.group(1), int(m.group(2))
        prev = set()
        for k in range(1, r + 1):                 # chain from round 1 so the ban tracks
            chosen = _draw(f"{cell}_r{k}", prev)   # what each round ACTUALLY drew
            prev = set(chosen)
    out = []
    for j in chosen:
        out.extend(PROBE_PAIRS[j])
    return out


AUDIT_PROMPT = """You are auditing a pool of scoring criteria that were written by other
models. Your job is a ROUTING decision, made blind: you are NOT told who wrote each
criterion or what it was written for.

For each criterion, decide:

  "quality_relevant" -- the criterion measures a property that a careful expert would
      count as genuine evidence about {construct}. It is about the substance of
      what is being communicated, the evidence behind it, and the content of the item.

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
