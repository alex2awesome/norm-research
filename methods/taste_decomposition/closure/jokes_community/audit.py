#!/usr/bin/env python3
"""Blind routing audit (prereg step 4) + routing finalisation, reddit-jokes community cell.

Identical machinery to mathse_vote/audit.py; what is cell-specific is the planted probe
pool, which the freeze requires to be CORPUS-MATCHED (the maps-batch-1 carry-forward,
notes/2026-08-06__spurious_maps_batch1.md 1d: probes authored for the wrong genre make
the auditor fail through no fault of its own).

Four PAIRS authored for a short forum joke; each pair is one unambiguously
craft-relevant criterion plus one unambiguously incidental look-alike that would
correlate with it (comic reinterpretation vs a short final line; single misleading
reading vs the question-and-answer riddle template; nothing-removable economy vs raw
length; specific premise vs shouting typography). Two pairs are drawn per round by
stable sha256 of the round tag, CHAINED off each round's realised draw so the fresh
auditor each round never audits the same planted pair as the previous auditor (the
small-pool repeat landmine found on mathse_vote r1 -> r2; see `probes_for`).

CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

PROBE_PAIRS = [
    # PAIR 1 -- the comic mechanism itself vs the layout habit that travels with it
    [{"probe": "quality_relevant",
      "name": "Reveal forces a rereading of the setup",
      "instruction": "Score 0-10 for how far the ending makes the reader go back and "
                     "reinterpret what the setup meant -- the setup supported one reading, "
                     "and the last move shows a second reading was there all along. 10 = "
                     "the whole setup flips meaning on the final move; 0 = the ending adds "
                     "information but nothing earlier is re-read. Judge the mechanism, not "
                     "whether the joke amused you."},
     {"probe": "incidental",
      "name": "Ends on a short final line",
      "instruction": "Score 0-10 for how short the last line is relative to the rest of "
                     "the text -- a clipped closing line set off on its own scores high, a "
                     "final line as long as the body scores low. 10 = the final line is a "
                     "few words on its own; 0 = there is no distinct short closing line. "
                     "Count the layout only; do not judge whether the ending works."}],
    # PAIR 2
    [{"probe": "quality_relevant",
      "name": "Setup commits to one misleading reading",
      "instruction": "Score 0-10 for how completely the setup commits the reader to a "
                     "single interpretation before the ending arrives -- no hedging, no "
                     "visible second meaning left showing, nothing that tips the reader off "
                     "early. 10 = the reader is fully committed to one reading; 0 = the "
                     "second meaning is visible from the first sentence or no particular "
                     "reading is set up at all."},
     {"probe": "incidental",
      "name": "Uses the question-and-answer riddle template",
      "instruction": "Score 0-10 for how far the text is laid out as a question followed by "
                     "an answer -- 'What do you call...', 'Why did...', a question mark "
                     "followed by a reply. 10 = a pure two-part question-and-answer; 0 = no "
                     "question form anywhere. Score the template only; do not judge whether "
                     "the riddle is any good."}],
    # PAIR 3
    [{"probe": "quality_relevant",
      "name": "Nothing in the text is removable",
      "instruction": "Score 0-10 for how much of the text is load-bearing: every clause "
                     "either plants something the ending uses or is required to make the "
                     "situation legible. 10 = removing any clause breaks the joke; 0 = whole "
                     "sentences could be deleted without the joke changing. Judge necessity, "
                     "not brevity -- a long text in which everything is used scores high."},
     {"probe": "incidental",
      "name": "Overall length of the post",
      "instruction": "Score 0-10 for how long the text is. 10 = many sentences or "
                     "paragraphs; 5 = a few sentences; 0 = a single short line. Judge extent "
                     "only -- do not consider whether the length is warranted, whether the "
                     "build is worth it, or whether anything in it lands."}],
    # PAIR 4
    [{"probe": "quality_relevant",
      "name": "Premise is specific rather than generic",
      "instruction": "Score 0-10 for how particular the situation is -- named details, a "
                     "concrete scene, a specific relationship or object doing real work -- "
                     "as against a stock frame filled with interchangeable placeholders. "
                     "10 = the specifics are what make the joke work; 0 = a template with "
                     "generic slots. Judge specificity, not subject matter."},
     {"probe": "incidental",
      "name": "Shouting typography and punctuation runs",
      "instruction": "Score 0-10 for how much the text uses capitalised shouting, repeated "
                     "exclamation or question marks, stretched-out letters, or emoji. 10 = "
                     "constant throughout; 0 = none at all. Count the typography only; do "
                     "not judge whether it is effective or whether the joke is funny."}],
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

    FIX (mathse_vote, round 2): with only 4 pairs and 2 drawn, the plain hash draw can hand
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
