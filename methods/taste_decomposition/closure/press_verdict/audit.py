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
from pathlib import Path

HERE = Path(__file__).resolve().parent

PROBE_PAIRS = [
    [{"probe": "quality_relevant",
      "name": "Quantified, concrete announcement of what actually changed",
      "instruction": "Score 0-10 for how concretely and specifically the release states "
                     "what has changed in the world -- what is being launched, closed, "
                     "priced, hired, recalled or reported -- with figures, dates or scope "
                     "that make the change checkable. 10 = the substantive change is stated "
                     "with specific magnitude and scope; 0 = the release never says what "
                     "concretely happened, only that something is exciting or ongoing."},
     {"probe": "incidental",
      "name": "Density of numerals and digit strings in the text",
      "instruction": "Score 0-10 for how many numerals and digit strings the text contains "
                     "relative to its length, counting any digits at all -- dates, phone "
                     "numbers, ticker symbols, percentages, footnote markers. 10 = digits "
                     "appear constantly; 0 = none at all. Count the characters only; do not "
                     "judge whether the numbers are informative or whether the release is "
                     "good."}],
    [{"probe": "quality_relevant",
      "name": "Supplies information not already public or self-evident",
      "instruction": "Score 0-10 for how much of the release is information a reader could "
                     "not already have -- new findings, previously undisclosed terms, "
                     "first-time figures, a decision not previously announced -- as opposed "
                     "to restating the organisation's mission, values or previously "
                     "announced plans. 10 = the substance is genuinely new; 0 = everything "
                     "in it is either already public or is generic self-description."},
     {"probe": "incidental",
      "name": "Presence of a standard boilerplate 'About' footer block",
      "instruction": "Score 0-10 for how fully the text carries the conventional trailing "
                     "furniture of a press release: an 'About <organisation>' paragraph, a "
                     "forward-looking-statements disclaimer, trademark or copyright lines, "
                     "a media-contact block. 10 = all of these are present and complete; "
                     "0 = none of them appear. Score the presence of the furniture only; "
                     "do not judge whether it is well written or whether the release is "
                     "newsworthy."}],
    [{"probe": "quality_relevant",
      "name": "Names who is affected and how, rather than asserting importance",
      "instruction": "Score 0-10 for how clearly the release identifies who is affected by "
                     "what it announces and in what specific way -- which customers, "
                     "patients, workers, markets or regions, and what changes for them. "
                     "10 = affected parties and the concrete effect on them are both named; "
                     "0 = the release only asserts that the news is significant, important "
                     "or industry-leading without saying for whom or how."},
     {"probe": "incidental",
      "name": "Extent of superlative and promotional adjectives",
      "instruction": "Score 0-10 for how heavily the text uses promotional intensifiers and "
                     "superlatives -- 'leading', 'world-class', 'revolutionary', "
                     "'best-in-class', 'unprecedented', 'proud to announce'. 10 = they "
                     "saturate the text; 0 = none appear. Count the vocabulary only; this "
                     "is explicitly not a judgement of whether the release is good or bad."}],
    [{"probe": "quality_relevant",
      "name": "Attributes claims to a checkable source or method",
      "instruction": "Score 0-10 for whether factual claims in the release are tied to "
                     "something a journalist could check -- a named study with its method, "
                     "a filing, a regulator's decision, a survey with its sample, an audited "
                     "figure -- rather than asserted by the organisation alone. 10 = the "
                     "central claims carry a checkable source or method; 0 = every claim "
                     "rests on the organisation's own say-so."},
     {"probe": "incidental",
      "name": "Extent of dateline and wire-service formatting conventions",
      "instruction": "Score 0-10 for how completely the text follows newswire layout "
                     "conventions: an all-caps city-and-date dateline, a '/PRNewswire/' or "
                     "equivalent wire marker, 'FOR IMMEDIATE RELEASE', a '###' or '-30-' end "
                     "mark, hyperlinked ticker notation. 10 = the full set is present; 0 = "
                     "none are. Score the layout conventions only; do not judge the "
                     "substance."}],
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
