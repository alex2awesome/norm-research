#!/usr/bin/env python3
"""Blind routing audit (prereg step 4) + routing finalisation, BBC most-read cell.

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

PROBE_PAIRS = [
    # PAIR 1 -- substance vs surface numerals
    [{"probe": "quality_relevant",
      "name": "Names the concrete new development that makes this news today",
      "instruction": "Score 0-10 for whether the headline states the specific thing that "
                     "changed -- the decision taken, the result announced, the event that "
                     "happened -- rather than gesturing at a topic or an ongoing "
                     "situation. 10 = a reader knows exactly what is new; 0 = a standing "
                     "topic label with no new development in it."},
     {"probe": "incidental",
      "name": "Contains a numeral",
      "instruction": "Score 0-10 for the presence and prominence of digits in the "
                     "headline -- counts, sums, ages, percentages. 10 = the numeral is "
                     "the first word or the headline contains several; 0 = no digits at "
                     "all. Count the digits only; do not judge whether the number makes "
                     "the story matter."}],
    # PAIR 2 -- earned curiosity vs mere length
    [{"probe": "quality_relevant",
      "name": "Withholds exactly the detail the headline makes the reader want",
      "instruction": "Score 0-10 for whether the headline sets up a specific, legitimate "
                     "question in the reader's mind whose answer is in the story -- who, "
                     "how much, what happened next -- without being deceptive about what "
                     "the story contains. 10 = a precise curiosity gap a reader can only "
                     "close by reading; 0 = either everything is already in the headline "
                     "or the gap is vague clickbait that the story cannot pay off."},
     {"probe": "incidental",
      "name": "Headline word count",
      "instruction": "Score 0-10 for how long the headline is in words. 10 = very long "
                     "(13+ words); 5 = around 8-9 words; 0 = 4 words or fewer. Judge "
                     "extent only -- do not consider whether the length serves the "
                     "story."}],
    # PAIR 3 -- reader consequence vs pronoun token
    [{"probe": "quality_relevant",
      "name": "Signals direct consequence for the reader's own life",
      "instruction": "Score 0-10 for whether the headline makes clear that the story "
                     "affects the reader personally -- their money, health, commute, "
                     "bills, weather, rights -- and roughly how. 10 = the personal stake "
                     "is explicit and concrete; 0 = no implication that the reader's own "
                     "circumstances are touched."},
     {"probe": "incidental",
      "name": "Contains the word 'you' or 'your'",
      "instruction": "Score 0-10 for occurrences of the literal tokens 'you', 'your' or "
                     "'yours' in the headline. 10 = several occurrences or one in the "
                     "opening two words; 0 = none. Count the tokens only; do not judge "
                     "whether the story actually concerns the reader."}],
    # PAIR 4 -- human stake vs quotation marks
    [{"probe": "quality_relevant",
      "name": "Centers a person with an emotionally legible stake",
      "instruction": "Score 0-10 for whether the headline puts a specific human being at "
                     "its centre and makes what they stand to gain or lose immediately "
                     "graspable -- a fight, a loss, a rescue, a triumph. 10 = a named or "
                     "vividly particularised person with a clear stake; 0 = institutions, "
                     "abstractions or processes with no human carrier."},
     {"probe": "incidental",
      "name": "Contains quotation marks",
      "instruction": "Score 0-10 for the presence of quotation marks in the headline -- "
                     "a quoted phrase, a scare-quoted word, a full quoted sentence. "
                     "10 = most of the headline is inside quotes; 0 = no quotation marks. "
                     "Count the punctuation only; do not judge what is quoted or whether "
                     "quoting it is warranted."}],
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
