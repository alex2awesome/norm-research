#!/usr/bin/env python3
"""Blind routing audit (prereg step 4) + routing finalisation, CAP_FINALIST cell.

Identical machinery to mathse_vote/audit.py; what is cell-specific is the planted probe
pool, which the freeze requires to be CORPUS-MATCHED (the maps-batch-1 carry-forward,
notes/2026-08-06__spurious_maps_batch1.md 1d: probes authored for the wrong genre make
the auditor fail through no fault of its own).

Four PAIRS authored FOR A CARTOON CAPTION CONTEST; each pair is one unambiguously
craft-relevant criterion plus one unambiguously incidental look-alike that would
correlate with it (unexpected speaker vs direct-speech punctuation; reframing the drawing
vs naming objects in it; withheld explanation vs raw word count; register clash vs the
presence of a proper noun).  Rounds 1-2 of this cell were audited with probes carried
over from the PEER-REVIEW genre -- round 1's planted quality probe was "Explicit statement
of the limitation or failure regime", scored against one-line captions -- which is exactly
the mismatch the freeze forbids; that is recorded in the campaign note and is one of the
three reasons rounds 1-2 are not at current standard. Two pairs are drawn per round by
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
    # PAIR 1 -- who is speaking vs how the speech is punctuated
    [{"probe": "quality_relevant",
      "name": "Line is assigned to the character it is least expected from",
      "instruction": "Score 0-10 for how far the caption puts the line in the mouth of "
                     "the figure in the drawing from whom it is least expected, so that "
                     "the surprise comes from the pairing of speaker and speech. 10 = the "
                     "speaker attribution is itself the joke; 0 = the line could be said "
                     "by any figure in the drawing, or by no one in particular. Judge the "
                     "speaker choice, not whether the caption amused you."},
     {"probe": "incidental",
      "name": "Punctuated as a completed sentence of direct speech",
      "instruction": "Score 0-10 for how far the caption is typed as a single complete "
                     "sentence of direct speech with standard terminal punctuation. 10 = "
                     "one full sentence, capitalised, ending in a period or question mark; "
                     "0 = a fragment, a list, or no terminal punctuation. Count the "
                     "typography only; do not judge whether the caption works."}],
    # PAIR 2 -- reframing the drawing vs merely naming what is in it
    [{"probe": "quality_relevant",
      "name": "Caption reframes what the drawing is a picture of",
      "instruction": "Score 0-10 for how far the caption makes the reader see the whole "
                     "drawing as a picture of a different situation from the one it "
                     "obviously depicts. 10 = the scene is completely relocated by the "
                     "line; 0 = the caption accepts the obvious reading of the scene and "
                     "comments inside it."},
     {"probe": "incidental",
      "name": "Explicitly names an object visible in the drawing",
      "instruction": "Score 0-10 for how many of the concrete objects, animals or "
                     "settings visible in the cartoon are named in so many words in the "
                     "caption. 10 = the caption names several of them explicitly; 0 = it "
                     "names none. Count the naming only; do not judge whether naming them "
                     "helps the joke."}],
    # PAIR 3 -- restraint as craft vs raw brevity
    [{"probe": "quality_relevant",
      "name": "Withholds the explanation the drawing seems to demand",
      "instruction": "Score 0-10 for how far the caption declines to explain the odd "
                     "thing in the drawing, and instead speaks as though it needed no "
                     "explanation, leaving the reader to supply the logic. 10 = the "
                     "strangeness is treated as entirely unremarkable; 0 = the caption "
                     "spells out why the scene is the way it is."},
     {"probe": "incidental",
      "name": "Number of words in the caption",
      "instruction": "Score 0-10 inversely with the caption's raw length in words: 10 = "
                     "five words or fewer, 0 = more than twenty-five words, scaling in "
                     "between. Count words only; do not judge whether the length suits "
                     "the joke."}],
    # PAIR 4 -- register clash vs the presence of a named entity
    [{"probe": "quality_relevant",
      "name": "Register of the words clashes with the situation drawn",
      "instruction": "Score 0-10 for how far the vocabulary and tone of the caption belong "
                     "to a different world from the scene in the drawing -- corporate "
                     "jargon in a jungle, therapeutic language at a crime scene, domestic "
                     "small talk in a catastrophe. 10 = a sustained clash of register runs "
                     "through the line; 0 = the words match the setting."},
     {"probe": "incidental",
      "name": "Contains a proper noun or brand name",
      "instruction": "Score 0-10 for how prominently the caption uses a proper noun -- a "
                     "person's name, a place, a company, a product, a title. 10 = a named "
                     "entity is the subject of the line; 0 = no proper noun appears. Count "
                     "the presence of the name only; do not judge whether it is apt."}],
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
