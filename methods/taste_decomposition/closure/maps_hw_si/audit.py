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

PROBES = [
    {"probe": "quality_relevant",
     "name": "Sets up an expectation and then turns it",
     "instruction": "Score 0-10 for whether the wording first leads the reader toward one "
                    "reading and then supplies a different one. 10 = a clear setup and a "
                    "clear turn away from it; 0 = a single flat statement with no setup "
                    "and nothing turned. Judge the comic structure, not the typography."},
    {"probe": "incidental",
     "name": "Extent of capitalised or shouted typography",
     "instruction": "Score 0-10 for how much of the text is set in capital letters or "
                    "otherwise typographically shouted (ALL-CAPS words, repeated letters, "
                    "strings of exclamation marks). 10 = the text is dominated by shouted "
                    "typography; 0 = ordinary sentence case throughout. Count the "
                    "typography only; do not judge whether it is good or whether the item "
                    "is funny."},
    {"probe": "quality_relevant",
     "name": "A specific concrete detail carries the joke",
     "instruction": "Score 0-10 for whether the comic work is done by a specific, concrete "
                    "particular (a named thing, an exact number, a precise action) rather "
                    "than by a generic category word. 10 = one vivid particular is doing "
                    "the work; 0 = only generic labels, nothing specific enough to picture. "
                    "Judge the specificity of the comic content, not the length."},
    {"probe": "incidental",
     "name": "Trailing furniture: hashtags, handles, parenthetical byline",
     "instruction": "Score 0-10 for how much non-sentence furniture is appended at the end "
                    "of the text -- trailing hashtags, @-handles, a parenthetical name and "
                    "town, or similar tags. 10 = several such tags appended; 0 = the text "
                    "ends with the sentence itself. Count the furniture only; this is not a "
                    "judgement of writing quality."},
]

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
    for j, p in enumerate(PROBES):
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
