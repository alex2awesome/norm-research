#!/usr/bin/env python3
"""Blind routing audit (prereg step 4) + routing finalisation, code_v3 cell.

Copied from maps_hw_si/audit.py with the two cell-specific changes the batch's own
carry-forward demands: **planted probes are CORPUS-MATCHED** (four probes authored for
a GitHub pull request, two unambiguously substantive and two unambiguously
submission-hygiene / shape), and the auditor's definition of "quality_relevant" is
written for a code contribution rather than a one-line joke.

The two probe PAIRS are the ones the campaign brief names:
  pair 1  "explains WHY the change is needed"      (real)
          vs "a pull-request template is filled in" (surface)
  pair 2  "names the concrete failure it prevents" (real)
          vs "conventional-commit / issue-reference formatting" (surface)

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
     "name": "States why the change is needed",
     "instruction": "Score 0-10 for whether the pull request says WHY the change is "
                    "needed -- the problem, need or goal that motivates it -- as opposed "
                    "to only describing what was changed. 10 = the motivating problem or "
                    "goal is stated explicitly and specifically enough that a reader who "
                    "did not know about it would understand it; 0 = the text restates the "
                    "diff, or says nothing about motivation at all. Judge whether the "
                    "reason is actually given, not how tidily the text is laid out and "
                    "not how long it is."},
    {"probe": "incidental",
     "name": "Extent to which a pull-request template is filled in",
     "instruction": "Score 0-10 for how much of a repository pull-request template is "
                    "visibly present and populated -- headed sections such as "
                    "'Description', 'Motivation', 'Type of change', 'How Has This Been "
                    "Tested?', 'Checklist', ticked or unticked checkbox lists, and "
                    "similar scaffolding. 10 = a full template with most sections and "
                    "checkboxes present; 0 = free prose or nothing, with no template "
                    "scaffolding at all. Count the scaffolding only; do not judge whether "
                    "what is written in it is any good."},
    {"probe": "quality_relevant",
     "name": "Names the concrete failure or limitation it addresses",
     "instruction": "Score 0-10 for whether the pull request identifies a specific, "
                    "concrete failure, limitation or behaviour that the change addresses "
                    "-- a particular error, a case that breaks, a measured shortcoming -- "
                    "rather than a generic aim. 10 = one specific failure or limitation "
                    "is named precisely enough to be reproduced or checked; 0 = only "
                    "generic aims ('improve', 'clean up', 'refactor') with nothing "
                    "specific. Judge the specificity of the substance, not the length of "
                    "the text."},
    {"probe": "incidental",
     "name": "Conventional-commit and issue-reference formatting",
     "instruction": "Score 0-10 for how much conventional formatting furniture the title "
                    "and description carry -- a conventional-commit prefix such as "
                    "'feat:', 'fix:', 'chore(scope):', a bracketed ticket tag, and "
                    "'#1234' / 'Fixes #1234' issue references. 10 = a conventional prefix "
                    "and one or more issue references are present; 0 = a plain title with "
                    "no prefix and no issue reference anywhere. Count the formatting "
                    "conventions only; this is not a judgement of whether the change or "
                    "its description is good."},
]

AUDIT_PROMPT = """You are auditing a pool of scoring criteria that were written by other
models. Your job is a ROUTING decision, made blind: you are NOT told who wrote each
criterion or what it was written for.

For each criterion, decide:

  "quality_relevant" -- the criterion measures a property that a careful maintainer
      would count as genuine evidence about {construct}. It is about the substance of
      the change, the engineering judgement it shows, the correctness and design of
      the code, the testing, or what the author actually tells a reviewer.

  "incidental" -- the criterion measures something that may well PREDICT how the item
      fares but is not itself {construct}: length or verbosity, formatting and layout
      habits, template scaffolding, checklists, conventional-commit prefixes and issue
      references, boilerplate or bot-generated phrasing, project- or ecosystem-specific
      style markers, subject-matter or language markers, temporal tells, or a textual
      fingerprint of some upstream circumstance of the change's production (who wrote
      it, how familiar they are with the maintainers, what tooling produced it, where
      it falls in the repository's history).

The corpus is: {corpus}. Each item is a {item}.

SECOND JUDGEMENT, only for criteria you label "incidental". Some incidental channels have
an upstream cause that plausibly ALSO produces genuinely better work (for example, a
long-standing contributor may both be treated more favourably AND actually write better
patches). Return `mixed`: true when you think the upstream cause plausibly produces real
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
