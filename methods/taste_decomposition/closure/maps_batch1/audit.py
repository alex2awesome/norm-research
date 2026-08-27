#!/usr/bin/env python3
"""Blind routing audit (prereg step 4) + routing finalisation, map-focused batch.

build     : pools the round's selected A and B criteria, STRIPS provenance,
            re-ids them in stable-hash order, plants 2 probe PAIRS (one obviously
            quality-relevant + one obviously incidental per pair, authored not
            sampled), and writes the auditor prompt.
finalize  : reads the auditor verdicts, compares each to the PROPOSING track,
            records the misrouting rate and the probe pass rate, and applies the
            audit's ruling (with an arbiter override file when one exists).

FREEZE ADDENDUM 2: for every B-routed channel the auditor also returns an
independent `mixed` judgement (does the conjectured upstream parent plausibly
cause real quality too).  A channel is treated as MIXED if EITHER the proposer or
the auditor says so (union rule -- the mixed flag only widens a sensitivity band,
so the conservative direction is to flag more).

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
     "name": "Explicit statement of the limitation or failure regime",
     "instruction": "Score 0-10 for how explicitly the item names a condition under "
                    "which its own claim would not hold, or a case it does not cover. "
                    "10 = a specific, checkable limitation is stated; 0 = no limitation "
                    "or scope condition is acknowledged anywhere."},
    {"probe": "incidental",
     "name": "Count of numerals and digit characters in the text",
     "instruction": "Score 0-10 for how many numerals and digit characters the text "
                    "contains relative to its length. 10 = digits everywhere; 0 = no "
                    "digits at all. Count characters only; do not judge whether the "
                    "numbers are meaningful or whether their presence is good."},
    {"probe": "quality_relevant",
     "name": "Reasoning given for why the approach should work",
     "instruction": "Score 0-10 for whether the item supplies a reason or mechanism for "
                    "why what it proposes should work, rather than only asserting that "
                    "it does. 10 = an explicit causal or conceptual account; 0 = bare "
                    "assertion with no reasoning."},
    {"probe": "incidental",
     "name": "Use of the Oxford (serial) comma",
     "instruction": "Score 0-10 for how consistently the text uses a comma before the "
                    "final conjunction in a list. 10 = used in every list; 0 = never "
                    "used or no lists present. Judge punctuation habit only; this is "
                    "not a judgement of writing quality."},
]

AUDIT_PROMPT = """You are auditing a pool of scoring criteria that were written by other
models. Your job is a ROUTING decision, made blind: you are NOT told who wrote each
criterion or what it was written for.

For each criterion, decide:

  "quality_relevant" -- the criterion measures a property that a careful expert would
      count as genuine evidence about {construct}. It is about the substance, the
      reasoning, the evidence, the craft, or the content of the item.

  "incidental" -- the criterion measures something that may well PREDICT how the item
      fares but is not itself {construct}: length or verbosity, formatting or layout,
      boilerplate or template phrasing, community/venue style markers, subject-matter
      or topic markers, temporal tells, orthographic habits, or a textual fingerprint of
      some upstream circumstance of the item's production (who made it, with what
      resources, when, with what help).

The corpus is: {corpus}. Each item is a {item}.

SECOND JUDGEMENT, only for criteria you label "incidental". Some incidental channels have
an upstream cause that plausibly ALSO produces genuinely better work (for example, a
well-resourced producer may both be favoured for irrelevant reasons AND actually do
better work). Return `mixed`: true when you think the upstream cause plausibly produces
real {construct} too, false when the channel looks like pure nuisance. For criteria you
label "quality_relevant", return mixed=false.

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

    body = "\n\n".join(
        f"[{r['audit_id']}] {r['name']}\n{r['instruction']}" for r in pool)
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

    probes_ok, probes_n = 0, 0
    final, disputes = [], []
    for r in key:
        v = vmap.get(r["audit_id"])
        if r["true_track"] == "PROBE":
            probes_n += 1
            if v and v["label"] == r["is_probe"]:
                probes_ok += 1
            continue
        audit_label = v["label"] if v else None
        audit_track = {"quality_relevant": "A", "incidental": "B"}.get(audit_label)
        proposed = r["true_track"]
        rec = {"blind_id": r["blind_id"], "name": r["name"],
               "proposed_track": proposed, "audit_label": audit_label,
               "audit_track": audit_track,
               "audit_confidence": (v or {}).get("confidence"),
               "audit_justification": (v or {}).get("justification"),
               "audit_mixed": bool((v or {}).get("mixed", False)),
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
        c = sel.get(r["blind_id"], {})
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
        s.add_argument("--round", type=int, default=1)
    a = ap.parse_args()
    {"build": cmd_build, "finalize": cmd_finalize}[a.cmd](a)
