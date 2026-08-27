#!/usr/bin/env python3
"""GEPA STAGE 2 — author label-blind rephrasings of the GEPA-TARGETED mined criteria.

Stage 1 (`gepa_phrasing.py targets`) flags a criterion for rephrasing when its judged
score distribution is degenerate: modal_share > .75 or na_rate > .20. A criterion can be
conceptually right and still be a bad instrument — if 85% of items get the same token, the
column carries almost no information whatever the idea behind it is. Stage 2 asks a sealed
rephraser to rewrite ONLY the instruction, holding the concept fixed, so the judge is
forced to use the range.

LABEL-BLIND, and the seal matters: the rephraser sees the criterion name, its instruction,
and its OWN degeneracy statistics (modal share, NA rate, the value histogram). It does NOT
see y, any AUC, any item text, or which criteria helped. Rewriting toward "more spread" is
a measurement-quality move, not a label-fitting move; rewriting toward "higher AUC" would
be label-fitting and is why the AUC is withheld.

  build  -> sealed rephraser prompt (3 variants per targeted criterion)
  merge  -> jokes_community_gepa_variants.json, ready for gepa_probe_score.py

CPU only.  Usage: python3 gepa_variants.py build --rounds 1,2,3,4,5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad") / HERE.name / "gepa"
N_VARIANTS = 3

HEAD = """You are the REPHRASER in a preregistered measurement protocol, working label-blind.

CORPUS: short jokes posted to a large public joke-sharing forum.
ITEM: one joke.

BACKGROUND. Each criterion below is scored 0-10 by an independent LLM judge that reads one
joke and one criterion and emits a single token. Every criterion below is CONCEPTUALLY
ACCEPTED — do not change what it is about. Each one has failed a DEGENERACY screen: the
judge puts far too many items on the same score, so the column carries almost no
information no matter how good the idea is. Its degeneracy statistics are shown.

YOUR TASK. For each criterion write exactly {k} REPHRASED VARIANTS of the scoring
INSTRUCTION that keep the concept identical but make the judge actually use the range.
Techniques that work: name concrete anchors for what a 10, a 5 and a 0 look like on THIS
kind of item; replace an all-or-nothing test with a graded one; split a conjunction the
judge is treating as a single gate; say explicitly which common case sits in the middle;
remove an escape hatch that makes the judge default to one value.

HARD CONSTRAINTS.
1. LABEL-BLIND. You have not been shown any outcome, vote, score or ranking for any item,
   and no variant may reference one. You have deliberately NOT been shown whether any
   criterion predicts anything -- do not speculate about it, and do not write toward it.
2. Keep the CONCEPT fixed. A variant that measures a different property is a failure, not
   an improvement.
3. Each variant is a complete standalone 0-10 instruction; say what a 10 and a 0 look
   like.
4. Do not reach for spread by inviting the judge to guess about the poster, the platform,
   the date or the item's history — that would turn a craft criterion into a nuisance
   channel.

CRITERIA TO REPHRASE:

{body}

OUTPUT. Emit exactly one JSON object and nothing else:

{{"variants": [
  {{"target_id": "<the id shown, copied exactly>", "variant": 1,
    "instruction": "<rewritten 0-10 instruction>",
    "rationale": "<which degeneracy this attacks and how>"}},
  ... exactly {k} entries per criterion, in the order shown ...
]}}
"""


def cmd_build(a):
    tgt = json.loads((HERE / f"{a.cell}_gepa_targets.json").read_text())
    targets = [c for c in tgt["criteria"] if c["gepa_targeted"] and not c["COLLAPSED"]]
    # collapsed criteria are EXCLUDED from the bank outright, so rephrasing them would be
    # reviving a column the collapse gate already removed; recorded, not silently dropped.
    excluded = [c for c in tgt["criteria"] if c["COLLAPSED"]]

    rows = []
    for c in targets:
        z = None
        rep = json.loads((HERE / f"{a.cell}_r{c['round']}_score_report.json").read_text())
        pc = rep["per_criterion"][c["blind_id"]]
        sel = {s["blind_id"]: s for s in
               json.loads((HERE / f"{a.cell}_r{c['round']}_species.json").read_text())["selected"]}
        instr = sel.get(c["blind_id"], {}).get("instruction", "")
        vc = pc.get("value_counts", {})
        hist = ", ".join(f"{k}:{v}" for k, v in sorted(vc.items(), key=lambda t: float(t[0])))
        rows.append({"tid": f"r{c['round']}_{c['blind_id']}", "name": c["name"],
                     "instruction": instr, "modal_share": c["modal_share"],
                     "na_rate": c["na_rate"], "hist": hist, "round": c["round"],
                     "blind_id": c["blind_id"]})

    body = "\n\n".join(
        f"[{r['tid']}] {r['name']}\n"
        f"  CURRENT INSTRUCTION: {r['instruction']}\n"
        f"  DEGENERACY: modal share {r['modal_share']:.3f}, NA rate {r['na_rate']:.3f}\n"
        f"  SCORE HISTOGRAM (score:count over 16,000 items): {r['hist']}"
        for r in rows)

    SCRATCH.mkdir(parents=True, exist_ok=True)
    (SCRATCH / "prompt_rephraser.txt").write_text(HEAD.format(k=N_VARIANTS, body=body))
    (HERE / f"{a.cell}_gepa_targets_expanded.json").write_text(json.dumps(
        {"cell": a.cell, "n_targeted": len(rows), "n_variants_each": N_VARIANTS,
         "excluded_collapsed": [f"r{c['round']}_{c['blind_id']}" for c in excluded],
         "targets": rows}, indent=1))
    print(f"{len(rows)} GEPA-targeted criteria x {N_VARIANTS} variants "
          f"({len(excluded)} collapsed criteria excluded, not rephrased)")
    for r in rows:
        print(f"  {r['tid']:9s} modal {r['modal_share']:.3f} na {r['na_rate']:.3f}  {r['name'][:48]}")
    print("prompt ->", SCRATCH / "prompt_rephraser.txt")


def cmd_merge(a):
    import re
    txt = (SCRATCH / "out_rephraser.txt").read_text().strip()
    if txt.startswith("```"):
        txt = txt.split("\n", 1)[1].rsplit("```", 1)[0]
    obj = json.loads(re.search(r"\{[\s\S]*\}", txt).group(0))
    exp = json.loads((HERE / f"{a.cell}_gepa_targets_expanded.json").read_text())
    by = {t["tid"]: t for t in exp["targets"]}
    out = []
    for v in obj["variants"]:
        t = by.get(v["target_id"])
        if t is None:
            continue
        out.append({"variant_id": f"{v['target_id']}_v{v['variant']}",
                    "target_id": v["target_id"], "round": t["round"],
                    "blind_id": t["blind_id"], "name": t["name"],
                    "instruction": v["instruction"].strip(),
                    "rationale": v.get("rationale", "")})
    missing = sorted(set(by) - {v["target_id"] for v in out})
    (HERE / f"{a.cell}_gepa_variants.json").write_text(json.dumps(
        {"cell": a.cell, "n_variants": len(out), "n_targets": len(by),
         "targets_without_variants": missing, "variants": out}, indent=1))
    print(f"{len(out)} variants over {len(by) - len(missing)}/{len(by)} targets "
          f"-> {a.cell}_gepa_variants.json")
    if missing:
        print("  WARNING no variants for:", missing)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("build", "merge"):
        s = sub.add_parser(n)
        s.add_argument("--cell", default="jokes_community")
        s.add_argument("--rounds", default="1,2,3,4,5")
    a = ap.parse_args()
    {"build": cmd_build, "merge": cmd_merge}[a.cmd](a)
