#!/usr/bin/env python3
"""SI PAIRWISE PROBE -- sealed judge prompts (batched).

One prompt = one QUESTION applied to a BATCH of pairs.  Batching is what makes a frontier
judge affordable here (371 pairs x 9 questions = 3,339 comparisons; at 25 pairs per call
that is ~134 calls instead of 3,339), and it costs nothing methodologically because each
pair is independent and the judge answers per pair_id.

QUESTIONS
  HOLISTIC   "which would the contest's editor pick?"  -- the separation test that
             decides the probe.  Run on EVERY pair, including the swap and anchor arms.
  8 CRITERIA "which better exemplifies <criterion>?"   -- run on the MATCHED arm plus the
             anchors.  These separate "the criteria are wrong" from "the judge cannot see
             it": if HOLISTIC separates and the criteria do not, the bank's criteria are
             the problem; if neither separates, the construct is not text-recoverable at
             this judge's capability.

CRITERION SUBSET, fixed here before any call, chosen to span the v2 bank's design
families rather than to maximise anything:
  a08 comic leap takes more than one step   (the bank's strongest absolute criterion, .528)
  a07 joke is available from the prompt alone            [negative -- obviousness family]
  a06 could serve a different prompt unchanged           [negative -- portability family]
  a11 unlikely to be independently duplicated            [contestability family]
  a01 continues past its own punch                       [negative -- length-cancelling]
  a15 punch word occupies the final position             [positional family]
  a19 every clause carries comic weight                  [ratio/length-orthogonal family]
  a32 ending is the strongest beat                       [positional family]
Four of the eight are NEGATIVELY oriented, so a judge that just picks the better entry
every time will score BELOW .5 on those four -- which is itself a useful check that the
judge is answering the question asked rather than the holistic one.

SEAL.  The judge sees the contest prompt and two entries labelled A and B.  It is never
told which entry won, what the tiers are, that one arm is length-matched, that anchors
are planted, or that a swap arm exists.  Position is pre-randomised in build_pairs.py.

CPU only.  Usage: python build_prompts.py --batch 25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SI = HERE.parent

CRITERIA = ["a08", "a07", "a06", "a11", "a01", "a15", "a19", "a32"]

HOLISTIC = (
    "For each pair below, two entries were submitted to the SAME humour contest, in "
    "answer to the SAME prompt. Both were good enough to be published. One of them was "
    "chosen by the contest's editor as the week's WINNER; the other was published only as "
    "an honourable mention.\n\n"
    "For each pair, say which entry the editor picked."
)

CRIT_TMPL = (
    "For each pair below, two entries were submitted to the SAME humour contest, in "
    "answer to the SAME prompt.\n\n"
    "For each pair, say which entry BETTER EXEMPLIFIES the following property.\n\n"
    "PROPERTY: {name}\n"
    "DEFINITION: {desc}\n\n"
    "Judge the property exactly as defined, NOT whether the entry is funnier or better "
    "overall. If the property is a flaw, the entry that exhibits the flaw MORE is the one "
    "that better exemplifies it."
)

TAIL = (
    "\n\nOUTPUT. Emit exactly one JSON object and nothing else:\n"
    '{{"answers": [{{"pair_id": "<id>", "choice": "A" or "B", '
    '"confidence": "high"/"medium"/"low"}}, ...]}}\n'
    "One entry per pair, using the pair_id shown, covering EVERY pair listed. "
    "You must choose A or B for every pair; there is no tie option.\n\nPAIRS:\n\n"
)


def render(pairs):
    out = []
    for p in pairs:
        out.append(
            f"--- pair_id={p['pair_id']} ---\n"
            f"CONTEST PROMPT: {p['prompt']}\n"
            f"ENTRY A: {p['entry_A']}\n"
            f"ENTRY B: {p['entry_B']}")
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=25)
    a = ap.parse_args()

    pairs = json.loads((HERE / "si_pairs.json").read_text())
    rubrics = {r["rubric_id"]: r for r in
               (json.loads(l) for l in open(SI / "va_v2" / "rubrics.jsonl") if l.strip())}

    jobs = []
    holistic_set = pairs                       # every arm
    crit_set = [p for p in pairs if p["arm"] in ("MATCHED", "ANCHOR_SCRAM",
                                                 "ANCHOR_FRAGMENT")]

    def chunk(seq, n):
        return [seq[i:i + n] for i in range(0, len(seq), n)]

    outdir = HERE / "prompts"
    outdir.mkdir(exist_ok=True)
    for b, ch in enumerate(chunk(holistic_set, a.batch)):
        tag = f"holistic_b{b:02d}"
        (outdir / f"{tag}.txt").write_text(HOLISTIC + TAIL.format() + render(ch))
        jobs.append({"tag": tag, "question": "holistic", "n_pairs": len(ch),
                     "pair_ids": [p["pair_id"] for p in ch]})
    for cid in CRITERIA:
        r = rubrics[cid]
        head = CRIT_TMPL.format(name=r["name"], desc=r["description"])
        for b, ch in enumerate(chunk(crit_set, a.batch)):
            tag = f"{cid}_b{b:02d}"
            (outdir / f"{tag}.txt").write_text(head + TAIL.format() + render(ch))
            jobs.append({"tag": tag, "question": cid, "criterion": r["name"],
                         "orientation": r["orientation"], "n_pairs": len(ch),
                         "pair_ids": [p["pair_id"] for p in ch]})

    man = {"n_jobs": len(jobs), "batch": a.batch,
           "n_holistic_pairs": len(holistic_set), "n_criterion_pairs": len(crit_set),
           "criteria": CRITERIA,
           "total_comparisons": len(holistic_set) + len(CRITERIA) * len(crit_set),
           "jobs": jobs}
    (HERE / "si_prompt_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man.items() if k != "jobs"}, indent=1))
    print(f"wrote {len(jobs)} prompt files -> {outdir}")


if __name__ == "__main__":
    main()
