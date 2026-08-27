#!/usr/bin/env python3
"""B-SIDE step -- borderline adjudication pass (the third, provenance-stripped pass
the task spec calls for: "every proposal judged against every held-out channel...,
borderline adjudicated in a provenance-stripped pass").

Reads the two full-recall judges' outputs per replicate, finds targets where they
DISAGREE, and builds a fresh sealed prompt -- same candidate pool, same neutral K-ids,
no proposer attribution, no hint of which judge said what -- asking a THIRD blind
adjudicator to rule on just those targets. This is the FINAL word for the primary
sensitivity statistic (recall_analyze_b.py uses it whenever it exists for a target).

CPU only. Usage: python adjudicate_build_b.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/robust_mm/bside")
REPS = ("bside_rep1", "bside_rep2", "bside_rep3")

TASK = """You are a BLIND adjudicator resolving a DISAGREEMENT between two earlier blind
adjudicators in a controlled measurement. You do not see their reasoning or their
verdicts -- form your own independent judgment from scratch.

Below are {nc} CANDIDATE channels (ids K01..K{nc:02d}) that a panel of writers proposed
while hunting for SUSPECTED-SPURIOUS scoring channels for machine-learning paper
abstracts (features that might predict how an abstract fares WITHOUT being a mark of
quality -- surface style, formatting, topic, or an upstream factor's textual
fingerprint), and {nt} TARGET channels (ids T-ids below) drawn from a separate,
earlier census of declared-nuisance channels for the same corpus.

For EACH target, decide whether any candidate names the SAME underlying channel -- i.e.
a judge scoring an abstract on the target would be scoring essentially the same
surface/upstream property as a judge scoring it on that candidate.

Calibration, and it matters:
  * Phrasing will differ -- the targets and candidates were written by different
    panels at different times. Judge the underlying property being scored, NOT the
    exact wording.
  * Do NOT accept mere topical adjacency. If the candidate is clearly broader and
    merely contains the target as one sub-case, or narrower and covers only a
    fragment of it, or shares a surface topic while scoring a different property --
    that is NOT a match.
  * A match means substitutable in practice: swapping one for the other as a scoring
    instruction would produce nearly the same per-abstract scores.
  * These specific targets were flagged as HARD CASES (two independent earlier judges
    disagreed), so read closely -- but do not force a match just because the case is
    hard. "No match" remains a fully valid, expected answer.

OUTPUT. Emit exactly one JSON object and nothing else -- no markdown fences, no
commentary:

{{"results": [
  {{"target": "T01",
   "match": true or false,
   "candidate_ids": ["K07", ...],
   "confidence": "high" or "low",
   "why": "<one sentence>"}},
  ... one entry for EACH target listed below ...
]}}

=== CANDIDATES ===

{cands}

=== TARGETS (hard cases only) ===

{targs}
"""


def load(rep, fname):
    p = SCRATCH / rep / fname
    if not p.exists():
        return None
    t = p.read_text().strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```\s*$", "", t)
    return {r["target"]: r for r in json.loads(t)["results"]}


def main():
    man = json.loads((HERE / "bside_recall_manifest.json").read_text())
    for rep in REPS:
        if rep not in man:
            continue
        A, B = load(rep, "recall_out_A.json"), load(rep, "recall_out_B.json")
        if not A or not B:
            print(f"{rep}: missing a judge pass, skip")
            continue
        cmap = man[rep]["candidate_map"]
        tmap = man[rep]["target_map"]
        borderline = [tid for tid in tmap if tid in A and tid in B
                     and bool(A[tid]["match"]) != bool(B[tid]["match"])]
        if not borderline:
            print(f"{rep}: 0 borderline targets, no adjudication needed")
            continue

        # rebuild the identical candidate listing from the original recall prompt so
        # K-ids line up exactly; re-derive candidate text from proposals + cand_map
        props = {p["pid"]: p for p in json.loads((HERE / f"proposals_{rep}.json").read_text())["proposals"]}
        cand_lines = []
        for kid in sorted(cmap, key=lambda k: int(k[1:])):
            p = props[cmap[kid]]
            cand_lines.append(f"{kid}. {p['name'].strip()}\n     {p['instruction'].strip()}")

        # target text: recover name/instruction from bside_census.json by channel
        census = json.loads((HERE / "bside_census.json").read_text())
        chan = {c["channel"]: c for c in census["channels"]}
        targ_lines = []
        for tid in borderline:
            c = tmap[tid]["channel"]
            targ_lines.append(f"{tid}. {chan[c]['name']}\n     {chan[c]['instruction'].strip()}")

        txt = TASK.format(nc=len(cand_lines), nt=len(borderline),
                          cands="\n\n".join(cand_lines), targs="\n\n".join(targ_lines))
        d = SCRATCH / rep
        (d / "adjudicate_prompt.txt").write_text(txt)
        print(f"{rep}: {len(borderline)} borderline target(s) {borderline} -> "
              f"{d / 'adjudicate_prompt.txt'} ({len(txt)} chars)")


if __name__ == "__main__":
    main()
