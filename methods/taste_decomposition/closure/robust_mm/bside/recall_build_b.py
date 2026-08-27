#!/usr/bin/env python3
"""B-SIDE step -- the FULL-RECALL rediscovery instrument, mirroring
../m3_recall_build.py exactly (never an embedding threshold; see that file's
docstring for why the mechanical tau detector is out of range on this corpus and
is not even attempted here).

For each target channel the judge sees EVERY proposal the replicate's fleet produced
and answers a recall question: does any of them name this same channel, and which?

Blinding: each replicate's 12 targets = 6 HELD-OUT channels (this replicate's
bside_census.json holdout set) + 6 stratum-matched RETAINED channels (channels this
replicate never held out -- the false-positive baseline), shuffled together by stable
hash and unlabeled. Candidate proposals are shown with neutral ids and no proposer
attribution.

CPU only. Usage: python recall_build_b.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/robust_mm/bside")
REPS = ("bside_rep1", "bside_rep2", "bside_rep3")

SALT = "bside-recall-blind-v1"


def h(s):
    return hashlib.sha256(f"{SALT}|{s}".encode()).hexdigest()


TASK = """You are a BLIND adjudicator in a controlled measurement.

Below are {nc} CANDIDATE channels (ids K01..K{nc:02d}) that a panel of writers proposed
while hunting for SUSPECTED-SPURIOUS scoring channels for machine-learning paper
abstracts (features that might predict how an abstract fares WITHOUT being a mark of
quality -- surface style, formatting, topic, or an upstream factor's textual
fingerprint), and {nt} TARGET channels (ids T01..T{nt:02d}) drawn from a separate,
earlier census of declared-nuisance channels for the same corpus.

For EACH target, decide whether any candidate names the SAME underlying channel --
i.e. a judge scoring an abstract on the target would be scoring essentially the same
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
  * Some targets will have NO match. Reporting no match is a perfectly good answer and
    is expected for a substantial share of them. Do not force matches.

OUTPUT. Emit exactly one JSON object and nothing else -- no markdown fences, no
commentary:

{{"results": [
  {{"target": "T01",
   "match": true or false,
   "candidate_ids": ["K07", ...],   // every candidate you judge to be the same channel; [] if none
   "confidence": "high" or "low",
   "why": "<one sentence>"}},
  ... one entry for every target T01..T{nt:02d} ...
]}}

=== CANDIDATES ===

{cands}

=== TARGETS ===

{targs}
"""


def main():
    cfg = json.loads((HERE / "bside_census.json").read_text())
    chan = {c["channel"]: c for c in cfg["channels"]}
    manifest = {}

    for rep in REPS:
        pf = HERE / f"proposals_{rep}.json"
        if not pf.exists():
            print(f"{rep}: no proposals file yet, skip")
            continue
        props = json.loads(pf.read_text())["proposals"]
        design_rep = rep.replace("bside_", "")  # rep1/rep2/rep3 in bside_census.json
        held = cfg["replicates"][design_rep]
        retained_pool = [c for c in chan if c not in held]
        ctrl = []
        for s, want in (("high", 2), ("mid", 2), ("low", 2)):
            pool = sorted([c for c in retained_pool if chan[c]["stratum"] == s],
                          key=lambda c: h(f"{rep}|ctrl|{c}"))
            ctrl += pool[:want]

        order = sorted(range(len(props)), key=lambda i: h(f"{rep}|cand|{props[i]['pid']}"))
        cand_lines, cand_map = [], {}
        for n, i in enumerate(order):
            kid = f"K{n+1:02d}"
            cand_map[kid] = props[i]["pid"]
            cand_lines.append(f"{kid}. {props[i]['name'].strip()}\n     {props[i]['instruction'].strip()}")

        targets = [(c, "heldout") for c in held] + [(c, "control_retained") for c in ctrl]
        targets.sort(key=lambda t: h(f"{rep}|targ|{t[0]}"))
        targ_lines, targ_map = [], {}
        for n, (c, kind) in enumerate(targets):
            tid = f"T{n+1:02d}"
            targ_map[tid] = {"channel": c, "kind": kind, "stratum": chan[c]["stratum"],
                             "alone_auc_fitmine": chan[c]["alone_auc_fitmine"]}
            targ_lines.append(f"{tid}. {chan[c]['name']}\n     {chan[c]['instruction'].strip()}")

        txt = TASK.format(nc=len(cand_lines), nt=len(targ_lines),
                          cands="\n\n".join(cand_lines), targs="\n\n".join(targ_lines))
        d = SCRATCH / rep
        d.mkdir(parents=True, exist_ok=True)
        (d / "recall_prompt.txt").write_text(txt)
        manifest[rep] = {"candidate_map": cand_map, "target_map": targ_map,
                         "n_candidates": len(cand_lines), "n_targets": len(targ_lines),
                         "prompt_path": str(d / "recall_prompt.txt"), "prompt_chars": len(txt)}
        print(f"{rep}: {len(cand_lines)} candidates x {len(targ_lines)} targets "
              f"({sum(1 for _, k in targets if k == 'heldout')} held-out / "
              f"{sum(1 for _, k in targets if k != 'heldout')} control), {len(txt)} chars")

    (HERE / "bside_recall_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote bside_recall_manifest.json")


if __name__ == "__main__":
    main()
