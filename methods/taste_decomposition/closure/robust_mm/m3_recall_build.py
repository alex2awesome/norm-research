#!/usr/bin/env python3
"""M3 step 4c -- the FULL-RECALL rediscovery instrument (primary readout).

Why this exists.  Two measured facts break the mechanical detector for this comparison:

  1. the maximum cosine between ANY of the 54 bank concepts and ANY fleet proposal is
     ~.72, below the tau >= .78 band the pilot's planted probes justify.  The 154-bank
     is written in general scientific-reporting register (CONSORT / PRISMA / STROBE /
     TIDieR items) and the fleet, reading ML abstracts, writes in ML register;
  2. within that compressed range the embedder barely RANKS: mean cosine falls only
     .64 -> .59 from the nearest candidate to the tenth.  So a top-3 pairwise window is
     an arbitrary slice of a nearly flat list.

The fix is to stop asking the embedder to shortlist.  For each target concept the judge
sees EVERY proposal the replicate's fleet produced and answers a recall question: does
any of them name this same concept, and which?  Coverage is then complete by
construction and the embedder is used for nothing but reporting.

Blinding: each replicate's 16 targets are 8 HELD-OUT concepts (dropped from the bank
before the slice was regenerated) and 8 stratum-matched RETAINED concepts (still in the
bank -- the false-positive baseline), shuffled together by stable hash and unlabeled.
Candidate proposals are shown with neutral ids and no proposer attribution.

CPU only.  Usage: python m3_recall_build.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/private/tmp/claude-502/-Users-spangher-Projects-stanford-research-norm-research"
               "/4af6bd48-d6eb-47fd-bcda-50f8ab197379/scratchpad/robust_mm")
sys.path.insert(0, str(HERE))
import embed_lib as E  # noqa: E402

SALT = "m3-recall-blind-v1"


def h(s):
    return hashlib.sha256(f"{SALT}|{s}".encode()).hexdigest()


TASK = """You are a BLIND adjudicator in a controlled measurement.

Below are {nc} CANDIDATE criteria (ids K01..K{nc:02d}) that a panel of writers proposed for
judging machine-learning paper abstracts, and {nt} TARGET criteria (ids T01..T{nt:02d}) drawn
from a separate, older rubric bank written in general scientific-reporting language.

For EACH target, decide whether any candidate names the SAME underlying evaluative
concept -- i.e. a judge scoring an abstract on the target would be scoring essentially
the same property as a judge scoring it on that candidate.

Calibration, and it matters:
  * Register differs by design. The targets are written in general scientific-reporting
    language (the vocabulary of reporting guidelines); the candidates are written in
    machine-learning language. Judge the underlying property, NOT the vocabulary.
  * Do NOT accept mere topical adjacency. If the candidate is clearly broader and merely
    contains the target as one sub-case, or narrower and covers only a fragment of it, or
    shares a topic while scoring a different property -- that is NOT a match.
  * A match means substitutable in practice: swapping one for the other in a scorecard
    would produce nearly the same per-abstract scores.
  * Some targets will have NO match. Reporting no match is a perfectly good answer and is
    expected for a substantial share of them. Do not force matches.

OUTPUT. Emit exactly one JSON object and nothing else -- no markdown fences, no
commentary:

{{"results": [
  {{"target": "T01",
   "match": true or false,
   "candidate_ids": ["K07", ...],   // every candidate you judge to be the same concept; [] if none
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="", help="restrict candidates to these families "
                                                   "(comma list); used for the GLM supplement pass")
    ap.add_argument("--suffix", default="", help="output suffix, e.g. _glm")
    args = ap.parse_args()
    fams = [f.strip() for f in args.families.split(",") if f.strip()]

    cfg = json.loads((HERE / "m3_concepts.json").read_text())
    bank = E.bank_concept_texts()
    conc = {r["concept"]: r for r in cfg["concepts"]}
    manifest = {}

    for rep in ("rep1", "rep2", "rep3"):
        props = json.loads((HERE / f"proposals_{rep}.json").read_text())["proposals"]
        if fams:
            props = [p for p in props if p["family"] in fams]
            if not props:
                print(f"{rep}: no candidates for families {fams}, skip")
                continue
        held = cfg["replicates"][rep]
        retained = [c for c in cfg["concept_footprints"] if c not in held]
        ctrl = []
        for s, want in (("high", 3), ("mid", 3), ("low", 2)):
            pool = sorted([c for c in retained if conc[c]["stratum"] == s],
                          key=lambda c: h(f"{rep}|ctrl|{c}"))
            ctrl += pool[:want]

        # candidates: hash-ordered so proposer identity is not recoverable from position
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
            targ_map[tid] = {"concept": c, "kind": kind, "stratum": conc[c]["stratum"],
                             "alone_auc_fitmine": conc[c]["alone_auc_fitmine"]}
            targ_lines.append(f"{tid}. {c}\n     {bank[c].strip()}")

        txt = TASK.format(nc=len(cand_lines), nt=len(targ_lines),
                          cands="\n\n".join(cand_lines), targs="\n\n".join(targ_lines))
        d = SCRATCH / rep
        d.mkdir(parents=True, exist_ok=True)
        (d / f"recall_prompt{args.suffix}.txt").write_text(txt)
        manifest[rep] = {"candidate_map": cand_map, "target_map": targ_map,
                         "n_candidates": len(cand_lines), "n_targets": len(targ_lines),
                         "prompt_path": str(d / f"recall_prompt{args.suffix}.txt"), "prompt_chars": len(txt)}
        print(f"{rep}: {len(cand_lines)} candidates x {len(targ_lines)} targets "
              f"({sum(1 for _, k in targets if k == 'heldout')} held-out / "
              f"{sum(1 for _, k in targets if k != 'heldout')} control), {len(txt)} chars")

    (HERE / f"m3_recall_manifest{args.suffix}.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote m3_recall_manifest{args.suffix}.json")


if __name__ == "__main__":
    main()
