#!/usr/bin/env python3
"""Articulation ladder for stage-1 candidates (metric foundry, 2026-07-08).

The decompression-rungs stream showed execution quality rises with articulation density
(name -> definition -> explanation -> exemplars). Stage-1 candidates are one-sentence rubrics;
this expands each to a RUNG-3 dense rubric via GLM: precise definition, why the community
rewards it, anchored scoring guidance (what 0 / 0.5 / 1 look like), and two short contrastive
exemplar sketches. Output: {name: dense_rubric} JSON per domain for replicate_candidates
--dense-rubrics. Label-free (articulation never sees labels; reconstruction-only discipline).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "methods")
sys.path.insert(0, "scripts/tools")

from replicate_candidates import stage1_candidates  # noqa: E402

PROMPT = """You are articulating an evaluation metric for {domain} so that a careful but
literal-minded judge can score any single item 0-1 reliably.

METRIC NAME: {name}
CURRENT ONE-SENTENCE RUBRIC: {rubric}
DESCRIPTION: {description}

Write a DENSE, self-contained scoring rubric with exactly these sections:
DEFINITION: one precise sentence saying what property is being measured (not why).
GUIDANCE: what a score of 1.0 looks like; what 0.5 looks like; what 0.0 looks like — concrete,
observable features a judge can check in one reading.
POSITIVE SKETCH: a 2-3 sentence sketch of a hypothetical item that clearly scores 1.0.
NEGATIVE SKETCH: a 2-3 sentence sketch of a hypothetical item that clearly scores 0.0.
BOUNDARY NOTE: the single most likely confusion with a neighboring property, and how to
resolve it.

Keep the whole rubric under 250 words. Return ONLY the rubric text (no preamble)."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", required=True)
    ap.add_argument("--skip-pooled", action="store_true")
    ap.add_argument("--domain-hint", required=True)
    ap.add_argument("--alpha1", type=float, default=0.05)
    ap.add_argument("--proposer-model", default="glm-5.2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                                 base_url=os.environ.get("ANTHROPIC_BASE_URL"))
    legs = [Path(p) for p in sorted(glob.glob(args.legs)) if Path(p).is_dir()]
    if args.skip_pooled:
        legs = [p for p in legs if "pooled" not in p.name]
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    dense = json.load(open(outp)) if outp.exists() else {}   # resume
    import time
    for leg in legs:
        for c in stage1_candidates(leg, args.alpha1):
            if c["name"] in dense:
                continue
            txt = ""
            for attempt, wait in enumerate((0, 20, 60, 180)):
                if wait:
                    time.sleep(wait)
                try:
                    r = client.messages.create(
                        model=args.proposer_model, max_tokens=800, temperature=0.4,
                        messages=[{"role": "user", "content": PROMPT.format(
                            domain=args.domain_hint, name=c["name"],
                            rubric=c["rubric"][:600], description=c["description"][:300])}])
                    txt = "".join(b.text for b in r.content if hasattr(b, "text")).strip()
                    break
                except Exception as e:
                    print(f"  retryable ({type(e).__name__}) attempt {attempt+1}", flush=True)
            if len(txt) > 100:
                dense[c["name"]] = txt
                json.dump(dense, open(outp, "w"), indent=1)   # incremental save
                print(f"articulated: {c['name'][:56]} ({len(txt)} chars)", flush=True)
            else:
                print(f"  FAILED to articulate: {c['name'][:56]}", flush=True)
    print(f"WROTE {len(dense)} dense rubrics -> {outp}", flush=True)


if __name__ == "__main__":
    main()
