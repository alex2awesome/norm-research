"""Build GLM calibration prompts — the REAL frozen instrument text over synthetic
constructs with known oracles.

Variants per construct (cycle 1, all in one batch):
  tf            — frozen readout template, straight
  exclusion_v1  — the FROZEN (contradictory) exclusion wording as deployed in W1a; its
                  E-tier miss-rate MEASURES the audit's defect #1 on a competent judge
  exclusion_fx  — W1c candidate fix: final question made variant-consistent
  negated       — frozen NEGATED_WRAPPER
  composed      — frozen COMPOSED_WRAPPER for the declared pairs (oracle = AND)
  confidence    — frozen confidence question after a stated answer (the stated answer is
                  the ORACLE answer, so confidence SHOULD be high on E-tier)

Output rows: {aspect_id: "<construct>::<variant>", datapoint_id: <item idx>, prompt}.
Readout = parsed YES/NO from text (the subscription endpoint has no logprobs; prompt
logic is what is being calibrated, and the logprob channel is separately acceptance-gated
on the frozen grids).

  python -m methods.tacit_channels.battery.synthetic.build_calibration_prompts \
      --out outputs/tacit_channels/battery_calibration/prompts_cycle1.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from methods.tacit_channels.battery.passes import (
    COMPOSED_WRAPPER, EXCLUSION_PREFIX, NEGATED_WRAPPER, confidence_prompt,
)
from methods.tacit_channels.battery.synthetic.constructs import (
    COMPOSED_PAIRS, CONSTRUCTS, ITEMS, oracle_vector,
)

TEMPLATE = Path("outputs/tacit_channels/exp_gtk1/readout_template.txt").read_text()

EXCLUSION_FIXED_QUESTION = (
    "Per the inversion instruction above: answer YES if the text FAILS the stated "
    "criterion, NO if it satisfies it. Answer with exactly one word: YES or NO.")
STRAIGHT_QUESTION = ("Does the text satisfy the criterion? Answer with exactly one word: "
                     "YES or NO.")


def rows_for(cid: str, variant: str, rubric: str, template: str = TEMPLATE) -> list:
    out = []
    for j, text in enumerate(ITEMS):
        out.append({"aspect_id": f"{cid}::{variant}", "datapoint_id": j,
                    "prompt": template.format(rubric=rubric, text=text)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for cid, c in CONSTRUCTS.items():
        base = c["prompt"]
        rows += rows_for(cid, "tf", base)
        rows += rows_for(cid, "exclusion_v1", EXCLUSION_PREFIX + base)
        fixed_tpl = TEMPLATE.replace(STRAIGHT_QUESTION, EXCLUSION_FIXED_QUESTION)
        rows += rows_for(cid, "exclusion_fx", EXCLUSION_PREFIX + base, fixed_tpl)
        rows += rows_for(cid, "negated", NEGATED_WRAPPER.format(content=base))
        # confidence: state the ORACLE answer where known, else YES
        vec = oracle_vector(cid)
        for j, text in enumerate(ITEMS):
            ans = "YES" if (vec[j] if vec is not None else True) else "NO"
            jp = TEMPLATE.format(rubric=base, text=text)
            rows.append({"aspect_id": f"{cid}::confidence", "datapoint_id": j,
                         "prompt": confidence_prompt(jp, ans)})
    for a, b in COMPOSED_PAIRS:
        rubric = COMPOSED_WRAPPER.format(content_a=CONSTRUCTS[a]["prompt"],
                                         content_b=CONSTRUCTS[b]["prompt"])
        rows += rows_for(f"{a}&&{b}", "composed", rubric)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(dict(Counter(r["aspect_id"].split("::")[-1] for r in rows)))
    print(f"{len(rows)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
