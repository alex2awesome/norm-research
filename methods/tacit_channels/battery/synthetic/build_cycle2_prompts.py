"""Cycle-2 calibration prompts — negation-wording iteration + E5 replacement construct +
graded-holistic pipeline validation (known-mixture blend).

Arms (extended 48-item corpus):
  neg_fx     — explicit-mapping negation wrapper (mirrors the validated exclusion fix)
  neg_direct — hand-negated predicate (upper bound on negation compliance)
  E5 tf / exclusion_fx / confidence — replacement for the bad word-count construct
  blend      — instructed known-mixture 0-10 rating (oracle = 0.5*E1+0.3*E3+0.2*E4)
  h1_graded  — H-tier charming 0-10 (scale-use, no oracle)
  composed   — E3&&E4 second clean AND pair
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from methods.tacit_channels.battery.passes import COMPOSED_WRAPPER, EXCLUSION_PREFIX, \
    confidence_prompt
from methods.tacit_channels.battery.synthetic.constructs import (
    BLEND_PROMPT, CONSTRUCTS, ITEMS, NEG_DIRECT, oracle_vector,
)

TEMPLATE = Path("outputs/tacit_channels/exp_gtk1/readout_template.txt").read_text()
STRAIGHT_Q = ("Does the text satisfy the criterion? Answer with exactly one word: "
              "YES or NO.")
NEG_FX_WRAPPER = ("INVERTED CRITERION: the criterion for this evaluation is the ABSENCE "
                  "of the following property.\n\nProperty: {content}")
NEG_FX_QUESTION = ("Is the property ABSENT from the text? Answer YES if the property is "
                   "absent, NO if it is present. Answer with exactly one word: YES or NO.")
EXCL_FX_QUESTION = ("Per the inversion instruction above: answer YES if the text FAILS "
                    "the stated criterion, NO if it satisfies it. Answer with exactly "
                    "one word: YES or NO.")
GRADED_TEMPLATE = ("You are rating a text.\n\n{rubric}\n\nText:\n{text}\n\n"
                   "Reply with a single integer 0-10.")


def rows_for(cid, variant, rubric, template=TEMPLATE):
    return [{"aspect_id": f"{cid}::{variant}", "datapoint_id": j,
             "prompt": template.format(rubric=rubric, text=t)}
            for j, t in enumerate(ITEMS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    neg_targets = list(NEG_DIRECT)
    for cid in neg_targets:
        base = CONSTRUCTS[cid]["prompt"]
        fx_tpl = TEMPLATE.replace(STRAIGHT_Q, NEG_FX_QUESTION)
        rows += rows_for(cid, "neg_fx", NEG_FX_WRAPPER.format(content=base), fx_tpl)
        rows += rows_for(cid, "neg_direct", NEG_DIRECT[cid])
    base5 = CONSTRUCTS["E5_qmark"]["prompt"]
    rows += rows_for("E5_qmark", "tf", base5)
    excl_tpl = TEMPLATE.replace(STRAIGHT_Q, EXCL_FX_QUESTION)
    rows += rows_for("E5_qmark", "exclusion_fx", EXCLUSION_PREFIX + base5, excl_tpl)
    vec = oracle_vector("E5_qmark")
    for j, t in enumerate(ITEMS):
        jp = TEMPLATE.format(rubric=base5, text=t)
        rows.append({"aspect_id": "E5_qmark::confidence", "datapoint_id": j,
                     "prompt": confidence_prompt(jp, "YES" if vec[j] else "NO")})
    for j, t in enumerate(ITEMS):
        rows.append({"aspect_id": "BLEND::graded", "datapoint_id": j,
                     "prompt": GRADED_TEMPLATE.format(rubric=BLEND_PROMPT, text=t)})
        rows.append({"aspect_id": "H1_charming::graded", "datapoint_id": j,
                     "prompt": GRADED_TEMPLATE.format(
                         rubric="Rate how CHARMING the text is, all things considered.",
                         text=t)})
    rubric = COMPOSED_WRAPPER.format(content_a=CONSTRUCTS["E3_animal"]["prompt"],
                                     content_b=CONSTRUCTS["E4_digit"]["prompt"])
    rows += rows_for("E3_animal&&E4_digit", "composed", rubric)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(dict(Counter(r["aspect_id"].split("::")[-1] for r in rows)))
    print(f"{len(rows)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
