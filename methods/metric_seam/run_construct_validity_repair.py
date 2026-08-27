#!/usr/bin/env python3
"""Replay the Math-a12 proxy failure as a construct-validity counterexample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from methods.metric_seam.verifiers.lifecycle import ConstructControl, UnitProposal
from methods.metric_seam.verifiers.math_a12_symbolic import (
    extract_equality_pairs,
    verify_pair,
)
from methods.metric_seam.verifiers.schema import Verdict


PROPOSAL = UnitProposal(
    unit_id="math.a12.rigor.contextual_equation_use",
    task="math",
    criterion_id="a12",
    construct_text="Precision and rigor in statements and proofs.",
    relation="An equation is used in a logically licensed role and does not introduce a rigor error.",
    occasion="The answer presents an equation as a definition, hypothesis, constraint, proof step, or conclusion.",
    satisfied_when="The equation is legitimate in its contextual logical role.",
    violated_when="The equation is asserted or used in a way that is invalid for its contextual logical role.",
    required_context="Full answer text, including the equation's surrounding logical-role language.",
    non_goals=("context-free symbolic identity", "whole-proof correctness"),
    proxy_risks=("treating nonidentical sides as an error", "rewarding only closed arithmetic tautologies"),
)


def controls() -> tuple[ConstructControl, ...]:
    proxy_on = (
        ("definition", "# Answer\nLet $b = s*x$ define $b$ for the remainder of the proof."),
        ("hypothesis", "# Answer\nAssume $f = g+h$. We derive the requested consequence under this hypothesis."),
        ("constraint", "# Answer\nWork subject to the constraint $a^2+b^2 = 1$."),
        ("solve", "# Answer\nTo solve the equation $2*r+4 = r-7$, collect the $r$ terms."),
        ("decomposition", "# Answer\nDefine the total by $n_b+n_s = n$."),
        ("boundary", "# Answer\nImpose the boundary condition $u = 0$ at the endpoint."),
        ("initial", "# Answer\nTake $x = 0$ as the initial condition."),
        ("recurrence", "# Answer\nDefine the next value by $q = p+1$."),
    )
    proxy_off = (
        ("unsupported", "# Answer\nTherefore the theorem is true. No derivation or justification is supplied."),
        ("false_generalization", "# Answer\nEvery finite group is abelian, which is obvious."),
        ("circular", "# Answer\nThe conclusion follows because the conclusion is correct."),
        ("missing_case", "# Answer\nThis proves the result; the excluded zero case is not considered."),
    )
    values = [
        ConstructControl(
            f"a12.proxy-on.{name}", text, "satisfied", True,
            "A definition, hypothesis, constraint, or equation-to-solve can be rigorous even when its sides are not identical.",
        )
        for name, text in proxy_on
    ]
    values.extend(
        ConstructControl(
            f"a12.proxy-off.{name}", text, "violated", False,
            "The prose has a rigor defect without presenting an adjacent symbolic equality for the identity proxy.",
        )
        for name, text in proxy_off
    )
    return tuple(values)


def _old_verifier(text: str, *, control_id: str) -> tuple[Verdict, int]:
    pairs = extract_equality_pairs(text, item_key=control_id)
    if not pairs:
        return Verdict(False, False), 0
    verdicts = [verify_pair(pair) for pair in pairs]
    applicable = [verdict for verdict in verdicts if verdict.applies]
    if not applicable:
        return Verdict(False, False), len(pairs)
    witnesses = tuple(span for verdict in applicable for span in verdict.witnesses)
    return Verdict(True, any(verdict.violated for verdict in applicable), witnesses), len(pairs)


def build_readout() -> dict[str, object]:
    rows = []
    for control in controls():
        verdict, pair_count = _old_verifier(control.ctext, control_id=control.control_id)
        rows.append({
            "control_id": control.control_id,
            "expected_construct_state": control.expected_state,
            "proxy_triggered": control.proxy_triggered,
            "symbolic_pair_count": pair_count,
            "old_verifier_state": verdict.state,
            "old_verifier_matches_construct": verdict.state == control.expected_state,
            "rationale": control.rationale,
            "ctext": control.ctext,
        })
    proxy_on = [row for row in rows if row["proxy_triggered"]]
    return {
        "schema": "metric-seam.construct-validity-repair.a12.v1",
        "status": "proxy_misconstrual_executably_demonstrated",
        "proposal": PROPOSAL.to_json_value(),
        "controls": rows,
        "summary": {
            "controls": len(rows),
            "old_verifier_construct_correct": sum(row["old_verifier_matches_construct"] for row in rows),
            "proxy_on_construct_satisfied": len(proxy_on),
            "proxy_on_called_violated": sum(row["old_verifier_state"] == "violated" for row in proxy_on),
        },
        "adjudication_status": "audit_and_author_expected_states_recorded; blinded_two-pass_prompt_adjudication_pending",
        "disposition": {
            "a12_rigor_unit": "rejected_before_freeze_construct_misconstrual",
            "narrow_symbolic_capability": "retain_as_context_free_rational_expression_identity_only",
            "prior_kappa_1_interpretation": "implementation_agreement_under_shared_misconstrual_not_validity",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(build_readout(), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
