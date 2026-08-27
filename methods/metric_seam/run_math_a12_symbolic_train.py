#!/usr/bin/env python3
"""Run the existing Math-a12 symbolic capability through the verifier TRAIN gate.

This is CPU-only and accepts only the frozen compiler TRAIN bundle.  The unit
of analysis is an extracted adjacent equality pair.  Synthetic nonidentity
pairs are probe diagnostics only and cannot alter natural-corpus base rates.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from methods.metric_seam.verifiers.math_a12_selection import SCHEMA as SELECTION_SCHEMA
from methods.metric_seam.verifiers.math_a12_symbolic import (
    EqualityPair,
    RELATION_ID,
    extract_equality_pairs,
    verify_pair,
)
from methods.metric_seam.verifiers.schema import Verdict
from methods.metric_seam.verifiers.train_gate import (
    ProbeOutcome,
    TrainDiscriminationResult,
    TrainObservation,
    evaluate_train_discrimination,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / (
    "outputs/metric_seam_pilot/reconstruction_v2/"
    "math_a12_symbolic_step_retrospective_prepare_001/compiler_bundle.json"
)
DEFAULT_SELECTION = ROOT / (
    "outputs/metric_seam_pilot/hierarchy_r123/math_a12_verifier_selection_v1.json"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/metric_seam_pilot/hierarchy_r123/results/"
    "math_a12_symbolic_train_v1/readout.json"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fraction(value: Fraction | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {"numerator": value.numerator, "denominator": value.denominator}


def _gate_value(gate: TrainDiscriminationResult) -> dict[str, Any]:
    return {
        "passed": gate.passed,
        "failure_reason": gate.failure_reason.value if gate.failure_reason else None,
        "failed_checks": list(gate.failed_checks),
        "natural_pair_candidates": gate.total_items,
        "natural_completed": gate.completed_items,
        "natural_applies": gate.applies_count,
        "natural_violated": gate.violated_count,
        "distinct_states_with_probes": gate.distinct_patterns,
        "natural_modal_count": gate.modal_count,
        "probe_total": gate.probe_total,
        "probe_correct": gate.probe_correct,
        "probe_inversions": gate.probe_inversions,
        "all_observations": gate.all_total_items,
        "completeness": _fraction(gate.completeness),
        "applies_rate": _fraction(gate.applies_rate),
        "violated_given_applies": _fraction(gate.violated_given_applies),
        "mode_rate": _fraction(gate.mode_rate),
        "probe_correct_rate": _fraction(gate.probe_correct_rate),
    }


def _mutant(pair: EqualityPair, index: int) -> EqualityPair:
    pair_id = f"probe-{index:03d}-{pair.pair_id}"
    return EqualityPair(
        item_key=pair.item_key,
        pair_id=pair_id,
        lhs=pair.lhs,
        rhs=f"({pair.rhs}) + 1",
        witness=type(pair.witness)(
            pair.witness.path,
            pair.witness.start_line,
            pair.witness.end_line,
            node_id=pair_id,
        ),
    )


def _probe_outcome(natural: Verdict, planted: Verdict) -> ProbeOutcome:
    if natural.applies and not natural.violated and planted.applies and planted.violated:
        return ProbeOutcome.CORRECT
    if natural.applies and natural.violated and planted.applies and not planted.violated:
        return ProbeOutcome.INVERTED
    return ProbeOutcome.INCORRECT


def run_train(rows: Sequence[dict[str, Any]], *, probe_cap: int = 40) -> dict[str, Any]:
    if not rows or probe_cap < 1:
        raise ValueError("TRAIN rows and a positive probe cap are required")
    if any(set(row) != {"ctext", "item_key"} for row in rows):
        raise ValueError("TRAIN rows exceed the ctext/item_key allowlist")

    pair_rows: list[dict[str, Any]] = []
    observations: list[TrainObservation] = []
    satisfied: list[tuple[EqualityPair, Verdict]] = []
    document_counts: list[dict[str, object]] = []
    for row in rows:
        pairs = extract_equality_pairs(row["ctext"], item_key=row["item_key"])
        document_counts.append(
            {"item_key": row["item_key"], "pair_candidate_count": len(pairs)}
        )
        for pair in pairs:
            verdict = verify_pair(pair)
            observation_id = f"natural:{pair.item_key}:{pair.pair_id}"
            observations.append(
                TrainObservation(observation_id, "natural_equality_pair", "natural", verdict)
            )
            pair_rows.append(
                {
                    **pair.to_request_value(),
                    "verdict": verdict.to_json_value(),
                    "state": verdict.state,
                }
            )
            if verdict.applies and not verdict.violated:
                satisfied.append((pair, verdict))

    probes: list[ProbeOutcome] = []
    probe_rows: list[dict[str, Any]] = []
    for index, (natural_pair, natural_verdict) in enumerate(satisfied[:probe_cap], 1):
        mutant = _mutant(natural_pair, index)
        planted = verify_pair(mutant)
        outcome = _probe_outcome(natural_verdict, planted)
        probes.append(outcome)
        observations.append(
            TrainObservation(
                f"planted:{mutant.item_key}:{mutant.pair_id}",
                "rhs_plus_one",
                "planted",
                planted,
            )
        )
        probe_rows.append(
            {
                "base_pair_id": natural_pair.pair_id,
                "planted_pair": mutant.to_request_value(),
                "verdict": planted.to_json_value(),
                "outcome": outcome.value,
            }
        )

    gate = evaluate_train_discrimination(observations, probes)
    states: dict[str, int] = {}
    for row in pair_rows:
        states[row["state"]] = states.get(row["state"], 0) + 1
    return {
        "schema": "metric-seam.math-a12-symbolic-train-verifier.v1",
        "status": "train_execution_complete",
        "task": "math",
        "criterion_id": "a12",
        "relation_id": RELATION_ID,
        "split": "compiler_train",
        "measurement_unit": "extracted adjacent equality pair",
        "verifier_class": "V_symbolic_sympy",
        "model_or_gpu_used": False,
        "heldout_accessed": False,
        "natural_document_count": len(rows),
        "natural_pair_count": len(pair_rows),
        "natural_state_counts": dict(sorted(states.items())),
        "gate": _gate_value(gate),
        "natural_pairs": pair_rows,
        "planted_probes": probe_rows,
        "document_pair_counts": document_counts,
        "claim_limits": [
            "The gate concerns relation-instance measurability, not whole-criterion fidelity.",
            "Exact nonidentity is not a document error without separately established claim scope.",
            "Synthetic probes test capability and never enter natural base-rate calculations.",
            "This is a retrospective adapter around an existing manually constructed capability.",
        ],
    }


def _validate_inputs(bundle: dict[str, Any], selection: dict[str, Any]) -> list[dict[str, Any]]:
    if bundle.get("schema") != "metric-seam.sanitized-ctext-train-compiler-view.v2":
        raise ValueError("unexpected compiler bundle schema")
    if bundle.get("task") != "math" or bundle.get("criterion_id") != "a12":
        raise ValueError("compiler bundle is not Math a12")
    if bundle.get("objective", {}).get("external_supervised_anchor") is not False:
        raise ValueError("external supervised anchors are forbidden")
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("unexpected verifier selection schema")
    if selection.get("relation_id") != RELATION_ID:
        raise ValueError("selection relation drift")
    rows = bundle.get("train_items")
    if not isinstance(rows, list):
        raise ValueError("compiler bundle has no TRAIN rows")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probe-cap", type=int, default=40)
    args = parser.parse_args(argv)
    if "heldout" in str(args.bundle).casefold() or "heldout" in str(args.output).casefold():
        raise ValueError("TRAIN runner refuses held-out paths")
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    rows = _validate_inputs(bundle, selection)
    result = run_train(rows, probe_cap=args.probe_cap)
    result["inputs"] = {
        "compiler_bundle": {"path": str(args.bundle), "sha256": _sha(args.bundle)},
        "selection": {"path": str(args.selection), "sha256": _sha(args.selection)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
