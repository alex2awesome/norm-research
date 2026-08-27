"""Run the isolated deterministic code-review verifiers on compiler TRAIN.

No model is called and no held-out path is accepted by this command.  The
candidate set is bound byte-for-byte to the frozen pilot selection manifest.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from methods.metric_seam.verifiers.code_review_ast import (
    ALL_AST_UNIT_SPECS,
    CONTROL_UNIT_SPECS,
    REAL_UNIT_SPECS,
    UnitSpec,
)
from methods.metric_seam.verifiers.code_review_ast_mutations import (
    mutation_templates_by_id,
)
from methods.metric_seam.verifiers.code_review_mutations import (
    MutationPair,
    build_train_violation_pair,
    validate_pair,
)
from methods.metric_seam.verifiers.schema import Verdict
from methods.metric_seam.verifiers.train_gate import (
    ProbeOutcome,
    TrainObservation,
    evaluate_train_discrimination,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / (
    "outputs/metric_seam_pilot/hierarchy_r123/items_v2/"
    "code-review/compiler_train.json"
)
DEFAULT_SELECTION = ROOT / (
    "outputs/metric_seam_pilot/hierarchy_r123/"
    "code_review_verifier_pilot_selection_v1.json"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/metric_seam_pilot/hierarchy_r123/results/"
    "code_review_ast_train_v1/readout.json"
)
EXPECTED_TRAIN_SHA256 = "0e6f68619dd1405b15ec99899edea50b539d2f12663fc7c0ff35abd2e5038167"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_train_binding(path: Path) -> None:
    """Bind execution to the frozen compiler-TRAIN bytes, not a filename hint."""

    if not path.is_file() or _sha256(path) != EXPECTED_TRAIN_SHA256:
        raise ValueError("V_ast TRAIN input does not match the frozen compiler_train bytes")


def validate_selection_binding(selection: dict[str, Any]) -> None:
    """Require exact real-unit and positive-control binding before execution."""

    if selection.get("schema") != "metric-seam.code-review-verifier-pilot-selection.v1":
        raise ValueError("unexpected pilot selection schema")
    selected = {row["pilot_unit_id"]: row for row in selection["real_units"]}
    expected_ids = {spec.unit_id for spec in REAL_UNIT_SPECS}
    if set(selected) != expected_ids:
        raise ValueError(
            f"V_ast real-unit ids do not match frozen selection: "
            f"expected={sorted(selected)}, implemented={sorted(expected_ids)}"
        )
    for spec in REAL_UNIT_SPECS:
        row = selected[spec.unit_id]
        if row["candidate_aspect_id"] != spec.aspect_id:
            raise ValueError(f"aspect mismatch for {spec.unit_id}")
        if row["relation"] != spec.source_cuf_span:
            raise ValueError(f"CUF relation drift for {spec.unit_id}")
        if int(row["cuf"]["node_id"]) != spec.source_cuf_node_id:
            raise ValueError(f"CUF node mismatch for {spec.unit_id}")
        if row["cuf"]["level"] != 1 or row["cuf"]["verdict"] != "CERTIFIED-UNIT":
            raise ValueError(f"CUF selection status drift for {spec.unit_id}")

    controls = {row["control_id"]: row for row in selection["controls"]}
    expected_controls = {spec.unit_id for spec in CONTROL_UNIT_SPECS}
    if not expected_controls <= set(controls):
        raise ValueError("frozen pilot selection is missing a V_ast positive control")
    for spec in CONTROL_UNIT_SPECS:
        if controls[spec.unit_id]["relation"] != spec.source_cuf_span:
            raise ValueError(f"control relation drift for {spec.unit_id}")
        if controls[spec.unit_id]["expected"] != "certify":
            raise ValueError(f"control expectation drift for {spec.unit_id}")


def _fraction(value: Fraction | None) -> dict[str, int] | None:
    if value is None:
        return None
    return {"numerator": value.numerator, "denominator": value.denominator}


def _verdict(value: Verdict | None) -> dict[str, Any] | None:
    return value.to_json_value() if value is not None else None


def _safe_verify(spec: UnitSpec, diff_text: str) -> tuple[Verdict | None, str | None]:
    try:
        return spec.verifier(diff_text), None
    except Exception as exc:  # execution failures belong in the TRAIN result
        return None, f"{type(exc).__name__}: {exc}"


def _probe_outcome(natural: Verdict | None, planted: Verdict | None) -> ProbeOutcome:
    if natural is None or planted is None:
        return ProbeOutcome.INCORRECT
    if not natural.violated and planted.violated:
        return ProbeOutcome.CORRECT
    if natural.violated and not planted.violated:
        return ProbeOutcome.INVERTED
    return ProbeOutcome.INCORRECT


def _build_pairs(
    spec: UnitSpec,
    natural_rows: Sequence[dict[str, Any]],
    natural_verdicts: dict[str, Verdict | None],
    *,
    plant_count: int,
) -> tuple[MutationPair, ...]:
    template = mutation_templates_by_id()[spec.unit_id]
    pairs: list[MutationPair] = []
    # A mutation probe requires a negative counterpart.  Selection depends
    # only on the frozen V_ast natural verdict, never on the mutant outcome.
    for row in natural_rows:
        natural = natural_verdicts[row["item_key"]]
        if natural is None or natural.violated:
            continue
        pair = build_train_violation_pair(
            row["ctext"],
            item_key=row["item_key"],
            unit_id=spec.unit_id,
            mutation_kind=template.mutation_kind,
            source_lines=template.source_lines,
            extension=template.extension,
        )
        validate_pair(pair)
        pairs.append(pair)
        if len(pairs) >= plant_count:
            break
    return tuple(pairs)


def run_train(
    natural_rows: Sequence[dict[str, Any]],
    *,
    plant_count: int = 40,
) -> dict[str, Any]:
    if plant_count < 1:
        raise ValueError("plant_count must be positive")
    item_keys = [row.get("item_key") for row in natural_rows]
    if not natural_rows or any(not isinstance(key, str) or not key for key in item_keys):
        raise ValueError("compiler_train must contain item_key/ctext rows")
    if len(set(item_keys)) != len(item_keys):
        raise ValueError("compiler_train item keys must be unique")
    if any(
        not isinstance(row.get("ctext"), str)
        or not row["ctext"].startswith("diff --git ")
        for row in natural_rows
    ):
        raise ValueError("compiler_train contains a non-diff ctext row")

    unit_results: list[dict[str, Any]] = []
    for spec in ALL_AST_UNIT_SPECS:
        natural_values: dict[str, Verdict | None] = {}
        natural_errors: dict[str, str] = {}
        observations: list[TrainObservation] = []
        natural_output: list[dict[str, Any]] = []
        for row in natural_rows:
            item_key = row["item_key"]
            value, error = _safe_verify(spec, row["ctext"])
            natural_values[item_key] = value
            if error is not None:
                natural_errors[item_key] = error
            observations.append(
                TrainObservation(
                    item_id=f"natural:{item_key}",
                    pattern_id="natural",
                    corpus_kind="natural",
                    verdict=value,
                )
            )
            natural_output.append(
                {"item_key": item_key, "verdict": _verdict(value), "error": error}
            )

        pairs = _build_pairs(
            spec, natural_rows, natural_values, plant_count=plant_count
        )
        probes: list[ProbeOutcome] = []
        planted_output: list[dict[str, Any]] = []
        for index, pair in enumerate(pairs):
            manifest = pair.manifest
            # The original digest uniquely recovers the selected natural row.
            base_row = next(
                row
                for row in natural_rows
                if hashlib.sha256(row["ctext"].encode("utf-8")).hexdigest()
                == manifest.original_sha256
            )
            item_key = base_row["item_key"]
            planted, error = _safe_verify(spec, pair.planted_violated)
            probes.append(_probe_outcome(natural_values[item_key], planted))
            observations.append(
                TrainObservation(
                    item_id=f"planted:{item_key}:{index}",
                    pattern_id=manifest.mutation_kind,
                    corpus_kind="planted",
                    verdict=planted,
                )
            )
            planted_output.append(
                {
                    "base_item_key": item_key,
                    "manifest": manifest.to_dict(),
                    "verdict": _verdict(planted),
                    "error": error,
                    "probe_outcome": probes[-1].value,
                }
            )

        gate = evaluate_train_discrimination(observations, probes)
        unit_result = {
                "unit_id": spec.unit_id,
                "unit_kind": (
                    "real_cuf_candidate" if spec.aspect_id is not None else "positive_control"
                ),
                "aspect_id": spec.aspect_id,
                "source_cuf_node_id": spec.source_cuf_node_id,
                "source_cuf_relation": spec.source_cuf_span,
                "implemented_relation": spec.implemented_relation,
                "relation_scope": spec.relation_scope,
                "natural": natural_output,
                "planted": planted_output,
                "gate": {
                    "passed": gate.passed,
                    "failure_reason": (
                        gate.failure_reason.value if gate.failure_reason is not None else None
                    ),
                    "failed_checks": list(gate.failed_checks),
                    "total_items": gate.total_items,
                    "completed_items": gate.completed_items,
                    "applies_count": gate.applies_count,
                    "violated_count": gate.violated_count,
                    "distinct_patterns": gate.distinct_patterns,
                    "corpus_kinds": sorted(gate.corpus_kinds),
                    "modal_count": gate.modal_count,
                    "probe_total": gate.probe_total,
                    "probe_correct": gate.probe_correct,
                    "probe_inversions": gate.probe_inversions,
                    "all_total_items": gate.all_total_items,
                    "all_completed_items": gate.all_completed_items,
                    "all_applies_count": gate.all_applies_count,
                    "all_violated_count": gate.all_violated_count,
                    "all_modal_count": gate.all_modal_count,
                    "completeness": _fraction(gate.completeness),
                    "applies_rate": _fraction(gate.applies_rate),
                    "violated_given_applies": _fraction(
                        gate.violated_given_applies
                    ),
                    "mode_rate": _fraction(gate.mode_rate),
                    "probe_correct_rate": _fraction(gate.probe_correct_rate),
                },
            }
        if spec.aspect_id is None:
            probe_rate = gate.probe_correct_rate
            unit_result["capability_diagnostic"] = {
                "planted_probe_separation_passed": bool(
                    probe_rate is not None
                    and probe_rate >= Fraction(75, 100)
                    and gate.probe_inversions == 0
                ),
                "probe_total": gate.probe_total,
                "probe_correct": gate.probe_correct,
                "probe_inversions": gate.probe_inversions,
                "probe_correct_rate": _fraction(probe_rate),
                "does_not_override_natural_gate": True,
            }
        unit_results.append(unit_result)
    real_results = [row for row in unit_results if row["unit_kind"] == "real_cuf_candidate"]
    control_results = [row for row in unit_results if row["unit_kind"] == "positive_control"]
    return {
        "schema": "metric-seam.code-review-ast-train.v1",
        "status": "train_execution_complete",
        "split": "compiler_train",
        "verifier_class": "V_ast",
        "model_or_gpu_used": False,
        "heldout_accessed": False,
        "real_unit_count": len(REAL_UNIT_SPECS),
        "positive_control_count": len(CONTROL_UNIT_SPECS),
        "natural_item_count": len(natural_rows),
        "requested_plant_count_per_unit": plant_count,
        "real_units": real_results,
        "positive_controls": control_results,
        "summary": {
            "real_train_gate_passed": sum(row["gate"]["passed"] for row in real_results),
            "real_train_gate_total": len(real_results),
            "control_natural_gate_passed": sum(
                row["gate"]["passed"] for row in control_results
            ),
            "control_natural_gate_total": len(control_results),
            "control_probe_separation_passed": sum(
                row["capability_diagnostic"]["planted_probe_separation_passed"]
                for row in control_results
            ),
            "control_probe_separation_total": len(control_results),
        },
        "claim_limits": [
            "Each implementation is a named structural sub-relation of its frozen CUF relation.",
            "Passing TRAIN discrimination is not a held-out certificate.",
            "Failure is bounded corpus/implementation failure, not tacitness.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plant-count", type=int, default=40)
    args = parser.parse_args()

    validate_train_binding(args.train)
    if "heldout" in str(args.selection).casefold():
        raise ValueError("V_ast TRAIN runner refuses held-out paths")
    selection = json.loads(args.selection.read_text())
    validate_selection_binding(selection)
    natural_rows = json.loads(args.train.read_text())
    result = run_train(natural_rows, plant_count=args.plant_count)
    result["inputs"] = {
        "compiler_train_path": str(args.train.relative_to(ROOT)),
        "compiler_train_sha256": _sha256(args.train),
        "selection_path": str(args.selection.relative_to(ROOT)),
        "selection_sha256": _sha256(args.selection),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
