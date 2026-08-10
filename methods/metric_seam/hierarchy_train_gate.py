"""Freeze a training-only operational gate for hierarchy code programs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping, Sequence

from methods.metric_seam.hierarchy_code_runner import (
    EXECUTION_SCHEMA,
    TRAIN_GATE_SCHEMA,
    build_execution_plan,
)


class TrainGateError(ValueError):
    """Raised when a training execution cannot support a frozen gate."""


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def build_train_gate(execution: Mapping, audit: Mapping, *, min_scored: int = 10,
                     min_coverage: float = 0.05, min_unique_scores: int = 2,
                     execution_source: str | None = None,
                     audit_source: str | None = None) -> dict:
    """Select programs using coverage/range on compiler-train only."""

    if execution.get("schema") != EXECUTION_SCHEMA or execution.get("phase") != "compiler_train":
        raise TrainGateError("gate requires a compiler_train hierarchy execution")
    if execution.get("reference_fields_passed_to_worker") is not False:
        raise TrainGateError("training execution passed reference fields")
    if execution.get("outcome_fields_passed_to_worker") is not False:
        raise TrainGateError("training execution passed outcome fields")
    if min_scored < 2 or not 0 <= min_coverage <= 1 or min_unique_scores < 2:
        raise TrainGateError("invalid operational thresholds")

    plans = build_execution_plan(audit)
    plan_by_id = {plan["aspect_id"]: plan for plan in plans}
    programs = execution.get("programs")
    if not isinstance(programs, list) or len(programs) != len(plans):
        raise TrainGateError("training execution/program plan count mismatch")
    execution_by_id = {program.get("aspect_id"): program for program in programs}
    if set(execution_by_id) != set(plan_by_id) or len(execution_by_id) != len(programs):
        raise TrainGateError("training execution/program identities drifted")

    program_rows = []
    selected_programs = []
    selected_relations = []
    for aspect_id, plan in sorted(plan_by_id.items()):
        program = execution_by_id[aspect_id]
        if program.get("source_path") != plan["source_path"]:
            raise TrainGateError(f"{aspect_id}: source path drift")
        summary = program.get("summary", {})
        status_counts = summary.get("status_counts", {})
        n_scored = summary.get("n_scored", 0)
        coverage = summary.get("coverage", 0.0)
        n_unique = summary.get("n_unique_scores", 0)
        item_failures = int(status_counts.get("execution_error", 0)) + int(
            status_counts.get("contract_error", 0)
        )
        selected = bool(
            program.get("worker_status") == "completed"
            and item_failures == 0
            and isinstance(n_scored, int) and n_scored >= min_scored
            and isinstance(coverage, (int, float)) and coverage >= min_coverage
            and isinstance(n_unique, int) and n_unique >= min_unique_scores
        )
        if program.get("worker_status") != "completed":
            reason = "worker_failure"
        elif item_failures:
            reason = "item_or_contract_failures"
        elif n_scored < min_scored or coverage < min_coverage:
            reason = "insufficient_train_coverage"
        elif n_unique < min_unique_scores:
            reason = "constant_train_measurement"
        else:
            reason = "selected"
        cell_ids = [relation["cell_id"] for relation in plan["relations"]]
        row = {
            "aspect_id": aspect_id,
            "source_path": plan["source_path"],
            "source_sha256": plan["source_sha256"],
            "cell_ids": cell_ids,
            "n_relation_mappings": len(cell_ids),
            "n_scored": n_scored,
            "coverage": coverage,
            "n_unique_scores": n_unique,
            "selected_for_heldout_pre_reference": selected,
            "decision": reason,
        }
        program_rows.append(row)
        if selected:
            selected_programs.append({
                key: row[key] for key in ("aspect_id", "source_path", "source_sha256", "cell_ids")
            })
            selected_relations.extend(plan["relations"])

    by_level = Counter(relation["level"] for relation in selected_relations)
    by_depth = Counter(str(relation["audited_depth"]) for relation in selected_relations)
    decisions = Counter(row["decision"] for row in program_rows)
    return {
        "schema": TRAIN_GATE_SCHEMA,
        "status": "frozen_before_heldout_program_execution",
        "selection_basis": "compiler_train_outputs_only",
        "training_execution_source": execution_source,
        "construct_fidelity_source": audit_source,
        "thresholds": {
            "min_scored": min_scored,
            "min_coverage": min_coverage,
            "min_unique_scores": min_unique_scores,
            "max_item_or_contract_failures": 0,
            "rationale": (
                "Ten scores and 5% coverage are a minimal operational replay gate. A later "
                "reconstruction estimate still requires at least 30 paired heldout scores."
            ),
        },
        "reference_values_used": False,
        "outcome_labels_used": False,
        "heldout_items_or_outputs_used": False,
        "interpretation": (
            "Selection means only that a relation-local scalar program had minimally usable "
            "training coverage and range. It is not whole-construct verification, prompt "
            "reconstruction, isomorphism, or evidence of tacitness for excluded programs."
        ),
        "summary": {
            "n_candidate_programs": len(program_rows),
            "n_selected_programs": len(selected_programs),
            "program_decision_counts": dict(sorted(decisions.items())),
            "n_static_relation_mappings": sum(len(plan["relations"]) for plan in plans),
            "n_selected_relation_mappings": len(selected_relations),
            "selected_relation_fraction_of_all_90_metrics": _fraction(len(selected_relations), 90),
            "selected_relation_fraction_of_static_fidelity_eligible": _fraction(
                len(selected_relations), sum(len(plan["relations"]) for plan in plans)
            ),
            "selected_relation_mappings_by_level": dict(sorted(by_level.items())),
            "selected_relation_mappings_by_depth": dict(sorted(by_depth.items())),
        },
        "selected_programs": selected_programs,
        "programs": program_rows,
    }


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-scored", type=int, default=10)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--min-unique-scores", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.out}; pass --force")
    payload = build_train_gate(
        _load(args.execution),
        _load(args.audit),
        min_scored=args.min_scored,
        min_coverage=args.min_coverage,
        min_unique_scores=args.min_unique_scores,
        execution_source=str(args.execution),
        audit_source=str(args.audit),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
