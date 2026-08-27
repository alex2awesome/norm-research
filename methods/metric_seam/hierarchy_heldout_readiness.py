"""Classify pre-reference code coverage before prompt reconstruction scoring."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Mapping, Sequence

from methods.metric_seam.hierarchy_code_runner import EXECUTION_SCHEMA, TRAIN_GATE_SCHEMA


SCHEMA = "metric-seam.hierarchy-code-heldout-readiness.v1"


class HeldoutReadinessError(ValueError):
    """Raised when heldout replay and its train gate do not match."""


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def build_heldout_readiness(execution: Mapping, train_gate: Mapping, *,
                            min_confirmatory_pairs: int = 30,
                            min_exploratory_pairs: int = 10,
                            execution_source: str | None = None,
                            gate_source: str | None = None) -> dict:
    if execution.get("schema") != EXECUTION_SCHEMA or execution.get("phase") != "heldout_pre_reference":
        raise HeldoutReadinessError("expected heldout_pre_reference execution")
    if train_gate.get("schema") != TRAIN_GATE_SCHEMA:
        raise HeldoutReadinessError("expected canonical compiler-train gate")
    if execution.get("reference_fields_passed_to_worker") is not False:
        raise HeldoutReadinessError("heldout execution passed reference fields")
    if execution.get("outcome_fields_passed_to_worker") is not False:
        raise HeldoutReadinessError("heldout execution passed outcome fields")
    if min_exploratory_pairs < 2 or min_confirmatory_pairs < min_exploratory_pairs:
        raise HeldoutReadinessError("invalid paired-score thresholds")

    selected = {row["aspect_id"]: row for row in train_gate.get("selected_programs", [])}
    programs = execution.get("programs")
    if not isinstance(programs, list) or {row.get("aspect_id") for row in programs} != set(selected):
        raise HeldoutReadinessError("heldout programs do not match frozen training selection")

    program_rows = []
    confirmatory = []
    relation_status_counts = Counter()
    relation_levels = Counter()
    relation_depths = Counter()
    for program in programs:
        aspect_id = program["aspect_id"]
        gate_row = selected[aspect_id]
        if (
            program.get("source_path") != gate_row["source_path"]
            or program.get("source_sha256") != gate_row["source_sha256"]
        ):
            raise HeldoutReadinessError(f"{aspect_id}: selected source identity drift")
        cell_ids = [relation["cell_id"] for relation in program.get("relations", [])]
        if cell_ids != gate_row["cell_ids"]:
            raise HeldoutReadinessError(f"{aspect_id}: relation mapping drift")
        summary = program.get("summary", {})
        n_scored = int(summary.get("n_scored", 0))
        n_unique = int(summary.get("n_unique_scores", 0))
        if program.get("worker_status") != "completed" or n_unique < 2:
            readiness = "not_evaluable"
        elif n_scored >= min_confirmatory_pairs:
            readiness = "confirmatory_reconstruction_evaluable"
        elif n_scored >= min_exploratory_pairs:
            readiness = "exploratory_sparse"
        else:
            readiness = "insufficient_paired_support"
        row = {
            "aspect_id": aspect_id,
            "source_path": program["source_path"],
            "source_sha256": program["source_sha256"],
            "cell_ids": cell_ids,
            "n_relation_mappings": len(cell_ids),
            "n_scored": n_scored,
            "coverage": summary.get("coverage"),
            "n_unique_scores": n_unique,
            "readiness": readiness,
        }
        program_rows.append(row)
        relation_status_counts[readiness] += len(cell_ids)
        if readiness == "confirmatory_reconstruction_evaluable":
            confirmatory.append({
                key: row[key] for key in (
                    "aspect_id", "source_path", "source_sha256", "cell_ids", "n_scored"
                )
            })
            for relation in program["relations"]:
                relation_levels[relation["level"]] += 1
                relation_depths[str(relation["audited_depth"])] += 1

    n_confirmatory_relations = relation_status_counts["confirmatory_reconstruction_evaluable"]
    n_static_relations = train_gate.get("summary", {}).get(
        "n_static_relation_mappings"
    )
    if not isinstance(n_static_relations, int) or n_static_relations <= 0:
        raise HeldoutReadinessError("train gate has no valid static-relation denominator")
    return {
        "schema": SCHEMA,
        "status": "frozen_before_prompt_reference_scoring",
        "heldout_execution_source": execution_source,
        "compiler_train_gate_source": gate_source,
        "thresholds": {
            "confirmatory_min_paired_scores": min_confirmatory_pairs,
            "exploratory_min_paired_scores": min_exploratory_pairs,
            "minimum_unique_scores": 2,
        },
        "reference_values_used": False,
        "outcome_labels_used": False,
        "prompt_outputs_used": False,
        "interpretation": (
            "Readiness means only that the frozen code vector has enough heldout support for a "
            "later prompt-reconstruction estimate. It is not itself reconstruction, verification "
            "of the whole construct, or isomorphism."
        ),
        "summary": {
            "n_train_selected_programs": len(selected),
            "n_confirmatory_programs": len(confirmatory),
            "n_confirmatory_relation_mappings": n_confirmatory_relations,
            "confirmatory_relation_fraction_of_all_90_metrics": _fraction(
                n_confirmatory_relations, 90
            ),
            "confirmatory_relation_fraction_of_static_fidelity_eligible": _fraction(
                n_confirmatory_relations, n_static_relations
            ),
            "relation_readiness_counts": dict(sorted(relation_status_counts.items())),
            "confirmatory_relation_mappings_by_level": dict(sorted(relation_levels.items())),
            "confirmatory_relation_mappings_by_depth": dict(sorted(relation_depths.items())),
        },
        "confirmatory_programs": confirmatory,
        "programs": sorted(program_rows, key=lambda row: row["aspect_id"]),
    }


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--train-gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.out}; pass --force")
    payload = build_heldout_readiness(
        _load(args.execution),
        _load(args.train_gate),
        execution_source=str(args.execution),
        gate_source=str(args.train_gate),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
