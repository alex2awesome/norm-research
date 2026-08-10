"""Freeze one train-measurable L-clamp profile per audited math program.

The gate reads only compiler-train conditional-slice outputs.  It selects the
first profile in the predeclared sentinel order that has sufficient measured
coverage, no item/contract failures, and at least two distinct scores.  It does
not inspect score direction, magnitude relative to a target, references,
outcomes, prompt values, or heldout items/outputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from methods.metric_seam.hierarchy_math_lclamp_runner import (
    EXECUTION_SCHEMA,
    PROFILE_GATE_SCHEMA,
    PROGRAMS_ROOT,
    SENTINELS,
    _capability_sources_from_audit,
    _content_fingerprint,
    _profile_summary,
    build_execution_plan,
    build_sentinel_profiles,
    validate_profiles,
)


_EXECUTION_TOP_FIELDS = {
    "schema", "status", "phase", "items_path", "n_items",
    "construct_fidelity_source", "construct_fidelity_fingerprint",
    "profile_selection_source", "scientific_object", "original_hybrid_execution",
    "pure_code_rewrite_claimed", "whole_construct_fidelity_claimed",
    "constant_profiles_frozen_within_each_run", "sentinel_grid", "sentinel_grid_rule",
    "reference_fields_passed_to_worker", "outcome_fields_passed_to_worker",
    "actual_llm_extractions_passed_to_worker", "models_or_apis_called_by_runner",
    "credentials_inherited_by_worker", "accelerators_visible_to_worker",
    "worker_process_isolated", "worker_filesystem_isolated", "worker_network_isolated",
    "candidate_trust_model", "capability_runtime", "ops_corpus_or_retrieval_state_loaded",
    "execution_provenance", "interpretation", "summary", "programs",
}
_PROGRAM_FIELDS = {
    "aspect_id", "selected_revision", "source_path", "worker_status",
    "measurement_status", "llm_field_names", "profiles", "summary",
    "program_sha256", "relations",
}
_PROFILE_RESULT_FIELDS = {
    "profile_id", "profile_index", "assignments",
    "constant_values_frozen_within_profile", "measurement_status", "rows", "summary",
}
_MEASUREMENT_ROW_FIELDS = {"item_key", "measurement_state", "status", "score"}


class MathLClampGateError(ValueError):
    """Raised when train execution cannot support a sealed profile gate."""


def _validate_profile_result(profile: Mapping, *, n_items: int) -> None:
    if set(profile) != _PROFILE_RESULT_FIELDS:
        raise MathLClampGateError("profile result contains forbidden or missing fields")
    if profile.get("constant_values_frozen_within_profile") is not True:
        raise MathLClampGateError("profile result did not freeze constant values")
    rows = profile.get("rows")
    if not isinstance(rows, list) or len(rows) != n_items:
        raise MathLClampGateError("profile result item count mismatch")
    item_keys = []
    for row in rows:
        if not isinstance(row, Mapping) or frozenset(row) not in {
            frozenset(_MEASUREMENT_ROW_FIELDS),
            frozenset(_MEASUREMENT_ROW_FIELDS | {"error_type"}),
        }:
            raise MathLClampGateError("invalid profile measurement row")
        item_key = row.get("item_key")
        if not isinstance(item_key, str) or not item_key:
            raise MathLClampGateError("invalid profile measurement item key")
        item_keys.append(item_key)
        state, status, score = (
            row.get("measurement_state"),
            row.get("status"),
            row.get("score"),
        )
        if state == "measured":
            if (
                status != "scored"
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or not 0.0 <= float(score) <= 1.0
            ):
                raise MathLClampGateError("invalid measured row")
        elif state == "abstained":
            if status != "abstained" or score is not None or "error_type" in row:
                raise MathLClampGateError("invalid abstained row")
        elif state == "failed":
            if status not in {"contract_error", "execution_error"} or score is not None or not isinstance(
                row.get("error_type"), str
            ):
                raise MathLClampGateError("invalid failed row")
        else:
            raise MathLClampGateError("invalid three-state measurement value")
    if len(set(item_keys)) != len(item_keys):
        raise MathLClampGateError("duplicate profile measurement item keys")
    expected_summary, expected_status = _profile_summary(rows)
    if profile.get("summary") != expected_summary or profile.get("measurement_status") != expected_status:
        raise MathLClampGateError("profile summary/status does not match its measurement rows")


def _validate_train_execution_shape(execution: Mapping, plans: Sequence[Mapping]) -> None:
    if set(execution) != _EXECUTION_TOP_FIELDS:
        raise MathLClampGateError(
            f"training execution fields differ: {sorted(set(execution) ^ _EXECUTION_TOP_FIELDS)}"
        )
    if execution.get("profile_selection_source") is not None:
        raise MathLClampGateError("compiler-train execution unexpectedly used a profile selection")
    if execution.get("sentinel_grid") != list(SENTINELS):
        raise MathLClampGateError("training execution sentinel grid drifted")
    n_items = execution.get("n_items")
    if isinstance(n_items, bool) or not isinstance(n_items, int) or n_items <= 0:
        raise MathLClampGateError("training execution has invalid item count")
    programs = execution.get("programs")
    if not isinstance(programs, list) or len(programs) != len(plans):
        raise MathLClampGateError("training execution/program plan count mismatch")
    plan_by_identity = {
        (
            plan["aspect_id"], plan["source_path"], plan["program_sha256"],
            plan["selected_revision"], tuple(plan["llm_field_names"]),
        ): plan
        for plan in plans
    }
    seen = set()
    for program in programs:
        if not isinstance(program, Mapping) or frozenset(program) not in {
            frozenset(_PROGRAM_FIELDS),
            frozenset(_PROGRAM_FIELDS | {"error_type"}),
        }:
            raise MathLClampGateError("invalid program execution record fields")
        identity = (
            program.get("aspect_id"), program.get("source_path"), program.get("program_sha256"),
            program.get("selected_revision"), tuple(program.get("llm_field_names", [])),
        )
        plan = plan_by_identity.get(identity)
        if plan is None or identity in seen:
            raise MathLClampGateError("training execution/program identities drifted")
        seen.add(identity)
        if program.get("relations") != plan["relations"]:
            raise MathLClampGateError("training execution relation mappings drifted")
        profiles = program.get("profiles")
        if program.get("worker_status") == "completed":
            if not isinstance(profiles, list):
                raise MathLClampGateError("completed program has invalid profiles")
            for profile in profiles:
                if not isinstance(profile, Mapping):
                    raise MathLClampGateError("invalid profile result")
                _validate_profile_result(profile, n_items=n_items)
        elif profiles != []:
            raise MathLClampGateError("failed worker emitted profile results")
    if seen != set(plan_by_identity):
        raise MathLClampGateError("training execution omitted a planned program")


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _profile_decision(
    profile: Mapping,
    *,
    min_measured: int,
    min_coverage: float,
    min_unique_scores: int,
    max_failed: int,
) -> tuple[bool, str, dict[str, Any]]:
    summary = profile.get("summary")
    if not isinstance(summary, Mapping):
        return False, "invalid_profile_summary", {
            "n_measured": 0,
            "coverage": 0.0,
            "n_unique_scores": 0,
            "n_failed": 0,
            "n_abstained": 0,
        }
    states = summary.get("state_counts")
    if not isinstance(states, Mapping) or set(states) != {"measured", "abstained", "failed"}:
        return False, "invalid_three_state_accounting", {
            "n_measured": 0,
            "coverage": 0.0,
            "n_unique_scores": 0,
            "n_failed": 0,
            "n_abstained": 0,
        }
    n_measured = summary.get("n_measured")
    coverage = summary.get("coverage")
    n_unique = summary.get("n_unique_scores")
    n_failed = states.get("failed")
    n_abstained = states.get("abstained")
    numeric_valid = (
        isinstance(n_measured, int)
        and not isinstance(n_measured, bool)
        and isinstance(coverage, (int, float))
        and not isinstance(coverage, bool)
        and isinstance(n_unique, int)
        and not isinstance(n_unique, bool)
        and isinstance(n_failed, int)
        and not isinstance(n_failed, bool)
        and isinstance(n_abstained, int)
        and not isinstance(n_abstained, bool)
    )
    measures = {
        "n_measured": n_measured if isinstance(n_measured, int) else 0,
        "coverage": coverage if isinstance(coverage, (int, float)) else 0.0,
        "n_unique_scores": n_unique if isinstance(n_unique, int) else 0,
        "n_failed": n_failed if isinstance(n_failed, int) else 0,
        "n_abstained": n_abstained if isinstance(n_abstained, int) else 0,
    }
    if not numeric_valid:
        return False, "invalid_profile_summary", measures
    if n_failed > max_failed:
        return False, "item_or_contract_failures", measures
    if n_measured < min_measured or coverage < min_coverage:
        return False, "insufficient_train_measurement", measures
    if n_unique < min_unique_scores:
        return False, "constant_train_measurement", measures
    return True, "eligible_train_measurement", measures


def build_train_profile_gate(
    execution: Mapping,
    audit: Mapping,
    *,
    min_measured: int = 10,
    min_coverage: float = 0.05,
    min_unique_scores: int = 2,
    max_failed: int = 0,
    execution_source: str | None = None,
    audit_source: str | None = None,
    program_root: Path = PROGRAMS_ROOT,
    require_canonical_programs: bool = True,
) -> dict[str, Any]:
    """Select one deterministic, train-measurable profile per program."""

    if execution.get("schema") != EXECUTION_SCHEMA or execution.get("phase") != "compiler_train":
        raise MathLClampGateError("profile gate requires compiler_train L-clamp execution")
    if execution.get("constant_profiles_frozen_within_each_run") is not True:
        raise MathLClampGateError("training execution did not freeze profile constants")
    if execution.get("original_hybrid_execution") is not False:
        raise MathLClampGateError("training artifact mislabels conditional slices")
    for field in (
        "reference_fields_passed_to_worker",
        "outcome_fields_passed_to_worker",
        "actual_llm_extractions_passed_to_worker",
        "models_or_apis_called_by_runner",
        "accelerators_visible_to_worker",
    ):
        if execution.get(field) is not False:
            raise MathLClampGateError(f"training execution violated {field}")
    for field in (
        "pure_code_rewrite_claimed",
        "whole_construct_fidelity_claimed",
        "credentials_inherited_by_worker",
        "ops_corpus_or_retrieval_state_loaded",
        "worker_filesystem_isolated",
        "worker_network_isolated",
    ):
        if execution.get(field) is not False:
            raise MathLClampGateError(f"training execution violated {field}")
    if execution.get("worker_process_isolated") is not True:
        raise MathLClampGateError("training execution was not process isolated")
    if (
        min_measured < 2
        or not 0 <= min_coverage <= 1
        or min_unique_scores < 2
        or max_failed < 0
    ):
        raise MathLClampGateError("invalid profile-gate thresholds")

    audit_fingerprint = _content_fingerprint(audit)
    if execution.get("construct_fidelity_fingerprint") != audit_fingerprint:
        raise MathLClampGateError("execution and construct audit fingerprints differ")
    plans = build_execution_plan(
        audit,
        program_root=program_root,
        require_canonical_programs=require_canonical_programs,
    )
    expected_capability_runtime = _capability_sources_from_audit(audit)
    if execution.get("capability_runtime") != expected_capability_runtime:
        raise MathLClampGateError("training execution capability runtime drifted")
    _validate_train_execution_shape(execution, plans)
    plan_by_identity = {
        (
            plan["aspect_id"],
            plan["source_path"],
            plan["program_sha256"],
            plan["selected_revision"],
            tuple(plan["llm_field_names"]),
        ): plan
        for plan in plans
    }
    programs = execution.get("programs")
    if not isinstance(programs, list) or len(programs) != len(plans):
        raise MathLClampGateError("training execution/program plan count mismatch")
    execution_by_identity = {}
    for program in programs:
        if not isinstance(program, Mapping):
            raise MathLClampGateError("invalid program execution record")
        identity = (
            program.get("aspect_id"),
            program.get("source_path"),
            program.get("program_sha256"),
            program.get("selected_revision"),
            tuple(program.get("llm_field_names", [])),
        )
        if identity in execution_by_identity:
            raise MathLClampGateError("duplicate program execution identity")
        execution_by_identity[identity] = program
    if set(execution_by_identity) != set(plan_by_identity):
        raise MathLClampGateError("training execution/program identities drifted")

    program_rows = []
    selected_program_profiles = []
    selected_relations = []
    all_profile_decisions = Counter()
    for identity, plan in sorted(plan_by_identity.items()):
        program = execution_by_identity[identity]
        expected_profiles = build_sentinel_profiles(plan["llm_field_names"])
        observed_profiles = program.get("profiles")
        if program.get("worker_status") == "completed":
            try:
                observed_identities = [
                    {
                        "profile_id": profile["profile_id"],
                        "profile_index": profile["profile_index"],
                        "assignments": profile["assignments"],
                    }
                    for profile in observed_profiles
                ]
                validate_profiles(
                    observed_identities,
                    plan["llm_field_names"],
                    require_complete_grid=True,
                )
            except Exception as exc:
                raise MathLClampGateError(
                    f"{plan['aspect_id']}: noncanonical train profile grid"
                ) from exc
            by_id = {profile.get("profile_id"): profile for profile in observed_profiles}
            if len(by_id) != len(expected_profiles):
                raise MathLClampGateError(f"{plan['aspect_id']}: duplicate/missing profiles")
        else:
            if observed_profiles != []:
                raise MathLClampGateError(
                    f"{plan['aspect_id']}: failed worker emitted profile results"
                )
            by_id = {}

        decisions = []
        first_eligible: dict[str, Any] | None = None
        for expected in expected_profiles:
            observed = by_id.get(expected["profile_id"])
            if observed is None:
                eligible = False
                decision = "worker_failure"
                measures = {
                    "n_measured": 0,
                    "coverage": 0.0,
                    "n_unique_scores": 0,
                    "n_failed": 0,
                    "n_abstained": 0,
                }
            else:
                eligible, decision, measures = _profile_decision(
                    observed,
                    min_measured=min_measured,
                    min_coverage=min_coverage,
                    min_unique_scores=min_unique_scores,
                    max_failed=max_failed,
                )
            all_profile_decisions[decision] += 1
            row = {
                "profile": expected,
                **measures,
                "eligible_by_train_measurability": eligible,
                "decision": decision,
            }
            decisions.append(row)
            if eligible and first_eligible is None:
                first_eligible = expected

        cell_ids = [relation["cell_id"] for relation in plan["relations"]]
        selected = first_eligible is not None
        program_row = {
            "aspect_id": plan["aspect_id"],
            "source_path": plan["source_path"],
            "program_sha256": plan["program_sha256"],
            "selected_revision": plan["selected_revision"],
            "llm_field_names": plan["llm_field_names"],
            "cell_ids": cell_ids,
            "n_relation_mappings": len(cell_ids),
            "n_grid_profiles": len(expected_profiles),
            "selected_for_heldout_pre_reference": selected,
            "selected_profile": first_eligible,
            "selection_rule": (
                "lowest canonical profile_index among profiles passing train-only "
                "coverage/nondegeneracy/failure thresholds"
            ),
            "profiles": decisions,
        }
        program_rows.append(program_row)
        if selected:
            selected_program_profiles.append(
                {
                    "aspect_id": plan["aspect_id"],
                    "source_path": plan["source_path"],
                    "program_sha256": plan["program_sha256"],
                    "selected_revision": plan["selected_revision"],
                    "llm_field_names": plan["llm_field_names"],
                    "cell_ids": cell_ids,
                    "profile": first_eligible,
                }
            )
            selected_relations.extend(plan["relations"])

    by_level = Counter(relation["level"] for relation in selected_relations)
    by_depth = Counter(str(relation["audited_depth"]) for relation in selected_relations)
    return {
        "schema": PROFILE_GATE_SCHEMA,
        "status": "frozen_before_heldout_profile_execution",
        "selection_basis": "compiler_train_profile_measurability_only",
        "training_execution_source": execution_source,
        "construct_fidelity_source": audit_source,
        "construct_fidelity_fingerprint": audit_fingerprint,
        "capability_runtime": expected_capability_runtime,
        "thresholds": {
            "min_measured": min_measured,
            "min_coverage": min_coverage,
            "min_unique_scores": min_unique_scores,
            "max_failed": max_failed,
            "profile_tie_break": "lowest fixed profile_index",
        },
        "reference_values_used": False,
        "outcome_labels_used": False,
        "heldout_items_or_outputs_used": False,
        "prompt_or_llm_values_used": False,
        "score_direction_or_target_used": False,
        "interpretation": (
            "Selection means only that one constant-field conditional slice had minimally usable "
            "train coverage and range. It does not validate the sentinel semantics, original hybrid, "
            "whole construct, prompt reconstruction, isomorphism, tacitness, or non-verifiability."
        ),
        "summary": {
            "n_candidate_programs": len(program_rows),
            "n_selected_programs": len(selected_program_profiles),
            "n_grid_profile_decisions": sum(row["n_grid_profiles"] for row in program_rows),
            "profile_decision_counts": dict(sorted(all_profile_decisions.items())),
            "n_static_relation_mappings": sum(len(plan["relations"]) for plan in plans),
            "n_selected_relation_mappings": len(selected_relations),
            "selected_relation_fraction_of_all_90_metrics": _fraction(
                len(selected_relations), 90
            ),
            "selected_relation_fraction_of_static_fidelity_eligible": _fraction(
                len(selected_relations), sum(len(plan["relations"]) for plan in plans)
            ),
            "selected_relation_mappings_by_level": dict(sorted(by_level.items())),
            "selected_relation_mappings_by_depth": dict(sorted(by_depth.items())),
        },
        "selected_program_profiles": selected_program_profiles,
        "programs": program_rows,
    }


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-measured", type=int, default=10)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--min-unique-scores", type=int, default=2)
    parser.add_argument("--max-failed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.out}; pass --force")
    payload = build_train_profile_gate(
        _load(args.execution),
        _load(args.audit),
        min_measured=args.min_measured,
        min_coverage=args.min_coverage,
        min_unique_scores=args.min_unique_scores,
        max_failed=args.max_failed,
        execution_source=str(args.execution),
        audit_source=str(args.audit),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
