#!/usr/bin/env python3
"""Freeze development-only authorization for progressive CE early exits.

Each supplied trial is a complete two-seed consensus over the same untouched
development norms.  Candidate trials are evaluated in a predeclared order;
their score and margin gates remain exactly those stored in the two completed
training reports.  An early exit is authorized only when both its exact-truth
error and its disagreement with the terminal complete-bank decision have a
simultaneous one-sided upper bound below the requested target.

No test/blind input option exists.  Failure merely disables early stopping at
that trial; every production norm continues to a later tier and ultimately to
the complete bank.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .aggregate_nemotron_ce_seed_consensus import (
    CONSENSUS_REPORT_SCHEMA,
    CONSENSUS_SCHEMA,
)
from .audit_false_abstentions import clopper_pearson_upper
from .common import normalize_space, read_jsonl, sha256_file


SCHEMA = "silver-match-v3-progressive-ce-dev-stop-policy-v1"
STATUS = "COMPLETE_DEV_FROZEN_PROGRESSIVE_STOP_POLICY"
TRUTH_DECISIONS = frozenset(
    {
        "MATCH",
        "MATCH_FAMILY_ONLY",
        "GENERIC_VERDICT",
        "NO_EXPLICIT_CRITERION",
        "NO_CANDIDATE_FITS",
        "CONTEXT_NEEDED",
        "NOISE",
    }
)


def _artifact(path: Path, *, count: int | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if count is not None:
        value["count"] = count
    return value


def _parse_trials(values: Sequence[str]) -> list[tuple[str, Path, Path]]:
    output = []
    seen: set[str] = set()
    for value in values:
        pieces = value.split("=", 2)
        if len(pieces) != 3:
            raise ValueError("--trial must be TRIAL_ID=CONSENSUS_JSONL=REPORT_JSON")
        trial_id = normalize_space(pieces[0])
        if not trial_id or trial_id in seen:
            raise ValueError(f"invalid/duplicate trial ID: {trial_id!r}")
        seen.add(trial_id)
        output.append((trial_id, Path(pieces[1]).resolve(), Path(pieces[2]).resolve()))
    if len(output) < 2:
        raise ValueError("at least one early trial and one terminal trial are required")
    return output


def _load_truth(path: Path, *, task: str) -> dict[str, dict[str, Any]]:
    truth: dict[str, dict[str, Any]] = {}
    groups: set[str] = set()
    for line_no, row in enumerate(read_jsonl(path), 1):
        uid = normalize_space(row.get("norm_uid"))
        decision = normalize_space(row.get("decision")).upper()
        split = normalize_space(row.get("split")).lower()
        role = normalize_space(row.get("collection_role") or row.get("selection_role")).lower()
        group = normalize_space(row.get("source_group") or row.get("split_group"))
        if (
            not uid
            or uid in truth
            or row.get("task") != task
            or decision not in TRUTH_DECISIONS
            or split != "dev"
            or role != "dev"
            or not group
            or row.get("training_eligible") is True
            or row.get("blind_evaluation_only") is True
        ):
            raise ValueError(f"truth is not unique untouched development data: {path}:{line_no}")
        metric_id = normalize_space(row.get("metric_id"))
        acceptable = row.get("acceptable_metric_ids") or row.get("equivalent_metric_ids") or []
        if isinstance(acceptable, str):
            acceptable = [acceptable]
        accepted_ids = {normalize_space(value) for value in acceptable if normalize_space(value)}
        if decision == "MATCH":
            if not metric_id:
                raise ValueError(f"development MATCH lacks metric_id: {uid}")
            accepted_ids.add(metric_id)
        elif metric_id:
            raise ValueError(f"development non-MATCH carries metric_id: {uid}")
        truth[uid] = {"decision": decision, "metric_ids": accepted_ids, "source_group": group}
        groups.add(group)
    if not truth:
        raise ValueError("development truth is empty")
    return truth


def _load_consensus(
    path: Path,
    report_path: Path,
    *,
    task: str,
    expected_uids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validation = report.get("validation") or {}
    if (
        report.get("schema_version") != CONSENSUS_REPORT_SCHEMA
        or report.get("status") != "COMPLETE"
        or report.get("output_sha256") != sha256_file(path)
        or int(report.get("norm_count", -1)) != len(expected_uids)
        or validation.get("all_thresholds_from_checkpoint_dev") is not True
        or validation.get("test_threshold_tuning_performed") is not False
        or validation.get("all_norms_preserved") is not True
        or validation.get("seed_norm_candidate_source_split_universes_identical") is not True
    ):
        raise ValueError(f"trial is not a complete dev-gated two-seed consensus: {path}")
    rows: dict[str, dict[str, Any]] = {}
    for line_no, row in enumerate(read_jsonl(path), 1):
        uid = normalize_space(row.get("norm_uid"))
        states = row.get("seed_decisions") or {}
        candidates = row.get("candidates") or []
        if (
            row.get("schema_version") != CONSENSUS_SCHEMA
            or row.get("task") != task
            or normalize_space(row.get("split")).lower() != "dev"
            or uid not in expected_uids
            or uid in rows
            or len(states) != 2
            or int(row.get("candidate_count", -1)) != len(candidates)
            or len({normalize_space(value.get("metric_id")) for value in candidates})
            != len(candidates)
        ):
            raise ValueError(f"invalid trial consensus row: {path}:{line_no}")
        automatic = row.get("automatic_match") is True
        if automatic:
            metric_id = normalize_space(row.get("metric_id"))
            if (
                row.get("decision") != "MATCH"
                or row.get("routing_category") != "MATCH"
                or not metric_id
                or {normalize_space(state.get("top_metric_id")) for state in states.values()}
                != {metric_id}
                or any(state.get("passes_frozen_gate") is not True for state in states.values())
            ):
                raise ValueError(f"trial automatic match is not same-leaf/two-gate: {uid}")
        rows[uid] = row
    if set(rows) != expected_uids:
        raise ValueError(f"trial consensus UID universe differs: {path}")
    return rows, report


def _error_upper(errors: int, total: int, alpha: float) -> float:
    return 1.0 if total == 0 else clopper_pearson_upper(errors, total, alpha=alpha)


def freeze_policy(
    *,
    task: str,
    truth_path: Path,
    trials: Sequence[tuple[str, Path, Path]],
    output_path: Path,
    target_error_upper: float = 0.05,
    family_alpha: float = 0.05,
) -> dict[str, Any]:
    task = normalize_space(task)
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    if not 0 < target_error_upper < 1 or not 0 < family_alpha < 1:
        raise ValueError("error target and alpha must lie in (0,1)")
    truth_path = truth_path.resolve()
    truth = _load_truth(truth_path, task=task)
    expected = set(truth)
    loaded = []
    for trial_id, consensus_path, report_path in trials:
        rows, report = _load_consensus(
            consensus_path,
            report_path,
            task=task,
            expected_uids=expected,
        )
        loaded.append((trial_id, consensus_path, report_path, rows, report))
        if any(
            normalize_space(rows[uid].get("source_group"))
            != truth[uid]["source_group"]
            for uid in expected
        ):
            raise ValueError(f"trial consensus source groups differ from dev truth: {trial_id}")
    seed_bindings = []
    for _, _, _, _, report in loaded:
        seed_bindings.append(
            sorted(
                (
                    normalize_space(seed.get("seed_id")),
                    normalize_space(seed.get("training_report_sha256")),
                    normalize_space(seed.get("checkpoint")),
                )
                for seed in report.get("seeds") or []
            )
        )
    if not seed_bindings or any(value != seed_bindings[0] for value in seed_bindings[1:]):
        raise ValueError("progressive development trials use different seed checkpoints")
    terminal_id, _, _, terminal_rows, terminal_report = loaded[-1]
    terminal_candidate_counts = {int(row.get("candidate_count", -1)) for row in terminal_rows.values()}
    if len(terminal_candidate_counts) != 1 or min(terminal_candidate_counts) < 1:
        raise ValueError("terminal development consensus is not a fixed complete-bank universe")
    terminal_depth = next(iter(terminal_candidate_counts))
    # Every candidate universe must be nested.  This is checked on development
    # before any policy can authorize an early production exit.
    prior_sets = {uid: set() for uid in expected}
    for trial_id, _, _, rows, _ in loaded:
        for uid in expected:
            ids = {normalize_space(value.get("metric_id")) for value in rows[uid].get("candidates") or []}
            if not prior_sets[uid] <= ids:
                raise ValueError(f"progressive candidate universe is not nested: {trial_id}/{uid}")
            prior_sets[uid] = ids
    if any(len(values) != terminal_depth for values in prior_sets.values()):
        raise ValueError("terminal development trial candidate depth is inconsistent")

    early_count = len(loaded) - 1
    # Two simultaneous claims (truth precision and terminal stability) are made
    # for every early trial.
    simultaneous_alpha = family_alpha / max(1, 2 * early_count)
    active = set(expected)
    authorized: list[str] = []
    reports = []
    for ordinal, (trial_id, consensus_path, report_path, rows, report) in enumerate(loaded, 1):
        terminal = ordinal == len(loaded)
        proposed = {
            uid: normalize_space(rows[uid].get("metric_id"))
            for uid in active
            if rows[uid].get("automatic_match") is True
        }
        truth_errors = sum(
            metric_id not in truth[uid]["metric_ids"]
            for uid, metric_id in proposed.items()
        )
        terminal_instability = sum(
            terminal_rows[uid].get("automatic_match") is not True
            or normalize_space(terminal_rows[uid].get("metric_id")) != metric_id
            for uid, metric_id in proposed.items()
        )
        truth_upper = _error_upper(truth_errors, len(proposed), simultaneous_alpha)
        stability_upper = _error_upper(
            terminal_instability, len(proposed), simultaneous_alpha
        )
        threshold_sources = {
            (
                seed.get("training_report_sha256"),
                (seed.get("frozen_gate") or {}).get("provenance"),
            )
            for seed in report.get("seeds") or []
        }
        if len(threshold_sources) != 2 or any(source[1] != "checkpoint.dev" for source in threshold_sources):
            raise ValueError(f"trial thresholds are not two independent checkpoint.dev gates: {trial_id}")
        passed = terminal or (
            bool(proposed)
            and truth_upper < target_error_upper
            and stability_upper < target_error_upper
        )
        if passed and not terminal:
            authorized.append(trial_id)
            active.difference_update(proposed)
        reports.append(
            {
                "trial_id": trial_id,
                "ordinal": ordinal,
                "terminal": terminal,
                "consensus": _artifact(consensus_path, count=len(rows)),
                "consensus_report": _artifact(report_path),
                "active_norm_count_before_trial": len(active) + (len(proposed) if passed and not terminal else 0),
                "proposed_exit_count": len(proposed),
                "proposed_exit_rate_among_active": len(proposed) / (len(active) + (len(proposed) if passed and not terminal else 0))
                if (len(active) + (len(proposed) if passed and not terminal else 0))
                else 0.0,
                "exact_truth_error_count": truth_errors,
                "exact_truth_error_upper_simultaneous": truth_upper,
                "terminal_decision_instability_count": terminal_instability,
                "terminal_instability_upper_simultaneous": stability_upper,
                "authorized_for_early_stop": passed and not terminal,
                "all_other_norms_continue": True,
                "threshold_provenance": "checkpoint.dev",
            }
        )
    # Estimate production compute under covariate stability only; the queue's
    # hard contract remains the exhaustive worst case.
    survival = 1.0
    expected_candidates_per_norm = 0.0
    for row in reports:
        mean_candidates = sum(
            int(value.get("candidate_count", 0)) for value in loaded[row["ordinal"] - 1][3].values()
        ) / len(expected)
        # Consensus inputs are cumulative; incremental work is their difference.
        previous = reports[row["ordinal"] - 2].get("cumulative_mean_candidates", 0.0) if row["ordinal"] > 1 else 0.0
        increment = max(0.0, mean_candidates - previous)
        expected_candidates_per_norm += survival * increment
        row["cumulative_mean_candidates"] = mean_candidates
        row["incremental_mean_candidates"] = increment
        if row["authorized_for_early_stop"]:
            rate = row["proposed_exit_rate_among_active"]
            survival *= max(0.0, 1.0 - rate)
    estimated_reduction = 1.0 - expected_candidates_per_norm / terminal_depth
    payload = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "task": task,
        "selection_split": "dev",
        "truth": _artifact(truth_path, count=len(truth)),
        "trial_order": [row[0] for row in loaded],
        "terminal_trial_id": terminal_id,
        "terminal_complete_bank_depth": terminal_depth,
        "authorized_early_stop_trials": authorized,
        "trial_audits": reports,
        "gate": {
            "target_error_upper": target_error_upper,
            "family_alpha": family_alpha,
            "multiple_claim_correction": "bonferroni_over_two_claims_per_predeclared_early_trial",
            "simultaneous_alpha_per_claim": simultaneous_alpha,
            "automatic_exit_rule": "both_checkpoint_dev_gates_pass_and_same_exact_metric",
            "threshold_search_performed": False,
        },
        "estimated_compute": {
            "basis": "untouched_dev_exit_rates_applied_to_production; estimate_not_a_coverage_assumption",
            "expected_single_seed_candidates_per_norm": expected_candidates_per_norm,
            "complete_bank_candidates_per_norm": terminal_depth,
            "estimated_pair_evaluation_reduction_rate": estimated_reduction,
            "worst_case_reduction_rate": 0.0,
            "terminal_complete_bank_rescue_mandatory_for_every_survivor": True,
        },
        "safety": {
            "test_or_blind_labels_read": False,
            "training_labels_used_for_stop_selection": False,
            "all_thresholds_from_training_reports_checkpoint_dev": True,
            "unauthorized_trial_exits_permitted": False,
            "disagreement_or_abstention_continues": True,
            "complete_bank_terminal_coverage": True,
        },
        "release_ready": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--trial", action="append", required=True)
    parser.add_argument("--target-error-upper", type=float, default=0.05)
    parser.add_argument("--family-alpha", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = freeze_policy(
        task=args.task,
        truth_path=Path(args.truth),
        trials=_parse_trials(args.trial),
        output_path=Path(args.output),
        target_error_upper=args.target_error_upper,
        family_alpha=args.family_alpha,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "authorized_early_stop_trials": result["authorized_early_stop_trials"],
                "estimated_pair_evaluation_reduction_rate": result["estimated_compute"]["estimated_pair_evaluation_reduction_rate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
