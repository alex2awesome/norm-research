#!/usr/bin/env python3
"""Freeze a completed task matcher and prepare the existing MI analysis handoff.

This is the fail-closed bridge from the learned post-inference stack to a
task-level analysis release.  It does not decide labels itself.  Instead it
joins the canonical final rows to the two-seed CE consensus, the selected
typed Gemma output, and the strict repeated-full-bank rescue output, then
checks that every promoted match has one of the two allowed evidence paths:

* both CE seeds passed their frozen development gates and selected the same
  leaf; or
* strict repeated-full-bank rescue independently verified the leaf.

Every non-CE row must be present in the typed Gemma output and must have gone
through repeated full-bank rescue.  The release is written only after the
independent production match/abstention risk release is recomputed as PASS.
The generated MI handoff invokes ``silver_mi_validation_v3`` against an
existing, hash-bound MI certificate; no scores or correlations are invented.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .adjudicate_gemma import DECISIONS
from .aggregate_nemotron_ce_seed_consensus import (
    CONSENSUS_REPORT_SCHEMA,
    CONSENSUS_SCHEMA,
)
from .audit_final_outputs import DECISION_ORDER, audit_outputs
from .common import normalize_space, read_jsonl, sha256_file
from .freeze_task_final_risk_release import verify_task_final_risk_release
from .silver_mi_validation_v3 import _load_certificate


SCHEMA = "silver-match-v3-postinference-analysis-release-v1"
HANDOFF_SCHEMA = "silver-match-v3-mi-validation-handoff-v1"
TYPED_ABSTENTIONS = tuple(
    decision
    for decision in DECISION_ORDER
    if decision not in {"MATCH", "UNSTABLE_MATCH", "INVALID_OUTPUT"}
)


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _resolve(raw: str | Path, anchor: Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (anchor.parent / path).resolve()


def _load_task(
    manifest_path: Path, task: str
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]], set[str], Path]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if task not in (manifest.get("banks") or {}):
        raise ValueError(f"manifest has no task {task!r}")
    # JSON object insertion order is the frozen manifest order.  Preserve it
    # in task releases and downstream denominators rather than re-sorting it.
    corpora = [
        corpus
        for corpus, meta in (manifest.get("corpora") or {}).items()
        if str(meta.get("task") or "") == task
    ]
    if not corpora:
        raise ValueError(f"manifest has no corpora for task {task!r}")
    canonical: list[dict[str, Any]] = []
    seen: set[str] = set()
    for corpus in corpora:
        meta = manifest["corpora"][corpus]
        norm_path = _resolve(str(meta["path"]), manifest_path)
        rows = list(read_jsonl(norm_path))
        if len(rows) != int(meta["count"]):
            raise ValueError(f"canonical count mismatch for {corpus}")
        for row in rows:
            uid = normalize_space(row.get("norm_uid"))
            if (
                not uid
                or uid in seen
                or str(row.get("task") or "") != task
                or str(row.get("corpus") or "") != corpus
            ):
                raise ValueError(f"invalid/duplicate/routing-drift canonical row: {uid!r}")
            seen.add(uid)
            canonical.append(row)
    bank_meta = manifest["banks"][task]
    bank_path = _resolve(str(bank_meta["path"]), manifest_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    metric_rows = list(bank.get("metrics") or [])
    metric_ids = {str(row.get("metric_id") or "") for row in metric_rows}
    if not metric_rows or "" in metric_ids or len(metric_ids) != len(metric_rows):
        raise ValueError("task bank has missing/duplicate metric IDs")
    if bank.get("source_sha256") not in (None, bank_meta["source_sha256"]):
        raise ValueError("bank payload/manifest source hash mismatch")
    if bank_meta.get("count") is not None and int(bank_meta["count"]) != len(metric_ids):
        raise ValueError("manifest bank count differs from bank payload")
    return manifest, corpora, canonical, metric_ids, bank_path


def _index_exact(
    path: Path,
    *,
    label: str,
    canonical: Sequence[Mapping[str, Any]],
    task: str,
) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    indexed: dict[str, dict[str, Any]] = {}
    expected_routing = {
        normalize_space(row["norm_uid"]): str(row["corpus"]) for row in canonical
    }
    for row in rows:
        uid = normalize_space(row.get("norm_uid"))
        if not uid or uid in indexed:
            raise ValueError(f"{label} has missing/duplicate norm UID: {uid!r}")
        if uid not in expected_routing:
            raise ValueError(f"{label} has foreign norm UID: {uid}")
        if (
            str(row.get("task") or "") != task
            or str(row.get("corpus") or "") != expected_routing[uid]
        ):
            raise ValueError(f"{label} routing mismatch for {uid}")
        indexed[uid] = row
    expected = set(expected_routing)
    if set(indexed) != expected:
        raise ValueError(
            f"{label} coverage mismatch: missing={len(expected-set(indexed))}, "
            f"extra={len(set(indexed)-expected)}"
        )
    return indexed


def _validate_ce(
    ce_path: Path,
    ce_report_path: Path,
    *,
    canonical: Sequence[Mapping[str, Any]],
    task: str,
    metric_ids: set[str],
) -> dict[str, dict[str, Any]]:
    report = json.loads(ce_report_path.read_text(encoding="utf-8"))
    progressive = report.get("progressive_queue") is True
    progressive_validation = report.get("validation") or {}
    if (
        report.get("schema_version") != CONSENSUS_REPORT_SCHEMA
        or report.get("status") != "COMPLETE"
        or report.get("output_sha256") != sha256_file(ce_path)
        or int(report.get("norm_count", -1)) != len(canonical)
        or (report.get("validation") or {}).get("all_norms_preserved") is not True
        or (report.get("validation") or {}).get("all_thresholds_from_checkpoint_dev")
        is not True
        or (report.get("validation") or {}).get(
            "seed_norm_candidate_source_split_universes_identical"
        )
        is not True
        or (report.get("validation") or {}).get("test_threshold_tuning_performed")
        is not False
        or (
            progressive
            and (
                progressive_validation.get("one_terminal_ce_decision_per_norm")
                is not True
                or progressive_validation.get(
                    "all_early_exits_dev_policy_authorized"
                )
                is not True
                or progressive_validation.get(
                    "every_survivor_reached_complete_bank"
                )
                is not True
                or not isinstance(report.get("progressive_pairs_manifest"), Mapping)
                or not isinstance(report.get("dev_stop_policy"), Mapping)
            )
        )
    ):
        raise ValueError("CE consensus report is incomplete, unbound, or not dev-gated")
    if progressive:
        for name in ("progressive_pairs_manifest", "dev_stop_policy"):
            ref = report[name]
            ref_path = _resolve(str(ref.get("path") or ""), ce_report_path)
            if not ref_path.is_file() or sha256_file(ref_path) != ref.get("sha256"):
                raise ValueError(f"progressive CE {name} artifact changed")
    rows = _index_exact(
        ce_path, label="CE consensus", canonical=canonical, task=task
    )
    for uid, row in rows.items():
        if row.get("schema_version") != CONSENSUS_SCHEMA:
            raise ValueError(f"CE schema mismatch for {uid}")
        if progressive and not isinstance(row.get("progressive"), Mapping):
            raise ValueError(f"progressive CE row lacks exit provenance for {uid}")
        if progressive:
            provenance = row["progressive"]
            automatic = row.get("automatic_match") is True
            if (
                provenance.get("all_prior_candidate_tiers_scored") is not True
                or automatic
                != bool(normalize_space(provenance.get("exit_trial_id")))
                or (
                    automatic
                    and provenance.get("dev_stop_authorized") is not True
                    and provenance.get("terminal_complete_bank_trial") is not True
                )
                or (
                    not automatic
                    and provenance.get("exit_trial_ordinal") is not None
                )
            ):
                raise ValueError(f"progressive CE exit provenance differs for {uid}")
        candidates = list(row.get("candidates") or [])
        candidate_ids = [str(value.get("metric_id") or "") for value in candidates]
        if (
            int(row.get("candidate_count", -1)) != len(candidates)
            or "" in candidate_ids
            or len(candidate_ids) != len(set(candidate_ids))
            or set(candidate_ids) - metric_ids
        ):
            raise ValueError(f"CE candidate bundle is missing/duplicate/out-of-bank: {uid}")
        states = row.get("seed_decisions") or {}
        if len(states) != 2:
            raise ValueError(f"CE row does not contain exactly two seeds: {uid}")
        automatic = row.get("automatic_match") is True
        metric_id = row.get("metric_id")
        if automatic:
            tops = {str(state.get("top_metric_id") or "") for state in states.values()}
            if (
                row.get("decision") != "MATCH"
                or row.get("routing_category") != "MATCH"
                or row.get("provisional_routing_only") is not False
                or str(metric_id or "") not in metric_ids
                or str(metric_id) not in candidate_ids
                or tops != {str(metric_id)}
                or any(
                    state.get("passes_frozen_gate") is not True
                    for state in states.values()
                )
            ):
                raise ValueError(f"CE automatic match is not same-leaf/two-gate: {uid}")
        elif (
            row.get("decision") != "ROUTE_TO_ADJUDICATION"
            or row.get("provisional_routing_only") is not True
            or metric_id is not None
            or row.get("human_abstention_subtype_assigned") is not False
        ):
            raise ValueError(f"CE provisional route was promoted or typed: {uid}")
    return rows


def _report_binds_output(
    report: Mapping[str, Any], output: Path, report_path: Path
) -> bool:
    """Accept the standard output bindings used by existing v3 batch reports."""

    expected_path = output.resolve()
    expected_sha = sha256_file(output)
    direct = report.get("output")
    if isinstance(direct, str):
        raw_sha = report.get("output_sha256")
        if _resolve(direct, report_path) == expected_path and raw_sha == expected_sha:
            return True
    if isinstance(direct, Mapping):
        raw_path = direct.get("path")
        if raw_path and _resolve(str(raw_path), report_path) == expected_path:
            if direct.get("sha256") == expected_sha:
                return True
    return False


def _validate_gemma(
    gemma_path: Path,
    gemma_report_path: Path,
    *,
    canonical: Sequence[Mapping[str, Any]],
    task: str,
    metric_ids: set[str],
    bank_sha: str,
) -> dict[str, dict[str, Any]]:
    report = json.loads(gemma_report_path.read_text(encoding="utf-8"))
    if not _report_binds_output(report, gemma_path, gemma_report_path):
        raise ValueError("Gemma report is not hash-bound to the typed output")
    if str(report.get("status") or "") not in {
        "PASS",
        "COMPLETE",
        "SELECTED_FOR_PRODUCTION",
    }:
        raise ValueError("Gemma typed output is not from a completed/selected run")
    rows = _index_exact(
        gemma_path, label="Gemma typed output", canonical=canonical, task=task
    )
    valid = set(DECISIONS) | {"INVALID_OUTPUT"}
    for uid, row in rows.items():
        decision = str(row.get("decision") or "")
        metric_id = row.get("metric_id")
        if decision not in valid:
            raise ValueError(f"invalid Gemma decision for {uid}: {decision!r}")
        if (decision == "MATCH") != (str(metric_id or "") in metric_ids):
            raise ValueError(f"Gemma metric/decision mismatch for {uid}")
        if decision != "MATCH" and metric_id is not None:
            raise ValueError(f"Gemma abstention carries a metric for {uid}")
        observed_bank = row.get("candidate_bank_source_sha256") or row.get(
            "bank_source_sha256"
        )
        if str(observed_bank or "") != bank_sha:
            raise ValueError(f"Gemma bank provenance mismatch for {uid}")
    return rows


def _validate_merged_partition(
    merged_path: Path,
    final_paths: Mapping[str, Path],
    corpora: Sequence[str],
) -> None:
    merged = list(read_jsonl(merged_path))
    partitioned = [
        row
        for corpus in corpora
        for row in read_jsonl(final_paths[corpus])
    ]
    if len(merged) != len(partitioned):
        raise ValueError("merged final and exact corpus partitions differ in row count")
    for position, (left, right) in enumerate(
        zip(merged, partitioned, strict=True), 1
    ):
        if left != right:
            raise ValueError(
                f"merged final/exact corpus partition drift at task position {position}"
            )


def _validate_rescue_report(
    report_path: Path,
    final_paths: Mapping[str, Path],
    corpora: Sequence[str],
    *,
    merged_final_path: Path | None,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "silver-match-v3-rescue-merge-v1":
        raise ValueError("unsupported rescue merge report schema")
    if report.get("strict_production") is not True or int(
        report.get("unresolved_rows", -1)
    ) != 0:
        raise ValueError("rescue merge is not strict or remains unresolved")
    if merged_final_path is not None:
        merged_final_path = merged_final_path.resolve()
        if (
            _resolve(str(report.get("output") or ""), report_path)
            != merged_final_path
            or report.get("output_sha256") != sha256_file(merged_final_path)
        ):
            raise ValueError("rescue report is not hash-bound to the merged task final")
        _validate_merged_partition(merged_final_path, final_paths, corpora)
    elif len(corpora) == 1:
        final_path = final_paths[corpora[0]].resolve()
        if (
            _resolve(str(report.get("output") or ""), report_path) != final_path
            or report.get("output_sha256") != sha256_file(final_path)
        ):
            raise ValueError("rescue report is not hash-bound to the canonical final")
    else:
        outputs = report.get("outputs") or {}
        if set(outputs) != set(corpora):
            raise ValueError(
                "multi-corpus rescue requires a bound merged final or exact corpus outputs"
            )
        for corpus in corpora:
            ref = outputs[corpus]
            path = final_paths[corpus].resolve()
            if (
                not isinstance(ref, Mapping)
                or _resolve(str(ref.get("path") or ""), report_path) != path
                or ref.get("sha256") != sha256_file(path)
            ):
                raise ValueError(f"rescue report corpus binding changed: {corpus}")
    return report


def _validate_final_evidence(
    *,
    canonical: Sequence[Mapping[str, Any]],
    final_paths: Mapping[str, Path],
    corpora: Sequence[str],
    ce: Mapping[str, Mapping[str, Any]],
    gemma: Mapping[str, Mapping[str, Any]],
    metric_ids: set[str],
) -> dict[str, Any]:
    final_rows = [
        row
        for corpus in corpora
        for row in read_jsonl(final_paths[corpus])
    ]
    if len(final_rows) != len(canonical):
        raise ValueError("final/canonical row count differs")
    counts = Counter()
    for expected, final in zip(canonical, final_rows, strict=True):
        uid = normalize_space(expected["norm_uid"])
        if normalize_space(final.get("norm_uid")) != uid:
            raise ValueError(f"final is not in canonical order at {uid}")
        if str(final.get("source_group") or "") != str(
            expected.get("source_group") or ""
        ):
            raise ValueError(f"final source-group provenance drift for {uid}")
        decision = str(final.get("decision") or "")
        metric_id = final.get("metric_id")
        ce_row = ce[uid]
        gemma_row = gemma[uid]
        ce_promoted = ce_row.get("automatic_match") is True
        if ce_promoted:
            if decision != "MATCH" or metric_id != ce_row.get("metric_id"):
                raise ValueError(f"same-leaf gated CE match not preserved for {uid}")
            counts["primary_same_leaf_gated_ce"] += 1
            continue

        status = str(final.get("rescue_status") or "")
        verification = str(final.get("verification_status") or "")
        if not status.startswith("EXHAUSTIVE_RESCUE_"):
            raise ValueError(f"non-CE row bypassed exhaustive rescue: {uid}")
        if decision == "MATCH":
            if (
                str(metric_id or "") not in metric_ids
                or verification
                not in {
                    "rescued_verified_exact_match",
                    "independent_blind_unresolved_resolution",
                }
            ):
                raise ValueError(f"rescued match lacks strict exact verification: {uid}")
            counts["rescued_verified_exact_match"] += 1
        elif decision in TYPED_ABSTENTIONS:
            if verification not in {
                "rescued_repeated_full_bank_typed_abstention",
                "independent_blind_unresolved_resolution",
            }:
                raise ValueError(f"typed abstention lacks repeated full-bank evidence: {uid}")
            counts["rescued_typed_abstention"] += 1
        elif decision in {"UNSTABLE_MATCH", "INVALID_OUTPUT"}:
            counts[f"rescued_{decision.lower()}"] += 1
        else:
            raise ValueError(f"unknown final decision for {uid}: {decision!r}")
        pre = final.get("pre_rescue")
        if not isinstance(pre, Mapping):
            raise ValueError(f"rescued row lacks pre_rescue evidence: {uid}")
        if (
            str(pre.get("decision") or "") != str(gemma_row.get("decision") or "")
            or pre.get("metric_id") != gemma_row.get("metric_id")
        ):
            raise ValueError(f"rescued row/Gemma typed decision drift for {uid}")
        if not isinstance(final.get("rescue_resolution"), Mapping):
            raise ValueError(f"rescued row lacks resolution evidence: {uid}")
    total = len(canonical)
    rescued = total - counts["primary_same_leaf_gated_ce"]
    return {
        "counts": dict(sorted(counts.items())),
        "primary_same_leaf_gated_ce_count": counts[
            "primary_same_leaf_gated_ce"
        ],
        "primary_same_leaf_gated_ce_rate": counts[
            "primary_same_leaf_gated_ce"
        ]
        / total,
        "repeated_full_bank_rescue_count": rescued,
        "repeated_full_bank_rescue_rate": rescued / total,
        "all_promoted_matches_have_allowed_evidence": True,
        "all_non_ce_rows_repeated_full_bank_rescued": True,
    }


def _validate_plan(
    plan_path: Path,
    *,
    task: str,
    manifest_path: Path,
    bank_sha: str,
) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("status") != "FROZEN_READY_FOR_UNLABELED_PRODUCTION"
        or plan.get("task") != task
        or (plan.get("manifest") or {}).get("sha256") != sha256_file(manifest_path)
        or plan.get("bank_source_sha256") != bank_sha
    ):
        raise ValueError("production plan is not frozen for this task/manifest/bank")


def _load_exclusions(
    paths: Iterable[Path], canonical_uids: set[str]
) -> tuple[set[str], dict[str, str]]:
    paths = [path.resolve() for path in paths]
    if not paths:
        raise ValueError("at least one train/dev/test analysis exclusion is required")
    uids: set[str] = set()
    artifacts: dict[str, str] = {}
    for path in paths:
        artifacts[str(path)] = sha256_file(path)
        for row in read_jsonl(path):
            uid = normalize_space(row.get("norm_uid"))
            if not uid:
                raise ValueError(f"analysis exclusion lacks norm_uid: {path}")
            uids.add(uid)
    foreign = uids - canonical_uids
    if foreign:
        raise ValueError(f"analysis exclusions contain foreign UIDs: {sorted(foreign)[:3]}")
    return uids, dict(sorted(artifacts.items()))


def freeze_release(
    *,
    manifest_path: Path,
    task: str,
    plan_path: Path,
    final_paths: Mapping[str, Path],
    final_audit_path: Path,
    ce_path: Path,
    ce_report_path: Path,
    gemma_path: Path,
    gemma_report_path: Path,
    rescue_report_path: Path,
    risk_release_path: Path,
    analysis_exclusion_paths: Sequence[Path],
    mi_certificate_path: Path,
    merged_final_path: Path | None = None,
    expected_rows: int | None = None,
    expected_bank_metrics: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    plan_path = plan_path.resolve()
    final_paths = {
        str(corpus): path.resolve() for corpus, path in final_paths.items()
    }
    merged_final_path = merged_final_path.resolve() if merged_final_path else None
    final_audit_path = final_audit_path.resolve()
    ce_path = ce_path.resolve()
    ce_report_path = ce_report_path.resolve()
    gemma_path = gemma_path.resolve()
    gemma_report_path = gemma_report_path.resolve()
    rescue_report_path = rescue_report_path.resolve()
    risk_release_path = risk_release_path.resolve()
    mi_certificate_path = mi_certificate_path.resolve()
    manifest, corpora, canonical, metric_ids, bank_path = _load_task(
        manifest_path, task
    )
    if expected_rows is not None and len(canonical) != expected_rows:
        raise ValueError(f"expected {expected_rows} canonical rows, found {len(canonical)}")
    if expected_bank_metrics is not None and len(metric_ids) != expected_bank_metrics:
        raise ValueError(
            f"expected {expected_bank_metrics} bank metrics, found {len(metric_ids)}"
        )
    if set(final_paths) != set(corpora):
        raise ValueError(
            f"final corpus coverage mismatch: missing={sorted(set(corpora)-set(final_paths))}, "
            f"extra={sorted(set(final_paths)-set(corpora))}"
        )
    bank_sha = str(manifest["banks"][task]["source_sha256"])
    _validate_plan(
        plan_path, task=task, manifest_path=manifest_path, bank_sha=bank_sha
    )
    ce = _validate_ce(
        ce_path,
        ce_report_path,
        canonical=canonical,
        task=task,
        metric_ids=metric_ids,
    )
    gemma = _validate_gemma(
        gemma_path,
        gemma_report_path,
        canonical=canonical,
        task=task,
        metric_ids=metric_ids,
        bank_sha=bank_sha,
    )
    _validate_rescue_report(
        rescue_report_path,
        final_paths,
        corpora,
        merged_final_path=merged_final_path,
    )
    audit = audit_outputs(
        manifest_path,
        [final_paths[corpus] for corpus in corpora],
        tasks={task},
        corpora=set(corpora),
    )
    frozen_audit = json.loads(final_audit_path.read_text(encoding="utf-8"))
    if frozen_audit != audit:
        raise ValueError("frozen final audit differs from exact recomputation")
    evidence = _validate_final_evidence(
        canonical=canonical,
        final_paths=final_paths,
        corpora=corpora,
        ce=ce,
        gemma=gemma,
        metric_ids=metric_ids,
    )
    canonical_uids = {normalize_space(row["norm_uid"]) for row in canonical}
    uid_to_corpus = {
        normalize_space(row["norm_uid"]): str(row["corpus"]) for row in canonical
    }
    exclusion_uids, exclusion_artifacts = _load_exclusions(
        analysis_exclusion_paths, canonical_uids
    )
    risk = verify_task_final_risk_release(
        risk_release_path,
        expected_manifest_path=manifest_path,
        expected_final_paths=[final_paths[corpus] for corpus in corpora],
    )
    if (
        risk.get("task") != task
        or risk.get("status") != "PASS"
        or risk.get("complete") is not True
        or risk.get("production_final_blind_audit") is not True
        or not risk.get("gates")
        or not all((risk.get("gates") or {}).values())
        or (risk.get("analysis_exclusions") or {}) != exclusion_artifacts
    ):
        raise ValueError("independent production risk release is not an exact linked PASS")

    bank_payload = json.loads(bank_path.read_text(encoding="utf-8"))
    _, mi_join = _load_certificate(
        mi_certificate_path, list(bank_payload["metrics"]), task
    )
    counts = audit["overall"]["decision_counts"]
    total = int(audit["audited_rows"])
    corpus_rows = {
        corpus: int(audit["by_corpus"][corpus]["count"]) for corpus in corpora
    }
    excluded_by_corpus = Counter(uid_to_corpus[uid] for uid in exclusion_uids)
    eligible_by_corpus = {
        corpus: corpus_rows[corpus] - int(excluded_by_corpus[corpus])
        for corpus in corpora
    }
    typed_count = sum(int(counts[name]) for name in TYPED_ABSTENTIONS)
    rollups = {
        "denominator_all_canonical_norms": total,
        "match_count": int(counts["MATCH"]),
        "match_rate": int(counts["MATCH"]) / total,
        "typed_abstention_count": typed_count,
        "typed_abstention_rate": typed_count / total,
        "abstention_or_failure_count": total - int(counts["MATCH"]),
        "abstention_or_failure_rate": (total - int(counts["MATCH"])) / total,
        "noise_count": int(counts["NOISE"]),
        "noise_rate": int(counts["NOISE"]) / total,
        "unstable_count": int(counts["UNSTABLE_MATCH"]),
        "unstable_rate": int(counts["UNSTABLE_MATCH"]) / total,
        "invalid_count": int(counts["INVALID_OUTPUT"]),
        "invalid_rate": int(counts["INVALID_OUTPUT"]) / total,
        "rescue_count": evidence["repeated_full_bank_rescue_count"],
        "rescue_rate": evidence["repeated_full_bank_rescue_rate"],
    }
    release = {
        "schema_version": SCHEMA,
        "status": "TASK_FROZEN_ANALYSIS_READY",
        "task": task,
        "corpora": corpora,
        "expected_rows": total,
        "manifest": _artifact(manifest_path),
        "bank_source_sha256": bank_sha,
        "production_plan": _artifact(plan_path),
        "final_audit": _artifact(final_audit_path),
        "final_outputs": [_artifact(final_paths[corpus]) for corpus in corpora],
        "postinference_evidence": {
            "two_seed_ce_consensus": _artifact(ce_path),
            "two_seed_ce_consensus_report": _artifact(ce_report_path),
            "typed_gemma": _artifact(gemma_path),
            "typed_gemma_report": _artifact(gemma_report_path),
            "strict_repeated_full_bank_rescue_report": _artifact(
                rescue_report_path
            ),
            "merged_task_final": (
                _artifact(merged_final_path) if merged_final_path else None
            ),
            "evidence_audit": evidence,
        },
        "blind_risk_audit": _artifact(risk_release_path),
        "blind_risk": {
            "status": risk["status"],
            "gates": risk["gates"],
            "thresholds": risk["thresholds"],
            "match": risk["match_audit"]["statistics"],
            "abstention": risk["abstention_audit"]["statistics"],
        },
        "analysis_exclusions": {
            "policy": (
                "exclude every labeled retriever/GEPA/verifier/CE/Gemma "
                "train/dev/test norm from MI and outcome estimation"
            ),
            "inputs": exclusion_artifacts,
            "count": len(exclusion_uids),
            "norm_uids": sorted(exclusion_uids),
        },
        "precision_claim_supported": True,
        "false_abstention_claim_supported": True,
        "coverage": {
            "task_count": 1,
            "task": task,
            "corpora_expected": len(corpora),
            "corpora_audited": int(audit["corpora_audited"]),
            "manifest_corpus_order": corpora,
            "canonical_rows_by_corpus": corpus_rows,
            "analysis_excluded_rows_by_corpus": {
                corpus: int(excluded_by_corpus[corpus]) for corpus in corpora
            },
            "analysis_eligible_rows_by_corpus": eligible_by_corpus,
            "canonical_rows_expected": total,
            "canonical_rows_audited": int(audit["audited_rows"]),
            "bank_metrics": len(metric_ids),
            "mi_certificate_metrics": int(mi_join["joined_metrics"]),
            "mi_certificate_bank_coverage": float(mi_join["bank_coverage"]),
        },
        "rates": {
            "nine_status_counts": counts,
            "nine_status_rates": audit["overall"]["decision_rates"],
            "micro_over_task": audit["overall"],
            "by_corpus": audit["by_corpus"],
            "macro_over_corpora": audit["macro_over_corpora"],
            **rollups,
        },
        "mi_certificate": {**_artifact(mi_certificate_path), **mi_join},
        "analysis_firewall": {
            "task_matcher_is_immutable": True,
            "may_join_mi_and_outcomes": True,
            "may_tune_this_or_other_task_matchers_from_results": False,
            "cross_task_prompt_or_threshold_transfer_after_release": False,
        },
    }
    handoff = {
        "schema_version": HANDOFF_SCHEMA,
        "status": "READY_TO_RUN_EXISTING_MI_VALIDATION",
        "task": task,
        "denominators": {
            "canonical_rows": total,
            "analysis_excluded_rows": len(exclusion_uids),
            "analysis_eligible_rows": total - len(exclusion_uids),
            "corpora": len(corpora),
            "canonical_rows_by_corpus": corpus_rows,
            "analysis_excluded_rows_by_corpus": {
                corpus: int(excluded_by_corpus[corpus]) for corpus in corpora
            },
            "analysis_eligible_rows_by_corpus": eligible_by_corpus,
            "bank_metrics": len(metric_ids),
            "certificate_metrics": int(mi_join["joined_metrics"]),
        },
        "coverage_requirements": {
            "one_task_only": True,
            "all_task_corpora_present": True,
            "all_canonical_norms_present": True,
            "all_nine_statuses_reported_including_zero_counts": True,
            "independent_final_risk_pass": True,
            "certificate_join_validated": True,
        },
        "command_module": "scripts.tools.silver_match_v3.silver_mi_validation_v3",
        "command_arguments": {
            "release": "<WRITTEN_RELEASE_PATH>",
            "certificate": str(mi_certificate_path),
            "n_permutations": 2000,
            "n_bootstrap": 1000,
            "seed": 1729,
            "output": "<MI_VALIDATION_OUTPUT.json>",
        },
        "command_argv": [
            "python",
            "-m",
            "scripts.tools.silver_match_v3.silver_mi_validation_v3",
            "--release",
            "<WRITTEN_RELEASE_PATH>",
            "--certificate",
            str(mi_certificate_path),
            "--n-permutations",
            "2000",
            "--n-bootstrap",
            "1000",
            "--seed",
            "1729",
            "--output",
            "<MI_VALIDATION_OUTPUT.json>",
        ],
        "expected_result_schema": "silver-match-v3-mi-validation-v1",
    }
    return release, handoff


def parse_final_bindings(
    values: Sequence[str], manifest_path: Path, task: str
) -> dict[str, Path]:
    """Parse exact ``CORPUS=PATH`` finals, retaining one-file Humor ergonomics."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = [
        corpus
        for corpus, meta in (manifest.get("corpora") or {}).items()
        if str(meta.get("task") or "") == task
    ]
    if not expected:
        raise ValueError(f"manifest has no corpora for task {task!r}")
    if len(expected) == 1 and len(values) == 1 and "=" not in values[0]:
        return {expected[0]: Path(values[0]).resolve()}
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("multi-corpus --final values must use CORPUS=PATH")
        corpus, raw_path = value.split("=", 1)
        corpus = corpus.strip()
        if not corpus or not raw_path:
            raise ValueError(f"invalid final binding: {value!r}")
        if corpus in output:
            raise ValueError(f"duplicate final corpus binding: {corpus}")
        output[corpus] = Path(raw_path).resolve()
    if set(output) != set(expected):
        raise ValueError(
            f"final corpus coverage mismatch: missing={sorted(set(expected)-set(output))}, "
            f"extra={sorted(set(output)-set(expected))}"
        )
    return output


def resolve_task(manifest_path: Path, requested: str | None) -> str:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = list((manifest.get("banks") or {}).keys())
    if requested:
        if requested not in tasks:
            raise ValueError(f"manifest has no task {requested!r}")
        return requested
    if len(tasks) != 1:
        raise ValueError("--task is required when the manifest contains multiple tasks")
    return str(tasks[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", help="derived when the manifest has exactly one task")
    parser.add_argument("--plan", required=True)
    parser.add_argument(
        "--final",
        action="append",
        required=True,
        help="CORPUS=PATH; a plain PATH is accepted for a one-corpus task",
    )
    parser.add_argument(
        "--merged-final",
        help="optional rescue-report-bound task file exactly partitioned by --final",
    )
    parser.add_argument("--final-audit", required=True)
    parser.add_argument("--ce-consensus", required=True)
    parser.add_argument("--ce-report", required=True)
    parser.add_argument("--gemma-typed", required=True)
    parser.add_argument("--gemma-report", required=True)
    parser.add_argument("--rescue-report", required=True)
    parser.add_argument("--risk-release", required=True)
    parser.add_argument("--analysis-exclusion", action="append", required=True)
    parser.add_argument("--mi-certificate", required=True)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-bank-metrics", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mi-handoff-output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    handoff_output = Path(args.mi_handoff_output).resolve()
    if output.exists() or handoff_output.exists():
        raise FileExistsError("refusing to overwrite release or MI handoff")
    manifest_path = Path(args.manifest).resolve()
    task = resolve_task(manifest_path, args.task)
    final_paths = parse_final_bindings(
        args.final, manifest_path, task
    )
    release, handoff = freeze_release(
        manifest_path=manifest_path,
        task=task,
        plan_path=Path(args.plan),
        final_paths=final_paths,
        final_audit_path=Path(args.final_audit),
        ce_path=Path(args.ce_consensus),
        ce_report_path=Path(args.ce_report),
        gemma_path=Path(args.gemma_typed),
        gemma_report_path=Path(args.gemma_report),
        rescue_report_path=Path(args.rescue_report),
        risk_release_path=Path(args.risk_release),
        analysis_exclusion_paths=[Path(path) for path in args.analysis_exclusion],
        mi_certificate_path=Path(args.mi_certificate),
        merged_final_path=(Path(args.merged_final) if args.merged_final else None),
        expected_rows=args.expected_rows,
        expected_bank_metrics=args.expected_bank_metrics,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
    handoff["release"] = _artifact(output)
    handoff["command_arguments"]["release"] = str(output)
    release_index = handoff["command_argv"].index("<WRITTEN_RELEASE_PATH>")
    handoff["command_argv"][release_index] = str(output)
    handoff_output.parent.mkdir(parents=True, exist_ok=True)
    handoff_output.write_text(
        json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": release["status"],
                "task": release["task"],
                "rows": release["expected_rows"],
                "release": _artifact(output),
                "mi_handoff": _artifact(handoff_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
