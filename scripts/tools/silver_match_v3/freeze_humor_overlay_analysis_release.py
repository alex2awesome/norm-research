#!/usr/bin/env python3
"""Freeze the canonical Humor overlay for descriptive, leakage-safe MI analysis.

This release path is intentionally advisory.  It binds the canonical overlay,
the frozen post-blind deployment-risk record, and the deterministic
production-only audit sample.  Because independent labels for that sample are
not part of this handoff, precision and false-abstention claims remain false.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .common import normalize_space, read_jsonl, sha256_file
from .silver_mi_validation_v3 import _load_certificate


OVERLAY_SCHEMA = "silver-match-v3-humor-authoritative-overlay-report-v1"
RISK_SCHEMA = "silver-match-v3-humor-hybrid-postblind-risk-v1"
SAMPLE_REPORT_SCHEMA = "silver-match-v3-humor-production-risk-sample-report-v1"
SAMPLE_SCHEMA = "silver-match-v3-humor-production-risk-sample-v1"


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _unique_uids(path: Path, *, expected: int, label: str) -> set[str]:
    values: set[str] = set()
    for row in read_jsonl(path):
        uid = normalize_space(row.get("norm_uid"))
        if not uid or uid in values:
            raise ValueError(f"{label} has missing/duplicate UID: {uid!r}")
        values.add(uid)
    if len(values) != expected:
        raise ValueError(f"{label} expected {expected} UIDs, got {len(values)}")
    return values


def _require_ref(reference: dict[str, Any], path: Path, label: str) -> None:
    if (
        Path(str(reference.get("path") or "")).resolve() != path.resolve()
        or reference.get("sha256") != sha256_file(path)
    ):
        raise ValueError(f"{label} path/hash binding differs")


def freeze(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = Path(args.manifest).resolve()
    plan_path = Path(args.plan).resolve()
    final_path = Path(args.final).resolve()
    final_audit_path = Path(args.final_audit).resolve()
    overlay_report_path = Path(args.overlay_report).resolve()
    predictions_path = Path(args.production_predictions).resolve()
    truth_path = Path(args.analysis_exclusion).resolve()
    risk_path = Path(args.risk_record).resolve()
    sample_report_path = Path(args.production_audit_report).resolve()
    certificate_path = Path(args.mi_certificate).resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_meta = (manifest.get("corpora") or {}).get("humor_multi") or {}
    bank_meta = (manifest.get("banks") or {}).get("humor") or {}
    if int(corpus_meta.get("count", -1)) != 77378 or int(bank_meta.get("count", -1)) != 285:
        raise ValueError("manifest Humor cardinalities differ")
    canonical_path = Path(str(corpus_meta["path"])).resolve()
    bank_path = Path(str(bank_meta["path"])).resolve()
    canonical_uids = _unique_uids(canonical_path, expected=77378, label="canonical")
    truth_uids = _unique_uids(truth_path, expected=22090, label="analysis exclusion")
    prediction_uids = _unique_uids(predictions_path, expected=55288, label="production predictions")
    if truth_uids & prediction_uids or truth_uids | prediction_uids != canonical_uids:
        raise ValueError("truth and production prediction UIDs do not exactly partition Humor")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("status") != "FROZEN_READY_FOR_UNLABELED_PRODUCTION"
        or plan.get("task") != "humor"
        or (plan.get("manifest") or {}).get("sha256") != sha256_file(manifest_path)
        or plan.get("bank_source_sha256") != bank_meta.get("source_sha256")
        or (plan.get("analysis_firewall") or {}).get("may_tune_matcher_from_mi") is not False
    ):
        raise ValueError("production plan is not the frozen Humor/manifest plan")

    overlay = json.loads(overlay_report_path.read_text(encoding="utf-8"))
    if (
        overlay.get("schema_version") != OVERLAY_SCHEMA
        or overlay.get("status") != "COMPLETE_CANONICAL_AUTHORITATIVE_OVERLAY_AUDITED"
        or int(((overlay.get("output") or {}).get("count", -1))) != 77378
        or ((overlay.get("partition_audit") or {}).get("union_equals_canonical")) is not True
    ):
        raise ValueError("overlay report is incomplete")
    _require_ref(overlay["output"], final_path, "overlay final")
    _require_ref(overlay["final_audit"], final_audit_path, "overlay final audit")
    _require_ref(overlay["inputs"]["production_predictions"], predictions_path, "overlay predictions")
    _require_ref(overlay["inputs"]["authoritative_truth"], truth_path, "overlay truth")

    final_audit = json.loads(final_audit_path.read_text(encoding="utf-8"))
    if (
        final_audit.get("schema_version") != "silver-match-v3-final-audit-v1"
        or final_audit.get("complete") is not True
        or int(final_audit.get("audited_rows", -1)) != 77378
        or (final_audit.get("scope") or {}).get("tasks") != ["humor"]
        or (final_audit.get("scope") or {}).get("corpora") != ["humor_multi"]
        or (final_audit.get("input_hashes") or {})
        != {str(final_path): sha256_file(final_path)}
    ):
        raise ValueError("standard final audit differs from the canonical final")

    risk = json.loads(risk_path.read_text(encoding="utf-8"))
    if (
        risk.get("schema_version") != RISK_SCHEMA
        or risk.get("status") != "FROZEN_ADVISORY_ONLY_AFTER_BLIND_PRECISION_CONTRACT_MISS"
        or risk.get("no_retraining_or_retuning_performed") is not True
        or int(risk.get("blind_or_test_rows_read_by_this_freeze", -1)) != 0
    ):
        raise ValueError("post-blind deployment risk record differs")

    sample_report = json.loads(sample_report_path.read_text(encoding="utf-8"))
    if (
        sample_report.get("schema_version") != SAMPLE_REPORT_SCHEMA
        or sample_report.get("status") != "COMPLETE_CREATE_ONLY_PRODUCTION_AUDIT_SAMPLE"
        or int(sample_report.get("blind_or_test_rows_read", -1)) != 0
    ):
        raise ValueError("postproduction sample report is incomplete")
    _require_ref(sample_report["risk_record"], risk_path, "sample risk record")
    _require_ref(sample_report["production_predictions"], predictions_path, "sample predictions")
    sample_path = Path(str(sample_report["sample"]["path"])).resolve()
    _require_ref(sample_report["sample"], sample_path, "sample rows")
    population = {str(key): int(value) for key, value in sample_report["population"].items()}
    requested = {str(key): int(value) for key, value in sample_report["requested"].items()}
    selected = {str(key): int(value) for key, value in sample_report["selected"].items()}
    if sum(population.values()) != 55288 or any(
        selected.get(lane, 0) != min(requested[lane], population.get(lane, 0))
        for lane in requested
    ):
        raise ValueError("postproduction sample lane cardinalities differ")
    sample_rows = list(read_jsonl(sample_path))
    if len(sample_rows) != sum(selected.values()) or any(
        row.get("schema_version") != SAMPLE_SCHEMA for row in sample_rows
    ):
        raise ValueError("postproduction sample contents differ")

    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    _, certificate_join = _load_certificate(certificate_path, list(bank["metrics"]), "humor")
    if (
        sha256_file(certificate_path) != args.mi_certificate_sha256
        or int(certificate_join["joined_metrics"]) != 285
        or float(certificate_join["bank_coverage"]) != 1.0
    ):
        raise ValueError("MI certificate identity or exact bank join differs")

    risk_context = {
        "status": risk["status"],
        "precision_contract_passed": False,
        "production_audit_status": "DETERMINISTIC_SAMPLE_FROZEN_LABELS_PENDING",
        "production_audit_population": population,
        "production_audit_selected": selected,
        "analysis_interpretation": (
            "Descriptive advisory-silver analysis only; independent production-audit "
            "labels are not available and no precision/false-abstention claim is supported."
        ),
        "risk_record": _artifact(risk_path),
        "production_audit_report": _artifact(sample_report_path),
    }
    release = {
        "schema_version": "silver-match-v3-task-analysis-release-v1",
        "status": "TASK_FROZEN_ANALYSIS_READY",
        "task": "humor",
        "corpora": ["humor_multi"],
        "expected_rows": 77378,
        "manifest": _artifact(manifest_path),
        "bank_source_sha256": bank_meta["source_sha256"],
        "production_plan": _artifact(plan_path),
        "final_audit": _artifact(final_audit_path),
        "final_outputs": [_artifact(final_path)],
        "blind_risk_audit": _artifact(sample_report_path),
        "blind_risk": risk_context,
        "analysis_exclusions": {
            "policy": "exclude exact joined truth/training/dev/test UIDs from MI and outcomes",
            "inputs": {str(truth_path): sha256_file(truth_path)},
            "count": 22090,
            "norm_uids": sorted(truth_uids),
        },
        "precision_claim_supported": False,
        "false_abstention_claim_supported": False,
        "coverage": {
            "canonical_rows": 77378,
            "analysis_excluded_rows": 22090,
            "analysis_eligible_rows": 55288,
            "bank_metrics": 285,
            "mi_certificate_metrics": 285,
        },
        "mi_certificate": {**_artifact(certificate_path), **certificate_join},
        "analysis_firewall": {
            "task_matcher_is_immutable": True,
            "may_join_mi_and_outcomes": True,
            "may_tune_this_or_other_task_matchers_from_results": False,
            "cross_task_prompt_or_threshold_transfer_after_release": False,
        },
    }
    handoff = {
        "schema_version": "silver-match-v3-mi-validation-handoff-v1",
        "status": "READY_TO_RUN_EXISTING_MI_VALIDATION",
        "task": "humor",
        "denominators": release["coverage"],
        "certificate": _artifact(certificate_path),
        "analysis_claim_scope": risk_context["analysis_interpretation"],
        "command_module": "scripts.tools.silver_match_v3.silver_mi_validation_v3",
        "expected_result_schema": "silver-match-v3-mi-validation-v1",
    }
    return release, handoff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--final", required=True)
    parser.add_argument("--final-audit", required=True)
    parser.add_argument("--overlay-report", required=True)
    parser.add_argument("--production-predictions", required=True)
    parser.add_argument("--analysis-exclusion", required=True)
    parser.add_argument("--risk-record", required=True)
    parser.add_argument("--production-audit-report", required=True)
    parser.add_argument("--mi-certificate", required=True)
    parser.add_argument("--mi-certificate-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mi-handoff-output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    handoff_output = Path(args.mi_handoff_output).resolve()
    if output.exists() or handoff_output.exists():
        raise FileExistsError("refusing to overwrite release or MI handoff")
    release, handoff = freeze(args)
    _write_new(output, release)
    handoff["release"] = _artifact(output)
    _write_new(handoff_output, handoff)
    print(json.dumps({"release": _artifact(output), "mi_handoff": _artifact(handoff_output)}, sort_keys=True))


if __name__ == "__main__":
    main()
