#!/usr/bin/env python3
"""Audit legacy content GEPA truth releases against the strict v2 contract.

This checker is intentionally release-local.  It reads only the declared role
freeze, identities, exact-consensus report, bound labeling passes, and their
transcript audits.  It never searches for or opens test, MI, or outcome data.
It reports blockers and does not relabel, rewrite roles, or promote a release.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


TASKS = ("creative-writing", "legal-outcome-prediction", "peer-review")
ROLES = ("optimize", "select")
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _require_ref(ref: dict[str, Any], label: str) -> Path:
    path = Path(str(ref.get("path") or "")).resolve()
    if not path.is_file() or sha256_file(path) != str(ref.get("sha256") or ""):
        raise ValueError(f"missing or drifted {label}: {path}")
    return path


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    uids = [str(row.get("norm_uid") or "") for row in rows]
    if not rows or "" in uids or len(uids) != len(set(uids)):
        raise ValueError(f"empty/missing/duplicate UID rows: {path}")
    return rows


def audit_role(root: Path, task: str, role: str) -> dict[str, Any]:
    release_path = root / "truth_releases" / role / "FREEZE.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    artifacts = release.get("artifacts") or {}
    truth_path = _require_ref(artifacts.get("truth") or {}, "released truth")
    report_path = _require_ref(
        artifacts.get("resolution_report") or {}, "exact consensus report"
    )
    role_freeze_path = _require_ref(
        artifacts.get("role_freeze") or {}, "role identity freeze"
    )
    truth = _rows(truth_path)
    truth_uids = {str(row["norm_uid"]) for row in truth}
    count = len(truth)
    if (
        release.get("schema_version")
        != "silver-match-v3-content-truth-release-freeze-v1"
        or release.get("status") != "FROZEN_COMPLETE_EXACT_TRUTH"
        or release.get("task") != task
        or release.get("role") != role
        or int(release.get("count") or -1) != count
    ):
        raise ValueError(f"legacy truth release drift: {task}/{role}")

    role_freeze = json.loads(role_freeze_path.read_text(encoding="utf-8"))
    identity_path = root / f"{role}_identity" / "identities.jsonl"
    identities = _rows(identity_path)
    identity_uids = {str(row["norm_uid"]) for row in identities}
    recorded_identity = (role_freeze.get("outputs") or {}).get("identities") or {}
    exclusion = role_freeze.get("exclusion_union") or {}
    role_freeze_valid = all(
        (
            role_freeze.get("schema_version")
            == "silver-match-v3-clean-gepa-panel-freeze-v1",
            role_freeze.get("status")
            == "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES",
            role_freeze.get("task") == task,
            role_freeze.get("role") == role,
            role_freeze.get("required_upstream_split") == "train",
            int(role_freeze.get("selected_count") or -1) == count,
            recorded_identity.get("sha256") == sha256_file(identity_path),
            identity_uids == truth_uids,
            int(exclusion.get("selected_uid_overlap", -1)) == 0,
            int(exclusion.get("selected_source_group_overlap", -1)) == 0,
            (role_freeze.get("content_contract") or {}).get("truth_fields_read")
            is False,
            (role_freeze.get("content_contract") or {}).get("downstream_outcomes_read")
            is False,
        )
    )
    if not role_freeze_valid:
        raise ValueError(f"role freeze/exclusion contract drift: {task}/{role}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    resolved = (report.get("outputs") or {}).get("resolved") or {}
    unresolved = (report.get("outputs") or {}).get("unresolved") or {}
    if (
        report.get("schema_version")
        != "silver-match-v3-exact-multi-pass-truth-report-v1"
        or report.get("complete") is not True
        or report.get("task") != task
        or report.get("gepa_role") != role
        or int(report.get("source_count") or -1) != count
        or int(report.get("resolved_count") or -1) != count
        or int(report.get("unresolved_count", -1)) != 0
        or resolved.get("sha256") != sha256_file(truth_path)
        or unresolved.get("sha256") != EMPTY_SHA256
    ):
        raise ValueError(f"exact consensus terminal report drift: {task}/{role}")

    passes = (report.get("inputs") or {}).get("passes") or {}
    rounds = report.get("rounds") or []
    if list(passes) != [str(row.get("pass") or "") for row in rounds]:
        raise ValueError(f"consensus pass/round order drift: {task}/{role}")
    pass_audit_rows = []
    for name, meta in passes.items():
        label_path = _require_ref(meta.get("labels") or {}, f"{name} labels")
        validation_path = _require_ref(
            meta.get("pack_validation") or {}, f"{name} pack validation"
        )
        pack = label_path.parent
        label_validation_path = pack / "labels.validation.json"
        label_validation = json.loads(label_validation_path.read_text(encoding="utf-8"))
        labels = _rows(label_path)
        if (
            len(labels) != int(meta.get("count") or -1)
            or label_validation.get("schema_version")
            != "silver-match-v3-independent-label-validation-v1"
            or label_validation.get("complete") is not True
            or ((label_validation.get("output") or {}).get("sha256"))
            != sha256_file(label_path)
            or ((label_validation.get("pack_validation") or {}).get("sha256"))
            != sha256_file(validation_path)
        ):
            raise ValueError(f"validated pass-label drift: {task}/{role}/{name}")
        transcript_path = (
            root
            / "strict_audits_v2"
            / "transcripts_pipeline_safe_v2"
            / role
            / f"{name}.json"
        )
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        if (
            transcript.get("schema_version")
            != "silver-match-v3-isolated-labeler-transcript-audit-v1"
            or transcript.get("status") != "PASS"
            or transcript.get("complete") is not True
            or transcript.get("violations") != []
            or transcript.get("full_pack_artifact_binding") is not True
            or (transcript.get("bank") or {}).get("sha256")
            != str(meta.get("pack_bank_sha256") or "")
            or (transcript.get("items") or {}).get("sha256")
            != str(meta.get("pack_items_sha256") or "")
            or (transcript.get("pack_validation") or {}).get("sha256")
            != sha256_file(validation_path)
        ):
            raise ValueError(f"strict transcript audit drift: {task}/{role}/{name}")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        pass_audit_rows.append(
            {
                "name": name,
                "count": len(labels),
                "pack_schema": validation.get("schema_version"),
                "labels_sha256": sha256_file(label_path),
                "pack_validation_sha256": sha256_file(validation_path),
                "transcript_audit": {
                    "path": str(transcript_path),
                    "sha256": sha256_file(transcript_path),
                    "status": "PASS",
                },
            }
        )

    independence_path = root / "label_workspaces" / f"{role.upper()}_INDEPENDENCE_AUDIT.json"
    independence = json.loads(independence_path.read_text(encoding="utf-8"))
    initial_schemas = [row["pack_schema"] for row in pass_audit_rows[:2]]
    blockers = []
    if initial_schemas != [
        "silver-match-v3-permuted-independent-teacher-pack-v1",
        "silver-match-v3-permuted-independent-teacher-pack-v1",
    ]:
        blockers.append(
            {
                "gate": "strict_initial_pack_provenance",
                "observed": initial_schemas,
                "required": "two permuted-independent teacher packs from one prelabel clean K50 source pack",
            }
        )
    if (
        independence.get("schema_version")
        != "silver-match-v3-independent-pack-view-audit-v1"
        or independence.get("status")
        != "FROZEN_MUTUALLY_PREDICTION_HIDDEN_BEFORE_LABELING"
    ):
        blockers.append(
            {
                "gate": "strict_prelabel_independence_provenance",
                "observed": {
                    "schema_version": independence.get("schema_version"),
                    "status": independence.get("status"),
                },
                "required": {
                    "schema_version": "silver-match-v3-independent-pack-view-audit-v1",
                    "status": "FROZEN_MUTUALLY_PREDICTION_HIDDEN_BEFORE_LABELING",
                },
            }
        )
    blockers.append(
        {
            "gate": "strict_clean_k50_candidate_release",
            "observed": "not bound by legacy truth release or initial pass validations",
            "required": "silver-match-v3-clean-gepa-label-pack-v1 frozen before labeling",
        }
    )
    strict_release = root / "truth_releases_strict_v2" / role / "FREEZE.json"
    if strict_release.exists():
        raise ValueError(f"unexpected preexisting strict release requires separate audit: {strict_release}")
    return {
        "task": task,
        "role": role,
        "count": count,
        "legacy_release": {
            "path": str(release_path),
            "sha256": sha256_file(release_path),
            "complete_exact_truth": True,
        },
        "role_identity_and_exclusion_freeze": {
            "path": str(role_freeze_path),
            "sha256": sha256_file(role_freeze_path),
            "identities_sha256": sha256_file(identity_path),
            "valid": True,
        },
        "consensus_terminal_report": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
            "complete": True,
            "unresolved_count": 0,
        },
        "pass_audits": pass_audit_rows,
        "strict_transcript_isolation_all_passes": True,
        "strict_v2_release_ready": False,
        "production_gepa_plan_may_be_frozen": False,
        "blockers": blockers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default="outputs/silver_match_v3")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    workspace = Path(args.workspace_root).resolve()
    results = []
    for task in TASKS:
        root = workspace / task / "gepa_clean_v1"
        for role in ROLES:
            results.append(audit_role(root, task, role))
    payload = {
        "schema_version": "silver-match-v3-content-gepa-strict-readiness-audit-v1",
        "status": "BLOCKED_STRICT_V2_RELEASE_AND_PRODUCTION_PLAN",
        "task_count": len(TASKS),
        "role_release_count": len(results),
        "legacy_exact_truth_count": sum(row["count"] for row in results),
        "strict_transcript_pass_count": sum(len(row["pass_audits"]) for row in results),
        "strict_transcript_fail_count": 0,
        "strict_v2_release_ready_count": 0,
        "production_gepa_plan_ready_count": 0,
        "results": results,
        "audit_contract": {
            "test_inputs_opened": False,
            "mi_or_outcome_inputs_opened": False,
            "truth_roles_changed": False,
            "labels_changed_or_created": False,
            "legacy_releases_overwritten": False,
            "strict_gate_failed_closed": True,
        },
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), **payload}))


if __name__ == "__main__":
    main()
