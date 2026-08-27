#!/usr/bin/env python3
"""Freeze one complete content-task GEPA truth panel for downstream jobs.

The release record binds the identity-only role freeze, truth-hidden source
pack, exact multi-pass resolution report, and final canonical truth JSONL.  It
is intentionally small so a remote GPU queue can verify identities and hashes
without guessing which ``resolved.after_*`` artifact is terminal.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adjudicate_gemma import DECISIONS
from .common import read_jsonl, sha256_file


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--role", choices=("optimize", "select"), required=True)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--role-freeze", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--resolution-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    pack_root = Path(args.pack_root).resolve()
    validation_path = pack_root / "validation.json"
    items_path = pack_root / "items.jsonl"
    bank_path = pack_root / "bank.json"
    freeze_path = Path(args.role_freeze).resolve()
    truth_path = Path(args.truth).resolve()
    report_path = Path(args.resolution_report).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite truth release: {output_path}")

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    role_freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        validation.get("task") != args.task
        or validation.get("truth_hidden") is not True
        or sha256_file(items_path) != validation["outputs"]["items"]["sha256"]
        or sha256_file(bank_path) != validation["outputs"]["bank"]["sha256"]
    ):
        raise ValueError("invalid or mutated truth-hidden source pack")

    identity_sha = str((validation.get("input_hashes") or {}).get("items") or "")
    frozen_identity = (role_freeze.get("outputs") or {}).get("identities") or {}
    contract = role_freeze.get("content_contract") or {}
    if (
        role_freeze.get("schema_version")
        != "silver-match-v3-clean-gepa-panel-freeze-v1"
        or role_freeze.get("status") != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or role_freeze.get("task") != args.task
        or role_freeze.get("role") != args.role
        or int(role_freeze.get("selected_count", -1)) != int(validation.get("count", -2))
        or str(frozen_identity.get("sha256") or "") != identity_sha
        or contract.get("selection_uses_identity_and_source_group_only") is not True
        or any(
            contract.get(key) is not False
            for key in (
                "downstream_outcomes_read",
                "metric_ids_read",
                "model_prediction_fields_read",
                "truth_fields_read",
            )
        )
    ):
        raise ValueError("role freeze does not bind the truth-hidden source identity")

    truth_sha = sha256_file(truth_path)
    report_role_freeze = (report.get("inputs") or {}).get("gepa_role_freeze") or {}
    report_truth = (report.get("outputs") or {}).get("resolved") or {}
    source_count = int(validation["count"])
    if (
        report.get("schema_version")
        != "silver-match-v3-exact-multi-pass-truth-report-v1"
        or report.get("complete") is not True
        or report.get("task") != args.task
        or report.get("gepa_role") != args.role
        or int(report.get("source_count", -1)) != source_count
        or int(report.get("resolved_count", -1)) != source_count
        or int(report.get("unresolved_count", -1)) != 0
        or str(report_role_freeze.get("sha256") or "") != sha256_file(freeze_path)
        or str(report_truth.get("sha256") or "") != truth_sha
    ):
        raise ValueError("resolution report is not a complete role-bound terminal truth")

    items = list(read_jsonl(items_path))
    truth = list(read_jsonl(truth_path))
    source_uids = [str(row["norm_uid"]) for row in items]
    truth_uids = [str(row["norm_uid"]) for row in truth]
    if (
        len(source_uids) != len(set(source_uids))
        or len(truth_uids) != len(set(truth_uids))
        or set(source_uids) != set(truth_uids)
    ):
        raise ValueError("terminal truth does not exactly cover the source panel")
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_ids = {str(row["metric_id"]) for row in bank["metrics"]}
    for row in truth:
        decision = str(row.get("decision") or "")
        metric_id = row.get("metric_id")
        if (
            row.get("task") != args.task
            or row.get("gepa_role") != args.role
            or decision not in DECISIONS
            or (decision == "MATCH" and str(metric_id) not in bank_ids)
            or (decision != "MATCH" and metric_id is not None)
        ):
            raise ValueError(f"invalid terminal truth row: {row.get('norm_uid')}")

    decision_counts = Counter(str(row["decision"]) for row in truth)
    release = {
        "schema_version": "silver-match-v3-content-truth-release-freeze-v1",
        "status": "FROZEN_COMPLETE_EXACT_TRUTH",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "role": args.role,
        "count": len(truth),
        "unique_uids": len(set(truth_uids)),
        "unique_source_groups": len(
            {str(row.get("source_group") or row.get("split_group")) for row in truth}
        ),
        "decision_counts": dict(sorted(decision_counts.items())),
        "match_count": decision_counts["MATCH"],
        "typed_nonmatch_count": len(truth) - decision_counts["MATCH"],
        "bank_source_sha256": validation["bank_source_sha256"],
        "identity": {
            "sha256": identity_sha,
            "selected_count": int(role_freeze["selected_count"]),
            "selected_source_groups": int(role_freeze["selected_source_groups"]),
            "remote_path": frozen_identity.get("path"),
        },
        "artifacts": {
            "role_freeze": _artifact(freeze_path),
            "source_pack_validation": _artifact(validation_path),
            "source_items": _artifact(items_path),
            "source_bank": _artifact(bank_path),
            "resolution_report": _artifact(report_path),
            "truth": _artifact(truth_path),
        },
        "contracts": {
            "identity_frozen_before_truth": True,
            "exact_source_coverage": True,
            "unresolved_count": 0,
            "select_may_not_mutate_prompts": args.role == "select",
            "optimize_may_mutate_prompts": args.role == "optimize",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {**release, "release_freeze_sha256": sha256_file(output_path)},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
