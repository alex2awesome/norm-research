#!/usr/bin/env python3
"""Freeze evidence that attempted independent label views were invalid.

This audit never parses label contents.  It records immutable file identities,
the shared item/bank order, and proposal visibility so the rejected outputs can
never be mistaken for canonical independent truth.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


def _files(root: Path, pattern: str) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.glob(pattern))
        if path.is_file()
    ]


def _view(root: Path) -> dict[str, Any]:
    validation = root / "validation.json"
    items = root / "items.jsonl"
    bank = root / "bank.json"
    item_rows = list(read_jsonl(items))
    bank_payload = json.loads(bank.read_text(encoding="utf-8"))
    uids = [str(row.get("norm_uid") or "") for row in item_rows]
    source_groups = [str(row.get("source_group") or "") for row in item_rows]
    metric_ids = [
        str(row.get("metric_id") or "") for row in bank_payload.get("metrics") or []
    ]
    if (
        not uids
        or "" in uids
        or len(set(uids)) != len(uids)
        or "" in source_groups
        or not metric_ids
        or "" in metric_ids
        or len(set(metric_ids)) != len(metric_ids)
    ):
        raise ValueError(f"invalid rejected-view identity inventory: {root}")
    candidate_files = _files(root, "*candidate*")
    raw_labels = _files(root, "raw_labels/*.json")
    raw_logs = _files(root, "logs/*.log")
    invalid_attempts = _files(root, "invalid_raw_labels/*")
    return {
        "root": str(root),
        "validation": {
            "path": str(validation),
            "sha256": sha256_file(validation),
        },
        "items": {"path": str(items), "sha256": sha256_file(items)},
        "bank": {"path": str(bank), "sha256": sha256_file(bank)},
        "uid_count": len(uids),
        "source_group_count": len(set(source_groups)),
        "metric_count": len(metric_ids),
        "uids_in_order": uids,
        "source_groups": sorted(set(source_groups)),
        "metric_ids_in_order": metric_ids,
        "candidate_files_visible": candidate_files,
        "raw_label_files": raw_labels,
        "raw_label_file_count": len(raw_labels),
        "raw_label_contents_parsed_by_auditor": False,
        "labeler_logs": raw_logs,
        "invalid_attempt_logs": invalid_attempts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--pass-a", required=True)
    parser.add_argument("--pass-b", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    left = _view(Path(args.pass_a).resolve())
    right = _view(Path(args.pass_b).resolve())
    if set(left["uids_in_order"]) != set(right["uids_in_order"]):
        raise ValueError("rejected views do not cover the same UID set")
    if set(left["metric_ids_in_order"]) != set(right["metric_ids_in_order"]):
        raise ValueError("rejected views do not cover the same bank leaves")
    evidence = {
        "same_item_order": left["uids_in_order"] == right["uids_in_order"],
        "same_bank_order": left["metric_ids_in_order"]
        == right["metric_ids_in_order"],
        "pass_a_candidate_file_count": len(left["candidate_files_visible"]),
        "pass_b_candidate_file_count": len(right["candidate_files_visible"]),
    }
    if not (
        evidence["same_item_order"]
        and evidence["same_bank_order"]
        and evidence["pass_a_candidate_file_count"] > 0
        and evidence["pass_b_candidate_file_count"] > 0
    ):
        raise ValueError("requested rejection reasons are not all evidenced")
    payload = {
        "schema_version": "silver-match-v3-rejected-nonindependent-label-views-v1",
        "status": "REJECTED_NONINDEPENDENT_VIEW",
        "panel": args.panel,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "reasons": [
            "identical_item_order_across_nominally_independent_passes",
            "identical_bank_order_across_nominally_independent_passes",
            "candidate_proposal_artifact_visible_in_each_label_workspace",
            "relocated_validation_chunk_paths_not_locally_auditable_verbatim",
        ],
        "evidence": evidence,
        "passes": {"A": left, "B": right},
        "canonical_usage": {
            "eligible_for_truth": False,
            "eligible_for_prompt_or_model_selection": False,
            "eligible_for_training": False,
            "eligible_for_reporting": False,
            "preserve_append_only_for_failure_analysis": True,
        },
        "exclusion_contract": {
            "panel_uids_and_source_groups_permanently_excluded_from_mi_and_outcome_analysis": True,
            "panel_uids_and_source_groups_permanently_excluded_from_retriever_or_ce_gradients": True,
            "same_frozen_panel_identities_may_be_relabelled_from_fresh_clean_views": True,
        },
        "auditor_read_label_contents": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "panel": args.panel,
                "output": str(output),
                "sha256": sha256_file(output),
                "raw_label_files": {
                    "A": left["raw_label_file_count"],
                    "B": right["raw_label_file_count"],
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
