#!/usr/bin/env python3
"""Seal and audit a frozen selection expansion against identity-only exclusions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .make_calibration import split_group_for


def _index(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    output = {str(row["norm_uid"]): row for row in rows}
    if not rows or len(output) != len(rows):
        raise ValueError(f"empty or duplicate norm_uid values: {path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--norms", required=True)
    parser.add_argument("--exclude-panel", action="append", default=[])
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--identity-output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if not args.exclude_panel:
        parser.error("at least one --exclude-panel is required")

    freeze_path = Path(args.freeze).resolve()
    selected_path = Path(args.selected).resolve()
    norms_path = Path(args.norms).resolve()
    exclusion_paths = [Path(value).resolve() for value in args.exclude_panel]
    identity_path = Path(args.identity_output).resolve()
    report_path = Path(args.report).resolve()
    if identity_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite selection audit outputs")

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_PREDICTIONS_OR_LABELS":
        raise ValueError("selection was not frozen before predictions/labels")
    role = freeze["roles"]["verifier_dev"]
    if role.get("count") != args.expected_count:
        raise ValueError("freeze role count mismatch")
    if sha256_file(selected_path) != role.get("sha256"):
        raise ValueError("selected item hash does not match freeze")

    norms = _index(norms_path)
    selected = _index(selected_path)
    if len(selected) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} selected rows, found {len(selected)}")
    selected_groups: dict[str, str] = {}
    identities = []
    for uid, row in selected.items():
        if uid not in norms:
            raise ValueError(f"selected UID absent from norm universe: {uid}")
        group = split_group_for(norms[uid])
        if str(row.get("source_group")) != group:
            raise ValueError(f"selected source group mismatch: {uid}")
        if group in selected_groups:
            raise ValueError(f"duplicate selected source group: {uid}/{selected_groups[group]}")
        selected_groups[group] = uid
        identities.append(
            {
                "norm_uid": uid,
                "source_group": group,
                "role": "new_humor_select_expansion",
                "permanently_excluded_from_training_gradients": True,
            }
        )

    panel_reports: dict[str, Any] = {}
    union_uids: set[str] = set()
    union_groups: set[str] = set()
    for path in exclusion_paths:
        panel = _index(path)
        missing = sorted(set(panel) - set(norms))
        if missing:
            raise ValueError(f"excluded UIDs absent from norms in {path}: {missing[:3]}")
        groups = {split_group_for(norms[uid]) for uid in panel}
        uid_overlap = set(selected) & set(panel)
        group_overlap = set(selected_groups) & groups
        panel_reports[str(path)] = {
            "sha256": sha256_file(path),
            "uids": len(panel),
            "source_groups": len(groups),
            "selected_uid_overlap": len(uid_overlap),
            "selected_source_group_overlap": len(group_overlap),
        }
        if uid_overlap or group_overlap:
            raise ValueError(f"selected rows overlap exclusion panel: {path}")
        union_uids.update(panel)
        union_groups.update(groups)

    identities.sort(key=lambda row: (row["source_group"], row["norm_uid"]))
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(identity_path, identities)
    report = {
        "schema_version": "silver-match-v3-frozen-select-expansion-audit-v1",
        "status": "SEALED_BEFORE_R4_PREDICTIONS_OR_LABELS",
        "task": freeze.get("task"),
        "selection_seed": freeze.get("selection_seed"),
        "selected": {
            "path": str(selected_path),
            "sha256": sha256_file(selected_path),
            "count": len(selected),
            "unique_source_groups": len(selected_groups),
            "identity_path": str(identity_path),
            "identity_sha256": sha256_file(identity_path),
            "permanently_excluded_from_training_gradients": True,
        },
        "exclusion_union": {
            "uids": len(union_uids),
            "source_groups": len(union_groups),
            "selected_uid_overlap": 0,
            "selected_source_group_overlap": 0,
        },
        "panels": panel_reports,
        "inputs": {
            "freeze": {"path": str(freeze_path), "sha256": sha256_file(freeze_path)},
            "norms": {"path": str(norms_path), "sha256": sha256_file(norms_path)},
        },
        "content_contract": {
            "selected_using_labels": False,
            "selected_using_adjudicator_outputs": False,
            "old_select_or_permanent_blind_content_exposed_by_report": False,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "identity_sha256": sha256_file(identity_path),
                "selected_count": len(selected),
                "selected_source_groups": len(selected_groups),
                "uid_overlap": 0,
                "source_group_overlap": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
