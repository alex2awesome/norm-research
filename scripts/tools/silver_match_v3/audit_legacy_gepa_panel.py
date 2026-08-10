#!/usr/bin/env python3
"""Audit a historical GEPA panel against authoritative upstream split identity.

The audit is deliberately identity-only.  It never reads metric decisions,
reasons, norm text, predictions, outcomes, or any sealed artifact.  Historical
``split``/``predeclared_split`` fields are compared with the upstream roles in
the supplied strong-label artifact.  Canonical source groups are still
recomputed from the manifest.  This distinction matters because retriever/LoRA
splits use a task training seed and are not the same as the older calibration
hash split.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file
from .make_calibration import split_for, split_group_for


def _resolve(path: str, anchor: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (anchor.parent / value).resolve()


def _unique_uids(rows: list[dict[str, Any]], *, name: str) -> set[str]:
    values = [str(row.get("norm_uid") or "") for row in rows]
    if not rows or "" in values or len(values) != len(set(values)):
        raise ValueError(f"{name} is empty or has missing/duplicate norm_uid values")
    return set(values)


def audit(
    *,
    manifest_path: Path,
    task: str,
    panel_path: Path,
    strong_labels_path: Path | None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical: dict[str, dict[str, Any]] = {}
    for corpus, meta in sorted((manifest.get("corpora") or {}).items()):
        if meta.get("task") != task:
            continue
        for row in read_jsonl(_resolve(str(meta["path"]), manifest_path)):
            uid = str(row.get("norm_uid") or "")
            if not uid or uid in canonical:
                raise ValueError(f"missing/duplicate canonical UID: {uid!r}")
            if row.get("task") != task or row.get("corpus") != corpus:
                raise ValueError(f"canonical task/corpus mismatch: {uid}")
            canonical[uid] = row
    if not canonical:
        raise ValueError(f"manifest has no canonical rows for {task}")

    panel = list(read_jsonl(panel_path))
    panel_uids = _unique_uids(panel, name="panel")
    unknown = sorted(panel_uids - set(canonical))
    if unknown:
        raise ValueError(f"panel contains noncanonical task UIDs: {unknown[:3]}")
    if any(row.get("task") != task for row in panel):
        raise ValueError("panel contains another task")

    strong: dict[str, Any] | None = None
    upstream_by_uid: dict[str, str]
    upstream_role_source: dict[str, Any]
    if strong_labels_path is not None:
        all_labels = list(read_jsonl(strong_labels_path))
        strong_rows = [
            row for row in all_labels if row.get("supervision_strength") == "strong"
        ]
        strong_uids = _unique_uids(strong_rows, name="strong-label subset")
        if any(row.get("task") != task for row in strong_rows):
            raise ValueError("strong-label subset contains another task")
        unknown_strong = sorted(strong_uids - set(canonical))
        if unknown_strong:
            raise ValueError(
                f"strong-label subset contains noncanonical task UIDs: {unknown_strong[:3]}"
            )
        if not panel_uids <= strong_uids:
            raise ValueError("strong-label subset does not cover every panel UID")
        role_by_uid = {
            str(row["norm_uid"]): str(row.get("split") or "") for row in strong_rows
        }
        invalid_roles = sorted(
            uid for uid, role in role_by_uid.items() if role not in {"train", "dev", "test"}
        )
        if invalid_roles:
            raise ValueError(f"strong labels have missing/invalid upstream roles: {invalid_roles[:3]}")
        upstream_by_uid = {uid: role_by_uid[uid] for uid in panel_uids}
        upstream_role_source = {
            "kind": "strong_label_explicit_split",
            "path": str(strong_labels_path),
            "sha256": sha256_file(strong_labels_path),
            "field": "split",
        }
        strong = {
            "path": str(strong_labels_path),
            "sha256": sha256_file(strong_labels_path),
            "count": len(strong_uids),
            "authoritative_upstream_split": dict(
                sorted(
                    Counter(role_by_uid[uid] for uid in strong_uids).items()
                )
            ),
            "panel_covers_every_strong_uid": strong_uids <= panel_uids,
            "panel_uid_set_equals_strong_uid_set": strong_uids == panel_uids,
            "strong_uids_missing_from_panel": len(strong_uids - panel_uids),
            "panel_uids_outside_strong_subset": len(panel_uids - strong_uids),
        }
    else:
        upstream_by_uid = {
            uid: split_for(split_group_for(canonical[uid])) for uid in panel_uids
        }
        upstream_role_source = {
            "kind": "fallback_make_calibration_hash_split",
            "field": None,
        }

    upstream_counts = Counter(upstream_by_uid.values())
    assigned_counts = Counter(str(row.get("split") or "missing") for row in panel)
    cross = Counter(
        (upstream_by_uid[str(row["norm_uid"])], str(row.get("split") or "missing"))
        for row in panel
    )
    false_train_declarations = sum(
        str(row.get("predeclared_split") or "") == "train"
        and upstream_by_uid[str(row["norm_uid"])] != "train"
        for row in panel
    )
    supplied_group_mismatches = 0
    assigned_groups: dict[str, set[str]] = {}
    for row in panel:
        uid = str(row["norm_uid"])
        canonical_group = split_group_for(canonical[uid])
        supplied = {
            str(row.get(field))
            for field in ("source_group", "split_group", "gepa_split_group")
            if row.get(field)
        }
        if supplied and canonical_group not in supplied:
            supplied_group_mismatches += 1
        assigned_groups.setdefault(str(row.get("split") or "missing"), set()).add(
            canonical_group
        )
    assigned_role_group_overlaps = {
        f"{left}__{right}": len(assigned_groups[left] & assigned_groups[right])
        for index, left in enumerate(sorted(assigned_groups))
        for right in sorted(assigned_groups)[index + 1 :]
    }

    nontrain = sum(value for key, value in upstream_counts.items() if key != "train")
    valid = nontrain == 0 and false_train_declarations == 0
    return {
        "schema_version": "silver-match-v3-legacy-gepa-panel-audit-v2",
        "status": "VALID_TRAIN_ONLY_GEPA" if valid else "INVALID_FOR_TRAIN_ONLY_GEPA",
        "task": task,
        "panel": {
            "path": str(panel_path),
            "sha256": sha256_file(panel_path),
            "count": len(panel),
            "canonical_source_groups": len(
                {split_group_for(canonical[uid]) for uid in panel_uids}
            ),
        },
        "authoritative_upstream_split": dict(sorted(upstream_counts.items())),
        "upstream_role_source": upstream_role_source,
        "historical_assigned_gepa_role": dict(sorted(assigned_counts.items())),
        "upstream_split_x_historical_role": {
            f"{upstream}__{role}": value
            for (upstream, role), value in sorted(cross.items())
        },
        "authoritative_nontrain_rows_in_panel": nontrain,
        "false_predeclared_train_rows": false_train_declarations,
        "supplied_source_group_mismatch_rows": supplied_group_mismatches,
        "historical_role_source_group_overlap": assigned_role_group_overlaps,
        "strong_label_universe": strong,
        "scientific_contract": {
            "may_reuse_panel_for_prompt_optimization": valid,
            "may_reuse_panel_for_prompt_selection": valid,
            "historical_outputs_remain_audit_evidence_only_if_invalid": not valid,
            "parsed_fields": [
                "norm_uid",
                "task",
                "source_group",
                "split_group",
                "gepa_split_group",
                "split",
                "predeclared_split",
                "supervision_strength",
            ],
            "metric_decisions_reasons_norm_text_predictions_outcomes_parsed": False,
            "sealed_artifacts_read": False,
        },
        "inputs": {
            "manifest": {
                "path": str(manifest_path),
                "sha256": sha256_file(manifest_path),
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--strong-labels")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit(
        manifest_path=Path(args.manifest).resolve(),
        task=args.task,
        panel_path=Path(args.panel).resolve(),
        strong_labels_path=(
            Path(args.strong_labels).resolve() if args.strong_labels else None
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
