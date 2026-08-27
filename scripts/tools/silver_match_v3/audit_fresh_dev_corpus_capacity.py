#!/usr/bin/env python3
"""Audit leakage-clean fresh-panel capacity by corpus and source group.

The audit reads only canonical identities, corpus names, source groups, and an
authoritative upstream role.  Historical artifacts may carry stale source-
group strings; their UIDs are therefore projected through the frozen manifest
before overlap is computed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file
from .make_calibration import split_for, split_group_for


def _resolve(value: str, anchor: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (anchor.parent / path).resolve()


def _spec(value: str) -> tuple[str, Path]:
    if "::" not in value:
        raise ValueError("exclusion specs must be NAME::PATH")
    name, path = value.split("::", 1)
    if not name.strip() or not path.strip():
        raise ValueError("exclusion specs require a nonempty name and path")
    return name.strip(), Path(path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--eligible-reference")
    parser.add_argument("--upstream-role-reference")
    parser.add_argument("--upstream-role-field", default="split")
    parser.add_argument("--required-upstream-role", default="train")
    parser.add_argument("--exclude-panel", action="append", default=[])
    parser.add_argument("--requested-count", type=int, required=True)
    parser.add_argument("--minimum-per-corpus", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.requested_count < 1 or args.minimum_per_corpus < 0:
        parser.error("requested count must be positive and minimum nonnegative")
    if not args.exclude_panel:
        parser.error("at least one --exclude-panel is required")

    manifest_path = Path(args.manifest).resolve()
    eligible_path = (
        Path(args.eligible_reference).resolve() if args.eligible_reference else None
    )
    role_path = (
        Path(args.upstream_role_reference).resolve()
        if args.upstream_role_reference
        else None
    )
    if (eligible_path is None) != (role_path is None):
        parser.error(
            "--eligible-reference and --upstream-role-reference must be supplied together"
        )
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    norms: dict[str, dict[str, Any]] = {}
    manifest_groups: dict[str, set[str]] = defaultdict(set)
    for corpus, meta in sorted((manifest.get("corpora") or {}).items()):
        if meta.get("task") != args.task:
            continue
        for row in read_jsonl(_resolve(str(meta["path"]), manifest_path)):
            uid = str(row.get("norm_uid") or "")
            if not uid or uid in norms:
                raise ValueError(f"missing or duplicate canonical UID: {uid!r}")
            if str(row.get("task")) != args.task or str(row.get("corpus")) != corpus:
                raise ValueError(f"canonical task/corpus mismatch: {uid}")
            norms[uid] = row
            manifest_groups[str(corpus)].add(split_group_for(row))
    if not norms:
        raise ValueError(f"manifest has no norms for task {args.task}")

    roles_by_uid: dict[str, str] = {}
    roles_by_group: dict[str, str] = {}
    if role_path is not None:
        role_rows = list(read_jsonl(role_path))
        for row in role_rows:
            uid = str(row.get("norm_uid") or "")
            if uid not in norms or row.get("task") != args.task:
                raise ValueError(f"role reference has noncanonical task UID: {uid}")
            role = str(row.get(args.upstream_role_field) or "")
            if not role:
                raise ValueError(
                    f"role reference is missing {args.upstream_role_field}: {uid}"
                )
            group = split_group_for(norms[uid])
            previous = roles_by_group.setdefault(group, role)
            if previous != role:
                raise ValueError(f"canonical group crosses upstream roles: {group}")
            roles_by_uid[uid] = role

        assert eligible_path is not None
        eligible_rows = list(read_jsonl(eligible_path))
        eligible_uids = [str(row.get("norm_uid") or "") for row in eligible_rows]
        if (
            not eligible_rows
            or "" in eligible_uids
            or len(eligible_uids) != len(set(eligible_uids))
        ):
            raise ValueError("eligible reference has empty, missing, or duplicate UIDs")
        missing = sorted(set(eligible_uids) - set(norms))
        if missing:
            raise ValueError(f"eligible reference has noncanonical UIDs: {missing[:3]}")
        missing_roles = sorted(set(eligible_uids) - set(roles_by_uid))
        if missing_roles:
            raise ValueError(
                f"eligible reference lacks authoritative roles: {missing_roles[:3]}"
            )
    else:
        eligible_uids = list(norms)
        roles_by_uid = {
            uid: split_for(split_group_for(norm)) for uid, norm in norms.items()
        }

    eligible_groups: dict[str, set[str]] = defaultdict(set)
    eligible_uid_count: Counter[str] = Counter()
    for uid in eligible_uids:
        if roles_by_uid[uid] != args.required_upstream_role:
            continue
        corpus = str(norms[uid]["corpus"])
        eligible_groups[corpus].add(split_group_for(norms[uid]))
        eligible_uid_count[corpus] += 1

    excluded_union_uids: set[str] = set()
    excluded_union_groups: set[str] = set()
    sources: dict[str, dict[str, Any]] = {}
    for raw in args.exclude_panel:
        name, path = _spec(raw)
        rows = list(read_jsonl(path))
        uids = [str(row.get("norm_uid") or "") for row in rows]
        if not rows or "" in uids or len(uids) != len(set(uids)):
            raise ValueError(f"exclusion has empty, missing, or duplicate UIDs: {path}")
        unknown = sorted(set(uids) - set(norms))
        if unknown:
            raise ValueError(f"exclusion has noncanonical UIDs: {path}: {unknown[:3]}")
        groups = {split_group_for(norms[uid]) for uid in uids}
        supplied_rows = supplied_mismatches = 0
        by_corpus_groups: dict[str, set[str]] = defaultdict(set)
        for row, uid in zip(rows, uids):
            group = split_group_for(norms[uid])
            by_corpus_groups[str(norms[uid]["corpus"])].add(group)
            supplied = {
                str(row.get(key))
                for key in ("source_group", "split_group", "gepa_split_group")
                if row.get(key)
            }
            if supplied:
                supplied_rows += 1
                if group not in supplied:
                    supplied_mismatches += 1
        excluded_union_uids.update(uids)
        excluded_union_groups.update(groups)
        sources[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "uids": len(uids),
            "source_groups": len(groups),
            "source_groups_by_corpus": {
                corpus: len(values) for corpus, values in sorted(by_corpus_groups.items())
            },
            "canonical_source_group_recomputed": True,
            "rows_with_supplied_source_group": supplied_rows,
            "supplied_source_group_mismatch_count": supplied_mismatches,
        }

    corpora = sorted(manifest_groups)
    capacity: dict[str, dict[str, Any]] = {}
    remaining_union: set[str] = set()
    for corpus in corpora:
        groups = eligible_groups.get(corpus, set())
        excluded = groups & excluded_union_groups
        remaining = groups - excluded_union_groups
        remaining_union.update(remaining)
        capacity[corpus] = {
            "manifest_source_groups": len(manifest_groups[corpus]),
            "eligible_reference_uids_in_required_role": eligible_uid_count[corpus],
            "eligible_source_groups_in_required_role": len(groups),
            "excluded_eligible_source_groups": len(excluded),
            "remaining_eligible_source_groups": len(remaining),
            "meets_minimum_per_corpus": len(remaining) >= args.minimum_per_corpus,
        }
    minimum_ok = all(
        row["meets_minimum_per_corpus"] for row in capacity.values()
    )
    total_ok = len(remaining_union) >= args.requested_count
    report = {
        "schema_version": "silver-match-v3-fresh-dev-corpus-capacity-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "requested_count": args.requested_count,
        "minimum_per_corpus": args.minimum_per_corpus,
        "required_upstream_role": args.required_upstream_role,
        "status": "FEASIBLE" if minimum_ok and total_ok else "INFEASIBLE",
        "feasibility": {
            "minimum_per_corpus_met": minimum_ok,
            "total_count_met": total_ok,
            "remaining_eligible_source_groups_total": len(remaining_union),
        },
        "capacity_by_corpus": capacity,
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "eligible_reference": (
                {"path": str(eligible_path), "sha256": sha256_file(eligible_path)}
                if eligible_path is not None
                else {"path": None, "universe": "all canonical task norms"}
            ),
            "upstream_role_reference": (
                {
                    "path": str(role_path),
                    "sha256": sha256_file(role_path),
                    "field": args.upstream_role_field,
                }
                if role_path is not None
                else {
                    "path": None,
                    "fallback": "make_calibration.split_for(canonical_source_group)",
                }
            ),
            "exclusions": sources,
        },
        "exclusion_union": {
            "uids": len(excluded_union_uids),
            "canonical_source_groups": len(excluded_union_groups),
        },
        "content_contract": {
            "fields_read": ["norm_uid", "task", "corpus", "source identity", args.upstream_role_field],
            "truth_decisions_metric_ids_predictions_reasons_and_outcomes_read": False,
            "canonical_source_groups_recomputed_from_manifest": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "output_sha256": sha256_file(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
