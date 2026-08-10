#!/usr/bin/env python3
"""Materialize immutable train/audit subpacks from a predeclared role file."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _index(rows: list[dict[str, Any]], *, source: Path) -> dict[str, dict[str, Any]]:
    output = {str(row["norm_uid"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate norm_uid values: {source}")
    return output


def _parse_role_map(values: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--role-map must be VALUE=NAME: {value!r}")
        role, name = value.split("=", 1)
        if not role or not name or role in output or not SAFE_NAME.fullmatch(name):
            raise ValueError(f"invalid/duplicate --role-map: {value!r}")
        output[role] = name
    if not output:
        raise ValueError("at least one --role-map is required")
    if len(set(output.values())) != len(output):
        raise ValueError("role-map output names must be unique")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-role-field", default="split")
    parser.add_argument(
        "--role-map",
        action="append",
        default=[],
        metavar="VALUE=NAME",
        help="Map a reference field value to an output subdirectory; repeat per role.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--chunk-size", type=int, default=25)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    pack_root = Path(args.pack_root).resolve()
    reference_path = Path(args.reference).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite partitioned pack: {output_root}")
    role_map = _parse_role_map(args.role_map)

    validation_path = pack_root / "validation.json"
    source_validation = json.loads(validation_path.read_text(encoding="utf-8"))
    source_items_path, source_bank_path = (
        pack_root / "items.jsonl",
        pack_root / "bank.json",
    )
    if sha256_file(source_items_path) != source_validation["outputs"]["items"]["sha256"]:
        raise ValueError("source items hash mismatch")
    if sha256_file(source_bank_path) != source_validation["outputs"]["bank"]["sha256"]:
        raise ValueError("source bank hash mismatch")

    source_items = list(read_jsonl(source_items_path))
    reference_rows = list(read_jsonl(reference_path))
    source_by_uid = _index(source_items, source=source_items_path)
    reference_by_uid = _index(reference_rows, source=reference_path)
    if set(source_by_uid) != set(reference_by_uid):
        missing = sorted(set(source_by_uid) - set(reference_by_uid))
        extra = sorted(set(reference_by_uid) - set(source_by_uid))
        raise ValueError(
            "predeclared role file must cover the source pack exactly; "
            f"missing={missing[:3]} extra={extra[:3]}"
        )

    task = str(source_validation["task"])
    bank_hash = str(source_validation["bank_source_sha256"])
    bank = json.loads(source_bank_path.read_text(encoding="utf-8"))
    if bank.get("task") != task or bank.get("source_sha256") != bank_hash:
        raise ValueError("source bank identity mismatch")

    by_role: dict[str, list[dict[str, Any]]] = {role: [] for role in role_map}
    for source in source_items:
        uid = str(source["norm_uid"])
        reference = reference_by_uid[uid]
        for field in ("task", "corpus", "row", "split_group"):
            if source.get(field) != reference.get(field):
                raise ValueError(f"reference provenance mismatch for {field}: {uid}")
        if reference.get("predeclared_split") not in (None, source.get("split")):
            raise ValueError(f"reference upstream split mismatch: {uid}")
        role = str(reference.get(args.reference_role_field) or "")
        if role not in role_map:
            raise ValueError(f"unmapped reference role {role!r}: {uid}")
        by_role[role].append(
            {
                **source,
                "teacher_partition_role": role_map[role],
                "teacher_partition_reference_field": args.reference_role_field,
                "teacher_partition_reference_value": role,
                "teacher_partition_reference_sha256": sha256_file(reference_path),
                "canonical_predeclared_split": source.get("split"),
            }
        )

    role_groups = {
        role: {str(row["split_group"]) for row in rows}
        for role, rows in by_role.items()
    }
    for left_index, left in enumerate(sorted(role_groups)):
        for right in sorted(role_groups)[left_index + 1 :]:
            overlap = role_groups[left] & role_groups[right]
            if overlap:
                raise ValueError(
                    f"predeclared roles overlap by source group: {left}/{right}: "
                    f"{sorted(overlap)[:3]}"
                )

    output_root.mkdir(parents=True, exist_ok=True)
    role_reports: dict[str, dict[str, Any]] = {}
    for role, rows in by_role.items():
        if not rows:
            raise ValueError(f"predeclared role is empty: {role}")
        name = role_map[role]
        root = output_root / name
        root.mkdir(parents=True, exist_ok=False)
        items_path, bank_path = root / "items.jsonl", root / "bank.json"
        write_jsonl(items_path, rows)
        bank_path.write_text(
            json.dumps(bank, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        chunks = []
        for start in range(0, len(rows), args.chunk_size):
            path = root / "chunks" / f"part-{start // args.chunk_size:03d}.jsonl"
            write_jsonl(path, rows[start : start + args.chunk_size])
            chunks.append(path)
        role_validation = {
            "schema_version": "silver-match-v3-predeclared-pack-partition-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task": task,
            "count": len(rows),
            "selected_source_groups": len(role_groups[role]),
            "selected_by_corpus": dict(
                sorted(Counter(str(row["corpus"]) for row in rows).items())
            ),
            "chunk_size": args.chunk_size,
            "chunk_count": len(chunks),
            "bank_source_sha256": bank_hash,
            "partition_role": name,
            "reference_role_field": args.reference_role_field,
            "reference_role_value": role,
            "canonical_predeclared_split": sorted(
                {str(row.get("canonical_predeclared_split")) for row in rows}
            ),
            "permanently_excluded_from_gradients": name in {"audit", "blind_audit"},
            "inputs": {
                "source_pack_validation": {
                    "path": str(validation_path),
                    "sha256": sha256_file(validation_path),
                },
                "reference": {
                    "path": str(reference_path),
                    "sha256": sha256_file(reference_path),
                },
            },
            "outputs": {
                "items": {"path": str(items_path), "sha256": sha256_file(items_path)},
                "bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
                "chunks": {str(path): sha256_file(path) for path in chunks},
            },
        }
        role_validation_path = root / "validation.json"
        role_validation_path.write_text(
            json.dumps(role_validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        role_reports[name] = {
            "reference_value": role,
            "count": len(rows),
            "source_groups": len(role_groups[role]),
            "validation": str(role_validation_path),
            "validation_sha256": sha256_file(role_validation_path),
        }

    report = {
        "schema_version": "silver-match-v3-predeclared-pack-partitions-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "source_count": len(source_items),
        "partition_count": sum(report["count"] for report in role_reports.values()),
        "source_group_overlap_across_roles": 0,
        "reference_role_field": args.reference_role_field,
        "role_map": role_map,
        "inputs": {
            "source_pack_validation": {
                "path": str(validation_path),
                "sha256": sha256_file(validation_path),
            },
            "reference": {
                "path": str(reference_path),
                "sha256": sha256_file(reference_path),
            },
        },
        "roles": role_reports,
    }
    report_path = output_root / "validation.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
