#!/usr/bin/env python3
"""Project complete independent labels onto one immutable partitioned subpack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


def _index(rows: list[dict[str, Any]], *, source: Path) -> dict[str, dict[str, Any]]:
    output = {str(row["norm_uid"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate norm_uid values: {source}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    labels_path = Path(args.labels).resolve()
    pack_root = Path(args.pack_root).resolve()
    output_path = Path(args.output).resolve()
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError(f"refusing to overwrite label subset: {output_path}")

    validation_path = pack_root / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("schema_version") != "silver-match-v3-predeclared-pack-partition-v1":
        raise ValueError("pack is not an immutable predeclared partition")
    items_path = pack_root / "items.jsonl"
    if sha256_file(items_path) != validation["outputs"]["items"]["sha256"]:
        raise ValueError("partition items hash mismatch")
    items = list(read_jsonl(items_path))
    labels = list(read_jsonl(labels_path))
    item_by_uid = _index(items, source=items_path)
    label_by_uid = _index(labels, source=labels_path)
    if not set(item_by_uid).issubset(label_by_uid):
        missing = sorted(set(item_by_uid) - set(label_by_uid))
        raise ValueError(f"labels do not cover partition: {missing[:3]}")

    task = str(validation["task"])
    bank_hash = str(validation["bank_source_sha256"])
    role = str(validation["partition_role"])
    permanently_excluded = bool(validation["permanently_excluded_from_gradients"])
    output = []
    for item in items:
        uid = str(item["norm_uid"])
        label = label_by_uid[uid]
        if label.get("task") != task or label.get("current_bank_source_sha256") != bank_hash:
            raise ValueError(f"label task/bank mismatch: {uid}")
        for field in ("corpus", "row", "split_group"):
            if label.get(field) != item.get(field):
                raise ValueError(f"label provenance mismatch for {field}: {uid}")
        output.append(
            {
                **label,
                "teacher_partition_role": role,
                "teacher_partition_validation_sha256": sha256_file(validation_path),
                "training_eligible": False,
                "training_blocked_pending_blind_audit": not permanently_excluded,
                "audit_permanently_excluded_from_gradients": permanently_excluded,
            }
        )
    write_jsonl(output_path, output)
    report = {
        "schema_version": "silver-match-v3-partitioned-label-subset-v1",
        "task": task,
        "partition_role": role,
        "count": len(output),
        "unique_source_groups": len({str(row["split_group"]) for row in output}),
        "permanently_excluded_from_gradients": permanently_excluded,
        "inputs": {
            "labels": {"path": str(labels_path), "sha256": sha256_file(labels_path)},
            "pack_validation": {
                "path": str(validation_path),
                "sha256": sha256_file(validation_path),
            },
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
