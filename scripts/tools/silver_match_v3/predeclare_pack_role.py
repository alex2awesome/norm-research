#!/usr/bin/env python3
"""Lock every row in an immutable pack to one downstream train/audit role."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True)
    parser.add_argument("--role", choices=("training", "blind_audit"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pack_root = Path(args.pack_root).resolve()
    output = Path(args.output).resolve()
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    if output.exists() or meta_path.exists():
        raise FileExistsError(f"refusing to overwrite predeclared role lock: {output}")
    validation_path, items_path = pack_root / "validation.json", pack_root / "items.jsonl"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if sha256_file(items_path) != validation["outputs"]["items"]["sha256"]:
        raise ValueError("pack items hash mismatch")
    items = list(read_jsonl(items_path))
    uids = [str(row["norm_uid"]) for row in items]
    if len(uids) != len(set(uids)):
        raise ValueError("pack contains duplicate UIDs")
    rows = [
        {
            **row,
            "predeclared_split": row.get("split"),
            "teacher_partition_role": args.role,
            "audit_permanently_excluded_from_gradients": args.role == "blind_audit",
        }
        for row in items
    ]
    write_jsonl(output, rows)
    report = {
        "schema_version": "silver-match-v3-predeclared-pack-role-v1",
        "task": validation["task"],
        "role": args.role,
        "count": len(rows),
        "unique_source_groups": len({str(row["split_group"]) for row in rows}),
        "permanently_excluded_from_gradients": args.role == "blind_audit",
        "inputs": {
            "pack_validation": {
                "path": str(validation_path),
                "sha256": sha256_file(validation_path),
            }
        },
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
