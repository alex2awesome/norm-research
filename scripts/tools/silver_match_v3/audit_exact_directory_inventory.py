#!/usr/bin/env python3
"""Require an exact, pycache-free recursive match to a frozen directory inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .common import sha256_file


def assert_exact_inventory(inventory_path: Path) -> dict[str, Any]:
    inventory_path = inventory_path.resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    root = Path(str(inventory.get("root") or "")).resolve()
    recorded = list(inventory.get("files") or [])
    if (
        inventory.get("schema_version")
        != "silver-match-v3-directory-content-inventory-v1"
        or inventory.get("status") != "FROZEN_CONTENT_HASH_INVENTORY"
        or not root.is_dir()
        or not recorded
    ):
        raise ValueError("directory inventory is absent or invalid")
    recorded_by_path = {str(row.get("relative_path") or ""): row for row in recorded}
    if "" in recorded_by_path or len(recorded_by_path) != len(recorded):
        raise ValueError("directory inventory has missing or duplicate relative paths")
    forbidden = [
        relative
        for relative in recorded_by_path
        if "__pycache__" in Path(relative).parts or relative.endswith((".pyc", ".pyo"))
    ]
    if forbidden:
        raise ValueError(f"inventory itself contains forbidden bytecode artifacts: {forbidden[:3]}")
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    actual_relatives = [str(path.relative_to(root)) for path in paths]
    if set(actual_relatives) != set(recorded_by_path):
        missing = sorted(set(recorded_by_path) - set(actual_relatives))
        extra = sorted(set(actual_relatives) - set(recorded_by_path))
        raise ValueError(f"recursive snapshot universe drift: missing={missing[:3]} extra={extra[:3]}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        relative = str(path.relative_to(root))
        row = {
            "relative_path": relative,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        recorded_row = recorded_by_path[relative]
        if (
            row["sha256"] != recorded_row.get("sha256")
            or row["size_bytes"] != int(recorded_row.get("size_bytes", -1))
        ):
            raise ValueError(f"snapshot file hash/size drift: {relative}")
        rows.append(row)
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["relative_path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["size_bytes"]).encode("ascii"))
        digest.update(b"\n")
    if (
        len(rows) != int(inventory.get("file_count", -1))
        or sum(int(row["size_bytes"]) for row in rows)
        != int(inventory.get("total_size_bytes", -1))
        or digest.hexdigest() != inventory.get("content_inventory_sha256")
    ):
        raise ValueError("recursive snapshot inventory aggregate drift")
    return {
        "schema_version": "silver-match-v3-exact-directory-inventory-audit-v1",
        "status": "EXACT_RECURSIVE_PYCACHE_FREE_INVENTORY_PASS",
        "inventory": {"path": str(inventory_path), "sha256": sha256_file(inventory_path)},
        "root": str(root),
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "content_inventory_sha256": digest.hexdigest(),
        "unexpected_files": 0,
        "missing_files": 0,
        "hash_or_size_drifted_files": 0,
        "pycache_or_bytecode_files": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = assert_exact_inventory(Path(args.inventory))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**result, "audit_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
