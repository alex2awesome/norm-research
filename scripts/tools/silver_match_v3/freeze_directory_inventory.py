#!/usr/bin/env python3
"""Freeze a parallel content-hash inventory for a model/runtime directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import sha256_file


def _file_row(root: Path, path: Path) -> dict[str, object]:
    return {
        "relative_path": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def build_inventory(root: Path, workers: int) -> dict[str, object]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    if not paths:
        raise ValueError(f"directory has no files: {root}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda path: _file_row(root, path), paths))
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["relative_path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row["sha256"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(row["size_bytes"]).encode("ascii"))
        digest.update(b"\n")
    return {
        "schema_version": "silver-match-v3-directory-content-inventory-v1",
        "status": "FROZEN_CONTENT_HASH_INVENTORY",
        "root": str(root),
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "content_inventory_sha256": digest.hexdigest(),
        "files": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = build_inventory(Path(args.root), args.workers)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": sha256_file(output),
                "root": payload["root"],
                "file_count": payload["file_count"],
                "total_size_bytes": payload["total_size_bytes"],
                "content_inventory_sha256": payload["content_inventory_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
