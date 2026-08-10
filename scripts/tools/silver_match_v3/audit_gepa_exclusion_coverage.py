#!/usr/bin/env python3
"""Audit that prior nonsealed GEPA derivatives are covered by an identity union."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


def _resolve(path: str, anchor: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (anchor.parent / value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--union", required=True)
    parser.add_argument("--scan-root", action="append", required=True)
    parser.add_argument(
        "--skip-path-regex",
        default=r"(?:^|[/_.-])test(?:[/_.-]|$)|(?:^|[/_.-])key(?:[/_.-]|$)|completed|/shards/",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path, union_path, output = (
        Path(args.manifest).resolve(), Path(args.union).resolve(), Path(args.output).resolve()
    )
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical: set[str] = set()
    for meta in (manifest.get("corpora") or {}).values():
        if meta.get("task") != args.task:
            continue
        canonical.update(str(row["norm_uid"]) for row in read_jsonl(_resolve(str(meta["path"]), manifest_path)))
    union = {str(row["norm_uid"]) for row in read_jsonl(union_path)}
    if not union or not union.issubset(canonical):
        raise ValueError("exclusion union is empty or outside canonical task norms")
    skip = re.compile(args.skip_path_regex, re.IGNORECASE)
    files: dict[str, Any] = {}
    skipped = []
    uncovered: dict[str, Any] = {}
    observation_sum = 0
    for root_raw in args.scan_root:
        root = Path(root_raw).resolve()
        for path in sorted(root.rglob("*.jsonl")):
            if skip.search(str(path)):
                skipped.append(str(path))
                continue
            task_uids: set[str] = set()
            for row in read_jsonl(path):
                uid = str(row.get("norm_uid") or "")
                if uid in canonical:
                    task_uids.add(uid)
            if not task_uids:
                continue
            missing = task_uids - union
            observation_sum += len(task_uids)
            files[str(path)] = {
                "sha256": sha256_file(path),
                "task_uids": len(task_uids),
                "uncovered_task_uids": len(missing),
            }
            if missing:
                uncovered[str(path)] = {
                    "count": len(missing),
                    "sample_sha256": __import__("hashlib").sha256("\n".join(sorted(missing)).encode()).hexdigest(),
                }
    report = {
        "schema_version": "silver-match-v3-gepa-exclusion-coverage-audit-v1",
        "status": "PASS" if not uncovered else "FAIL_UNCOVERED_PRIOR_EXPOSURES",
        "task": args.task,
        "canonical_uids": len(canonical),
        "union_uids": len(union),
        "derivative_files_scanned": len(files),
        "task_uid_observation_sum": observation_sum,
        "files_with_uncovered_task_uids": uncovered,
        "skip_path_regex": args.skip_path_regex,
        "skipped_path_count": len(skipped),
        "skipped_paths": skipped,
        "files": files,
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "union": {"path": str(union_path), "sha256": sha256_file(union_path)},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "output_sha256": sha256_file(output)}, sort_keys=True))
    if uncovered:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
