#!/usr/bin/env python3
"""Hash-pin exact UID/source-group overlap between two labeled artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_jsonl, sha256_file


def identities(path: Path) -> tuple[set[str], set[str]]:
    uids: set[str] = set()
    groups: set[str] = set()
    for row in read_jsonl(path):
        uid = str(row.get("norm_uid") or "")
        group = str(row.get("source_group") or row.get("gepa_split_group") or "")
        if not uid or uid in uids or not group:
            raise ValueError(f"missing/duplicate UID or source group in {path}: {uid!r}")
        uids.add(uid)
        groups.add(group)
    if not uids:
        raise ValueError(f"empty labels: {path}")
    return uids, groups


def audit(left_path: Path, right_path: Path) -> dict:
    left_uids, left_groups = identities(left_path)
    right_uids, right_groups = identities(right_path)
    uid_overlap = left_uids & right_uids
    group_overlap = left_groups & right_groups
    return {
        "schema_version": "silver-match-v3-label-overlap-audit-v1",
        "left": {
            "path": str(left_path),
            "sha256": sha256_file(left_path),
            "uids": len(left_uids),
            "source_groups": len(left_groups),
        },
        "right": {
            "path": str(right_path),
            "sha256": sha256_file(right_path),
            "uids": len(right_uids),
            "source_groups": len(right_groups),
        },
        "overlap": {
            "uids": len(uid_overlap),
            "left_uid_fraction": len(uid_overlap) / len(left_uids),
            "source_groups": len(group_overlap),
            "left_source_group_fraction": len(group_overlap) / len(left_groups),
        },
        "left_only": {
            "uids": len(left_uids - right_uids),
            "source_groups": len(left_groups - right_groups),
        },
        "right_only": {
            "uids": len(right_uids - left_uids),
            "source_groups": len(right_groups - left_groups),
        },
        "left_adds_new_exact_labels": bool(left_uids - right_uids),
        "status": (
            "LEFT_ADDS_NEW_EXACT_LABELS"
            if left_uids - right_uids
            else "LEFT_FULLY_RECYCLED_FROM_RIGHT"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    left, right, output = map(
        lambda value: Path(value).resolve(), (args.left, args.right, args.output)
    )
    if output.exists():
        raise FileExistsError(output)
    report = audit(left, right)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**report, "output_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
