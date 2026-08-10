#!/usr/bin/env python3
"""Freeze a source-group-safe GEPA train/dev split from train-only labels.

This splitter is intentionally narrower than the corpus calibration splitter:
it may only subdivide source groups that already belong to the predeclared
``train`` universe.  It therefore cannot turn external dev/test labels into
prompt-development examples.  The output's ``split`` field is the local GEPA
role; ``predeclared_split`` records the immutable upstream role.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .make_calibration import split_for, split_group_for


def gepa_split(group: str, *, seed: int, dev_percent: int) -> str:
    if not 1 <= dev_percent <= 50:
        raise ValueError("dev_percent must be in [1, 50]")
    value = int(
        hashlib.sha256(f"gepa-train-only-v1\x1f{seed}\x1f{group}".encode()).hexdigest()[:16],
        16,
    ) % 100
    return "dev" if value < dev_percent else "train"


def split_panel(
    labels: list[dict[str, Any]],
    norms: dict[str, dict[str, Any]],
    *,
    task: str,
    seed: int,
    dev_percent: int,
    excluded_groups: set[str] | None = None,
    minimum_train: int = 1,
    minimum_dev: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if minimum_train < 1 or minimum_dev < 1:
        raise ValueError("minimum train/dev support must be positive")
    excluded_groups = excluded_groups or set()
    selected = [row for row in labels if row.get("task") == task]
    if not selected:
        raise ValueError(f"no labels for task {task}")
    uids = [str(row.get("norm_uid") or "") for row in selected]
    if "" in uids or len(uids) != len(set(uids)):
        raise ValueError("labels have missing/duplicate norm_uid")

    group_roles: dict[str, str] = {}
    output = []
    excluded_uids: list[str] = []
    for row in sorted(selected, key=lambda value: str(value["norm_uid"])):
        uid = str(row["norm_uid"])
        norm = norms.get(uid)
        if norm is None:
            raise ValueError(f"label UID absent from canonical task norms: {uid}")
        group = split_group_for(norm)
        upstream = split_for(group)
        if upstream != "train":
            raise ValueError(
                f"GEPA panel contains non-train source group: {uid} -> {upstream}"
            )
        # Exclusions are source-group wide.  Dropping just a labeled row would
        # allow another quote/review from the same document to leak a sealed
        # audit or prior selection group into prompt development.
        if group in excluded_groups:
            excluded_uids.append(uid)
            continue
        role = group_roles.setdefault(
            group, gepa_split(group, seed=seed, dev_percent=dev_percent)
        )
        output.append(
            {
                **row,
                "predeclared_split": upstream,
                "split": role,
                "gepa_split_group": group,
                "gepa_split_seed": seed,
                "gepa_dev_percent": dev_percent,
            }
        )
    counts = Counter(str(row["split"]) for row in output)
    if counts["train"] < minimum_train or counts["dev"] < minimum_dev:
        raise ValueError(
            "GEPA split lacks predeclared minimum support: "
            f"counts={dict(counts)} minimum_train={minimum_train} "
            f"minimum_dev={minimum_dev}"
        )
    groups_by_role = {
        role: {row["gepa_split_group"] for row in output if row["split"] == role}
        for role in ("train", "dev")
    }
    if groups_by_role["train"] & groups_by_role["dev"]:
        raise AssertionError("source group crossed GEPA train/dev roles")
    report = {
        "task": task,
        "seed": seed,
        "dev_percent": dev_percent,
        "count": len(output),
        "split_counts": dict(sorted(counts.items())),
        "source_group_counts": {
            role: len(groups_by_role[role]) for role in ("train", "dev")
        },
        "source_group_overlap": 0,
        "upstream_role": "train_only",
        "minimum_support": {"train": minimum_train, "dev": minimum_dev},
        "excluded_source_groups": len(excluded_groups),
        "excluded_labeled_uids": len(excluded_uids),
        "selected_source_group_overlap_with_exclusions": 0,
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=910247)
    parser.add_argument("--dev-percent", type=int, default=25)
    parser.add_argument(
        "--exclude-reference",
        action="append",
        default=[],
        help=(
            "JSONL containing permanently excluded norm_uid values; every "
            "canonical source group represented by a UID is excluded"
        ),
    )
    parser.add_argument(
        "--require-exclusions",
        action="store_true",
        help="fail closed unless at least one exclusion reference is supplied",
    )
    parser.add_argument("--minimum-train", type=int, default=1)
    parser.add_argument("--minimum-dev", type=int, default=1)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    label_path = Path(args.labels).resolve()
    output_path = Path(args.output).resolve()
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    norms: dict[str, dict[str, Any]] = {}
    for corpus, meta in manifest["corpora"].items():
        if meta.get("task") != args.task:
            continue
        for row in read_jsonl(Path(meta["path"])):
            uid = str(row["norm_uid"])
            if uid in norms:
                raise ValueError(f"duplicate canonical task norm UID: {uid}")
            norms[uid] = row
    exclusion_paths = [Path(value).resolve() for value in args.exclude_reference]
    if args.require_exclusions and not exclusion_paths:
        raise ValueError("--require-exclusions needs at least one --exclude-reference")
    excluded_uids: set[str] = set()
    exclusion_counts: dict[str, int] = {}
    for path in exclusion_paths:
        rows = list(read_jsonl(path))
        path_uids = {str(row.get("norm_uid") or "") for row in rows}
        if "" in path_uids:
            raise ValueError(f"missing norm_uid in exclusion reference: {path}")
        missing = sorted(path_uids - set(norms))
        if missing:
            raise ValueError(
                f"exclusion reference contains UIDs outside task {args.task}: "
                f"{path}: {missing[:3]}"
            )
        excluded_uids.update(path_uids)
        exclusion_counts[str(path)] = len(path_uids)
    excluded_groups = {
        split_group_for(norms[uid]) for uid in excluded_uids
    }
    rows, report = split_panel(
        list(read_jsonl(label_path)),
        norms,
        task=args.task,
        seed=args.seed,
        dev_percent=args.dev_percent,
        excluded_groups=excluded_groups,
        minimum_train=args.minimum_train,
        minimum_dev=args.minimum_dev,
    )
    write_jsonl(output_path, rows)
    meta = {
        "schema_version": "silver-match-v3-train-only-gepa-panel-v1",
        "report": report,
        "input_hashes": {
            "manifest": sha256_file(manifest_path),
            "labels": sha256_file(label_path),
            "exclusion_references": {
                str(path): sha256_file(path) for path in exclusion_paths
            },
        },
        "exclusion_counts": exclusion_counts,
        "excluded_uid_count": len(excluded_uids),
        "excluded_source_group_count": len(excluded_groups),
        "output_sha256": sha256_file(output_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
