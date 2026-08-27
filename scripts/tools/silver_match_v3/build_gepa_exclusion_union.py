#!/usr/bin/env python3
"""Build a hash-bound identity union for all prior task-local GEPA exposures.

JSONL panels and newline UID files are parsed only for canonical norm identity
and source group.  Sealed tests/outcomes are accepted only as ``--hash-only``
artifacts: their bytes are hashed, but their structured content is never
loaded.  The output is suitable as one fail-closed ``--exclude-panel`` input
to :mod:`freeze_clean_gepa_panel`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .make_calibration import split_for, split_group_for


def _resolve(path: str, anchor: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (anchor.parent / value).resolve()


def _spec(raw: str) -> tuple[str, Path]:
    if "::" not in raw:
        raise ValueError("artifact specs must be CATEGORY::PATH")
    category, path = raw.split("::", 1)
    if not category.strip() or not path.strip():
        raise ValueError("artifact specs require nonempty category and path")
    return category.strip(), Path(path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--panel", action="append", default=[])
    parser.add_argument("--uid-file", action="append", default=[])
    parser.add_argument("--hash-only", action="append", default=[])
    parser.add_argument("--required-category", action="append", default=[])
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    if not args.panel and not args.uid_file:
        parser.error("at least one parsed panel or UID file is required")

    manifest_path = Path(args.manifest).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    norms: dict[str, dict[str, Any]] = {}
    for corpus, meta in sorted((manifest.get("corpora") or {}).items()):
        if meta.get("task") != args.task:
            continue
        for row in read_jsonl(_resolve(str(meta["path"]), manifest_path)):
            uid = str(row.get("norm_uid") or "")
            if not uid or uid in norms:
                raise ValueError(f"missing/duplicate canonical task UID: {uid!r}")
            norms[uid] = row
    if not norms:
        raise ValueError(f"manifest contains no norms for task {args.task}")

    sources: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {}
    categories: set[str] = set()

    def add(category: str, path: Path, uids: list[str], fmt: str) -> None:
        if str(path) in sources:
            raise ValueError(f"duplicate inventory source: {path}")
        if not uids or len(uids) != len(set(uids)):
            raise ValueError(f"empty or duplicate UIDs in {path}")
        missing = sorted(set(uids) - set(norms))
        if missing:
            raise ValueError(f"inventory source has noncanonical task UIDs: {missing[:3]}")
        groups = set()
        for uid in uids:
            norm = norms[uid]
            group = split_group_for(norm)
            groups.add(group)
            identities.setdefault(
                uid,
                {
                    "schema_version": "silver-match-v3-gepa-exclusion-identity-v1",
                    "norm_uid": uid,
                    "task": args.task,
                    "corpus": str(norm["corpus"]),
                    "source_group": group,
                    "upstream_split": split_for(group),
                },
            )
        categories.add(category)
        sources[str(path)] = {
            "category": category,
            "format": fmt,
            "sha256": sha256_file(path),
            "uids": len(uids),
            "source_groups": len(groups),
            "structured_content_parsed": True,
            "fields_used": ["norm_uid", "source_group"],
        }

    for raw in args.panel:
        category, path = _spec(raw)
        rows = list(read_jsonl(path))
        uids = [str(row.get("norm_uid") or "") for row in rows]
        add(category, path, uids, "jsonl")
        supplied_group_rows = supplied_group_mismatches = 0
        for row in rows:
            uid = str(row["norm_uid"])
            supplied = {
                str(row.get(key))
                for key in ("source_group", "split_group", "gepa_split_group")
                if row.get(key)
            }
            if supplied:
                supplied_group_rows += 1
                if split_group_for(norms[uid]) not in supplied:
                    supplied_group_mismatches += 1
        # Canonical UID membership is authoritative.  Historical teacher rows
        # can carry a stale unit-separator source_group alongside a correct
        # split_group.  Recompute every exclusion group from canonical norms and
        # preserve mismatch counts for audit instead of trusting legacy fields.
        sources[str(path)]["canonical_source_group_recomputed"] = True
        sources[str(path)]["rows_with_supplied_source_group"] = supplied_group_rows
        sources[str(path)]["supplied_source_group_mismatch_count"] = supplied_group_mismatches

    for raw in args.uid_file:
        category, path = _spec(raw)
        uids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        add(category, path, uids, "newline_delimited_uids")

    for raw in args.hash_only:
        category, path = _spec(raw)
        if str(path) in sources:
            raise ValueError(f"duplicate inventory source: {path}")
        categories.add(category)
        sources[str(path)] = {
            "category": category,
            "format": "hash_only",
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "structured_content_parsed": False,
            "fields_used": [],
        }

    missing_categories = sorted(set(args.required_category) - categories)
    if missing_categories:
        raise ValueError(f"required exclusion categories absent: {missing_categories}")

    rows = sorted(identities.values(), key=lambda row: (row["source_group"], row["norm_uid"]))
    groups = {str(row["source_group"]) for row in rows}
    output_root.mkdir(parents=True, exist_ok=False)
    identity_path = output_root / "identities.jsonl"
    write_jsonl(identity_path, rows)
    report = {
        "schema_version": "silver-match-v3-gepa-exclusion-union-v1",
        "status": "FROZEN_BEFORE_NEW_PANEL_SELECTION_PREDICTIONS_OR_LABELS",
        "task": args.task,
        "required_categories": sorted(set(args.required_category)),
        "observed_categories": sorted(categories),
        "all_required_categories_present": not missing_categories,
        "identity_union": {
            "uids": len(rows),
            "source_groups": len(groups),
            "by_upstream_split": dict(sorted(Counter(row["upstream_split"] for row in rows).items())),
            "by_corpus": dict(sorted(Counter(row["corpus"] for row in rows).items())),
            "path": str(identity_path),
            "sha256": sha256_file(identity_path),
        },
        "sources": dict(sorted(sources.items())),
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
        },
        "content_contract": {
            "parsed_sources_used_only_identity_fields": True,
            "sealed_test_or_outcome_structured_content_parsed": False,
            "model_predictions_metric_ids_reasons_and_outcomes_used": False,
        },
    }
    report_path = output_root / "EXCLUSION_INVENTORY.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report_sha256": sha256_file(report_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
