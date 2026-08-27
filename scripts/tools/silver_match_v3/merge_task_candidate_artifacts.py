#!/usr/bin/env python3
"""Merge per-corpus K-candidate artifacts into one validated task universe.

Rows are copied byte-for-byte after validation so the merged artifact does not
silently normalize scores, ranks, or floating-point representations.  The
companion manifest binds every source, the merged bytes, the current metric
bank, UID uniqueness, and corpus coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import sha256_file


def _ref(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def merge(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    sources = [Path(value).resolve() for value in args.input]
    output = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite candidate merge outputs")
    temporary = output.parent / f".{output.name}.partial"
    if temporary.exists():
        raise FileExistsError(f"stale candidate merge temporary exists: {temporary}")
    if len(sources) < 2 or len(sources) != len(set(sources)):
        raise ValueError("candidate merge requires at least two distinct inputs")
    if any(not path.is_file() for path in sources):
        raise FileNotFoundError("one or more candidate inputs are missing")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank_hash = str(
        ((manifest.get("banks") or {}).get(args.task) or {}).get("source_sha256")
        or ""
    )
    if not bank_hash:
        raise ValueError(f"manifest lacks bank identity for task {args.task}")
    expected_corpora = {
        str(corpus)
        for corpus, meta in (manifest.get("corpora") or {}).items()
        if meta.get("task") == args.task
    }
    if not expected_corpora:
        raise ValueError(f"manifest contains no corpora for task {args.task}")

    seen_uids: set[str] = set()
    counts: Counter[str] = Counter()
    source_reports: list[dict[str, Any]] = []
    output_hash = hashlib.sha256()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as merged:
            for source in sources:
                source_hash = hashlib.sha256()
                source_count = 0
                source_corpora: Counter[str] = Counter()
                with source.open("rb") as handle:
                    for line_number, raw in enumerate(handle, 1):
                        source_hash.update(raw)
                        if not raw.strip():
                            raise ValueError(f"blank candidate row: {source}:{line_number}")
                        if not raw.endswith(b"\n"):
                            raise ValueError(
                                f"candidate row lacks terminal newline: {source}:{line_number}"
                            )
                        try:
                            row = json.loads(raw)
                        except json.JSONDecodeError as error:
                            raise ValueError(
                                f"invalid candidate JSON: {source}:{line_number}: {error}"
                            ) from error
                        uid = str(row.get("norm_uid") or "")
                        corpus = str(row.get("corpus") or "")
                        candidates = list(row.get("candidates") or [])
                        metric_ids = [
                            str(value.get("metric_id") or "") for value in candidates
                        ]
                        if not uid or uid in seen_uids:
                            raise ValueError(f"missing/duplicate candidate UID: {uid!r}")
                        if row.get("task") != args.task:
                            raise ValueError(f"candidate task mismatch: {uid}")
                        if corpus not in expected_corpora:
                            raise ValueError(
                                f"candidate corpus absent from manifest: {uid}/{corpus}"
                            )
                        if str(row.get("bank_source_sha256") or "") != bank_hash:
                            raise ValueError(f"candidate bank identity mismatch: {uid}")
                        if len(candidates) < args.minimum_k:
                            raise ValueError(
                                f"candidate row shorter than K={args.minimum_k}: {uid}"
                            )
                        if "" in metric_ids or len(metric_ids) != len(set(metric_ids)):
                            raise ValueError(
                                f"candidate row has missing/duplicate metric IDs: {uid}"
                            )
                        seen_uids.add(uid)
                        counts[corpus] += 1
                        source_corpora[corpus] += 1
                        source_count += 1
                        merged.write(raw)
                        output_hash.update(raw)
                source_reports.append(
                    {
                        "path": str(source),
                        "sha256": source_hash.hexdigest(),
                        "bytes": source.stat().st_size,
                        "rows": source_count,
                        "by_corpus": dict(sorted(source_corpora.items())),
                    }
                )

        observed_corpora = set(counts)
        if args.require_all_task_corpora and observed_corpora != expected_corpora:
            missing = sorted(expected_corpora - observed_corpora)
            extra = sorted(observed_corpora - expected_corpora)
            raise ValueError(
                "candidate merge does not cover all task corpora: "
                f"missing={missing} extra={extra}"
            )
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if output_hash.hexdigest() != sha256_file(output):
        raise ValueError("merged candidate hash drift after write")

    report = {
        "schema_version": "silver-match-v3-task-candidate-merge-v1",
        "status": "FROZEN_VALIDATED_BYTE_PRESERVING_MERGE",
        "task": args.task,
        "minimum_k": args.minimum_k,
        "bank_source_sha256": bank_hash,
        "row_count": len(seen_uids),
        "unique_uid_count": len(seen_uids),
        "expected_task_corpora": sorted(expected_corpora),
        "observed_corpora": sorted(observed_corpora),
        "all_task_corpora_required": bool(args.require_all_task_corpora),
        "by_corpus": dict(sorted(counts.items())),
        "content_contract": {
            "source_rows_copied_byte_for_byte": True,
            "candidate_scores_used": False,
            "candidate_metric_ids_used_only_for_k_and_uniqueness_validation": True,
            "predictions_labels_truth_and_outcomes_used": False,
        },
        "inputs": {
            "manifest": _ref(manifest_path),
            "candidate_sources": source_reports,
        },
        "output": {
            "path": str(output),
            "sha256": output_hash.hexdigest(),
            "bytes": output.stat().st_size,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**report, "report_sha256": sha256_file(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--input", action="append", default=[], required=True)
    parser.add_argument("--minimum-k", type=int, default=50)
    parser.add_argument("--require-all-task-corpora", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    if args.minimum_k < 1:
        parser.error("--minimum-k must be positive")
    print(json.dumps(merge(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
