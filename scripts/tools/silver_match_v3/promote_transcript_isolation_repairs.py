#!/usr/bin/env python3
"""Promote clean original chunks plus audited repairs into one label pass."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl
from .prepare_transcript_isolation_repair_pack import AUDIT_SCHEMA, REPAIR_SCHEMA


VALIDATION_SCHEMA = "silver-match-v3-independent-label-validation-v1"
PROMOTION_SCHEMA = "silver-match-v3-composite-transcript-clean-label-promotion-v1"


def _load_unique(path: Path, name: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = list(read_jsonl(path))
    by_uid = {str(row.get("norm_uid") or ""): row for row in rows}
    if "" in by_uid or len(by_uid) != len(rows):
        raise ValueError(f"{name} has missing or duplicate norm_uid values")
    return rows, by_uid


def _validation_binding(report_path: Path, labels_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != VALIDATION_SCHEMA
        or report.get("complete") is not True
        or (report.get("output") or {}).get("sha256") != sha256_file(labels_path)
    ):
        raise ValueError(f"label validation does not bind labels: {report_path}")
    return report


def promote(
    *,
    source_pack: Path,
    base_labels: Path,
    base_validation_path: Path,
    failed_audit_path: Path,
    repair_pack: Path,
    repair_labels: Path,
    repair_validation_path: Path,
    repair_audit_path: Path,
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    paths = [
        source_pack,
        base_labels,
        base_validation_path,
        failed_audit_path,
        repair_pack,
        repair_labels,
        repair_validation_path,
        repair_audit_path,
        output,
        report_path,
    ]
    (
        source_pack,
        base_labels,
        base_validation_path,
        failed_audit_path,
        repair_pack,
        repair_labels,
        repair_validation_path,
        repair_audit_path,
        output,
        report_path,
    ) = [path.resolve() for path in paths]
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite transcript-clean promotion")

    source_validation_path = source_pack / "validation.json"
    source_validation = json.loads(source_validation_path.read_text(encoding="utf-8"))
    source_bank = source_pack / "bank.json"
    source_items = source_pack / "items.jsonl"
    if (
        sha256_file(source_bank) != source_validation["outputs"]["bank"]["sha256"]
        or sha256_file(source_items) != source_validation["outputs"]["items"]["sha256"]
    ):
        raise ValueError("source pack hash mismatch")
    source_chunks = {
        path.stem: path for path in sorted((source_pack / "chunks").glob("part-*.jsonl"))
    }
    source_rows, source_by_uid = _load_unique(source_items, "source items")
    uid_chunk: dict[str, str] = {}
    for chunk, path in source_chunks.items():
        for row in read_jsonl(path):
            uid = str(row["norm_uid"])
            if uid in uid_chunk:
                raise ValueError(f"source UID appears in multiple chunks: {uid}")
            uid_chunk[uid] = chunk
    if set(uid_chunk) != set(source_by_uid):
        raise ValueError("source items and chunks cover different UIDs")

    failed_audit = json.loads(failed_audit_path.read_text(encoding="utf-8"))
    if (
        failed_audit.get("schema_version") != AUDIT_SCHEMA
        or failed_audit.get("status") != "FAIL"
        or not failed_audit.get("violations")
        or (failed_audit.get("bank") or {}).get("sha256") != sha256_file(source_bank)
    ):
        raise ValueError("invalid failed source transcript audit")
    violating_chunks = sorted({str(row.get("chunk") or "") for row in failed_audit["violations"]})
    if "" in violating_chunks or not set(violating_chunks) <= set(source_chunks):
        raise ValueError("failed audit names invalid chunks")

    repair_pack_validation_path = repair_pack / "validation.json"
    repair_pack_validation = json.loads(
        repair_pack_validation_path.read_text(encoding="utf-8")
    )
    repair_chunks = {
        path.stem: path for path in sorted((repair_pack / "chunks").glob("part-*.jsonl"))
    }
    if (
        repair_pack_validation.get("schema_version") != REPAIR_SCHEMA
        or repair_pack_validation.get("selected_chunks") != violating_chunks
        or repair_pack_validation.get("label_content_read_for_selection") is not False
        or (repair_pack_validation.get("inputs", {}).get("failed_transcript_audit") or {}).get(
            "sha256"
        )
        != sha256_file(failed_audit_path)
        or set(repair_chunks) != set(violating_chunks)
        or sha256_file(repair_pack / "bank.json") != sha256_file(source_bank)
    ):
        raise ValueError("repair pack is not the exact audit-violation-only selection")
    repair_uids = {
        str(row["norm_uid"])
        for chunk in violating_chunks
        for row in read_jsonl(repair_chunks[chunk])
    }
    expected_repair_uids = {uid for uid, chunk in uid_chunk.items() if chunk in violating_chunks}
    if repair_uids != expected_repair_uids:
        raise ValueError("repair pack UID selection mismatch")

    repair_audit = json.loads(repair_audit_path.read_text(encoding="utf-8"))
    audit_chunks = {str(row.get("chunk") or ""): row for row in repair_audit.get("chunks") or []}
    if (
        repair_audit.get("schema_version") != AUDIT_SCHEMA
        or repair_audit.get("status") != "PASS"
        or repair_audit.get("complete") is not True
        or repair_audit.get("violations")
        or set(audit_chunks) != set(violating_chunks)
        or (repair_audit.get("bank") or {}).get("sha256") != sha256_file(repair_pack / "bank.json")
    ):
        raise ValueError("repair transcript audit is not complete and clean")
    for chunk, path in repair_chunks.items():
        if audit_chunks[chunk].get("chunk_sha256") != sha256_file(path):
            raise ValueError(f"repair transcript audit chunk mismatch: {chunk}")

    base_validation = _validation_binding(base_validation_path, base_labels)
    repair_validation = _validation_binding(repair_validation_path, repair_labels)
    if (
        (base_validation.get("pack_validation") or {}).get("sha256")
        != sha256_file(source_validation_path)
        or (repair_validation.get("pack_validation") or {}).get("sha256")
        != sha256_file(repair_pack_validation_path)
        or (repair_validation.get("transcript_audit") or {}).get("sha256")
        != sha256_file(repair_audit_path)
    ):
        raise ValueError("label validations are not bound to the source and repair audits")
    _, base_by_uid = _load_unique(base_labels, "base labels")
    _, repair_by_uid = _load_unique(repair_labels, "repair labels")
    if set(base_by_uid) != set(source_by_uid) or set(repair_by_uid) != repair_uids:
        raise ValueError("base or repair label coverage mismatch")

    promoted: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for item in source_rows:
        uid = str(item["norm_uid"])
        chunk = uid_chunk[uid]
        repaired = chunk in violating_chunks
        row = dict(repair_by_uid[uid] if repaired else base_by_uid[uid])
        acceptance = "audit_selected_isolation_repair" if repaired else "original_clean_chunk"
        row["transcript_acceptance"] = {
            "source": acceptance,
            "chunk": chunk,
            "failed_source_audit_sha256": sha256_file(failed_audit_path),
            "repair_audit_sha256": sha256_file(repair_audit_path) if repaired else None,
        }
        promoted.append(row)
        source_counts[acceptance] += 1
    write_jsonl(output, promoted)

    report = {
        "schema_version": PROMOTION_SCHEMA,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "status": "PASS_COMPOSITE_TRANSCRIPT_CLEAN_LABELS",
        "task": source_validation["task"],
        "bank_source_sha256": source_validation["bank_source_sha256"],
        "complete": True,
        "count": len(promoted),
        "accepted_original_chunks": sorted(set(source_chunks) - set(violating_chunks)),
        "accepted_repair_chunks": violating_chunks,
        "excluded_failed_base_chunks": violating_chunks,
        "excluded_failed_base_uid_count": len(repair_uids),
        "label_source_counts": dict(sorted(source_counts.items())),
        "selection_used_label_content": False,
        # Compatibility surface for downstream full-bank consensus validators:
        # this composite is the authoritative validated form of the original
        # pack, with every failed base chunk replaced by a clean repair.
        "pack_validation": {
            "path": str(source_validation_path),
            "sha256": sha256_file(source_validation_path),
        },
        "transcript_audit": {
            "status": "PASS_COMPOSITE_TRANSCRIPT_CLEAN",
            "failed_source_audit_sha256": sha256_file(failed_audit_path),
            "repair_audit_sha256": sha256_file(repair_audit_path),
            "excluded_failed_base_chunks": violating_chunks,
            "accepted_repair_chunks": violating_chunks,
        },
        "inputs": {
            "source_pack_validation": {
                "path": str(source_validation_path),
                "sha256": sha256_file(source_validation_path),
            },
            "base_labels": {"path": str(base_labels), "sha256": sha256_file(base_labels)},
            "base_validation": {
                "path": str(base_validation_path),
                "sha256": sha256_file(base_validation_path),
            },
            "failed_source_audit": {
                "path": str(failed_audit_path),
                "sha256": sha256_file(failed_audit_path),
            },
            "repair_pack_validation": {
                "path": str(repair_pack_validation_path),
                "sha256": sha256_file(repair_pack_validation_path),
            },
            "repair_labels": {
                "path": str(repair_labels),
                "sha256": sha256_file(repair_labels),
            },
            "repair_validation": {
                "path": str(repair_validation_path),
                "sha256": sha256_file(repair_validation_path),
            },
            "repair_audit": {
                "path": str(repair_audit_path),
                "sha256": sha256_file(repair_audit_path),
            },
        },
        "output": {"path": str(output), "sha256": sha256_file(output)},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**report, "report": str(report_path), "report_sha256": sha256_file(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pack", required=True)
    parser.add_argument("--base-labels", required=True)
    parser.add_argument("--base-validation", required=True)
    parser.add_argument("--failed-audit", required=True)
    parser.add_argument("--repair-pack", required=True)
    parser.add_argument("--repair-labels", required=True)
    parser.add_argument("--repair-validation", required=True)
    parser.add_argument("--repair-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result = promote(
        source_pack=Path(args.source_pack),
        base_labels=Path(args.base_labels),
        base_validation_path=Path(args.base_validation),
        failed_audit_path=Path(args.failed_audit),
        repair_pack=Path(args.repair_pack),
        repair_labels=Path(args.repair_labels),
        repair_validation_path=Path(args.repair_validation),
        repair_audit_path=Path(args.repair_audit),
        output=Path(args.output),
        report_path=Path(args.report),
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
