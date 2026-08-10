#!/usr/bin/env python3
"""Build a label-content-blind repair pack for failed transcript chunks."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file, write_jsonl


AUDIT_SCHEMA = "silver-match-v3-isolated-labeler-transcript-audit-v1"
REPAIR_SCHEMA = "silver-match-v3-transcript-isolation-repair-pack-v1"


def _chunk_paths(root: Path) -> dict[str, Path]:
    paths = {path.stem: path for path in sorted((root / "chunks").glob("part-*.jsonl"))}
    if not paths:
        raise ValueError("source pack has no chunks")
    return paths


def build(source: Path, failed_audit_path: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    failed_audit_path = failed_audit_path.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite repair pack: {output}")

    source_validation_path = source / "validation.json"
    source_items_path = source / "items.jsonl"
    source_bank_path = source / "bank.json"
    source_validation = json.loads(source_validation_path.read_text(encoding="utf-8"))
    if sha256_file(source_items_path) != source_validation["outputs"]["items"]["sha256"]:
        raise ValueError("source items hash mismatch")
    if sha256_file(source_bank_path) != source_validation["outputs"]["bank"]["sha256"]:
        raise ValueError("source bank hash mismatch")

    chunks = _chunk_paths(source)
    failed_audit = json.loads(failed_audit_path.read_text(encoding="utf-8"))
    if (
        failed_audit.get("schema_version") != AUDIT_SCHEMA
        or failed_audit.get("status") != "FAIL"
        or not failed_audit.get("violations")
        or (failed_audit.get("bank") or {}).get("sha256") != sha256_file(source_bank_path)
    ):
        raise ValueError("audit is not a failed isolation audit for the source bank")
    audit_rows = {str(row.get("chunk") or ""): row for row in failed_audit.get("chunks") or []}
    for chunk, row in audit_rows.items():
        if chunk not in chunks or row.get("chunk_sha256") != sha256_file(chunks[chunk]):
            raise ValueError(f"failed audit chunk binding mismatch: {chunk}")

    selected = sorted({str(row.get("chunk") or "") for row in failed_audit["violations"]})
    if "" in selected or not selected or not set(selected) <= set(chunks):
        raise ValueError("failed audit names an invalid repair chunk")

    selected_rows: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    for chunk in selected:
        for row in read_jsonl(chunks[chunk]):
            uid = str(row.get("norm_uid") or "")
            if not uid or uid in seen_uids:
                raise ValueError(f"missing or duplicate repair UID: {uid}")
            seen_uids.add(uid)
            selected_rows.append(row)

    output.mkdir(parents=True, exist_ok=True)
    output_bank = output / "bank.json"
    shutil.copyfile(source_bank_path, output_bank)
    output_items = output / "items.jsonl"
    write_jsonl(output_items, selected_rows)
    output_chunks: dict[str, str] = {}
    for chunk in selected:
        destination = output / "chunks" / f"{chunk}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(chunks[chunk], destination)
        output_chunks[str(destination)] = sha256_file(destination)

    report = {
        "schema_version": REPAIR_SCHEMA,
        "status": "FROZEN_AUDIT_VIOLATION_ONLY_REPAIR_PACK",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": source_validation["task"],
        "count": len(selected_rows),
        "chunk_count": len(selected),
        "selected_chunks": selected,
        "source_chunk_count": len(chunks),
        "bank_source_sha256": source_validation["bank_source_sha256"],
        "truth_hidden": True,
        "label_content_read_for_selection": False,
        "selection_rule": "all_and_only_chunks_with_fail_closed_transcript_audit_violations",
        "inputs": {
            "source_pack_validation": {
                "path": str(source_validation_path),
                "sha256": sha256_file(source_validation_path),
            },
            "failed_transcript_audit": {
                "path": str(failed_audit_path),
                "sha256": sha256_file(failed_audit_path),
                "status": "FAIL",
            },
        },
        "outputs": {
            "items": {"path": str(output_items), "sha256": sha256_file(output_items)},
            "bank": {"path": str(output_bank), "sha256": sha256_file(output_bank)},
            "chunks": output_chunks,
        },
    }
    validation = output / "validation.json"
    validation.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**report, "validation": str(validation), "validation_sha256": sha256_file(validation)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pack", required=True)
    parser.add_argument("--failed-audit", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = build(Path(args.source_pack), Path(args.failed_audit), Path(args.output_root))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
