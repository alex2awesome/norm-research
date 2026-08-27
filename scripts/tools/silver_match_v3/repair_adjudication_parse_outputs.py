#!/usr/bin/env python3
"""Repair parser-only INVALID_OUTPUT rows while preserving failed artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from .adjudicate_gemma import parse_response
from .common import read_jsonl, sha256_file, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quarantine-dir", required=True)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    quarantine = Path(args.quarantine_dir).resolve()
    audit_output = Path(args.audit_output).resolve()
    if not output.is_file() or not meta_path.is_file():
        raise FileNotFoundError("output and metadata are required")
    if quarantine.exists() or audit_output.exists():
        raise FileExistsError("refusing to overwrite repair provenance")
    rows = list(read_jsonl(output))
    uids = [str(row.get("norm_uid") or "") for row in rows]
    if "" in uids or len(uids) != len(set(uids)):
        raise ValueError("output has empty or duplicate norm_uid values")
    invalid_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("decision") == "INVALID_OUTPUT" or row.get("parse_error")
    ]
    if not invalid_indices:
        raise ValueError("no parser-invalid rows to repair")
    repaired_rows = list(rows)
    repair_records = []
    for index in invalid_indices:
        row = dict(rows[index])
        raw = str(row.get("raw_response") or "")
        parsed, error = parse_response(
            raw, set(map(str, row.get("candidate_ids") or []))
        )
        if parsed is None or error is not None:
            raise ValueError(
                f"stored raw response remains invalid: {row['norm_uid']}: {error}"
            )
        for key in ("decision", "metric_id", "confidence", "reason"):
            row[key] = parsed[key]
        row["parse_error"] = None
        repaired_rows[index] = row
        repair_records.append(
            {
                "norm_uid": row["norm_uid"],
                "old_decision": rows[index].get("decision"),
                "new_decision": row["decision"],
                "new_metric_id": row["metric_id"],
                "repair_kind": "parse_stored_raw_response_only",
            }
        )

    original_output_sha = sha256_file(output)
    original_meta_sha = sha256_file(meta_path)
    quarantine.mkdir(parents=True, exist_ok=False)
    shutil.copy2(output, quarantine / output.name)
    shutil.copy2(meta_path, quarantine / meta_path.name)

    temporary_output = output.with_name(output.name + ".parser-repair.tmp")
    temporary_meta = meta_path.with_name(meta_path.name + ".parser-repair.tmp")
    if temporary_output.exists() or temporary_meta.exists():
        raise FileExistsError("stale parser-repair temporary file")
    write_jsonl(temporary_output, repaired_rows)
    meta = json.loads(meta_path.read_text())
    meta.update(
        {
            "output_sha256": sha256_file(temporary_output),
            "eligible_count": len(repaired_rows),
            "new_count": len(repaired_rows),
            "invalid_count": 0,
            "parser_only_repair_count": len(repair_records),
            "parser_only_repair_original_output_sha256": original_output_sha,
            "metadata_counts_reconstructed_from_complete_output": True,
        }
    )
    temporary_meta.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_output, output)
    os.replace(temporary_meta, meta_path)
    audit = {
        "schema_version": "silver-match-v3-parser-only-output-repair-v1",
        "status": "REPAIRED_FROM_STORED_RAW_RESPONSE_WITHOUT_NEW_INFERENCE",
        "originals": {
            "output": {
                "path": str(quarantine / output.name),
                "sha256": original_output_sha,
            },
            "metadata": {
                "path": str(quarantine / meta_path.name),
                "sha256": original_meta_sha,
            },
        },
        "repaired": {
            "output": {"path": str(output), "sha256": sha256_file(output)},
            "metadata": {"path": str(meta_path), "sha256": sha256_file(meta_path)},
        },
        "repairs": repair_records,
        "new_model_inference_used": False,
        "prompt_candidates_thresholds_or_model_changed": False,
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {**audit, "audit_sha256": sha256_file(audit_output)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
