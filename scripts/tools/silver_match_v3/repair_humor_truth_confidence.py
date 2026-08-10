#!/usr/bin/env python3
"""Repair the 31 null-confidence rows in the joined Humor truth (v1 -> v2).

The 22,090-row joined truth carries 31 legacy ``sonnet_audit`` bridged MATCH
rows whose audit schema never collected a confidence
(``notes.confidence_basis == "not_collected_by_audit_schema"``).  The overlay
merge validator requires confidence in {high, medium, low}, so the v1 file
fails closed.  This repair recovers a documented, conservative confidence:

- If the row's ``notes.superseded_labels`` contain labels that agree with the
  audited decision AND exact metric_id, take the MINIMUM confidence among the
  agreeing labels (low < medium < high).
- Otherwise assign the conservative floor ``low``.

Every repaired row gains explicit provenance fields; every other row is copied
through byte-identically.  The v1 file is never modified.  Create-only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .adjudicate_gemma import CONFIDENCES
from .common import sha256_file


SCHEMA = "silver-match-v3-humor-truth-confidence-repair-v1"
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
EXPECTED_BASIS = "not_collected_by_audit_schema"


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def repair(args: argparse.Namespace) -> dict[str, Any]:
    truth_path = Path(args.truth).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report_output).resolve()
    for path in (output_path, report_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite repair artifact: {path}")
    observed = sha256_file(truth_path)
    if observed != args.truth_sha256:
        raise ValueError(f"v1 truth SHA mismatch: {observed}")

    repaired: list[dict[str, Any]] = []
    rows = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        with truth_path.open(encoding="utf-8") as source, temporary.open(
            "x", encoding="utf-8"
        ) as sink:
            for line in source:
                if not line.strip():
                    raise ValueError("blank line in v1 truth")
                rows += 1
                row = json.loads(line)
                confidence = (row.get("confidence") or "").strip().lower()
                if confidence in CONFIDENCES:
                    sink.write(line if line.endswith("\n") else line + "\n")
                    continue
                if confidence:
                    raise ValueError(
                        f"unexpected non-null invalid confidence: {row.get('norm_uid')}"
                    )
                notes = row.get("notes") or {}
                if notes.get("confidence_basis") != EXPECTED_BASIS:
                    raise ValueError(
                        "null-confidence row lacks the expected audit basis: "
                        f"{row.get('norm_uid')}"
                    )
                agreeing = sorted(
                    {
                        str(label.get("confidence")).strip().lower()
                        for label in (notes.get("superseded_labels") or [])
                        if label.get("decision") == row.get("decision")
                        and label.get("metric_id") == row.get("metric_id")
                        and str(label.get("confidence")).strip().lower() in CONFIDENCES
                    }
                )
                if agreeing:
                    backfill = min(agreeing, key=CONFIDENCE_RANK.__getitem__)
                    basis = "min_confidence_among_agreeing_superseded_labels"
                else:
                    backfill = "low"
                    basis = "conservative_floor_no_agreeing_superseded_label"
                row["confidence"] = backfill
                row["confidence_backfilled"] = True
                row["confidence_backfill_basis"] = basis
                row["confidence_backfill_agreeing_confidences"] = agreeing
                row["confidence_backfill_repair"] = SCHEMA
                sink.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                repaired.append(
                    {
                        "norm_uid": row.get("norm_uid"),
                        "split": row.get("split"),
                        "decision": row.get("decision"),
                        "metric_id": row.get("metric_id"),
                        "backfilled_confidence": backfill,
                        "basis": basis,
                        "agreeing_superseded_confidences": agreeing,
                    }
                )
            sink.flush()
            os.fsync(sink.fileno())
        if rows != args.expected_rows:
            raise ValueError(f"expected {args.expected_rows} rows, saw {rows}")
        if len(repaired) != args.expected_repairs:
            raise ValueError(
                f"expected {args.expected_repairs} repairs, made {len(repaired)}"
            )
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    report = {
        "schema_version": SCHEMA,
        "status": "COMPLETE_CREATE_ONLY_CONFIDENCE_REPAIR",
        "input": {"path": str(truth_path), "sha256": observed, "rows": rows},
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "rows": rows,
        },
        "policy": {
            "recoverable": "min confidence among superseded labels agreeing on decision+metric_id",
            "unrecoverable": "conservative floor 'low'",
            "unmodified_rows": "byte-identical passthrough",
        },
        "repaired_count": len(repaired),
        "repaired_rows": repaired,
    }
    try:
        _write_json_new(report_path, report)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--truth-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--expected-rows", type=int, default=22090)
    parser.add_argument("--expected-repairs", type=int, default=31)
    args = parser.parse_args()
    report = repair(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": report["output"],
                "repaired_count": report["repaired_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
