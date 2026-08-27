#!/usr/bin/env python3
"""Apply independent manual labels to a blinded audit packet immutably."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .common import read_jsonl, sha256_file, write_jsonl
from .score_teacher_verification_audit import MANUAL_DECISIONS


def apply_labels(
    packet: Sequence[dict[str, Any]], labels: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    packet_by_uid = {str(row["norm_uid"]): row for row in packet}
    labels_by_uid = {str(row["norm_uid"]): row for row in labels}
    if len(packet_by_uid) != len(packet) or len(labels_by_uid) != len(labels):
        raise ValueError("duplicate UID in audit packet or labels")
    if packet_by_uid.keys() != labels_by_uid.keys():
        missing = sorted(packet_by_uid.keys() - labels_by_uid.keys())[:5]
        extra = sorted(labels_by_uid.keys() - packet_by_uid.keys())[:5]
        raise ValueError(f"audit label UID mismatch; missing={missing} extra={extra}")
    output = []
    for uid in sorted(packet_by_uid):
        label = labels_by_uid[uid]
        decision = str(label.get("manual_decision") or "")
        if decision not in MANUAL_DECISIONS:
            raise ValueError(f"invalid manual_decision for {uid}: {decision}")
        reason = str(label.get("manual_reason") or "").strip()
        auditor = str(label.get("auditor") or "").strip()
        if decision != "UNCERTAIN" and not reason:
            raise ValueError(f"missing manual_reason for {uid}")
        if not auditor:
            raise ValueError(f"missing auditor for {uid}")
        row = dict(packet_by_uid[uid])
        row.update(
            {
                "manual_decision": decision,
                "manual_metric_id": label.get("manual_metric_id"),
                "manual_reason": reason,
                "auditor": auditor,
            }
        )
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--labels", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    packet_path = Path(args.packet).resolve()
    label_paths = [Path(path).resolve() for path in args.labels]
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    label_rows = [row for path in label_paths for row in read_jsonl(path)]
    rows = apply_labels(list(read_jsonl(packet_path)), label_rows)
    write_jsonl(output, rows)
    report = {
        "count": len(rows),
        "input_hashes": {
            "packet": sha256_file(packet_path),
            **{f"labels:{path}": sha256_file(path) for path in label_paths},
        },
        "output_sha256": sha256_file(output),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
