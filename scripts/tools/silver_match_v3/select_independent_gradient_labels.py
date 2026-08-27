#!/usr/bin/env python3
"""Select high-precision, non-anchor exact labels for an immutable gradient set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .common import read_jsonl, sha256_file, write_jsonl


def select(
    labels: list[dict], excluded_uids: set[str], allowed_confidences: set[str]
) -> tuple[list[dict], Counter[str]]:
    output = []
    audit: Counter[str] = Counter()
    for row in labels:
        uid = str(row.get("norm_uid") or "")
        if uid in excluded_uids:
            audit["excluded_hidden_anchor"] += 1
            continue
        if row.get("decision") != "MATCH":
            audit[f"excluded_decision:{row.get('decision')}"] += 1
            continue
        confidence = str(row.get("confidence") or "").lower()
        if confidence not in allowed_confidences:
            audit[f"excluded_confidence:{confidence}"] += 1
            continue
        output.append(row)
        audit["selected"] += 1
    return output, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--exclude-uids-from", required=True)
    parser.add_argument("--allow-confidence", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    labels_path = Path(args.labels).resolve()
    exclude_path = Path(args.exclude_uids_from).resolve()
    output_path = Path(args.output).resolve()
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    if output_path.exists() or meta_path.exists():
        raise FileExistsError(f"immutable output already exists: {output_path}")
    excluded_uids = {str(row["norm_uid"]) for row in read_jsonl(exclude_path)}
    allowed = {value.lower() for value in args.allow_confidence}
    rows, audit = select(list(read_jsonl(labels_path)), excluded_uids, allowed)
    if not rows:
        raise ValueError("selection produced no gradient labels")
    write_jsonl(output_path, rows)
    meta = {
        "schema_version": "silver-match-v3-independent-gradient-selection-v1",
        "labels": str(labels_path),
        "labels_sha256": sha256_file(labels_path),
        "exclude_uids_from": str(exclude_path),
        "exclude_uids_sha256": sha256_file(exclude_path),
        "allowed_confidences": sorted(allowed),
        "audit": dict(sorted(audit.items())),
        "selected_metric_coverage": len({row["metric_id"] for row in rows}),
        "output_sha256": sha256_file(output_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
