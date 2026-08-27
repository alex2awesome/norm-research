#!/usr/bin/env python3
"""Freeze the rejected PR R4 run whose prompt/runner interfaces disagreed."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from .common import read_jsonl, sha256_file


def _last_json_object(raw: str) -> dict | None:
    decoder = json.JSONDecoder()
    found = None
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "decision" in value:
            found = value
    return found


def freeze(plan_path: Path, output_path: Path, audit_path: Path) -> dict:
    plan_path = plan_path.resolve()
    output_path = output_path.resolve()
    audit_path = audit_path.resolve()
    if audit_path.exists():
        raise FileExistsError(audit_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if Path(plan["outputs"]["original"]).resolve() != output_path:
        raise ValueError("output is not the frozen original-order output")
    for order in ("hashed", "reverse"):
        if Path(plan["outputs"][order]).exists():
            raise ValueError(f"unexpected post-failure output exists: {order}")

    rows = list(read_jsonl(output_path))
    if not rows:
        raise ValueError("failed output is empty")
    numeric_raw = 0
    parse_errors: collections.Counter[str] = collections.Counter()
    for row in rows:
        if (
            row.get("decision") != "INVALID_OUTPUT"
            or row.get("parse_error") not in {"unknown_confidence", "no_json"}
            or row.get("prompt_sha256")
            != plan["inputs"]["adjudicator_prompt"]["sha256"]
            or row.get("order_mode") != "original"
        ):
            raise ValueError("run contains a non-interface-failure row")
        parse_errors[str(row["parse_error"])] += 1
        parsed = _last_json_object(str(row.get("raw_response") or ""))
        confidence = None if parsed is None else parsed.get("confidence")
        raw_response = str(row.get("raw_response") or "")
        if (
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
        ) or re.search(r'"confidence"\s*:\s*(?:0(?:\.\d+)?|1(?:\.0+)?)', raw_response):
            numeric_raw += 1
    if numeric_raw != len(rows):
        raise ValueError("not every raw response used numeric confidence")

    record = {
        "schema_version": "silver-match-v3-pr-r4-rejected-interface-run-v1",
        "status": "REJECTED_INTERFACE_CONTRACT_BEFORE_TRUTH_JOIN",
        "task": "press-releases",
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "prompt": plan["inputs"]["adjudicator_prompt"],
        "runner": plan["inputs"]["runner"],
        "partial_output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "count": len(rows),
            "invalid_count": len(rows),
            "parse_error_counts": dict(sorted(parse_errors.items())),
            "numeric_confidence_raw_count": numeric_raw,
        },
        "unlaunched_orders": ["hashed", "reverse"],
        "root_causes": [
            "frozen prompt requested numeric [0,1] confidence while frozen runner accepted only high|medium|low",
            "the frozen 220-token generation cap truncated a minority of otherwise JSON-shaped responses",
        ],
        "contracts": {
            "truth_or_truth_predictions_read": False,
            "partial_output_must_never_be_reused": True,
            "new_plan_output_root_and_seed_required": True,
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**record, "audit_sha256": sha256_file(audit_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(args.plan, args.output, args.audit),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
