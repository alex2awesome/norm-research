#!/usr/bin/env python3
"""Seal a post-test, dev-only fusion choice without reusing frozen test."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .common import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--dense-promotion-record", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    selection_path = Path(args.selection).resolve()
    promotion_path = Path(args.dense_promotion_record).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"append-only fusion record exists: {output}")
    selection = json.loads(selection_path.read_text())
    promotion = json.loads(promotion_path.read_text())
    if selection.get("task") != args.task or promotion.get("task") != args.task:
        raise ValueError("task mismatch")
    if selection.get("selection_split") != "external_dev_only":
        raise ValueError("fusion selection was not external-dev-only")
    if selection.get("frozen_test_consumed") is not False:
        raise ValueError("fusion selection unexpectedly claims test consumption")
    if promotion.get("frozen_test_consumption", {}).get("status") != "CONSUMED_EXACTLY_ONCE":
        raise ValueError("dense adapter frozen-test audit is not sealed")
    chosen = selection["chosen"]
    fusion_path = Path(chosen["fusion_report"])
    record = {
        "schema_version": "silver-match-v3.dev-only-fusion.1",
        "task": args.task,
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "selection_split": "external_dev_only",
        "chosen": chosen,
        "selection_artifact": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "fusion_artifact": {
            "path": str(fusion_path),
            "sha256": sha256_file(fusion_path),
        },
        "test_status": {
            "fusion_test_evaluated": False,
            "dense_adapter_test_was_previously_consumed": True,
            "dense_promotion_record": str(promotion_path),
            "dense_promotion_record_sha256": sha256_file(promotion_path),
            "policy": (
                "DO NOT RUN OR REPORT TEST METRICS FOR THIS FUSION. The only frozen "
                "test result is the earlier pure-dense adapter confirmation; fusion is "
                "dev-selected and deliberately unretested."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
