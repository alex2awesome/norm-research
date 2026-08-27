#!/usr/bin/env python3
"""Seal dev selection and one-time frozen-test consumption for an adapter."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .common import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dev-report", required=True)
    parser.add_argument("--test-report", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        "dev_report": Path(args.dev_report).resolve(),
        "test_report": Path(args.test_report).resolve(),
        "training_report": Path(args.training_report).resolve(),
        "adapter": Path(args.adapter).resolve(),
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"append-only promotion record already exists: {output}")
    dev = json.loads(paths["dev_report"].read_text())
    test = json.loads(paths["test_report"].read_text())
    training = json.loads(paths["training_report"].read_text())
    if dev.get("task") != args.task or test.get("task") != args.task or training.get("task") != args.task:
        raise ValueError("task mismatch across promotion evidence")
    if dev.get("selection_role") != "promotion_dev" or dev.get("split") != "dev":
        raise ValueError("dev report is not promotion_dev")
    if not (dev.get("promotion_gate") or {}).get("passed"):
        raise ValueError("external dev promotion gate did not pass")
    if test.get("selection_role") != "frozen_test_once" or test.get("split") != "test":
        raise ValueError("test report is not frozen_test_once")
    if dev.get("input_hashes", {}).get("adapter") != test.get("input_hashes", {}).get("adapter"):
        raise ValueError("dev/test adapter hashes differ")
    if dev.get("input_hashes", {}).get("labels") != test.get("input_hashes", {}).get("labels"):
        raise ValueError("dev/test frozen label input differs")
    siblings = sorted(paths["test_report"].parent.glob(f"{args.task}.test*.json"))
    if siblings != [paths["test_report"]]:
        raise ValueError(f"expected exactly one frozen-test artifact, found: {siblings}")
    adapter_hashes = {
        path.name: sha256_file(path)
        for path in sorted(paths["adapter"].iterdir())
        if path.is_file()
    }
    if adapter_hashes != test.get("input_hashes", {}).get("adapter"):
        raise ValueError("current adapter bytes differ from frozen-test input")
    stat = paths["test_report"].stat()
    record = {
        "schema_version": "silver-match-v3.adapter-promotion.1",
        "task": args.task,
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "decision": "PROMOTE_TASK_SPECIFIC_ADAPTER",
        "selection": {
            "dev_report": str(paths["dev_report"]),
            "dev_report_sha256": sha256_file(paths["dev_report"]),
            "promotion_gate": dev["promotion_gate"],
            "adapter_hashes": adapter_hashes,
            "training_report": str(paths["training_report"]),
            "training_report_sha256": sha256_file(paths["training_report"]),
        },
        "frozen_test_consumption": {
            "status": "CONSUMED_EXACTLY_ONCE",
            "input_labels_sha256": test["input_hashes"]["labels"],
            "output": str(paths["test_report"]),
            "output_sha256": sha256_file(paths["test_report"]),
            "output_mtime_utc": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "matching_test_artifact_count_at_seal": len(siblings),
            "rerun_policy": "FORBIDDEN; this ONCE artifact is final and must not be overwritten or repeated",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
