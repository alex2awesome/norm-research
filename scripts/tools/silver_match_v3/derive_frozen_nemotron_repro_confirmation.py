#!/usr/bin/env python3
"""Derive the predeclared deterministic confirmation from a frozen retry queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace(command: list[str], flag: str, value: str) -> None:
    command[command.index(flag) + 1] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-queue", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--repro-rule", required=True)
    parser.add_argument("--trigger-dev-report", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source_queue).resolve()
    if sha256_file(source) != args.expected_source_sha256:
        raise ValueError("source queue mismatch")
    queue = json.loads(source.read_text())
    command = list(queue["command"])
    replace(command, "--attention", "eager")
    replace(command, "--output-root", str(Path(args.output_root).resolve()))
    queue["command"] = command
    queue["frozen_at"] = "2026-07-13T01:14:00-07:00"
    queue["environment"]["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    rule = Path(args.repro_rule).resolve()
    report = Path(args.trigger_dev_report).resolve()
    queue["bindings"] = sorted(
        [
            row
            for row in queue["bindings"]
            if row["name"] not in {"reproducibility_rule", "primary_external_dev_trigger"}
        ]
        + [
            {"name": "reproducibility_rule", "path": str(rule), "sha256": sha256_file(rule)},
            {"name": "primary_external_dev_trigger", "path": str(report), "sha256": sha256_file(report)},
        ],
        key=lambda row: row["name"],
    )
    log_root = Path(args.log_root).resolve()
    queue["outputs"] = {
        "training_output_root": str(Path(args.output_root).resolve()),
        "launch_record": str(log_root / "launch_record.json"),
        "pid": str(log_root / "training.pid"),
        "log": str(log_root / "training.log"),
    }
    queue["confirmation"] = {
        "same_data_split_hyperparameters_and_seeds": True,
        "attention": "eager",
        "cublas_workspace_config": ":4096:8",
        "external_test_consumed": False,
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
