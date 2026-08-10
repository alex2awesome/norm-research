#!/usr/bin/env python3
"""Derive an append-only nested-split retry queue from a failed frozen queue."""

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


def replace_argument(command: list[str], name: str, value: str) -> None:
    index = command.index(name)
    command[index + 1] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-queue", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--predeclaration", required=True)
    parser.add_argument("--failure-record", required=True)
    parser.add_argument("--feasibility-record", required=True)
    parser.add_argument("--nested-split-seed", required=True, type=int)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_path = Path(args.source_queue).resolve()
    if sha256_file(source_path) != args.expected_source_sha256:
        raise ValueError("source queue differs from frozen v1 queue")
    queue = json.loads(source_path.read_text())
    if queue.get("schema_version") != "silver-match-v3-frozen-nemotron-retry-queue-v1":
        raise ValueError("unsupported source queue")

    predeclaration = Path(args.predeclaration).resolve()
    failure = Path(args.failure_record).resolve()
    feasibility = Path(args.feasibility_record).resolve()
    bindings = [row for row in queue["bindings"] if row["name"] != "predeclaration"]
    bindings.extend(
        [
            {
                "name": "predeclaration",
                "path": str(predeclaration),
                "sha256": sha256_file(predeclaration),
            },
            {
                "name": "failed_v1_record",
                "path": str(failure),
                "sha256": sha256_file(failure),
            },
            {
                "name": "nested_split_feasibility",
                "path": str(feasibility),
                "sha256": sha256_file(feasibility),
            },
        ]
    )
    queue["bindings"] = sorted(bindings, key=lambda row: row["name"])
    queue["frozen_at"] = "2026-07-13T00:54:00-07:00"
    queue["retry"] = {
        "reason": "v1 reused the upstream-role seed and therefore had empty internal holdouts",
        "source_queue_sha256": args.expected_source_sha256,
        "nested_split_seed": args.nested_split_seed,
        "seed_search_performed": False,
    }
    command = list(queue["command"])
    replace_argument(command, "--split-seed", str(args.nested_split_seed))
    replace_argument(command, "--output-root", str(Path(args.output_root).resolve()))
    queue["command"] = command
    log_root = Path(args.log_root).resolve()
    queue["outputs"] = {
        "training_output_root": str(Path(args.output_root).resolve()),
        "launch_record": str(log_root / "launch_record.json"),
        "pid": str(log_root / "training.pid"),
        "log": str(log_root / "training.log"),
    }

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
