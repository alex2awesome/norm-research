#!/usr/bin/env python3
"""Validate and run one hash-pinned complete-bank retrieval lane."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .common import sha256_file
from .run_frozen_retrieval_queue import run_queue, validate_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    lane_path = Path(args.lane).resolve()
    lane = json.loads(lane_path.read_text(encoding="utf-8"))
    if (
        lane.get("schema_version") != "silver-match-v3-retrieval-lane-execution-v1"
        or lane.get("status") != "FROZEN_NOT_LAUNCHED"
        or lane.get("release_ready") is not False
    ):
        raise ValueError("unsupported/unfrozen lane queue")
    plan_path = Path(lane["task_plan"]["path"])
    if sha256_file(plan_path) != lane["task_plan"]["sha256"]:
        raise ValueError("task retrieval plan hash mismatch")
    runner_path = Path(lane["runner"]["path"])
    if sha256_file(runner_path) != lane["runner"]["sha256"]:
        raise ValueError("frozen lane runner hash mismatch")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)
    matches = [
        step
        for step in plan["steps"]
        if step["kind"] == "retrieve"
        and step["corpus"] == lane["corpus"]
        and step["system"] == lane["system"]
    ]
    if len(matches) != 1 or any(
        lane[key] != matches[0][key]
        for key in ("candidate", "audit", "expected_k", "command")
    ):
        raise ValueError("lane queue differs from its task plan")
    if not args.run:
        print(json.dumps({"status": "VALIDATED_NOT_LAUNCHED", "lane": str(lane_path)}))
        return
    execution_plan = copy.deepcopy(plan)
    execution_plan["execution"]["gpu_index"] = int(lane["gpu_index"])
    run_queue(
        execution_plan,
        only_corpus=str(lane["corpus"]),
        only_system=str(lane["system"]),
        retrieval_only=True,
    )


if __name__ == "__main__":
    main()
