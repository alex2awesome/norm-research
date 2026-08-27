#!/usr/bin/env python3
"""Freeze one independently executable lane from a full task retrieval plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .common import sha256_file
from .gpu_host_policy import validate_gpu_indices_for_host
from .run_frozen_retrieval_queue import validate_plan


def freeze(plan_path: Path, corpus: str, system: str, gpu_index: int) -> dict[str, Any]:
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_plan(plan)
    validate_gpu_indices_for_host([gpu_index])
    steps = [
        step
        for step in plan["steps"]
        if step["kind"] == "retrieve"
        and step["corpus"] == corpus
        and step["system"] == system
    ]
    if len(steps) != 1:
        raise ValueError("lane identity does not select exactly one retrieval")
    step = steps[0]
    runner = (
        Path(plan["execution"]["repo_root"])
        / "scripts/tools/silver_match_v3/run_frozen_retrieval_lane.py"
    ).resolve()
    if not runner.is_file():
        raise FileNotFoundError(runner)
    return {
        "schema_version": "silver-match-v3-retrieval-lane-execution-v1",
        "status": "FROZEN_NOT_LAUNCHED",
        "release_ready": False,
        "task": plan["task"],
        "task_plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "corpus": corpus,
        "system": system,
        "gpu_index": gpu_index,
        "candidate": step["candidate"],
        "audit": step["audit"],
        "expected_k": step["expected_k"],
        "command": step["command"],
        "scientific_plan_changed": False,
        "uses_batched_encoder_inference": True,
        "uses_openai_server": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = freeze(Path(args.plan), args.corpus, args.system, args.gpu_index)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
