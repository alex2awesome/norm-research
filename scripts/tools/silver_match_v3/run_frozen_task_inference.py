#!/usr/bin/env python3
"""Run one frozen task-plan inference shard without hand-copied settings.

This wrapper turns a hash-bound production plan into exactly one adjudicator or
verifier command.  It validates every scientific input and implementation pin
before execution, so large tasks can be distributed across GPUs without each
launcher re-specifying prompts, render limits, seeds, or model snapshots.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import sha256_file


def _check_artifact(value: dict[str, Any], label: str) -> Path:
    path = Path(str(value.get("path") or ""))
    expected = str(value.get("sha256") or "")
    if not path.exists() or sha256_file(path) != expected:
        raise ValueError(f"frozen artifact changed: {label}={path}")
    return path


def _resolve_prompt(raw: str, repo_root: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path


def _validate_plan(plan_path: Path, stage: str) -> tuple[dict[str, Any], Path]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("schema_version") != "silver-match-v3-task-production-plan-v1"
        or plan.get("status") != "FROZEN_READY_FOR_UNLABELED_PRODUCTION"
    ):
        raise ValueError("plan is not frozen for unlabeled production")
    for key in ("manifest", "candidate_union", "candidate_union_meta"):
        _check_artifact(plan[key], key)
    block = plan[stage]
    implementation = _check_artifact(block["implementation"], f"{stage}.implementation")
    _check_artifact(block["selection"], f"{stage}.selection")
    if stage == "verifier":
        _check_artifact(block["production_policy"], "verifier.production_policy")
    for raw_path, value in (block.get("prompt_components") or {}).items():
        _check_artifact({"path": raw_path, **value}, f"{stage}.prompt_component")
    repo_root = implementation.parents[3]
    return plan, repo_root


def build_command(
    *,
    plan_path: Path,
    stage: str,
    order: str,
    output_path: Path,
    shard_id: int,
    num_shards: int,
    batch_size: int,
    gpu_memory_utilization: float,
    primary_path: Path | None = None,
    resume: bool = False,
) -> tuple[list[str], Path, str]:
    if order not in {"original", "hashed"}:
        raise ValueError("production order must be original or hashed")
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError("invalid shard coordinates")
    if batch_size < 1 or not 0 < gpu_memory_utilization < 1:
        raise ValueError("invalid inference resource settings")
    plan, repo_root = _validate_plan(plan_path, stage)
    block = plan[stage]
    prompt = _resolve_prompt(str(block["prompt"]), repo_root).resolve()
    addons = [
        _resolve_prompt(str(path), repo_root).resolve()
        for path in block.get("prompt_addons") or []
    ]
    command = [
        sys.executable,
        "-u",
        "-m",
        f"scripts.tools.silver_match_v3.{'adjudicate_gemma' if stage == 'adjudicator' else 'verify_gemma'}",
        "--manifest",
        str(Path(plan["manifest"]["path"])),
        "--candidates",
        str(Path(plan["candidate_union"]["path"])),
        "--output",
        str(output_path),
        "--prompt",
        str(prompt),
        "--model",
        str(block["model"] if stage == "adjudicator" else block["rendering"]["model"]),
        "--order-mode",
        order,
        "--batch-size",
        str(batch_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--shard-id",
        str(shard_id),
        "--num-shards",
        str(num_shards),
    ]
    for addon in addons:
        command.extend(("--prompt-addon", str(addon)))
    if stage == "adjudicator":
        rendering = block["prompt_rendering"]
        sampling = block["production_sampling"]
        command.extend(
            (
                "--max-candidates",
                str(block["candidate_depth"]),
                "--context-chars",
                str(rendering["context_chars"]),
                "--description-chars",
                str(rendering["description_chars"]),
                "--example-chars",
                str(rendering["example_chars"]),
                "--max-examples",
                str(rendering["max_examples"]),
                "--max-model-len",
                str(sampling["max_model_len"]),
                "--max-tokens",
                str(sampling["max_tokens"]),
                "--seed",
                str(sampling["seed"]),
            )
        )
    else:
        if primary_path is None or not primary_path.exists():
            raise ValueError("verifier stage requires an existing --primary artifact")
        rendering = block["rendering"]
        command.extend(
            (
                "--primary",
                str(primary_path),
                "--max-alternatives",
                str(rendering["max_alternatives"]),
                "--context-chars",
                str(rendering["context_chars"]),
                "--description-chars",
                str(rendering["description_chars"]),
                "--example-chars",
                str(rendering["example_chars"]),
                "--max-examples",
                str(rendering["max_examples"]),
                "--max-model-len",
                str(rendering["max_model_len"]),
                "--max-tokens",
                str(rendering["max_tokens"]),
                "--seed",
                str(rendering["seed"]),
            )
        )
    if resume:
        command.append("--resume")
    implementation_sha = str(block["implementation"]["sha256"])
    return command, repo_root, implementation_sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--stage", choices=("adjudicator", "verifier"), required=True)
    parser.add_argument("--order", choices=("original", "hashed"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--primary")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    output_path = Path(args.output).resolve()
    command, repo_root, implementation_sha = build_command(
        plan_path=plan_path,
        stage=args.stage,
        order=args.order,
        output_path=output_path,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        primary_path=Path(args.primary).resolve() if args.primary else None,
        resume=args.resume,
    )
    payload = {
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "stage": args.stage,
        "order": args.order,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "command": command,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)
    if args.dry_run:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=repo_root, check=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    implementation = Path(plan[args.stage]["implementation"]["path"])
    if sha256_file(implementation) != implementation_sha:
        raise RuntimeError("inference implementation changed during execution")


if __name__ == "__main__":
    main()
