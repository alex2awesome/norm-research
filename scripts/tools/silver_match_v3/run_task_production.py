#!/usr/bin/env python3
"""Run one frozen task's policy-faithful Gemma production chain.

This is the task-generic counterpart of the original Notice-and-Comment shell
runner.  It consumes only a rendering-bound production plan emitted by
``freeze_task_production_plan`` and launches the local batch-vLLM modules
directly (never an OpenAI-compatible server).  Every stage is append-only and
resume-safe: completed metadata/artifacts are validated or reused, while
unfinished inference outputs are resumed in place.

The runner deliberately stops at ``final_pre_rescue``.  Exact MATCHes still
need the task's blind precision audit, and provisional abstentions still need
the repeated-capture/full-bank rescue and typed-abstention pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from .audit_production_adjudications import _check_plan_artifacts
from .common import sha256_file
from .gpu_host_policy import validate_gpu_indices_for_host, validate_launch_gpus
from .merge_inference_shards import merge_shards


def _resolve(path: str, repo_root: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (repo_root / value).resolve()


def _meta(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def _component_args(block: dict[str, Any], repo_root: Path) -> tuple[Path, list[Path]]:
    prompt = _resolve(str(block.get("prompt") or ""), repo_root)
    addons = [
        _resolve(str(value), repo_root) for value in block.get("prompt_addons") or []
    ]
    if not prompt.is_file() or any(not value.is_file() for value in addons):
        raise FileNotFoundError("a frozen prompt component is unavailable")
    return prompt, addons


def _append_addons(command: list[str], addons: Iterable[Path]) -> None:
    for addon in addons:
        command.extend(("--prompt-addon", str(addon)))


def adjudicator_command(
    *,
    plan: dict[str, Any],
    repo_root: Path,
    gemma_python: Path,
    output: Path,
    order: str,
    batch_size: int,
    gpu_memory_utilization: float,
    candidates: Path | None = None,
    max_candidates: int | None = None,
    shard_id: int = 0,
    num_shards: int = 1,
) -> list[str]:
    block = plan["adjudicator"]
    rendering = block["prompt_rendering"]
    sampling = block["production_sampling"]
    prompt, addons = _component_args(block, repo_root)
    command = [
        str(gemma_python),
        "-u",
        "-m",
        "scripts.tools.silver_match_v3.adjudicate_gemma",
        "--manifest",
        str(Path(plan["manifest"]["path"])),
        "--candidates",
        str(candidates or Path(plan["candidate_union"]["path"])),
        "--output",
        str(output),
        "--prompt",
        str(prompt),
        "--model",
        str(block["model"]),
        "--max-candidates",
        str(max_candidates or block["candidate_depth"]),
        "--context-chars",
        str(rendering["context_chars"]),
        "--description-chars",
        str(rendering["description_chars"]),
        "--example-chars",
        str(rendering["example_chars"]),
        "--max-examples",
        str(rendering["max_examples"]),
        "--batch-size",
        str(batch_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(sampling["max_model_len"]),
        "--max-tokens",
        str(sampling["max_tokens"]),
        "--seed",
        str(sampling["seed"]),
        "--order-mode",
        order,
        "--shard-id",
        str(shard_id),
        "--num-shards",
        str(num_shards),
        "--resume",
    ]
    _append_addons(command, addons)
    return command


def verifier_command(
    *,
    plan: dict[str, Any],
    repo_root: Path,
    gemma_python: Path,
    primary: Path,
    output: Path,
    order: str,
    batch_size: int,
    gpu_memory_utilization: float,
    candidates: Path | None = None,
    max_alternatives: int | None = None,
    shard_id: int = 0,
    num_shards: int = 1,
) -> list[str]:
    block = plan["verifier"]
    rendering = block["rendering"]
    prompt, addons = _component_args(block, repo_root)
    command = [
        str(gemma_python),
        "-u",
        "-m",
        "scripts.tools.silver_match_v3.verify_gemma",
        "--manifest",
        str(Path(plan["manifest"]["path"])),
        "--candidates",
        str(candidates or Path(plan["candidate_union"]["path"])),
        "--primary",
        str(primary),
        "--output",
        str(output),
        "--prompt",
        str(prompt),
        "--model",
        str(rendering["model"]),
        "--max-alternatives",
        str(max_alternatives or rendering["max_alternatives"]),
        "--context-chars",
        str(rendering["context_chars"]),
        "--description-chars",
        str(rendering["description_chars"]),
        "--example-chars",
        str(rendering["example_chars"]),
        "--max-examples",
        str(rendering["max_examples"]),
        "--batch-size",
        str(batch_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(rendering["max_model_len"]),
        "--max-tokens",
        str(rendering["max_tokens"]),
        "--seed",
        str(rendering["seed"]),
        "--order-mode",
        order,
        "--shard-id",
        str(shard_id),
        "--num-shards",
        str(num_shards),
        "--resume",
    ]
    _append_addons(command, addons)
    if rendering.get("enforce_eager"):
        command.append("--enforce-eager")
    return command


def _run(command: list[str], *, cwd: Path, log: Path | None = None) -> None:
    if log is None:
        subprocess.run(command, cwd=cwd, check=True)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + shlex.join(command) + "\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=cwd,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def _run_parallel_gpu(
    jobs: list[tuple[str, list[str], Path]],
    *,
    gpus: list[int],
    cwd: Path,
    enforce_live_gpu_policy: bool = False,
) -> None:
    if not jobs:
        return
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("GPU pool must contain distinct devices")
    # A three-order verifier remains scientifically identical when only two
    # devices are available: execute it in deterministic waves instead of
    # silently dropping the reverse order or refusing a resumable run.
    for start in range(0, len(jobs), len(gpus)):
        wave = jobs[start : start + len(gpus)]
        if enforce_live_gpu_policy:
            validate_launch_gpus(gpus[: len(wave)])
        processes: list[tuple[str, subprocess.Popen[str], Any]] = []
        try:
            for (name, command, log), gpu in zip(
                wave, gpus[: len(wave)], strict=True
            ):
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open("a", encoding="utf-8")
                handle.write("COMMAND " + shlex.join(command) + "\n")
                handle.flush()
                env = dict(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                processes.append((name, process, handle))
            failures = []
            for name, process, _ in processes:
                return_code = process.wait()
                if return_code:
                    failures.append((name, return_code))
            if failures:
                raise RuntimeError(f"parallel inference failed: {failures}")
        finally:
            for _, _, handle in processes:
                handle.close()


def _stage_outputs(
    *,
    output_dir: Path,
    task: str,
    stem: str,
    shards_per_order: int,
    orders: list[str],
) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    combined = {
        order: output_dir / f"{task}.{stem}.{order}.jsonl"
        for order in orders
    }
    shards = {
        order: [
            output_dir
            / "shards"
            / order
            / f"{task}.{stem}.{order}.part-{shard_id:03d}-of-{shards_per_order:03d}.jsonl"
            for shard_id in range(shards_per_order)
        ]
        for order in orders
    }
    return combined, shards


def _merge_completed_shards(
    *, combined: dict[str, Path], shards: dict[str, list[Path]]
) -> None:
    for order, output in combined.items():
        meta = _meta(output)
        if output.exists() != meta.exists():
            raise ValueError(f"partial combined inference artifact: {output}")
        if output.exists():
            payload = json.loads(meta.read_text(encoding="utf-8"))
            if payload.get("output_sha256") != sha256_file(output):
                raise ValueError(f"combined inference artifact changed: {output}")
            continue
        merge_shards(input_paths=shards[order], output_path=output)


def _validate_plan(plan_path: Path, repo_root: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("status") != "FROZEN_READY_FOR_UNLABELED_PRODUCTION":
        raise ValueError("task plan is not frozen for unlabeled production")
    if not plan.get("task") or not plan.get("corpora"):
        raise ValueError("task plan lacks task/corpus scope")
    _check_plan_artifacts(plan)
    for role in ("adjudicator", "verifier"):
        implementation = Path(plan[role]["implementation"]["path"])
        if sha256_file(implementation) != plan[role]["implementation"]["sha256"]:
            raise ValueError(f"{role} implementation changed")
        _component_args(plan[role], repo_root)
    return plan


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    plan_path = Path(args.plan).resolve()
    plan = _validate_plan(plan_path, repo_root)
    task = str(plan["task"])
    adjudicator_orders = [str(value) for value in plan["adjudicator"].get("orders") or []]
    verifier_orders = [str(value) for value in plan["verifier"].get("orders") or []]
    if adjudicator_orders != ["original", "hashed"]:
        raise ValueError("production adjudicator must retain original+hashed exact consensus")
    if verifier_orders not in (
        ["original", "hashed"],
        ["original", "hashed", "reverse"],
    ):
        raise ValueError("production verifier has unsupported frozen order topology")
    adj_sampling = plan["adjudicator"]["production_sampling"]
    verifier_rendering = plan["verifier"]["rendering"]
    adj_batch_size = int(adj_sampling.get("batch_size", args.batch_size))
    adj_gpu_utilization = float(
        adj_sampling.get("gpu_memory_utilization", args.gpu_memory_utilization)
    )
    verifier_batch_size = int(verifier_rendering.get("batch_size", args.batch_size))
    verifier_gpu_utilization = float(
        verifier_rendering.get(
            "gpu_memory_utilization", args.gpu_memory_utilization
        )
    )
    output_root = Path(args.output_root).resolve() / task
    adj_dir = output_root / "adjudicator"
    verifier_dir = output_root / "verifier"
    subset_dir = output_root / "subsets"
    final_dir = output_root / "final_pre_rescue"
    for value in (adj_dir, verifier_dir, subset_dir, final_dir):
        value.mkdir(parents=True, exist_ok=True)

    gemma_python = Path(args.gemma_python).resolve()
    cpu_python = Path(args.cpu_python).resolve()
    if not gemma_python.is_file() or not cpu_python.is_file():
        raise FileNotFoundError("configured Python interpreter is unavailable")

    if args.shards_per_order == 1:
        adj = {
            order: adj_dir / f"{task}.primary.{order}.jsonl"
            for order in adjudicator_orders
        }
        jobs = []
        for order, output in adj.items():
            if not _meta(output).exists():
                jobs.append(
                    (
                        f"adjudicator-{order}",
                        adjudicator_command(
                            plan=plan,
                            repo_root=repo_root,
                            gemma_python=gemma_python,
                            output=output,
                            order=order,
                            batch_size=adj_batch_size,
                            gpu_memory_utilization=adj_gpu_utilization,
                        ),
                        adj_dir / f"{task}.primary.{order}.log",
                    )
                )
        _run_parallel_gpu(
            jobs, gpus=args.gpus, cwd=repo_root, enforce_live_gpu_policy=True
        )
    else:
        adj, adj_shards = _stage_outputs(
            output_dir=adj_dir,
            task=task,
            stem="primary",
            shards_per_order=args.shards_per_order,
            orders=adjudicator_orders,
        )
        jobs = []
        for order in adjudicator_orders:
            if _meta(adj[order]).exists():
                continue
            for shard_id, output in enumerate(adj_shards[order]):
                if not _meta(output).exists():
                    jobs.append(
                        (
                            f"adjudicator-{order}-shard-{shard_id}",
                            adjudicator_command(
                                plan=plan,
                                repo_root=repo_root,
                                gemma_python=gemma_python,
                                output=output,
                                order=order,
                                batch_size=adj_batch_size,
                                gpu_memory_utilization=adj_gpu_utilization,
                                shard_id=shard_id,
                                num_shards=args.shards_per_order,
                            ),
                            adj_dir
                            / "shards"
                            / order
                            / f"{task}.primary.{order}.part-{shard_id:03d}.log",
                        )
                    )
        _run_parallel_gpu(
            jobs, gpus=args.gpus, cwd=repo_root, enforce_live_gpu_policy=True
        )
        _merge_completed_shards(combined=adj, shards=adj_shards)
    if any(not _meta(value).is_file() for value in adj.values()):
        raise RuntimeError("adjudicator stage lacks successful completion metadata")

    adj_audit = adj_dir / f"{task}.two-order.audit.json"
    if not adj_audit.exists():
        _run(
            [
                str(cpu_python),
                "-m",
                "scripts.tools.silver_match_v3.audit_production_adjudications",
                "--plan",
                str(plan_path),
                "--original",
                str(adj["original"]),
                "--hashed",
                str(adj["hashed"]),
                "--output",
                str(adj_audit),
            ],
            cwd=repo_root,
        )

    if args.shards_per_order == 1:
        verify = {
            order: verifier_dir / f"{task}.primary.verify.{order}.jsonl"
            for order in verifier_orders
        }
        jobs = []
        for order, output in verify.items():
            if not _meta(output).exists():
                jobs.append(
                    (
                        f"verifier-{order}",
                        verifier_command(
                            plan=plan,
                            repo_root=repo_root,
                            gemma_python=gemma_python,
                            primary=adj["original"],
                            output=output,
                            order=order,
                            batch_size=verifier_batch_size,
                            gpu_memory_utilization=verifier_gpu_utilization,
                        ),
                        verifier_dir / f"{task}.primary.verify.{order}.log",
                    )
                )
        _run_parallel_gpu(
            jobs, gpus=args.gpus, cwd=repo_root, enforce_live_gpu_policy=True
        )
    else:
        verify, verify_shards = _stage_outputs(
            output_dir=verifier_dir,
            task=task,
            stem="primary.verify",
            shards_per_order=args.shards_per_order,
            orders=verifier_orders,
        )
        jobs = []
        for order in verifier_orders:
            if _meta(verify[order]).exists():
                continue
            for shard_id, output in enumerate(verify_shards[order]):
                if not _meta(output).exists():
                    jobs.append(
                        (
                            f"verifier-{order}-shard-{shard_id}",
                            verifier_command(
                                plan=plan,
                                repo_root=repo_root,
                                gemma_python=gemma_python,
                                primary=adj["original"],
                                output=output,
                                order=order,
                                batch_size=verifier_batch_size,
                                gpu_memory_utilization=verifier_gpu_utilization,
                                shard_id=shard_id,
                                num_shards=args.shards_per_order,
                            ),
                            verifier_dir
                            / "shards"
                            / order
                            / f"{task}.primary.verify.{order}.part-{shard_id:03d}.log",
                        )
                    )
        _run_parallel_gpu(
            jobs, gpus=args.gpus, cwd=repo_root, enforce_live_gpu_policy=True
        )
        _merge_completed_shards(combined=verify, shards=verify_shards)
    if any(not _meta(value).is_file() for value in verify.values()):
        raise RuntimeError("verifier stage lacks successful completion metadata")

    combined = verifier_dir / f"{task}.primary.verify.strict-all-orders.jsonl"
    if not combined.exists():
        combine_command = [
                str(cpu_python),
                "-m",
                "scripts.tools.silver_match_v3.combine_ordered_verifications",
                "--primary",
                str(adj["original"]),
                "--selection",
                str(Path(plan["verifier"]["selection"]["path"])),
                "--policy",
                str(Path(plan["verifier"]["production_policy"]["path"])),
                "--plan",
                str(plan_path),
                "--output",
                str(combined),
            ]
        for order in verifier_orders:
            combine_command.extend(("--verification", f"{order}={verify[order]}"))
        _run(combine_command, cwd=repo_root)

    final_artifacts: dict[str, dict[str, Any]] = {}
    for corpus in plan["corpora"]:
        primary_subset = subset_dir / f"{corpus}.primary.original.jsonl"
        order_subset = subset_dir / f"{corpus}.primary.hashed.jsonl"
        verification_subset = subset_dir / f"{corpus}.verify.strict-all-orders.jsonl"
        for source, output in (
            (adj["original"], primary_subset),
            (adj["hashed"], order_subset),
            (combined, verification_subset),
        ):
            if not output.exists():
                _run(
                    [
                        str(cpu_python),
                        "-m",
                        "scripts.tools.silver_match_v3.filter_labels",
                        "--input",
                        str(source),
                        "--output",
                        str(output),
                        "--where",
                        f"corpus={corpus}",
                    ],
                    cwd=repo_root,
                )
        final = final_dir / f"{corpus}.jsonl"
        if not Path(str(final) + ".report.json").exists():
            _run(
                [
                    str(cpu_python),
                    "-m",
                    "scripts.tools.silver_match_v3.finalize_adjudications",
                    "--manifest",
                    str(Path(plan["manifest"]["path"])),
                    "--corpus",
                    str(corpus),
                    "--primary",
                    str(primary_subset),
                    "--order-check",
                    str(order_subset),
                    "--verification",
                    str(verification_subset),
                    "--adjudicator-selection",
                    str(Path(plan["adjudicator"]["selection"]["path"])),
                    "--verifier-selection",
                    str(Path(plan["verifier"]["selection"]["path"])),
                    "--verifier-policy",
                    str(Path(plan["verifier"]["production_policy"]["path"])),
                    "--production-plan",
                    str(plan_path),
                    "--strict-production",
                    "--output",
                    str(final),
                ],
                cwd=repo_root,
            )
        final_report = Path(str(final) + ".report.json")
        final_artifacts[str(corpus)] = {
            "output": {"path": str(final), "sha256": sha256_file(final)},
            "report": {
                "path": str(final_report),
                "sha256": sha256_file(final_report),
            },
        }

    report = {
        "schema_version": "silver-match-v3-task-production-run-v2",
        "task": task,
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "candidate_count": int(plan["expected_count"]),
        "adjudicator_audit": {
            "path": str(adj_audit),
            "sha256": sha256_file(adj_audit),
        },
        "strict_verification": {
            "path": str(combined),
            "sha256": sha256_file(combined),
            "orders": verifier_orders,
        },
        "final_pre_rescue": final_artifacts,
        "status": "COMPLETE_PRE_RESCUE_ONLY",
    }
    report_path = output_root / "production_run.report.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing != report:
            raise ValueError(
                "existing production report differs from recomputed report"
            )
    else:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--gemma-python", required=True)
    parser.add_argument("--cpu-python", default=sys.executable)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--shards-per-order", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    args = parser.parse_args()
    validate_gpu_indices_for_host(args.gpus)
    if args.shards_per_order < 1:
        parser.error("--shards-per-order must be positive")
    if not args.gpus or len(args.gpus) != len(set(args.gpus)):
        parser.error("--gpus must name one or more distinct devices")
    if len(args.gpus) > 4:
        parser.error("--gpus may name at most four devices")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be in (0,1)")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
