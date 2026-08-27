#!/usr/bin/env python3
"""Run a frozen task's repeated full-bank rescue and typed finalization."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import sha256_file
from .gpu_host_policy import validate_gpu_indices_for_host, validate_launch_gpus
from .run_task_production import (
    _meta,
    _run,
    _run_parallel_gpu,
    adjudicator_command,
    verifier_command,
)


def _check_artifact(value: dict[str, Any], label: str) -> Path:
    path = Path(str(value.get("path") or "")).resolve()
    if not path.is_file() or sha256_file(path) != value.get("sha256"):
        raise ValueError(f"frozen rescue artifact changed: {label}={path}")
    return path


def _validate_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (
        plan.get("schema_version") not in {
            "silver-match-v3-task-rescue-plan-v2",
            "silver-match-v3-task-rescue-plan-v3",
        }
        or plan.get("status") != "FROZEN_READY_FOR_REPEATED_FULL_BANK_RESCUE"
    ):
        raise ValueError("rescue plan is not frozen for production")
    for key in ("manifest", "production_plan", "production_report"):
        _check_artifact(plan[key], key)
    for corpus, values in (plan.get("primary_final_pre_rescue") or {}).items():
        _check_artifact(values["output"], f"primary.{corpus}.output")
        _check_artifact(values["report"], f"primary.{corpus}.report")
    for system, values in (plan.get("candidate_systems") or {}).items():
        for index, item in enumerate(values.get("inputs") or []):
            _check_artifact(item["candidate"], f"candidate.{system}.{index}")
            _check_artifact(item["audit"], f"candidate_audit.{system}.{index}")
    _check_artifact(plan["abstention_verifier"]["prompt"], "abstention_prompt")
    for index, value in enumerate(plan.get("blind_audit_exclusions") or []):
        _check_artifact(value, f"blind_audit_exclusion.{index}")
    if not plan.get("blind_audit_exclusions"):
        raise ValueError("rescue plan lacks blind-audit exclusions")
    for name, value in (plan.get("implementations") or {}).items():
        _check_artifact(value, f"implementation.{name}")
    policy = plan.get("rescue_policy") or {}
    legacy_policy = plan.get("schema_version") == "silver-match-v3-task-rescue-plan-v2"
    if (
        int(policy.get("coverage_repeats", 0)) < 2
        or policy.get("reinclude_primary") is not True
        or policy.get("include_all_abstentions") is not True
        or policy.get("strict_two_order_finalist_adjudication") is not True
        or policy.get("strict_two_order_typed_abstention_verification") is not True
    ):
        raise ValueError("rescue plan weakens the repeated/strict production contract")
    if legacy_policy:
        if policy.get("strict_two_order_contrastive_verification") is not True:
            raise ValueError("legacy rescue plan weakens two-order match verification")
    else:
        verifier_orders = list(policy.get("contrastive_verification_orders") or [])
        if (
            policy.get("finalist_adjudication_orders") != ["original", "hashed"]
            or verifier_orders not in (
                ["original", "hashed"],
                ["original", "hashed", "reverse"],
            )
            or policy.get("typed_abstention_verification_orders")
            != ["original", "hashed"]
            or policy.get("strict_all_selected_order_contrastive_verification")
            is not True
            or (plan.get("verifier") or {}).get("orders") != verifier_orders
        ):
            raise ValueError("rescue plan weakens selected verifier order topology")
    risk = plan.get("final_risk_policy") or {}
    if (
        risk.get("sample_schema") != "silver-match-v3-final-decision-sample-v2"
        or int(risk.get("uniform_match_sample_n", 0)) < 60
        or risk.get("uniform_match_sample_n")
        != risk.get("uniform_abstention_sample_n")
        or int(risk.get("independent_full_bank_passes_minimum", 0)) < 2
        or risk.get("unique_exact_two_vote_consensus_required") is not True
        or risk.get("disagreement_only_resolvers_required") is not True
        or risk.get("unresolved_gold_rows_may_not_be_dropped_from_sample") is not True
        or risk.get("strict_transcript_isolation_required_for_every_pass") is not True
        or float(risk.get("alpha_one_sided", 1.0)) != 0.05
        or float(risk.get("false_abstention_upper_target", 1.0)) > 0.05
        or float(risk.get("match_exact_precision_lower_target", 0.0)) < 0.90
        or float(risk.get("typed_abstention_exact_lower_target", 0.0)) < 0.80
    ):
        raise ValueError("rescue plan weakens the frozen final-risk contract")
    return plan


def _gpu_run(command: list[str], *, gpu: int, cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + shlex.join(command) + "\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def _trial_worker(jobs: list[tuple[list[str], Path]], *, gpu: int, cwd: Path) -> None:
    for command, log in jobs:
        _gpu_run(command, gpu=gpu, cwd=cwd, log=log)


def _artifact_args(flag: str, paths: list[Path]) -> list[str]:
    output: list[str] = []
    for path in paths:
        output.extend((flag, str(path)))
    return output


def _abstention_command(
    *,
    plan: dict[str, Any],
    gemma_python: Path,
    audits: Path,
    output: Path,
    order: str,
    batch_size: int,
    gpu_memory_utilization: float,
) -> list[str]:
    block = plan["abstention_verifier"]
    return [
        str(gemma_python),
        "-u",
        "-m",
        "scripts.tools.silver_match_v3.verify_abstention_gemma",
        "--manifest",
        str(Path(plan["manifest"]["path"])),
        "--audits",
        str(audits),
        "--output",
        str(output),
        "--prompt",
        str(Path(block["prompt"]["path"])),
        "--model",
        str(block["model"]),
        "--batch-size",
        str(batch_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(block["max_model_len"]),
        "--max-tokens",
        str(block["max_tokens"]),
        "--seed",
        str(block["seed"]),
        "--order-mode",
        order,
        "--resume",
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    plan_path = Path(args.plan).resolve()
    plan = _validate_plan(plan_path)
    frozen_blind_audit_n = int(
        plan["final_risk_policy"]["uniform_match_sample_n"]
    )
    if (
        args.blind_audit_n is not None
        and args.blind_audit_n != frozen_blind_audit_n
    ):
        raise ValueError("runtime blind-audit size differs from the frozen rescue plan")
    production_plan = json.loads(
        Path(plan["production_plan"]["path"]).read_text(encoding="utf-8")
    )
    task = str(plan["task"])
    manifest = Path(plan["manifest"]["path"])
    output_root = Path(args.output_root).resolve() / task
    captures = output_root / "captures"
    trial_dir = output_root / "trial_adjudications"
    aggregate = output_root / "aggregate"
    finalists_dir = output_root / "finalists"
    abstention_dir = output_root / "typed_abstentions"
    final_dir = output_root / "final_by_corpus"
    for value in (output_root, trial_dir, finalists_dir, abstention_dir, final_dir):
        value.mkdir(parents=True, exist_ok=True)
    cpu_python = Path(args.cpu_python).resolve()
    gemma_python = Path(args.gemma_python).resolve()
    if not cpu_python.is_file() or not gemma_python.is_file():
        raise FileNotFoundError("configured Python interpreter is unavailable")

    primary_paths = [
        Path(plan["primary_final_pre_rescue"][corpus]["output"]["path"])
        for corpus in plan["corpora"]
    ]
    candidate_paths = [
        Path(item["candidate"]["path"])
        for system in sorted(plan["candidate_systems"])
        for item in sorted(
            plan["candidate_systems"][system]["inputs"],
            key=lambda value: str(value["corpus"]),
        )
    ]
    policy = plan["rescue_policy"]
    finalist_adjudication_orders = list(
        policy.get("finalist_adjudication_orders") or ["original", "hashed"]
    )
    finalist_verifier_orders = list(
        policy.get("contrastive_verification_orders") or ["original", "hashed"]
    )
    typed_abstention_orders = list(
        policy.get("typed_abstention_verification_orders")
        or ["original", "hashed"]
    )
    adj_sampling = production_plan["adjudicator"]["production_sampling"]
    verifier_rendering = production_plan["verifier"]["rendering"]
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
    rescue_manifest = captures / "rescue_manifest.json"
    if not rescue_manifest.exists():
        command = [
            str(cpu_python),
            "-m",
            "scripts.tools.silver_match_v3.build_abstention_rescue",
            "--manifest",
            str(manifest),
            *_artifact_args("--candidates", candidate_paths),
            *_artifact_args("--primary", primary_paths),
            "--output-root",
            str(captures),
            "--block-size",
            str(policy["block_size"]),
            "--primary-k",
            str(policy["primary_k"]),
            "--all-abstentions",
            "--coverage-repeats",
            str(policy["coverage_repeats"]),
            "--reinclude-primary",
        ]
        _run(command, cwd=repo_root)
    rescue = json.loads(rescue_manifest.read_text(encoding="utf-8"))
    if (
        rescue.get("manifest_sha256") != plan["manifest"]["sha256"]
        or rescue.get("coverage_repeats") != policy["coverage_repeats"]
        or rescue.get("reinclude_primary") is not True
        or set(rescue.get("candidate_inputs") or {})
        != {str(path.resolve()) for path in candidate_paths}
        or set(rescue.get("primary_inputs") or {})
        != {str(path.resolve()) for path in primary_paths}
    ):
        raise ValueError("materialized rescue capture differs from the frozen plan")
    trial_paths = [Path(path) for path in sorted((rescue.get("outputs") or {}))]
    if not trial_paths:
        raise ValueError("rescue produced no eligible abstention trials")

    worker_jobs: list[list[tuple[list[str], Path]]] = [
        [] for _ in args.gpus
    ]
    trial_outputs = []
    for index, trial in enumerate(trial_paths):
        output = trial_dir / f"{trial.stem}.original.jsonl"
        trial_outputs.append(output)
        if not _meta(output).exists():
            command = adjudicator_command(
                plan=production_plan,
                repo_root=repo_root,
                gemma_python=gemma_python,
                output=output,
                order="original",
                batch_size=adj_batch_size,
                gpu_memory_utilization=adj_gpu_utilization,
                candidates=trial,
                max_candidates=int(policy["block_size"]),
            )
            worker_jobs[index % len(worker_jobs)].append(
                (command, trial_dir / f"{trial.stem}.original.log")
            )
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        active_trial_gpus = [
            gpu
            for jobs, gpu in zip(worker_jobs, args.gpus, strict=True)
            if jobs
        ]
        if active_trial_gpus:
            validate_launch_gpus(active_trial_gpus)
        futures = [
            executor.submit(_trial_worker, jobs, gpu=gpu, cwd=repo_root)
            for jobs, gpu in zip(worker_jobs, args.gpus, strict=True)
            if jobs
        ]
        for future in futures:
            future.result()
    if any(not _meta(path).is_file() for path in trial_outputs):
        raise RuntimeError("rescue trial stage lacks successful completion metadata")

    finalists = aggregate / "match_finalists.jsonl"
    no_match = aggregate / "no_match_provisional.jsonl"
    if not finalists.exists() or not no_match.exists():
        command = [
            str(cpu_python),
            "-m",
            "scripts.tools.silver_match_v3.aggregate_abstention_rescue",
            "--manifest",
            str(manifest),
            "--rescue-manifest",
            str(rescue_manifest),
            *_artifact_args("--primary", primary_paths),
            *_artifact_args("--adjudication", trial_outputs),
            "--output-root",
            str(aggregate),
            "--max-finalists",
            str(policy["max_finalists"]),
        ]
        _run(command, cwd=repo_root)

    finalist_adj = {
        order: finalists_dir / f"adjudicate.{order}.jsonl"
        for order in finalist_adjudication_orders
    }
    jobs = []
    for order, output in finalist_adj.items():
        if not _meta(output).exists():
            jobs.append(
                (
                    f"finalist-adjudicator-{order}",
                    adjudicator_command(
                        plan=production_plan,
                        repo_root=repo_root,
                        gemma_python=gemma_python,
                        output=output,
                        order=order,
                        batch_size=adj_batch_size,
                        gpu_memory_utilization=adj_gpu_utilization,
                        candidates=finalists,
                        max_candidates=int(policy["max_finalists"]),
                    ),
                    finalists_dir / f"adjudicate.{order}.log",
                )
            )
    _run_parallel_gpu(
        jobs, gpus=args.gpus, cwd=repo_root, enforce_live_gpu_policy=True
    )
    if any(not _meta(path).is_file() for path in finalist_adj.values()):
        raise RuntimeError("finalist adjudication lacks successful metadata")

    finalist_verify = {
        order: finalists_dir / f"verify.{order}.jsonl"
        for order in finalist_verifier_orders
    }
    jobs = []
    for order, output in finalist_verify.items():
        if not _meta(output).exists():
            jobs.append(
                (
                    f"finalist-verifier-{order}",
                    verifier_command(
                        plan=production_plan,
                        repo_root=repo_root,
                        gemma_python=gemma_python,
                        primary=finalist_adj["original"],
                        output=output,
                        order=order,
                        batch_size=verifier_batch_size,
                        gpu_memory_utilization=verifier_gpu_utilization,
                        candidates=finalists,
                        max_alternatives=max(1, int(policy["max_finalists"]) - 1),
                    ),
                    finalists_dir / f"verify.{order}.log",
                )
            )
    _run_parallel_gpu(
        jobs, gpus=args.gpus, cwd=repo_root, enforce_live_gpu_policy=True
    )
    if any(not _meta(path).is_file() for path in finalist_verify.values()):
        raise RuntimeError("finalist verification lacks successful metadata")

    finalist_combined = finalists_dir / "verify.strict-all-selected-orders.jsonl"
    if not finalist_combined.exists():
        combine_command = [
                str(cpu_python),
                "-m",
                "scripts.tools.silver_match_v3.combine_ordered_verifications",
                "--primary",
                str(finalist_adj["original"]),
                "--selection",
                str(Path(plan["verifier"]["selection"]["path"])),
                "--policy",
                str(Path(plan["verifier"]["production_policy"]["path"])),
                "--rescue-plan",
                str(plan_path),
                "--output",
                str(finalist_combined),
            ]
        for order in finalist_verifier_orders:
            combine_command.extend(
                ("--verification", f"{order}={finalist_verify[order]}")
            )
        _run(combine_command, cwd=repo_root)

    abstention_verify = {
        order: abstention_dir / f"verify.{order}.jsonl"
        for order in typed_abstention_orders
    }
    jobs = []
    for order, output in abstention_verify.items():
        if not _meta(output).exists():
            jobs.append(
                (
                    f"abstention-verifier-{order}",
                    _abstention_command(
                        plan=plan,
                        gemma_python=gemma_python,
                        audits=no_match,
                        output=output,
                        order=order,
                        batch_size=args.batch_size,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                    ),
                    abstention_dir / f"verify.{order}.log",
                )
            )
    _run_parallel_gpu(
        jobs, gpus=args.gpus, cwd=repo_root, enforce_live_gpu_policy=True
    )
    if any(not _meta(path).is_file() for path in abstention_verify.values()):
        raise RuntimeError("typed-abstention verification lacks successful metadata")
    abstention_combined = abstention_dir / "verify.strict-combined.jsonl"
    if not abstention_combined.exists():
        _run(
            [
                str(cpu_python),
                "-m",
                "scripts.tools.silver_match_v3.combine_two_order_abstention_verifications",
                "--audits",
                str(no_match),
                "--original",
                str(abstention_verify["original"]),
                "--hashed",
                str(abstention_verify["hashed"]),
                "--output",
                str(abstention_combined),
            ],
            cwd=repo_root,
        )

    final_all = output_root / "final.all-corpora.jsonl"
    unresolved = output_root / "unresolved.jsonl"
    merge = [
        str(cpu_python),
        "-m",
        "scripts.tools.silver_match_v3.merge_rescue_decisions",
        "--manifest",
        str(manifest),
        *_artifact_args("--primary", primary_paths),
        "--finalist-candidates",
        str(finalists),
        "--finalist-adjudications",
        str(finalist_adj["original"]),
        "--finalist-order-check",
        str(finalist_adj["hashed"]),
        "--finalist-verification",
        str(finalist_combined),
        "--no-match-audits",
        str(no_match),
        "--abstention-verifications",
        str(abstention_combined),
        "--adjudicator-selection",
        str(Path(plan["adjudicator"]["selection"]["path"])),
        "--verifier-selection",
        str(Path(plan["verifier"]["selection"]["path"])),
        "--verifier-policy",
        str(Path(plan["verifier"]["production_policy"]["path"])),
        "--rescue-plan",
        str(plan_path),
        "--unresolved-output",
        str(unresolved),
        "--strict-production",
        "--output",
        str(final_all),
    ]
    if args.manual_unresolved_labels or args.manual_unresolved_validation:
        if not args.manual_unresolved_labels or not args.manual_unresolved_validation:
            raise ValueError(
                "manual unresolved labels and validation must be supplied together"
            )
        merge.extend(
            (
                "--manual-unresolved-labels",
                str(Path(args.manual_unresolved_labels).resolve()),
            )
        )
        merge.extend(
            (
                "--manual-unresolved-validation",
                str(Path(args.manual_unresolved_validation).resolve()),
            )
        )
    if not final_all.exists():
        completed = subprocess.run(merge, cwd=repo_root, check=False)
        if completed.returncode:
            if not unresolved.is_file() or unresolved.stat().st_size == 0:
                raise RuntimeError(
                    "strict rescue merge failed without an unresolved ledger"
                )
            pack = output_root / "unresolved_blind_pack"
            if not (pack / "validation.json").exists():
                _run(
                    [
                        str(cpu_python),
                        "-m",
                        "scripts.tools.silver_match_v3.prepare_unresolved_decision_pack",
                        "--manifest",
                        str(manifest),
                        "--unresolved",
                        str(unresolved),
                        "--output-root",
                        str(pack),
                        "--chunk-size",
                        "25",
                        "--seed",
                        "161803",
                    ],
                    cwd=repo_root,
                )
            pending = {
                "schema_version": "silver-match-v3-task-rescue-run-v3",
                "status": "AWAITING_BLIND_UNRESOLVED_LABELS",
                "task": task,
                "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
                "unresolved": {
                    "path": str(unresolved),
                    "sha256": sha256_file(unresolved),
                },
                "blind_pack_validation": {
                    "path": str(pack / "validation.json"),
                    "sha256": sha256_file(pack / "validation.json"),
                },
                "verification_topology": {
                    "finalist_adjudication_orders": finalist_adjudication_orders,
                    "contrastive_verification_orders": finalist_verifier_orders,
                    "typed_abstention_verification_orders": typed_abstention_orders,
                },
            }
            (output_root / "rescue_run.pending.json").write_text(
                json.dumps(pending, indent=2, sort_keys=True) + "\n"
            )
            return pending

    final_paths = []
    for corpus in plan["corpora"]:
        output = final_dir / f"{corpus}.jsonl"
        if not output.exists():
            _run(
                [
                    str(cpu_python),
                    "-m",
                    "scripts.tools.silver_match_v3.filter_labels",
                    "--input",
                    str(final_all),
                    "--output",
                    str(output),
                    "--where",
                    f"corpus={corpus}",
                ],
                cwd=repo_root,
            )
        final_paths.append(output)
    final_audit = output_root / "final.audit.json"
    if not final_audit.exists():
        _run(
            [
                str(cpu_python),
                "-m",
                "scripts.tools.silver_match_v3.audit_final_outputs",
                "--manifest",
                str(manifest),
                "--task",
                task,
                *_artifact_args("--final", final_paths),
                "--output",
                str(final_audit),
            ],
            cwd=repo_root,
        )
    exclusion_paths = [Path(value["path"]) for value in plan["blind_audit_exclusions"]]
    for kind, seed in (("match", "271828"), ("abstention", "314159")):
        # v2 adds a standard transcript-auditable per-task full-bank label
        # pack. Keep any legacy v1 samples intact beside this append-only
        # root; they are not eligible for the production risk release.
        audit_root = output_root / f"blind_audit_{kind}_v2"
        if not (audit_root / "sample_report.json").exists():
            _run(
                [
                    str(cpu_python),
                    "-m",
                    "scripts.tools.silver_match_v3.prepare_final_decision_audit",
                    "--manifest",
                    str(manifest),
                    *_artifact_args("--final", final_paths),
                    *_artifact_args("--exclude", exclusion_paths),
                    "--output-root",
                    str(audit_root),
                    "--global-n",
                    str(frozen_blind_audit_n),
                    "--per-task-n",
                    str(frozen_blind_audit_n),
                    "--seed",
                    seed,
                    "--sample-kind",
                    kind,
                ],
                cwd=repo_root,
            )
    result = {
        "schema_version": "silver-match-v3-task-rescue-run-v3",
        "status": "COMPLETE_PENDING_BLIND_RISK_LABELS",
        "task": task,
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "final_all": {"path": str(final_all), "sha256": sha256_file(final_all)},
        "final_audit": {"path": str(final_audit), "sha256": sha256_file(final_audit)},
        "verification_topology": {
            "finalist_adjudication_orders": finalist_adjudication_orders,
            "contrastive_verification_orders": finalist_verifier_orders,
            "typed_abstention_verification_orders": typed_abstention_orders,
        },
        "blind_match_sample": _check_and_return(
            output_root / "blind_audit_match_v2" / "sample_report.json"
        ),
        "blind_abstention_sample": _check_and_return(
            output_root / "blind_audit_abstention_v2" / "sample_report.json"
        ),
    }
    result_path = output_root / "rescue_run.report_v2.json"
    if result_path.exists():
        if json.loads(result_path.read_text(encoding="utf-8")) != result:
            raise ValueError("existing rescue run report differs")
    else:
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _check_and_return(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--gemma-python", required=True)
    parser.add_argument("--cpu-python", default=sys.executable)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--blind-audit-n", type=int)
    parser.add_argument("--manual-unresolved-labels")
    parser.add_argument("--manual-unresolved-validation")
    args = parser.parse_args()
    validate_gpu_indices_for_host(args.gpus)
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must name one or more distinct devices")
    if len(args.gpus) > 4:
        parser.error("--gpus may name at most four devices")
    if args.batch_size < 1 or (
        args.blind_audit_n is not None and args.blind_audit_n < 1
    ):
        parser.error("batch/sample sizes must be positive")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
