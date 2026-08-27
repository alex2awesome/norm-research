#!/usr/bin/env python3
"""Freeze a multi-order truth-blind full-bank Gemma direct-batch queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .adjudicate_gemma import build_item_prompt, ordered_candidates
from .audit_exact_directory_inventory import assert_exact_inventory
from .common import read_jsonl, sha256_file
from .freeze_python_runtime_dependency_inventory import (
    assert_exact_runtime_dependencies,
)


def _ref(path: Path, **extra: Any) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        **extra,
    }


def _resolve(raw: str, anchor: Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (anchor.parent / path).resolve()


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = Path(args.manifest).resolve()
    candidate_path = Path(args.candidates).resolve()
    candidate_freeze_path = Path(args.candidate_freeze).resolve()
    source_pack_validation_path = Path(args.source_pack_validation).resolve()
    source_pack_audit_path = Path(args.source_pack_audit).resolve()
    partition_path = Path(args.partition).resolve()
    partition_freeze_path = Path(args.partition_freeze).resolve()
    runner_path = Path(args.runner).resolve()
    implementation_inventory_path = Path(args.implementation_inventory).resolve()
    model_inventory_path = Path(args.model_inventory).resolve()
    runtime_dependency_inventory_path = Path(args.runtime_dependency_inventory).resolve()
    python_path = Path(args.python).resolve()
    model_path = Path(args.model).resolve()
    prompt_paths = [Path(args.prompt).resolve(), *[Path(p).resolve() for p in args.prompt_addon]]
    orders = list(args.order)
    if len(orders) < 2 or len(orders) != len(set(orders)):
        raise ValueError("at least two distinct deterministic orders are required")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_freeze = json.loads(candidate_freeze_path.read_text(encoding="utf-8"))
    source_validation = json.loads(source_pack_validation_path.read_text(encoding="utf-8"))
    source_audit = json.loads(source_pack_audit_path.read_text(encoding="utf-8"))
    partition_freeze = json.loads(partition_freeze_path.read_text(encoding="utf-8"))
    if (
        sha256_file(manifest_path) != args.manifest_sha256
        or source_validation.get("task") != args.task
        or source_validation.get("truth_hidden") is not True
        or source_validation.get("candidate_proposals_hidden") is not True
        or source_audit.get("status") != "EXACT_TRUTH_AND_CANDIDATE_HIDDEN_PACK_PASS"
        or source_audit.get("task") != args.task
        or source_audit.get("labels_predictions_mi_and_outcomes_read") is not False
        or partition_freeze.get("status")
        != "FROZEN_BEFORE_ANY_DISTILLATION_LABELS_OR_PREDICTIONS"
        or (partition_freeze.get("output") or {}).get("sha256")
        != sha256_file(partition_path)
        or (partition_freeze.get("content_contract") or {}).get(
            "labels_predictions_metric_ids_reasons_mi_or_outcomes_used"
        )
        is not False
        or candidate_freeze.get("schema_version")
        != "silver-match-v3-partition-role-full-bank-candidates-freeze-v1"
        or candidate_freeze.get("status") != "FROZEN_BEFORE_INFERENCE"
        or candidate_freeze.get("task") != args.task
        or candidate_freeze.get("partition_role") != args.role
        or candidate_freeze.get("truth_hidden") is not True
        or candidate_freeze.get("select_rows_read") is not False
        or candidate_freeze.get(
            "prior_decisions_metric_ids_predictions_proposals_mi_and_outcomes_read"
        )
        is not False
        or (candidate_freeze.get("output") or {}).get("sha256")
        != sha256_file(candidate_path)
    ):
        raise ValueError("queue inputs are not exact truth-blind frozen artifacts")

    bank_meta = (manifest.get("banks") or {}).get(args.task) or {}
    bank_path = _resolve(str(bank_meta.get("path") or ""), manifest_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    metrics = list(bank.get("metrics") or [])
    metric_ids = [str(row.get("metric_id") or "") for row in metrics]
    if (
        not metric_ids
        or "" in metric_ids
        or len(metric_ids) != len(set(metric_ids))
        or bank.get("source_sha256") != candidate_freeze.get("bank_source_sha256")
        or len(metric_ids) != int(candidate_freeze.get("candidate_depth", -1))
    ):
        raise ValueError("candidate freeze does not bind the complete current bank")
    candidates = list(read_jsonl(candidate_path))
    uids = [str(row.get("norm_uid") or "") for row in candidates]
    if (
        len(candidates) != args.expected_count
        or "" in uids
        or len(uids) != len(set(uids))
        or any(
            row.get("task") != args.task
            or row.get("partition_role") != args.role
            or row.get("truth_hidden") is not True
            or row.get("prior_predictions_hidden") is not True
            or [str(card.get("metric_id") or "") for card in row.get("candidates") or []]
            != metric_ids
            for row in candidates
        )
    ):
        raise ValueError("candidate rows are not an exact complete-bank role pack")

    implementation_inventory = json.loads(
        implementation_inventory_path.read_text(encoding="utf-8")
    )
    implementation_inventory_audit = assert_exact_inventory(implementation_inventory_path)
    runner_relative = str(runner_path.relative_to(Path(implementation_inventory["root"]).resolve()))
    inventory_files = {
        str(row["relative_path"]): row for row in implementation_inventory.get("files") or []
    }
    if (
        implementation_inventory.get("status") != "FROZEN_CONTENT_HASH_INVENTORY"
        or (inventory_files.get(runner_relative) or {}).get("sha256")
        != sha256_file(runner_path)
    ):
        raise ValueError("runner is not hash-bound inside the immutable implementation snapshot")
    model_inventory = json.loads(model_inventory_path.read_text(encoding="utf-8"))
    model_inventory_audit = assert_exact_inventory(model_inventory_path)
    runtime_dependency_audit = assert_exact_runtime_dependencies(
        runtime_dependency_inventory_path
    )
    if (
        model_inventory.get("status") != "FROZEN_CONTENT_HASH_INVENTORY"
        or Path(str(model_inventory.get("root") or "")).resolve() != model_path
        or model_inventory_audit.get("status")
        != "EXACT_RECURSIVE_PYCACHE_FREE_INVENTORY_PASS"
        or runtime_dependency_audit.get("python_sha256") != sha256_file(python_path)
    ):
        raise ValueError("model inventory does not bind the requested model snapshot")
    if not python_path.is_file() or not model_path.is_dir():
        raise FileNotFoundError("frozen Python or model runtime is absent")
    for path in prompt_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    system_prompt = "\n\n".join(
        path.read_text(encoding="utf-8").rstrip() for path in prompt_paths
    ) + "\n"
    combined_prompt_sha256 = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()

    norms: dict[str, dict[str, Any]] = {}
    wanted = set(uids)
    for corpus, meta in sorted((manifest.get("corpora") or {}).items()):
        if meta.get("task") != args.task:
            continue
        source = _resolve(str(meta["path"]), manifest_path)
        for row in read_jsonl(source):
            uid = str(row.get("norm_uid") or "")
            if uid in wanted:
                if uid in norms:
                    raise ValueError(f"duplicate canonical queue UID: {uid}")
                norms[uid] = row
    if set(norms) != wanted:
        raise ValueError("queue UIDs are absent from the canonical manifest")
    metric_by_id = {str(row["metric_id"]): row for row in metrics}

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    token_counts: dict[str, list[int]] = {order: [] for order in orders}
    for candidate in candidates:
        norm = norms[str(candidate["norm_uid"])]
        for order in orders:
            cards = ordered_candidates(candidate["candidates"], order, norm["norm_uid"])
            prompt = build_item_prompt(
                system_prompt,
                norm,
                cards,
                metric_by_id,
                context_chars=args.context_chars,
                description_chars=args.description_chars,
                example_chars=args.example_chars,
                max_examples=args.max_examples,
            )
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            token_ids = encoded.get("input_ids") if hasattr(encoded, "get") else encoded
            if (
                not isinstance(token_ids, list)
                or not token_ids
                or isinstance(token_ids[0], list)
            ):
                raise ValueError("tokenizer returned an unexpected chat-template shape")
            token_counts[order].append(len(token_ids))
    maximum_prompt_tokens = max(max(values) for values in token_counts.values())
    if maximum_prompt_tokens + args.max_tokens > args.max_model_len:
        raise ValueError("full-bank prompt exceeds the frozen model-length budget")

    output_root = output.parent / "runs"
    outputs: dict[str, dict[str, Any]] = {}
    commands: dict[str, list[str]] = {}
    for order in orders:
        run_output = output_root / (
            f"gemma4.{args.task}.{args.role}.full-bank{len(metric_ids)}.{order}.jsonl"
        )
        if run_output.exists() or run_output.with_suffix(run_output.suffix + ".meta.json").exists():
            raise FileExistsError(f"baseline output already exists: {run_output}")
        command = [
            str(python_path),
            "-u",
            "-m",
            "scripts.tools.silver_match_v3.adjudicate_gemma",
            "--manifest",
            str(manifest_path),
            "--candidates",
            str(candidate_path),
            "--output",
            str(run_output),
            "--prompt",
            str(prompt_paths[0]),
        ]
        for addon in prompt_paths[1:]:
            command.extend(["--prompt-addon", str(addon)])
        command.extend(
            [
                "--model",
                str(model_path),
                "--max-candidates",
                str(len(metric_ids)),
                "--context-chars",
                str(args.context_chars),
                "--description-chars",
                str(args.description_chars),
                "--example-chars",
                str(args.example_chars),
                "--max-examples",
                str(args.max_examples),
                "--batch-size",
                str(args.batch_size),
                "--max-model-len",
                str(args.max_model_len),
                "--max-tokens",
                str(args.max_tokens),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--seed",
                str(args.seed),
                "--order-mode",
                order,
                "--keep-raw",
                "--resume",
            ]
        )
        outputs[order] = {"path": str(run_output), "existed_before_freeze": False}
        commands[order] = command

    result = {
        "schema_version": "silver-match-v3-truth-blind-full-bank-gemma-baseline-queue-v1",
        "status": "FROZEN_QUEUED_WAITING_FOR_PROJECT_GPU_SLOT",
        "task": args.task,
        "role": args.role,
        "scientific_contract": {
            "row_count": len(candidates),
            "candidate_depth": len(metric_ids),
            "bank_source_sha256": bank["source_sha256"],
            "all_current_bank_leaves_present_in_every_order": True,
            "deterministic_order_count": len(orders),
            "orders": orders,
            "truth_labels_select_rows_prior_predictions_mi_and_outcomes_hidden": True,
            "baseline_may_not_be_scored_before_optimize_truth_release": True,
            "baseline_outputs_may_not_enter_truth_consensus": True,
            "task_local_immutable_batch_runner_required": True,
        },
        "inputs": {
            "manifest": _ref(manifest_path),
            "bank": _ref(bank_path, source_sha256=bank["source_sha256"], metric_count=len(metric_ids)),
            "source_pack_validation": _ref(source_pack_validation_path),
            "source_pack_audit": _ref(source_pack_audit_path),
            "partition": _ref(partition_path),
            "partition_freeze": _ref(partition_freeze_path),
            "candidate_freeze": _ref(candidate_freeze_path),
            "candidates": _ref(candidate_path, rows=len(candidates), candidate_depth=len(metric_ids)),
            "prompt_components": [_ref(path) for path in prompt_paths],
            "runner": _ref(runner_path),
            "python_executable": _ref(python_path),
            "implementation_inventory": _ref(
                implementation_inventory_path,
                content_inventory_sha256=implementation_inventory["content_inventory_sha256"],
                exact_recursive_audit_status=implementation_inventory_audit["status"],
            ),
            "model_inventory": _ref(
                model_inventory_path,
                content_inventory_sha256=model_inventory["content_inventory_sha256"],
                exact_recursive_audit_status=model_inventory_audit["status"],
            ),
            "runtime_dependency_inventory": _ref(
                runtime_dependency_inventory_path,
                exact_runtime_audit_status=runtime_dependency_audit["status"],
                core_content_inventory_sha256=runtime_dependency_audit[
                    "core_content_inventory_sha256"
                ],
            ),
        },
        "prompt": {
            "combined_sha256": combined_prompt_sha256,
            "token_preflight": {
                order: {
                    "count": len(values),
                    "prompt_tokens_min": min(values),
                    "prompt_tokens_max": max(values),
                }
                for order, values in token_counts.items()
            },
            "max_tokens": args.max_tokens,
            "max_total_budget": maximum_prompt_tokens + args.max_tokens,
            "max_model_len": args.max_model_len,
            "all_rows_fit": True,
        },
        "runtime": {
            "host": args.host,
            "python": str(python_path),
            "model": str(model_path),
            "requested_gpu": args.requested_gpu,
            "launch_condition": "project occupancy below four active GPUs and requested GPU physically free",
            "environment": {
                "PYTHONPATH": str(Path(implementation_inventory["root"]).resolve()),
                "CUDA_VISIBLE_DEVICES": str(args.requested_gpu),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "temperature": 0.0,
            "seed": args.seed,
            "orders": orders,
            "max_candidates": len(metric_ids),
            "context_chars": args.context_chars,
            "description_chars": args.description_chars,
            "example_chars": args.example_chars,
            "max_examples": args.max_examples,
            "batch_size": args.batch_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "keep_raw": True,
            "resume": True,
        },
        "outputs": outputs,
        "commands": commands,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**result, "queue_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--source-pack-validation", required=True)
    parser.add_argument("--source-pack-audit", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--partition-freeze", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--candidate-freeze", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-addon", action="append", default=[])
    parser.add_argument("--runner", required=True)
    parser.add_argument("--implementation-inventory", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-inventory", required=True)
    parser.add_argument("--runtime-dependency-inventory", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--requested-gpu", type=int, required=True)
    parser.add_argument("--order", action="append", required=True, choices=("original", "hashed", "reverse"))
    parser.add_argument("--context-chars", type=int, default=800)
    parser.add_argument("--description-chars", type=int, default=40)
    parser.add_argument("--example-chars", type=int, default=40)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.expected_count < 1 or args.batch_size < 1:
        parser.error("counts and batch size must be positive")
    print(json.dumps(freeze(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
