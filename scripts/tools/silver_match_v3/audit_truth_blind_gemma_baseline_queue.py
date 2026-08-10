#!/usr/bin/env python3
"""Audit an unstarted multi-order truth-blind full-bank Gemma queue."""

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


def _bound(entry: dict[str, Any]) -> Path:
    path = Path(str(entry.get("path") or "")).resolve()
    if (
        not path.is_file()
        or sha256_file(path) != str(entry.get("sha256") or "")
        or path.stat().st_size != int(entry.get("bytes", -1))
    ):
        raise ValueError(f"missing or hash-drifted queue artifact: {path}")
    return path


def _resolve(raw: str, anchor: Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (anchor.parent / path).resolve()


def audit(queue_path: Path) -> dict[str, Any]:
    queue_path = queue_path.resolve()
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    contract = queue.get("scientific_contract") or {}
    runtime = queue.get("runtime") or {}
    orders = list(runtime.get("orders") or [])
    if (
        queue.get("schema_version")
        != "silver-match-v3-truth-blind-full-bank-gemma-baseline-queue-v1"
        or queue.get("status") != "FROZEN_QUEUED_WAITING_FOR_PROJECT_GPU_SLOT"
        or not str(queue.get("task") or "")
        or queue.get("role") != "optimize"
        or int(contract.get("row_count", -1)) < 1
        or int(contract.get("candidate_depth", -1)) < 1
        or not str(contract.get("bank_source_sha256") or "")
        or contract.get("all_current_bank_leaves_present_in_every_order") is not True
        or int(contract.get("deterministic_order_count", -1)) < 2
        or len(orders) < 2
        or len(orders) != len(set(orders))
        or contract.get(
            "truth_labels_select_rows_prior_predictions_mi_and_outcomes_hidden"
        )
        is not True
        or contract.get("baseline_may_not_be_scored_before_optimize_truth_release")
        is not True
        or contract.get("baseline_outputs_may_not_enter_truth_consensus") is not True
        or contract.get("task_local_immutable_batch_runner_required") is not True
        or runtime.get("resume") is not True
    ):
        raise ValueError("queue scope or truth-blind scientific contract is invalid")
    if orders != list(contract.get("orders") or []):
        raise ValueError("runtime and contract order topology differ")

    inputs = queue.get("inputs") or {}
    expected_inputs = {
        "manifest",
        "bank",
        "source_pack_validation",
        "source_pack_audit",
        "partition",
        "partition_freeze",
        "candidate_freeze",
        "candidates",
        "prompt_components",
        "runner",
        "python_executable",
        "implementation_inventory",
        "model_inventory",
        "runtime_dependency_inventory",
    }
    if set(inputs) != expected_inputs:
        raise ValueError("queue contains an unrecognized or missing frozen input")
    paths = {
        name: _bound(inputs[name])
        for name in expected_inputs - {"prompt_components"}
    }
    prompt_paths = [_bound(entry) for entry in inputs["prompt_components"]]
    if not prompt_paths:
        raise ValueError("queue has no prompt components")

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    bank = json.loads(paths["bank"].read_text(encoding="utf-8"))
    bank_ids = [str(row.get("metric_id") or "") for row in bank.get("metrics") or []]
    if (
        len(bank_ids) != int(contract["candidate_depth"])
        or "" in bank_ids
        or len(bank_ids) != len(set(bank_ids))
        or bank.get("task") != queue["task"]
        or bank.get("source_sha256") != contract.get("bank_source_sha256")
        or inputs["bank"].get("source_sha256") != bank.get("source_sha256")
        or int(inputs["bank"].get("metric_count", -1)) != len(bank_ids)
        or ((manifest.get("banks") or {}).get(queue["task"]) or {}).get("source_sha256")
        != bank.get("source_sha256")
    ):
        raise ValueError("queue does not bind the exact complete current task bank")

    source_validation = json.loads(paths["source_pack_validation"].read_text(encoding="utf-8"))
    source_audit = json.loads(paths["source_pack_audit"].read_text(encoding="utf-8"))
    partition_freeze = json.loads(paths["partition_freeze"].read_text(encoding="utf-8"))
    candidate_freeze = json.loads(paths["candidate_freeze"].read_text(encoding="utf-8"))
    if (
        source_validation.get("truth_hidden") is not True
        or source_validation.get("candidate_proposals_hidden") is not True
        or source_validation.get("prior_labels_predictions_mi_and_outcomes_not_read")
        is not True
        or source_audit.get("status") != "EXACT_TRUTH_AND_CANDIDATE_HIDDEN_PACK_PASS"
        or source_audit.get("labels_predictions_mi_and_outcomes_read") is not False
        or partition_freeze.get("status")
        != "FROZEN_BEFORE_ANY_DISTILLATION_LABELS_OR_PREDICTIONS"
        or (partition_freeze.get("output") or {}).get("sha256")
        != sha256_file(paths["partition"])
        or (partition_freeze.get("content_contract") or {}).get(
            "labels_predictions_metric_ids_reasons_mi_or_outcomes_used"
        )
        is not False
        or int((partition_freeze.get("role_counts") or {}).get(queue["role"], -1))
        != int(contract["row_count"])
        or candidate_freeze.get("status") != "FROZEN_BEFORE_INFERENCE"
        or candidate_freeze.get("partition_role") != queue["role"]
        or candidate_freeze.get("truth_hidden") is not True
        or candidate_freeze.get("select_rows_read") is not False
        or candidate_freeze.get(
            "prior_decisions_metric_ids_predictions_proposals_mi_and_outcomes_read"
        )
        is not False
        or (candidate_freeze.get("output") or {}).get("sha256")
        != sha256_file(paths["candidates"])
    ):
        raise ValueError("queue lineage reads labels, select rows, predictions, MI, or outcomes")

    rows = list(read_jsonl(paths["candidates"]))
    uids = [str(row.get("norm_uid") or "") for row in rows]
    if (
        len(rows) != int(contract["row_count"])
        or "" in uids
        or len(uids) != len(set(uids))
        or any(
            row.get("task") != queue["task"]
            or row.get("partition_role") != queue["role"]
            or row.get("truth_hidden") is not True
            or row.get("prior_predictions_hidden") is not True
            or row.get("bank_source_sha256") != bank.get("source_sha256")
            or [str(card.get("metric_id") or "") for card in row.get("candidates") or []]
            != bank_ids
            for row in rows
        )
    ):
        raise ValueError("candidate rows are not exact complete-bank role rows")

    implementation_inventory = json.loads(
        paths["implementation_inventory"].read_text(encoding="utf-8")
    )
    implementation_inventory_audit = assert_exact_inventory(
        paths["implementation_inventory"]
    )
    inventory_root = Path(str(implementation_inventory.get("root") or "")).resolve()
    runner = paths["runner"]
    runner_relative = str(runner.relative_to(inventory_root))
    inventory_files = {
        str(row["relative_path"]): row
        for row in implementation_inventory.get("files") or []
    }
    if (
        implementation_inventory.get("status") != "FROZEN_CONTENT_HASH_INVENTORY"
        or implementation_inventory.get("content_inventory_sha256")
        != inputs["implementation_inventory"].get("content_inventory_sha256")
        or (inventory_files.get(runner_relative) or {}).get("sha256")
        != sha256_file(runner)
        or Path(str((runtime.get("environment") or {}).get("PYTHONPATH") or "")).resolve()
        != inventory_root
        or (runtime.get("environment") or {}).get("PYTHONDONTWRITEBYTECODE") != "1"
        or inputs["implementation_inventory"].get("exact_recursive_audit_status")
        != "EXACT_RECURSIVE_PYCACHE_FREE_INVENTORY_PASS"
        or implementation_inventory_audit.get("status")
        != "EXACT_RECURSIVE_PYCACHE_FREE_INVENTORY_PASS"
    ):
        raise ValueError("batch runner is not inside the bound immutable snapshot")
    model_inventory = json.loads(paths["model_inventory"].read_text(encoding="utf-8"))
    model_inventory_audit = assert_exact_inventory(paths["model_inventory"])
    runtime_dependency_audit = assert_exact_runtime_dependencies(
        paths["runtime_dependency_inventory"]
    )
    if (
        model_inventory.get("status") != "FROZEN_CONTENT_HASH_INVENTORY"
        or model_inventory.get("content_inventory_sha256")
        != inputs["model_inventory"].get("content_inventory_sha256")
        or Path(str(model_inventory.get("root") or "")).resolve()
        != Path(str(runtime.get("model") or "")).resolve()
        or inputs["model_inventory"].get("exact_recursive_audit_status")
        != "EXACT_RECURSIVE_PYCACHE_FREE_INVENTORY_PASS"
        or model_inventory_audit.get("status")
        != "EXACT_RECURSIVE_PYCACHE_FREE_INVENTORY_PASS"
        or paths["python_executable"]
        != Path(str(runtime.get("python") or "")).resolve()
        or inputs["runtime_dependency_inventory"].get("exact_runtime_audit_status")
        != "EXACT_PYTHON_RUNTIME_DEPENDENCIES_PASS"
        or runtime_dependency_audit.get("status")
        != "EXACT_PYTHON_RUNTIME_DEPENDENCIES_PASS"
        or runtime_dependency_audit.get("python_sha256")
        != inputs["python_executable"].get("sha256")
        or inputs["runtime_dependency_inventory"].get(
            "core_content_inventory_sha256"
        )
        != runtime_dependency_audit.get("core_content_inventory_sha256")
    ):
        raise ValueError("Gemma model snapshot is not inventory-bound")

    system_prompt = "\n\n".join(
        path.read_text(encoding="utf-8").rstrip() for path in prompt_paths
    ) + "\n"
    if hashlib.sha256(system_prompt.encode("utf-8")).hexdigest() != (
        queue.get("prompt") or {}
    ).get("combined_sha256"):
        raise ValueError("combined prompt hash drift")
    preflight = queue.get("prompt") or {}
    wanted = set(uids)
    norms: dict[str, dict[str, Any]] = {}
    for corpus, meta in sorted((manifest.get("corpora") or {}).items()):
        if meta.get("task") != queue["task"]:
            continue
        for row in read_jsonl(_resolve(str(meta["path"]), paths["manifest"])):
            uid = str(row.get("norm_uid") or "")
            if uid in wanted:
                if uid in norms:
                    raise ValueError(f"duplicate canonical queue UID: {uid}")
                norms[uid] = row
    if set(norms) != wanted:
        raise ValueError("queue UIDs are absent from the canonical manifest")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(runtime["model"], trust_remote_code=True)
    metric_by_id = {str(row["metric_id"]): row for row in bank["metrics"]}
    observed_counts: dict[str, list[int]] = {order: [] for order in orders}
    for candidate in rows:
        norm = norms[str(candidate["norm_uid"])]
        for order in orders:
            cards = ordered_candidates(candidate["candidates"], order, norm["norm_uid"])
            rendered_prompt = build_item_prompt(
                system_prompt,
                norm,
                cards,
                metric_by_id,
                context_chars=int(runtime["context_chars"]),
                description_chars=int(runtime["description_chars"]),
                example_chars=int(runtime["example_chars"]),
                max_examples=int(runtime["max_examples"]),
            )
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": rendered_prompt}],
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
            observed_counts[order].append(len(token_ids))
    recorded_counts = preflight.get("token_preflight") or {}
    if (
        preflight.get("all_rows_fit") is not True
        or int(preflight.get("max_total_budget", 10**9))
        > int(preflight.get("max_model_len", -1))
        or int(preflight.get("max_model_len", -1)) != int(runtime.get("max_model_len", -2))
        or int(preflight.get("max_tokens", -1)) != int(runtime.get("max_tokens", -2))
        or set(recorded_counts) != set(orders)
        or any(
            int(row.get("count", -1)) != len(rows)
            for row in recorded_counts.values()
        )
        or any(
            int(recorded_counts[order].get("prompt_tokens_min", -1))
            != min(observed_counts[order])
            or int(recorded_counts[order].get("prompt_tokens_max", -1))
            != max(observed_counts[order])
            for order in orders
        )
        or max(max(values) for values in observed_counts.values())
        + int(runtime["max_tokens"])
        > int(runtime["max_model_len"])
    ):
        raise ValueError("full-bank token preflight is invalid")

    commands = queue.get("commands") or {}
    outputs = queue.get("outputs") or {}
    if set(commands) != set(orders) or set(outputs) != set(orders):
        raise ValueError("command/output topology differs from frozen orders")
    for order in orders:
        command = [str(value) for value in commands[order]]
        output_path = Path(str((outputs[order] or {}).get("path") or "")).resolve()
        meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
        if (
            command[:4]
            != [runtime["python"], "-u", "-m", "scripts.tools.silver_match_v3.adjudicate_gemma"]
            or outputs[order].get("existed_before_freeze") is not False
            or output_path.exists()
            or meta_path.exists()
            or "--keep-raw" not in command
            or "--resume" not in command
        ):
            raise ValueError(f"{order} is not a clean unstarted direct-batch command")

        def arg(flag: str) -> str:
            return command[command.index(flag) + 1]

        singleton_flags = (
            "--manifest",
            "--candidates",
            "--output",
            "--prompt",
            "--model",
            "--max-candidates",
            "--context-chars",
            "--description-chars",
            "--example-chars",
            "--max-examples",
            "--batch-size",
            "--max-model-len",
            "--max-tokens",
            "--gpu-memory-utilization",
            "--seed",
            "--order-mode",
            "--keep-raw",
            "--resume",
        )
        if (
            any(command.count(flag) != 1 for flag in singleton_flags)
            or Path(arg("--manifest")).resolve() != paths["manifest"]
            or Path(arg("--candidates")).resolve() != paths["candidates"]
            or Path(arg("--output")).resolve() != output_path
            or Path(arg("--prompt")).resolve() != prompt_paths[0]
            or [Path(command[i + 1]).resolve() for i, value in enumerate(command) if value == "--prompt-addon"]
            != prompt_paths[1:]
            or Path(arg("--model")).resolve() != Path(runtime["model"]).resolve()
            or arg("--order-mode") != order
            or int(arg("--max-candidates")) != len(bank_ids)
            or int(arg("--context-chars")) != int(runtime["context_chars"])
            or int(arg("--description-chars")) != int(runtime["description_chars"])
            or int(arg("--example-chars")) != int(runtime["example_chars"])
            or int(arg("--max-examples")) != int(runtime["max_examples"])
            or int(arg("--batch-size")) != int(runtime["batch_size"])
            or int(arg("--max-model-len")) != int(runtime["max_model_len"])
            or int(arg("--max-tokens")) != int(runtime["max_tokens"])
            or float(arg("--gpu-memory-utilization"))
            != float(runtime["gpu_memory_utilization"])
            or int(arg("--seed")) != int(runtime["seed"])
            or float(runtime.get("temperature", -1)) != 0.0
            or str((runtime.get("environment") or {}).get("CUDA_VISIBLE_DEVICES"))
            != str(runtime.get("requested_gpu"))
        ):
            raise ValueError(f"{order} command differs from the frozen queue")

    return {
        "schema_version": "silver-match-v3-truth-blind-gemma-baseline-queue-audit-v1",
        "status": "EXACT_HASH_PINNED_MULTI_ORDER_QUEUE_PASS_WAITING_FOR_GPU",
        "queue": {"path": str(queue_path), "sha256": sha256_file(queue_path)},
        "task": queue["task"],
        "role": queue["role"],
        "row_count": len(rows),
        "bank_metric_count": len(bank_ids),
        "bank_source_sha256": bank["source_sha256"],
        "orders": orders,
        "truth_labels_select_rows_predictions_mi_and_outcomes_hidden": True,
        "task_local_immutable_batch_runner": True,
        "implementation_snapshot_exact_recursive_pycache_free": True,
        "model_snapshot_exact_recursive_hash_bound": True,
        "python_executable_hash_and_size_bound": True,
        "python_runtime_dependencies_exactly_bound": True,
        "outputs_absent_before_launch": True,
        "interruption_safe_append_only_resume": True,
        "all_bound_artifact_hashes_pass": True,
        "prompt_token_budget_pass": True,
        "score_before_optimize_truth_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = audit(Path(args.queue))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**result, "audit_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
