#!/usr/bin/env python3
"""Wait for the fixed Humor Gemma LoRA, then run its sealed select evaluation."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


EXPECTED_DATASET_SHA = "679ee4d2feb5a35beb977788382a81dff2402bb78c13c3b7fd1ac68b32f887f2"
EXPECTED_TRAINER_SHA = "d5f811f7662af5701f04a6dcb2c9c10419cb577c798eb9ba1c4fdd3d46db71cc"
SK3_ALLOWED_GPU_INDICES = frozenset({0, 5, 6, 7})
SK3_PROHIBITED_GPU_INDICES = frozenset({1, 2, 3, 4})


def validate_target_gpu_for_host(target_gpu: int) -> None:
    host = socket.gethostname().split(".", 1)[0].lower()
    is_sk3 = host in {"sk3", "skampere3"} or host.startswith("skampere3-")
    if is_sk3 and target_gpu not in SK3_ALLOWED_GPU_INDICES:
        raise ValueError(
            f"sk3 GPU policy prohibits target {target_gpu}; "
            f"allowed={sorted(SK3_ALLOWED_GPU_INDICES)}"
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def gpu_state() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gpu_raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpus = []
    for line in gpu_raw.splitlines():
        index, uuid, memory, utilization = [value.strip() for value in line.split(",")]
        gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "memory_used_mib": int(memory),
                "utilization_percent": int(utilization),
            }
        )
    process_raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    processes = []
    for line in process_raw.splitlines():
        if not line.strip():
            continue
        uuid, pid, memory, name = [value.strip() for value in line.split(",", 3)]
        owner = subprocess.run(
            ["ps", "-o", "user=", "-p", pid],
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
        processes.append(
            {
                "gpu_uuid": uuid,
                "pid": int(pid),
                "used_memory_mib": int(memory),
                "process_name": name,
                "owner": owner,
            }
        )
    return gpus, processes


def target_is_available(
    target_gpu: int,
    idle_memory_mib: int,
) -> tuple[bool, dict[str, Any]]:
    gpus, processes = gpu_state()
    by_index = {row["index"]: row for row in gpus}
    if target_gpu not in by_index:
        raise ValueError(f"target GPU does not exist: {target_gpu}")
    target = by_index[target_gpu]
    target_processes = [row for row in processes if row["gpu_uuid"] == target["uuid"]]
    available = (
        not target_processes
        and target["memory_used_mib"] <= idle_memory_mib
        and target["utilization_percent"] == 0
    )
    return available, {
        "target": target,
        "target_processes": target_processes,
        "gpu_count_gate_applied": False,
    }


def validate_static_bindings(plan: Mapping[str, Any]) -> None:
    for name, binding in (plan.get("static_bindings") or {}).items():
        path = Path(str(binding["path"]))
        actual = sha256_file(path)
        if actual != binding["sha256"]:
            raise ValueError(f"static binding drift: {name}: {actual} != {binding['sha256']}")


def validate_training_result(report_path: Path, adapter: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recipe = report.get("recipe") or {}
    adapter_report = report.get("adapter") or {}
    config = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    if (
        report.get("schema_version") != "silver-match-v3-gemma4-typed-lora-train-report-v1"
        or report.get("status") != "COMPLETE_ADAPTER_ONLY_RELOAD_VERIFIED"
        or ((report.get("dataset") or {}).get("sha256")) != EXPECTED_DATASET_SHA
        or ((report.get("trainer_script") or {}).get("sha256")) != EXPECTED_TRAINER_SHA
        or recipe.get("epochs") != 1
        or recipe.get("per_device_batch_size") != 2
        or recipe.get("gradient_accumulation_steps") != 8
        or recipe.get("learning_rate") != 1e-4
        or recipe.get("seed") != 94137
        or (recipe.get("lora") or {}).get("r") != 16
        or (recipe.get("lora") or {}).get("alpha") != 32
        or (recipe.get("lora") or {}).get("dropout") != 0.05
        or adapter_report.get("adapter_only") is not True
        or adapter_report.get("inference_reload_verified") is not True
        or Path(str(adapter_report.get("directory") or "")).resolve() != adapter.resolve()
        or (adapter_report.get("config") or {}).get("sha256") != sha256_file(config)
        or (adapter_report.get("weights") or {}).get("sha256") != sha256_file(weights)
    ):
        raise ValueError("completed Gemma adapter does not satisfy the frozen training contract")
    return report


def validate_truth_blind_eval_inputs(paths: Mapping[str, Path]) -> None:
    candidate_meta = json.loads(paths["candidate_meta"].read_text(encoding="utf-8"))
    preflight = json.loads(paths["prompt_preflight"].read_text(encoding="utf-8"))
    candidate_input = (preflight.get("inputs") or {}).get("candidates") or {}
    prompt_inputs = (preflight.get("inputs") or {}).get("prompt_components") or []
    if (
        candidate_meta.get("status") != "COMPLETE_TRUTH_BLIND_CANDIDATES"
        or candidate_meta.get("truth_fields_read") is not False
        or int(candidate_meta.get("count", -1)) != 300
        or int(candidate_meta.get("top_k", -1)) != 16
        or ((candidate_meta.get("output") or {}).get("sha256"))
        != sha256_file(paths["candidates"])
        or preflight.get("status") != "PASS_NO_CONTEXT_OVERFLOW"
        or preflight.get("truth_read") is not False
        or int(preflight.get("violation_count", -1)) != 0
        or int((preflight.get("generation") or {}).get("max_model_len", -1)) != 4096
        or int((preflight.get("generation") or {}).get("max_tokens", -1)) != 160
        or candidate_input.get("sha256") != sha256_file(paths["candidates"])
        or [row.get("sha256") for row in prompt_inputs]
        != [sha256_file(paths["base_prompt"]), sha256_file(paths["prompt_addon"])]
    ):
        raise ValueError("truth-blind candidate or exact-token preflight contract failed")


def run_logged(command: list[str], environment: Mapping[str, str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            env=dict(environment),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        handle.flush()
        os.fsync(handle.fileno())
    return int(process.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--python-overlay", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-gpu", type=int, default=0)
    parser.add_argument("--idle-memory-mib", type=int, default=1024)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-polls", type=int, default=1200)
    args = parser.parse_args()
    validate_target_gpu_for_host(args.target_gpu)

    runtime = Path(args.runtime_root).resolve()
    training = Path(args.training_root).resolve()
    python = Path(args.python).resolve()
    model = Path(args.model).resolve()
    code_root = runtime / "code"
    module_root = code_root / "scripts" / "tools" / "silver_match_v3"
    paths = {
        "training_queue": training / "queues" / "humor_gemma4_typed_v1.queue.retry1.json",
        "training_report": training / "outputs" / "humor_gemma4_typed_v1_retry1.train.report.json",
        "adapter_config": training / "outputs" / "humor_gemma4_typed_v1_retry1" / "adapter_config.json",
        "adapter_weights": training / "outputs" / "humor_gemma4_typed_v1_retry1" / "adapter_model.safetensors",
        "manifest": runtime / "inputs" / "manifest.sk2.json",
        "candidates": runtime / "inputs" / "candidates.top16.jsonl",
        "candidate_meta": runtime / "inputs" / "candidates.top16.meta.json",
        "prompt_preflight": runtime / "inputs" / "prompt_token_preflight.json",
        "base_prompt": runtime / "prompts" / "gepa_round1_candidate.txt",
        "prompt_addon": runtime / "prompts" / "gepa_humor_k50_r2_cleantrain.txt",
        "model_inventory": training / "inputs" / "model_inventory.json",
        "runner": module_root / "run_paired_gemma_lora_batch.py",
        "scorer": module_root / "score_humor_gemma_lora_select.py",
        "watcher": Path(__file__).resolve(),
        "truth": runtime / "truth_locked" / "resolved293.jsonl",
        "truth_report": runtime / "truth_locked" / "consensus.report.json",
        "unresolved": runtime / "truth_locked" / "unresolved7.jsonl",
    }
    adapter = paths["adapter_config"].parent
    paired_root = runtime / "outputs" / "paired_v1"
    score_output = runtime / "outputs" / "fresh_select_score_v1.json"
    row_audit = runtime / "outputs" / "fresh_select_score_v1.rows.jsonl"
    inference_log = runtime / "logs" / "paired_inference_v1.log"
    scoring_log = runtime / "logs" / "fresh_select_score_v1.log"
    freeze_path = runtime / "queues" / "train_then_eval_v1.freeze.json"
    run_record = runtime / "logs" / "train_then_eval_v1.run_record.json"
    events = runtime / "logs" / "train_then_eval_v1.events.jsonl"
    if run_record.exists():
        raise FileExistsError(run_record)

    existing_static = {
        name: ref(path)
        for name, path in paths.items()
        if name not in {"training_report", "adapter_config", "adapter_weights"}
    }
    inference_command = [
        str(python), "-u", "-m", "scripts.tools.silver_match_v3.run_paired_gemma_lora_batch",
        "--manifest", str(paths["manifest"]),
        "--candidates", str(paths["candidates"]),
        "--prompt", str(paths["base_prompt"]),
        "--prompt-addon", str(paths["prompt_addon"]),
        "--model", str(model),
        "--model-inventory", str(paths["model_inventory"]),
        "--adapter", str(adapter),
        "--adapter-name", "humor_typed_v1",
        "--adapter-id", "1",
        "--output-root", str(paired_root),
        "--max-candidates", "16",
        "--context-chars", "1400",
        "--description-chars", "520",
        "--example-chars", "180",
        "--max-examples", "2",
        "--batch-size", "128",
        "--max-model-len", "4096",
        "--max-tokens", "160",
        "--gpu-memory-utilization", "0.88",
        "--max-lora-rank", "16",
        "--seed", "17",
        "--resume",
    ]
    scoring_command = [
        str(python), "-u", "-m", "scripts.tools.silver_match_v3.score_humor_gemma_lora_select",
        "--truth", str(paths["truth"]),
        "--truth-consensus-report", str(paths["truth_report"]),
        "--unresolved-exclusions", str(paths["unresolved"]),
        "--candidates", str(paths["candidates"]),
        "--paired-original", str(paired_root / "paired.original.jsonl"),
        "--paired-hashed", str(paired_root / "paired.hashed.jsonl"),
        "--inference-freeze", str(paired_root / "truth_blind_inference.freeze.json"),
        "--inference-meta", str(paired_root / "paired_inference.meta.json"),
        "--output", str(score_output),
        "--row-audit-output", str(row_audit),
        "--minimum-exact-gain", "0.03",
        "--minimum-stability", "0.90",
        "--maximum-invalid-rate", "0.01",
        "--alpha", "0.05",
    ]
    freeze = {
        "schema_version": "silver-match-v3-humor-gemma4-train-then-eval-freeze-v1",
        "status": "FROZEN_DURING_TRAINING_BEFORE_ADAPTER_RESULT",
        "frozen_at": utc_now(),
        "task": "humor",
        "static_bindings": existing_static,
        "pending_training_outputs": {
            name: str(paths[name])
            for name in ("training_report", "adapter_config", "adapter_weights")
        },
        "truth_firewall": {
            "inference_command_contains_truth_path": False,
            "scoring_is_separate_post_inference_process": True,
            "truth_may_not_select_hyperparameters_or_seed": True,
        },
        "inference_command": inference_command,
        "scoring_command": scoring_command,
        "gpu_policy": {
            "target_gpu": args.target_gpu,
            "gpu_count_gate_applied": False,
            "idle_memory_mib": args.idle_memory_mib,
            "stable_idle_polls_required": 2,
            "co_location_forbidden": True,
            "sk3_allowed_gpu_indices": sorted(SK3_ALLOWED_GPU_INDICES),
            "sk3_prohibited_gpu_indices": sorted(SK3_PROHIBITED_GPU_INDICES),
        },
    }
    forbidden = {str(paths["truth"]), str(paths["truth_report"]), str(paths["unresolved"])}
    if forbidden & set(inference_command):
        raise ValueError("truth artifact leaked into the inference command")
    if freeze_path.exists():
        current = json.loads(freeze_path.read_text(encoding="utf-8"))
        comparable_current = dict(current)
        comparable_current.pop("frozen_at", None)
        comparable_freeze = dict(freeze)
        comparable_freeze.pop("frozen_at", None)
        if comparable_current != comparable_freeze:
            raise ValueError("existing watcher freeze differs from current plan")
    else:
        write_json_new(freeze_path, freeze)
    validate_static_bindings(freeze)
    validate_truth_blind_eval_inputs(paths)

    def event(payload: Mapping[str, Any]) -> None:
        with events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": utc_now(), **payload}, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    event({"status": "WAITING_FOR_TRAINING_REPORT", "freeze_sha256": sha256_file(freeze_path)})
    for _ in range(args.max_polls):
        validate_static_bindings(freeze)
        validate_truth_blind_eval_inputs(paths)
        if all(paths[name].is_file() for name in ("training_report", "adapter_config", "adapter_weights")):
            break
        time.sleep(args.poll_seconds)
    else:
        event({"status": "TIMEOUT_WAITING_FOR_TRAINING"})
        raise TimeoutError("training did not finish within watcher horizon")

    training_report = validate_training_result(paths["training_report"], adapter)
    event(
        {
            "status": "TRAINING_RESULT_VERIFIED",
            "training_report_sha256": sha256_file(paths["training_report"]),
            "adapter_config_sha256": sha256_file(paths["adapter_config"]),
            "adapter_weights_sha256": sha256_file(paths["adapter_weights"]),
        }
    )
    stable = 0
    launch_gpu_state: dict[str, Any] | None = None
    for _ in range(args.max_polls):
        available, state = target_is_available(args.target_gpu, args.idle_memory_mib)
        stable = stable + 1 if available else 0
        if stable >= 2:
            launch_gpu_state = state
            break
        time.sleep(args.poll_seconds)
    else:
        event({"status": "TIMEOUT_WAITING_FOR_IDLE_GPU"})
        raise TimeoutError("target GPU did not become stably idle")

    environment = dict(os.environ)
    library_path = os.pathsep.join(
        value
        for value in (
            "/usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs",
            "/usr/lib/x86_64-linux-gnu",
            environment.get("LIBRARY_PATH", ""),
        )
        if value
    )
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.target_gpu),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HF_HOME": str(training / "home" / ".cache" / "huggingface"),
            "HF_HUB_OFFLINE": "1",
            "HOME": str(training / "home"),
            "LIBRARY_PATH": library_path,
            "PYTHONPATH": f"{code_root}:{Path(args.python_overlay).resolve()}",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "XDG_CACHE_HOME": str(training / "home" / ".cache"),
            "FLASHINFER_WORKSPACE_BASE": str(training),
            "VLLM_USE_FLASHINFER_MOE_FP8": "0",
        }
    )
    event({"status": "LAUNCHING_PAIRED_INFERENCE", "gpu_state": launch_gpu_state})
    inference_started = utc_now()
    inference_returncode = run_logged(inference_command, environment, inference_log)
    if inference_returncode != 0:
        record = {
            "schema_version": "silver-match-v3-humor-gemma4-train-then-eval-run-v1",
            "status": "INFERENCE_FAILED",
            "completed_at": utc_now(),
            "inference_returncode": inference_returncode,
            "inference_log": ref(inference_log),
            "freeze": ref(freeze_path),
        }
        write_json_new(run_record, record)
        raise RuntimeError(f"paired inference failed: {inference_returncode}")
    inference_meta = paired_root / "paired_inference.meta.json"
    if not inference_meta.is_file():
        raise FileNotFoundError(inference_meta)
    event({"status": "PAIRED_INFERENCE_COMPLETE", "meta_sha256": sha256_file(inference_meta)})

    scoring_started = utc_now()
    scoring_returncode = run_logged(scoring_command, environment, scoring_log)
    if scoring_returncode != 0:
        status = "SCORING_FAILED"
    else:
        status = "COMPLETE"
    artifacts = {
        name: ref(path)
        for name, path in {
            "training_report": paths["training_report"],
            "adapter_config": paths["adapter_config"],
            "adapter_weights": paths["adapter_weights"],
            "inference_freeze": paired_root / "truth_blind_inference.freeze.json",
            "paired_original": paired_root / "paired.original.jsonl",
            "paired_hashed": paired_root / "paired.hashed.jsonl",
            "inference_meta": inference_meta,
            "inference_log": inference_log,
            "scoring_log": scoring_log,
            **(
                {"score": score_output, "row_audit": row_audit}
                if scoring_returncode == 0
                else {}
            ),
        }.items()
    }
    record = {
        "schema_version": "silver-match-v3-humor-gemma4-train-then-eval-run-v1",
        "status": status,
        "completed_at": utc_now(),
        "training_completed_at": training_report.get("completed_at"),
        "inference_started_at": inference_started,
        "scoring_started_at": scoring_started,
        "inference_returncode": inference_returncode,
        "scoring_returncode": scoring_returncode,
        "freeze": ref(freeze_path),
        "artifacts": artifacts,
    }
    write_json_new(run_record, record)
    event({"status": status, "run_record_sha256": sha256_file(run_record)})
    if scoring_returncode != 0:
        raise RuntimeError(f"scoring failed: {scoring_returncode}")


if __name__ == "__main__":
    main()
