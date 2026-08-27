#!/usr/bin/env python3
"""Freeze the two predeclared Legal Nemotron retry queues."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "lock", "feasibility", "manifest", "bank", "teacher", "teacher_meta",
                 "external_dev", "external_dev_test", "optimize_freeze", "select_freeze",
                 "model", "model_inventory", "source_model_inventory", "runtime_inventory",
                 "trainer", "launcher", "teacher_freezer", "python", "output_root", "log_root",
                 "queue_root"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    args = parser.parse_args()
    values = vars(args)
    paths = {name: Path(values[name]).resolve() for name in values if name not in {"output_root", "log_root", "queue_root"}}
    lock = json.loads(paths["lock"].read_text())
    if lock.get("status") != "FROZEN_BEFORE_INTERNAL_SPLIT_MATERIALIZATION_TRAINING_OR_EXTERNAL_DEV":
        raise ValueError("Legal retry lock is not launchable")
    expected = lock["inputs"]
    checks = {
        "manifest": expected["relocated_manifest_sha256"], "bank": expected["bank_artifact_sha256"],
        "teacher": expected["exact_teacher_union_sha256"], "teacher_meta": expected["exact_teacher_union_meta_sha256"],
        "external_dev": expected["external_dev_sha256"], "external_dev_test": expected["external_dev_test_sha256"],
        "optimize_freeze": expected["optimize_truth_release_sha256"], "select_freeze": expected["select_truth_release_sha256"],
        "trainer": lock["implementation"]["trainer_sha256"], "launcher": lock["implementation"]["launcher_sha256"],
        "teacher_freezer": lock["implementation"]["teacher_union_freezer_sha256"],
    }
    for name, expected_hash in checks.items():
        if sha256_file(paths[name]) != expected_hash:
            raise ValueError(f"frozen input mismatch: {name}")
    model_inventory = json.loads(paths["model_inventory"].read_text())
    if model_inventory.get("content_inventory_sha256") != lock["base_model"]["content_inventory_sha256"]:
        raise ValueError("model inventory mismatch")
    output_root, log_root, queue_root = (Path(values[name]).resolve() for name in ("output_root", "log_root", "queue_root"))
    if queue_root.exists():
        raise FileExistsError(queue_root)
    queue_root.mkdir(parents=True)
    fixed = lock["fixed_training"]
    records = []
    for variant in lock["predeclared_variants"]:
        name = variant["name"]
        gpu = int(variant["gpu_preference"])
        variant_output = output_root / name
        variant_log = log_root / name
        bindings = []
        for binding_name, path in paths.items():
            if binding_name in {"repo", "model", "python"}:
                continue
            row = {"name": binding_name, "path": str(path), "sha256": sha256_file(path)}
            if binding_name in {"external_dev", "external_dev_test"}:
                row["training_access"] = "FORBIDDEN"
            bindings.append(row)
        command = [
            str(paths["python"]), "-u", "-m", "scripts.tools.silver_match_v3.train_nemotron_lora",
            "--task", lock["task"], "--manifest", str(paths["manifest"]), "--teachers", str(paths["teacher"]),
            "--model", str(paths["model"]), "--output-root", str(variant_output), "--device", "cuda",
            "--attention", fixed["attention"], "--max-seq-length", str(fixed["max_seq_length"]),
            "--epochs", str(fixed["epochs"]), "--batch-size", str(fixed["batch_size"]),
            "--gradient-accumulation-steps", str(fixed["gradient_accumulation_steps"]),
            "--eval-batch-size", str(fixed["eval_batch_size"]), "--learning-rate", str(variant["learning_rate"]),
            "--weight-decay", str(fixed["weight_decay"]), "--warmup-ratio", str(fixed["warmup_ratio"]),
            "--margin", str(fixed["margin"]), "--hard-negative-pool", str(fixed["hard_negative_pool"]),
            "--negatives-per-positive", str(fixed["negatives_per_positive"]), "--lora-rank", str(fixed["lora_rank"]),
            "--lora-alpha", str(fixed["lora_alpha"]), "--lora-dropout", str(fixed["lora_dropout"]),
            "--seed", str(variant["seed"]), "--split-seed", str(fixed["split_seed"]),
            "--train-percent", str(fixed["train_percent"]), "--dev-percent", str(fixed["dev_percent"]),
            "--min-dev-recall-gain", "0.0", "--selection-k", str(fixed["selection_k"]),
            "--epoch-selection-policy", fixed["epoch_selection_policy"], "--no-enforce-promotion-gate",
        ]
        queue = {
            "schema_version": "silver-match-v3-frozen-nemotron-retry-queue-v1", "status": "FROZEN_READY",
            "task": lock["task"], "repo": str(paths["repo"]), "model": str(paths["model"]),
            "model_inventory": {"path": str(paths["model_inventory"]), "sha256": sha256_file(paths["model_inventory"]),
                                "content_inventory_sha256": model_inventory["content_inventory_sha256"]},
            "gpu": {"index": gpu, "uuid": "GPU-d040097c-ad5e-bd3b-6e43-8ec396c65c89" if gpu == 4 else "GPU-7cc58f63-1fda-ed17-3af3-a0c659675257"},
            "bindings": sorted(bindings, key=lambda row: row["name"]), "command": command,
            "environment": {"CUDA_VISIBLE_DEVICES": str(gpu), "HOME": "/lfs/skampere2/0/alexspan",
                            "HF_HOME": "/lfs/skampere2/0/alexspan/.cache/huggingface",
                            "XDG_CACHE_HOME": "/lfs/skampere2/0/alexspan/.cache",
                            "HF_MODULES_CACHE": "/lfs/skampere2/0/alexspan/.cache/huggingface/modules",
                            "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
                            "CUBLAS_WORKSPACE_CONFIG": fixed["cublas_workspace_config"],
                            "PYTHONPATH": f"{paths['repo']}/vendor:{paths['repo']}"},
            "outputs": {"training_output_root": str(variant_output), "launch_record": str(variant_log / "launch_record.json"),
                        "pid": str(variant_log / "training.pid"), "log": str(variant_log / "training.log")},
        }
        queue_path = queue_root / f"{name}.queue.json"
        queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n")
        records.append({"variant": name, "gpu": gpu, "path": str(queue_path), "sha256": sha256_file(queue_path)})
    freeze = queue_root / "FREEZE.json"
    freeze.write_text(json.dumps({"schema_version": "silver-match-v3-legal-nemotron-queue-freeze-v1",
                                  "status": "FROZEN_READY", "lock_sha256": sha256_file(paths["lock"]),
                                  "variants": records}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"freeze": str(freeze), "freeze_sha256": sha256_file(freeze), "variants": records}, sort_keys=True))


if __name__ == "__main__":
    main()
