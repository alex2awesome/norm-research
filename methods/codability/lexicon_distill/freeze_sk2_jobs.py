#!/usr/bin/env python3
"""Freeze the sk2-only Gemma similarity training/evaluation job DAG."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL = (
    "/lfs/skampere2/0/shared_hf_cache/"
    "models--google--gemma-4-31b-it/snapshots/518276fb130dc81caf9a4f772e65e63ef2526493"
)
DEFAULT_REPO = "/lfs/skampere2/0/alexspan/norm-research"
DEFAULT_PYTHON = "/lfs/skampere2/0/alexspan/envs/gemma4-similarity-lora-v1/bin/python"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def has_auxiliary_targets(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            families = set((json.loads(line).get("family_distributions") or {}).keys())
            if families.intersection({"opus", "glm"}):
                return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", default="outputs/lexicon/similarity_distill_v1/inventory.json")
    parser.add_argument("--dataset-manifest", default="outputs/lexicon/similarity_distill_v1/manifest.json")
    parser.add_argument("--output", default="outputs/lexicon/similarity_distill_v1/sk2_jobs.json")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory_path = Path(args.inventory).resolve()
    manifest_path = Path(args.dataset_manifest).resolve()
    output = Path(args.output).resolve()
    if output.exists() and not args.replace:
        raise FileExistsError(output)
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    powered = [row for row in inventory["powered_cells"] if row["powered"]]
    repo = Path(args.repo)
    data = repo / "outputs/lexicon/similarity_distill_v1"
    model_inventory = data / "sk2_model_inventory.json"
    run = repo / "outputs/lexicon/similarity_lora_v1"
    trainer = "methods.codability.lexicon_distill.train_gemma4_similarity_lora"
    evaluator = "methods.codability.lexicon_distill.evaluate_similarity_lora"
    gpu_for_level = {"R1": 0, "R2": 1, "R3": 6}
    jobs: list[dict[str, Any]] = []
    for level in ("R1", "R2", "R3"):
        base = [
            args.python, "-m", trainer,
            "--dataset", str(data / f"{level}_train.jsonl"),
            "--protocols", str(data / "protocols.json"),
            "--model", args.model, "--level", level,
            "--model-inventory", str(model_inventory),
            "--max-length", "1024", "--epochs", "1", "--batch-size", "8",
            "--gradient-accumulation-steps", "2", "--lora-r", "16",
            "--lora-alpha", "32", "--lora-dropout", "0.05",
        ]
        preflight_id = f"preflight_{level}"
        preflight_report = run / "reports" / f"{preflight_id}.json"
        jobs.append(
            {
                "job_id": preflight_id, "kind": "preflight", "gpu": None,
                "depends_on": [],
                "argv": base + ["--preflight-only", "--report", str(preflight_report)],
                "outputs": [str(preflight_report)],
            }
        )
        local_train_path = inventory_path.parent / f"{level}_train.jsonl"
        auxiliary_available = has_auxiliary_targets(local_train_path)
        variants: list[tuple[str, list[str], list[str]]] = []
        if auxiliary_available:
            auxiliary_id = f"pooled_{level}_auxiliary"
            auxiliary_adapter = run / "adapters" / auxiliary_id
            auxiliary_report = run / "reports" / f"{auxiliary_id}.train.json"
            jobs.append(
                {
                    "job_id": auxiliary_id, "kind": "train_pooled_auxiliary",
                    "gpu": gpu_for_level[level], "depends_on": [preflight_id],
                    "argv": base + [
                        "--auxiliary-only", "--output", str(auxiliary_adapter),
                        "--report", str(auxiliary_report),
                    ],
                    "outputs": [str(auxiliary_adapter / "adapter_model.safetensors"), str(auxiliary_report)],
                }
            )
            variants.extend(
                [
                    (
                        "full",
                        ["--primary-only", "--init-adapter", str(auxiliary_adapter), "--learning-rate", "2e-5"],
                        [auxiliary_id],
                    ),
                    ("primary", ["--primary-only"], [preflight_id]),
                ]
            )
        else:
            variants.append(("full", ["--primary-only"], [preflight_id]))
        for variant, extra, dependencies in variants:
            job_id = f"pooled_{level}_{variant}"
            adapter = run / "adapters" / job_id
            report = run / "reports" / f"{job_id}.train.json"
            jobs.append(
                {
                    "job_id": job_id,
                    "kind": "train_pooled",
                    "gpu": gpu_for_level[level],
                    "depends_on": dependencies,
                    "argv": base + extra + ["--output", str(adapter), "--report", str(report)],
                    "outputs": [str(adapter / "adapter_model.safetensors"), str(report)],
                }
            )
        eval_variants = ("base", "full", "primary") if auxiliary_available else ("base", "full")
        for variant in eval_variants:
            job_id = f"eval_{level}_{variant}"
            predictions = run / "predictions" / f"{job_id}.jsonl"
            report = run / "reports" / f"{job_id}.json"
            argv = [
                args.python, "-m", evaluator, "evaluate",
                "--dataset", str(data / f"{level}_eval.jsonl"),
                "--protocols", str(data / "protocols.json"),
                "--model", args.model, "--level", level,
                "--predictions", str(predictions), "--report", str(report),
                "--batch-size", "16",
            ]
            dependency: list[str] = []
            if variant != "base":
                dependency = [f"pooled_{level}_{variant}"]
                argv += ["--adapter", str(run / "adapters" / f"pooled_{level}_{variant}")]
            jobs.append(
                {
                    "job_id": job_id, "kind": "evaluate", "gpu": gpu_for_level[level],
                    "depends_on": dependency, "argv": argv,
                    "outputs": [str(predictions), str(report)],
                }
            )
    task_gpus = (0, 1, 6, 7)
    for index, cell in enumerate(powered):
        task, level = cell["task"], cell["level"]
        slug = task.replace("-", "_")
        job_id = f"task_{slug}_{level}"
        pooled_id = f"pooled_{level}_full"
        adapter = run / "adapters" / job_id
        report = run / "reports" / f"{job_id}.train.json"
        jobs.append(
            {
                "job_id": job_id, "kind": "train_task", "gpu": task_gpus[index % len(task_gpus)],
                "depends_on": [pooled_id],
                "argv": [
                    args.python, "-m", trainer,
                    "--dataset", str(data / f"{level}_train.jsonl"),
                    "--protocols", str(data / "protocols.json"),
                    "--model", args.model, "--level", level, "--task", task,
                    "--model-inventory", str(model_inventory),
                    "--init-adapter", str(run / "adapters" / pooled_id),
                    "--primary-only",
                    "--output", str(adapter), "--report", str(report),
                    "--max-length", "1024", "--epochs", "1", "--batch-size", "8",
                    "--gradient-accumulation-steps", "2", "--learning-rate", "2e-5",
                    "--lora-r", "16", "--lora-alpha", "32", "--lora-dropout", "0.05",
                ],
                "outputs": [str(adapter / "adapter_model.safetensors"), str(report)],
            }
        )
        eval_id = f"eval_{slug}_{level}_task"
        predictions = run / "predictions" / f"{eval_id}.jsonl"
        eval_report = run / "reports" / f"{eval_id}.json"
        jobs.append(
            {
                "job_id": eval_id, "kind": "evaluate_task", "gpu": task_gpus[index % len(task_gpus)],
                "depends_on": [job_id],
                "argv": [
                    args.python, "-m", evaluator, "evaluate",
                    "--dataset", str(data / f"{level}_eval.jsonl"),
                    "--protocols", str(data / "protocols.json"), "--model", args.model,
                    "--adapter", str(adapter), "--level", level, "--task", task,
                    "--predictions", str(predictions), "--report", str(eval_report),
                    "--batch-size", "16",
                ],
                "outputs": [str(predictions), str(eval_report)],
            }
        )
        compare_id = f"compare_{slug}_{level}"
        jobs.append(
            {
                "job_id": compare_id, "kind": "compare", "gpu": None,
                "depends_on": [eval_id, f"eval_{level}_full"],
                "argv": [
                    args.python, "-m", evaluator, "compare",
                    "--pooled-predictions", str(run / "predictions" / f"eval_{level}_full.jsonl"),
                    "--task-predictions", str(predictions),
                    "--report", str(run / "reports" / f"{compare_id}.json"),
                ],
                "outputs": [str(run / "reports" / f"{compare_id}.json")],
            }
        )
    payload = {
        "schema_version": "gemma4-similarity-sk2-job-dag-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host_allowlist": ["sk2", "skampere2", "skampere2.stanford.edu"],
        "sk3_forbidden": True,
        "repo": args.repo,
        "python": args.python,
        "model": args.model,
        "dataset_inventory": {
            "path": str(inventory_path), "remote_path": str(data / "inventory.json"),
            "sha256": sha256_file(inventory_path),
        },
        "dataset_manifest": {
            "path": str(manifest_path), "remote_path": str(data / "manifest.json"),
            "sha256": sha256_file(manifest_path),
        },
        "model_inventory_remote_path": str(model_inventory),
        "powered_cells": powered,
        "jobs": jobs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "jobs": len(jobs), "powered_cells": len(powered)}))


if __name__ == "__main__":
    main()
