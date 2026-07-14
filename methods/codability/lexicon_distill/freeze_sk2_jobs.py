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
    parser.add_argument(
        "--include-r1-auxiliary", action="store_true",
        help="opt in to the numerically fragile R1 auxiliary curriculum (off by default)",
    )
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
    task_cells = [
        row for row in inventory["powered_cells"]
        if float(row["weighted_train_pairs"]) > 0
        and int(row["test_pairs"]) >= 100
        and int(row["test_same"]) >= 20
    ]
    repo = Path(args.repo)
    data = repo / "outputs/lexicon/similarity_distill_v1"
    model_inventory = data / "sk2_model_inventory.json"
    local_model_inventory = inventory_path.parent / "sk2_model_inventory.json"
    if not local_model_inventory.is_file():
        raise FileNotFoundError(local_model_inventory)
    run = repo / "outputs/lexicon/similarity_lora_v1"
    trainer = "methods.codability.lexicon_distill.train_gemma4_similarity_lora"
    evaluator = "methods.codability.lexicon_distill.evaluate_similarity_lora"
    local_repo = Path(__file__).resolve().parents[3]
    implementation_relatives = (
        "methods/codability/lexicon_distill/dataset.py",
        "methods/codability/lexicon_distill/train_gemma4_similarity_lora.py",
        "methods/codability/lexicon_distill/evaluate_similarity_lora.py",
        "methods/codability/lexicon_distill/calibrate_threshold.py",
        "methods/codability/lexicon_distill/hierarchy_contracts.py",
        "methods/codability/lexicon_distill/score_hierarchy_pairs.py",
        "methods/codability/lexicon_distill/build_hierarchy_candidate.py",
        "methods/codability/lexicon_distill/frontier_calibration.py",
        "methods/codability/lexicon_distill/freeze_sk2_jobs.py",
        "methods/codability/lexicon_distill/run_sk2_jobs.py",
    )
    gpu_for_level = {"R1": 0, "R2": 1, "R3": 6}
    # Keep expensive untouched evaluations off the training lanes.  R1 owns
    # evaluation GPU 2; shorter R2/R3 evaluations serialize on GPU 3.
    eval_gpu_for_level = {"R1": 2, "R2": 3, "R3": 3}
    jobs: list[dict[str, Any]] = []
    auxiliary_by_level: dict[str, bool] = {}
    headline_variant_by_level: dict[str, str] = {}
    for level in ("R1", "R2", "R3"):
        base = [
            args.python, "-m", trainer,
            "--dataset", str(data / f"{level}_train.jsonl"),
            "--protocols", str(data / "protocols.json"),
            "--model", args.model, "--level", level,
            "--model-inventory", str(model_inventory),
            "--max-length", "1024", "--epochs", "1", "--batch-size", "8",
            "--gradient-accumulation-steps", "2", "--learning-rate", "2e-5", "--lora-r", "16",
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
        raw_auxiliary_available = has_auxiliary_targets(local_train_path)
        auxiliary_available = raw_auxiliary_available and not (
            level == "R1" and not args.include_r1_auxiliary)
        auxiliary_by_level[level] = raw_auxiliary_available
        # Auxiliary curricula are ablations, not critical-path dependencies.  R1's much larger
        # primary corpus is independently sufficient and its auxiliary initialization has proved
        # numerically fragile, so downstream work follows the primary-only fit.  Other levels keep
        # the historical full name when no split exists.
        headline_variant_by_level[level] = (
            "primary" if level == "R1" and raw_auxiliary_available else "full"
        )
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
                        # The lower-trust curriculum gets one quarter of the
                        # primary learning rate; all examples remain present.
                        "--auxiliary-only", "--learning-rate", "5e-6",
                        "--output", str(auxiliary_adapter),
                        "--report", str(auxiliary_report),
                    ],
                    "outputs": [str(auxiliary_adapter / "adapter_model.safetensors"), str(auxiliary_report)],
                }
            )
            variants.extend(
                [
                    (
                        "full",
                        ["--primary-only", "--init-adapter", str(auxiliary_adapter)],
                        [auxiliary_id],
                    ),
                    ("primary", ["--primary-only"], [preflight_id]),
                ]
            )
        elif raw_auxiliary_available:
            variants.append(("primary", ["--primary-only"], [preflight_id]))
        else:
            variants.append(("full", ["--primary-only"], [preflight_id]))
        for variant, extra, dependencies in variants:
            job_id = f"pooled_{level}_{variant}"
            adapter = run / "adapters" / job_id
            report = run / "reports" / f"{job_id}.train.json"
            # R1 has two independent 104k-row primary fits.  Once the short
            # R3 fit releases GPU 6, run the primary-only ablation there while
            # the auxiliary-initialized headline fit continues on GPU 0.
            # Keeping both on GPU 0 needlessly serialized the longest jobs.
            training_gpu = 6 if level == "R1" and variant == "primary" else gpu_for_level[level]
            if level == "R1" and variant == "primary":
                # Let the much shorter R3 fit establish its adapter before R1
                # occupies their shared lane for several hours.
                dependencies = [*dependencies, "pooled_R3_full"]
            jobs.append(
                {
                    "job_id": job_id,
                    "kind": "train_pooled",
                    "gpu": training_gpu,
                    "depends_on": dependencies,
                    "argv": base + extra + ["--output", str(adapter), "--report", str(report)],
                    "outputs": [str(adapter / "adapter_model.safetensors"), str(report)],
                }
            )
        if auxiliary_available:
            eval_variants = ("base", "full", "primary")
        elif raw_auxiliary_available:
            eval_variants = ("base", "primary")
        else:
            eval_variants = ("base", "full")
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
                    "job_id": job_id, "kind": "evaluate", "gpu": eval_gpu_for_level[level],
                    "depends_on": dependency, "argv": argv,
                    "outputs": [str(predictions), str(report)],
                }
            )
            if variant != "base" and level in {"R1", "R3"}:
                dev_id = f"eval_{level}_{variant}_dev"
                dev_predictions = run / "predictions" / f"{dev_id}.jsonl"
                dev_report = run / "reports" / f"{dev_id}.json"
                jobs.append({
                    "job_id": dev_id, "kind": "evaluate_development",
                    "gpu": eval_gpu_for_level[level],
                    "depends_on": [f"pooled_{level}_{variant}"],
                    "argv": [
                        args.python, "-m", evaluator, "evaluate",
                        "--dataset", str(data / f"{level}_pair_dev.jsonl"),
                        "--protocols", str(data / "protocols.json"),
                        "--model", args.model, "--level", level,
                        "--adapter", str(run / "adapters" / f"pooled_{level}_{variant}"),
                        "--predictions", str(dev_predictions), "--report", str(dev_report),
                        "--batch-size", "16",
                    ],
                    "outputs": [str(dev_predictions), str(dev_report)],
                })
                calibration_id = f"calibrate_{level}_{variant}"
                calibration_report = run / "reports" / f"{calibration_id}.json"
                protocol_id = (
                    "r1-narrow-construct-v1" if level == "R1"
                    else "r3-top-level-category-v1"
                )
                jobs.append({
                    "job_id": calibration_id, "kind": "calibrate_threshold", "gpu": None,
                    "depends_on": [dev_id],
                    "argv": [
                        args.python, "-m", "methods.codability.lexicon_distill.calibrate_threshold",
                        "--predictions", str(dev_predictions), "--report", str(calibration_report),
                        "--target-precision", "0.60", "--minimum-recall", "0.50",
                        "--protocol-id", protocol_id,
                        "--adapter-file",
                        str(run / "adapters" / f"pooled_{level}_{variant}"
                            / "adapter_model.safetensors"),
                        "--protocols", str(data / "protocols.json"),
                    ],
                    "outputs": [str(calibration_report)],
                })
        variant_comparisons = [("full_vs_base", "base", "full")]
        if auxiliary_available:
            variant_comparisons.extend(
                [
                    ("primary_vs_base", "base", "primary"),
                    ("full_vs_primary", "primary", "full"),
                ]
            )
        for comparison, reference, candidate in variant_comparisons:
            job_id = f"compare_{level}_{comparison}"
            report = run / "reports" / f"{job_id}.json"
            jobs.append(
                {
                    "job_id": job_id, "kind": "compare_variant", "gpu": None,
                    "depends_on": [f"eval_{level}_{reference}", f"eval_{level}_{candidate}"],
                    "argv": [
                        args.python, "-m", evaluator, "compare-variants",
                        "--reference-predictions",
                        str(run / "predictions" / f"eval_{level}_{reference}.jsonl"),
                        "--candidate-predictions",
                        str(run / "predictions" / f"eval_{level}_{candidate}.jsonl"),
                        "--reference-label", reference,
                        "--candidate-label", candidate,
                        "--report", str(report),
                    ],
                    "outputs": [str(report)],
                }
            )
    task_gpus = (0, 1, 6)
    task_eval_gpus = (2, 3)
    for index, cell in enumerate(task_cells):
        task, level = cell["task"], cell["level"]
        slug = task.replace("-", "_")
        job_id = f"task_{slug}_{level}"
        headline_variant = headline_variant_by_level[level]
        pooled_id = f"pooled_{level}_{headline_variant}"
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
                "job_id": eval_id, "kind": "evaluate_task",
                "gpu": task_eval_gpus[index % len(task_eval_gpus)],
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
        compare_argv = [
            args.python, "-m", evaluator, "compare",
            "--pooled-predictions",
            str(run / "predictions" / f"eval_{level}_{headline_variant}.jsonl"),
            "--task-predictions", str(predictions),
            "--report", str(run / "reports" / f"{compare_id}.json"),
        ]
        if not bool(cell["powered"]):
            compare_argv.append("--descriptive-only")
        jobs.append(
            {
                "job_id": compare_id, "kind": "compare", "gpu": None,
                "depends_on": [eval_id, f"eval_{level}_{headline_variant}"],
                "argv": compare_argv,
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
        "model_inventory": {
            "remote_path": str(model_inventory),
            "sha256": sha256_file(local_model_inventory),
        },
        "implementation_files": {
            relative: {
                "sha256": sha256_file(local_repo / relative),
                "remote_path": str(repo / relative),
            }
            for relative in implementation_relatives
        },
        "powered_cells": powered,
        "task_cells": task_cells,
        "headline_variant_by_level": headline_variant_by_level,
        "auxiliary_available_by_level": auxiliary_by_level,
        "jobs": jobs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "jobs": len(jobs),
        "powered_cells": len(powered), "task_cells": len(task_cells),
    }))


if __name__ == "__main__":
    main()
