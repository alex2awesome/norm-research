#!/usr/bin/env python3
"""Freeze the leakage-controlled Humor Nemotron LoRA training queue."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


SCHEMA = "silver-match-v3-frozen-nemotron-retry-queue-v1"
EXPECTED_TASK = "humor"


def _binding(
    name: str, path: Path, *, training_access: str | None = None
) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    row = {"name": name, "path": str(path), "sha256": sha256_file(path)}
    if training_access is not None:
        row["training_access"] = training_access
    return row


def _source_groups(rows: list[dict[str, Any]]) -> set[str]:
    groups = {str(row.get("source_group") or "").strip() for row in rows}
    groups.discard("")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repo",
        "manifest",
        "bank",
        "teacher",
        "teacher_meta",
        "external_dev",
        "external_dev_meta",
        "fresh_select",
        "optimize_truth",
        "blind_identity",
        "blind_freeze",
        "negative_gate",
        "model",
        "model_inventory",
        "trainer",
        "launcher",
        "python",
        "output_root",
        "log_root",
        "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--seed", type=int, default=94131)
    parser.add_argument("--split-seed", type=int, default=874192)
    args = parser.parse_args()

    raw = vars(args)
    non_paths = {"gpu", "gpu_uuid", "seed", "split_seed", "output_root", "log_root"}
    paths = {
        name: Path(value).resolve()
        for name, value in raw.items()
        if name not in non_paths
    }
    output = paths["output"]
    if output.exists():
        raise FileExistsError(output)

    teacher_rows = list(read_jsonl(paths["teacher"]))
    external_rows = list(read_jsonl(paths["external_dev"]))
    if not teacher_rows or not external_rows:
        raise ValueError("teacher and external-dev artifacts must be nonempty")
    decisions = Counter(str(row.get("decision") or "") for row in teacher_rows)
    if decisions != {"MATCH": len(teacher_rows)}:
        raise ValueError(f"retriever teachers must be MATCH-only: {dict(decisions)}")
    if any(row.get("task") != EXPECTED_TASK for row in teacher_rows + external_rows):
        raise ValueError("non-Humor row in frozen supervision")
    if any(row.get("ce_weak_forced_positive") is True for row in teacher_rows):
        raise ValueError("weak forced positives are forbidden")
    if any(row.get("metric_id") in (None, "") for row in teacher_rows):
        raise ValueError("MATCH teacher is missing metric_id")
    teacher_uids = {str(row.get("norm_uid") or "") for row in teacher_rows}
    external_uids = {str(row.get("norm_uid") or "") for row in external_rows}
    uid_overlap = teacher_uids & external_uids
    group_overlap = _source_groups(teacher_rows) & _source_groups(external_rows)
    if uid_overlap or group_overlap:
        raise ValueError(
            f"training/external-dev leakage: uids={len(uid_overlap)} groups={len(group_overlap)}"
        )
    if {str(row.get("split") or "") for row in external_rows} != {"dev"}:
        raise ValueError("external-dev artifact must contain only the frozen dev role")

    teacher_meta = json.loads(paths["teacher_meta"].read_text())
    if teacher_meta.get("output_sha256") != sha256_file(paths["teacher"]):
        raise ValueError("teacher meta does not bind the teacher artifact")
    external_meta = json.loads(paths["external_dev_meta"].read_text())
    if external_meta.get("output_sha256") != sha256_file(paths["external_dev"]):
        raise ValueError("external-dev meta does not bind the dev artifact")

    model_inventory = json.loads(paths["model_inventory"].read_text())
    if Path(model_inventory["root"]).resolve() != paths["model"]:
        raise ValueError("base-model inventory root mismatch")
    if model_inventory.get("content_inventory_sha256") != (
        "629023c4f3aaf30a29d6de547628d246fc34bd290ea7523d7b4f3052b4e3e172"
    ):
        raise ValueError("unexpected Nemotron base-model content inventory")

    output_root = Path(args.output_root).resolve()
    log_root = Path(args.log_root).resolve()
    forbidden_names = {
        "external_dev",
        "external_dev_meta",
        "fresh_select",
        "optimize_truth",
        "blind_identity",
        "blind_freeze",
        "negative_gate",
    }
    bindings: list[dict[str, str]] = []
    for name, path in paths.items():
        if name in {"repo", "model", "python", "output"}:
            continue
        bindings.append(
            _binding(
                name,
                path,
                training_access="FORBIDDEN" if name in forbidden_names else None,
            )
        )

    command = [
        str(paths["python"]),
        "-u",
        "-m",
        "scripts.tools.silver_match_v3.train_nemotron_lora",
        "--task",
        EXPECTED_TASK,
        "--manifest",
        str(paths["manifest"]),
        "--teachers",
        str(paths["teacher"]),
        "--model",
        str(paths["model"]),
        "--output-root",
        str(output_root),
        "--device",
        "cuda",
        "--attention",
        "eager",
        "--max-seq-length",
        "512",
        "--epochs",
        "5",
        "--batch-size",
        "8",
        "--gradient-accumulation-steps",
        "4",
        "--eval-batch-size",
        "32",
        "--learning-rate",
        "5e-05",
        "--weight-decay",
        "0.01",
        "--warmup-ratio",
        "0.1",
        "--margin",
        "0.15",
        "--hard-negative-pool",
        "32",
        "--negatives-per-positive",
        "6",
        "--lora-rank",
        "32",
        "--lora-alpha",
        "64",
        "--lora-dropout",
        "0.05",
        "--seed",
        str(args.seed),
        "--split-seed",
        str(args.split_seed),
        "--train-percent",
        "80",
        "--dev-percent",
        "10",
        "--min-dev-recall-gain",
        "0.0",
        "--selection-k",
        "50",
        "--epoch-selection-policy",
        "depth_lexicographic",
        "--no-enforce-promotion-gate",
    ]
    forbidden_paths = {
        row["path"] for row in bindings if row.get("training_access") == "FORBIDDEN"
    }
    if forbidden_paths.intersection(command):
        raise ValueError("training-forbidden artifact appears in the command")

    queue = {
        "schema_version": SCHEMA,
        "status": "FROZEN_READY",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "task": EXPECTED_TASK,
        "repo": str(paths["repo"]),
        "model": str(paths["model"]),
        "model_inventory": {
            "path": str(paths["model_inventory"]),
            "sha256": sha256_file(paths["model_inventory"]),
            "content_inventory_sha256": model_inventory["content_inventory_sha256"],
        },
        "gpu": {"index": args.gpu, "uuid": args.gpu_uuid},
        "bindings": sorted(bindings, key=lambda row: row["name"]),
        "command": command,
        "environment": {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "HOME": "/lfs/skampere3/0/alexspan",
            "HF_HOME": "/lfs/skampere3/0/alexspan/.cache/huggingface",
            "XDG_CACHE_HOME": "/lfs/skampere3/0/alexspan/.cache",
            "HF_MODULES_CACHE": "/lfs/skampere3/0/alexspan/.cache/huggingface/modules",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONPATH": f"{paths['repo']}/vendor:{paths['repo']}",
        },
        "outputs": {
            "training_output_root": str(output_root),
            "launch_record": str(log_root / "launch_record.json"),
            "pid": str(log_root / "training.pid"),
            "log": str(log_root / "training.log"),
        },
        "supervision_audit": {
            "teacher_rows": len(teacher_rows),
            "teacher_source_groups": len(_source_groups(teacher_rows)),
            "teacher_metric_coverage": len({row["metric_id"] for row in teacher_rows}),
            "weak_forced_positives": 0,
            "external_dev_rows": len(external_rows),
            "external_dev_exact_matches": sum(
                row.get("decision") == "MATCH" for row in external_rows
            ),
            "cross_role_uid_overlap": 0,
            "cross_role_source_group_overlap": 0,
        },
        "frozen_promotion_policy": {
            "internal_epoch_selection": "depth_lexicographic_at_50",
            "external_dev_exact_recall_at_50_min_gain": 0.03,
            "external_dev_exact_recall_at_80_may_decrease": False,
            "fresh_select_consumed_for_training_or_epoch_selection": False,
            "permanent_blind_consumed": False,
            "seed_search_performed": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"output": str(output), "sha256": sha256_file(output), **queue["supervision_audit"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
