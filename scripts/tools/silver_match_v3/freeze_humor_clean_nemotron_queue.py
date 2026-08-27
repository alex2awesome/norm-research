#!/usr/bin/env python3
"""Freeze one clean-truth Humor Nemotron-LoRA v1 training queue."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file
from .train_nemotron_lora import load_universe


SCHEMA = "silver-match-v3-frozen-nemotron-retry-queue-v1"
CONTENT_INVENTORY_SHA = "629023c4f3aaf30a29d6de547628d246fc34bd290ea7523d7b4f3052b4e3e172"


def _binding(name: str, path: Path, *, forbidden: bool = False) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    row = {"name": name, "path": str(path), "sha256": sha256_file(path)}
    if forbidden:
        row["training_access"] = "FORBIDDEN"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "repo",
        "manifest",
        "bank",
        "norms",
        "teacher",
        "teacher_report",
        "gate",
        "frozen_k50",
        "frozen_k50_meta",
        "select_identities",
        "select_freeze",
        "model",
        "model_inventory",
        "trainer",
        "launcher",
        "python",
        "runtime_home",
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
    non_files = {
        "gpu",
        "gpu_uuid",
        "seed",
        "split_seed",
        "runtime_home",
        "output_root",
        "log_root",
    }
    paths = {
        name: Path(value).absolute() if name == "python" else Path(value).resolve()
        for name, value in raw.items()
        if name not in non_files
    }
    output = paths["output"]
    if output.exists():
        raise FileExistsError(output)
    for name in ("repo", "model"):
        if not paths[name].is_dir():
            raise FileNotFoundError(paths[name])
    if not paths["python"].is_file():
        raise FileNotFoundError(paths["python"])

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    bank_meta = (manifest.get("banks") or {}).get("humor") or {}
    corpus_meta = (manifest.get("corpora") or {}).get("humor_multi") or {}
    if (
        Path(str(bank_meta.get("path") or "")).resolve() != paths["bank"]
        or bank_meta.get("sha256") != sha256_file(paths["bank"])
        or Path(str(corpus_meta.get("path") or "")).resolve() != paths["norms"]
        or corpus_meta.get("sha256") != sha256_file(paths["norms"])
    ):
        raise ValueError("runtime manifest does not bind the staged bank/norm bytes")

    teacher = list(read_jsonl(paths["teacher"]))
    teacher_report = json.loads(paths["teacher_report"].read_text(encoding="utf-8"))
    split_counts = Counter(str(row.get("split") or "") for row in teacher)
    if (
        len(teacher) != 388
        or Counter(str(row.get("decision") or "") for row in teacher) != {"MATCH": 388}
        or split_counts != {"train": 327, "dev": 28, "test": 33}
        or any(row.get("gradient_eligible") is not True for row in teacher)
        or any(row.get("supervision_strength") != "strong" for row in teacher)
        or (teacher_report.get("output") or {}).get("sha256")
        != sha256_file(paths["teacher"])
    ):
        raise ValueError("clean teacher is not the frozen 388-row explicit-role artifact")

    universe = load_universe(
        paths["manifest"],
        [paths["teacher"]],
        "humor",
        split_seed=args.split_seed,
        train_percent=80,
        dev_percent=10,
        require_bank_hash=True,
        respect_teacher_splits=True,
    )
    if len(universe.labels) != 388 or universe.split_audit["rows"] != {
        "train": 327,
        "dev": 28,
        "test": 33,
    }:
        raise ValueError("trainer preflight changed clean teacher coverage/splits")

    gate = json.loads(paths["gate"].read_text(encoding="utf-8"))
    if gate.get("status") != "FROZEN_BEFORE_SELECT_TRUTH_OR_TRAINING_RESULT":
        raise ValueError("promotion gate was not predeclared")
    gate_bindings = gate.get("bindings") or {}
    for name in ("teacher", "teacher_report", "candidates", "candidates_meta"):
        runtime_name = {
            "candidates": "frozen_k50",
            "candidates_meta": "frozen_k50_meta",
        }.get(name, name)
        if (gate_bindings.get(name) or {}).get("sha256") != sha256_file(paths[runtime_name]):
            raise ValueError(f"runtime artifact differs from predeclared gate: {name}")

    inventory = json.loads(paths["model_inventory"].read_text(encoding="utf-8"))
    if (
        Path(str(inventory.get("root") or "")).resolve() != paths["model"]
        or inventory.get("content_inventory_sha256") != CONTENT_INVENTORY_SHA
        or inventory.get("status") != "FROZEN_CONTENT_HASH_INVENTORY"
    ):
        raise ValueError("base Nemotron inventory mismatch")

    output_root = Path(args.output_root).resolve()
    log_root = Path(args.log_root).resolve()
    runtime_home = Path(args.runtime_home).resolve()
    if not runtime_home.is_dir():
        raise FileNotFoundError(runtime_home)
    forbidden_names = {
        "frozen_k50",
        "frozen_k50_meta",
        "select_identities",
        "select_freeze",
        "gate",
    }
    bindings = []
    for name, path in paths.items():
        if name in {"repo", "model", "python", "output"}:
            continue
        bindings.append(_binding(name, path, forbidden=name in forbidden_names))

    command = [
        str(paths["python"]),
        "-u",
        "-m",
        "scripts.tools.silver_match_v3.train_nemotron_lora",
        "--task",
        "humor",
        "--manifest",
        str(paths["manifest"]),
        "--teachers",
        str(paths["teacher"]),
        "--respect-teacher-splits",
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
    forbidden = {row["path"] for row in bindings if row.get("training_access") == "FORBIDDEN"}
    if forbidden & set(command):
        raise ValueError("training-forbidden evaluation/reference artifact appears in command")

    queue = {
        "schema_version": SCHEMA,
        "status": "FROZEN_READY",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "task": "humor",
        "repo": str(paths["repo"]),
        "model": str(paths["model"]),
        "model_inventory": {
            "path": str(paths["model_inventory"]),
            "sha256": sha256_file(paths["model_inventory"]),
            "content_inventory_sha256": CONTENT_INVENTORY_SHA,
        },
        "gpu": {"index": args.gpu, "uuid": args.gpu_uuid},
        "bindings": sorted(bindings, key=lambda row: row["name"]),
        "command": command,
        "environment": {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "HOME": str(runtime_home),
            "HF_HOME": str(runtime_home / ".cache" / "huggingface"),
            "XDG_CACHE_HOME": str(runtime_home / ".cache"),
            "HF_MODULES_CACHE": str(
                runtime_home / ".cache" / "huggingface" / "modules"
            ),
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
            "teacher_rows": 388,
            "teacher_split_counts": dict(sorted(split_counts.items())),
            "teacher_metric_coverage": len({str(row["metric_id"]) for row in teacher}),
            "weak_or_forced_positive_rows": 0,
            "fresh_select_uid_overlap": 0,
            "fresh_select_source_group_overlap": 0,
            "frozen_k50_reference_rows": 896,
        },
        "frozen_promotion_policy": gate["promotion_gate"],
        "safety": {
            "single_predeclared_seed": True,
            "select_truth_consumed_at_freeze": False,
            "select_truth_consumed_by_training_or_epoch_selection": False,
            "frozen_k50_consumed_by_training": False,
            "append_only_output": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "teacher_rows": 388,
                "split_counts": dict(sorted(split_counts.items())),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
