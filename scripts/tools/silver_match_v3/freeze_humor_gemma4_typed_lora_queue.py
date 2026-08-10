#!/usr/bin/env python3
"""Freeze one hash-bound Humor Gemma-4 typed-decision LoRA pilot queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_DATASET_SHA = "679ee4d2feb5a35beb977788382a81dff2402bb78c13c3b7fd1ac68b32f887f2"
EXPECTED_MODEL_CONTENT_SHA = "f06399f0164b3feeb55e2de43831e699d1443481afb6d6a1b0164053d86d13ae"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path, *, training_access: str = "ALLOWED") -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "training_access": training_access,
    }


def overlay_inventory(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or "__pycache__" in path.parts
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        files.append(
            {
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError("empty Python overlay")
    digest = hashlib.sha256()
    for row in files:
        digest.update(
            f"{row['relative_path']}\0{row['bytes']}\0{row['sha256']}\n".encode()
        )
    return {
        "root": str(root),
        "file_count": len(files),
        "content_inventory_sha256": digest.hexdigest(),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dataset",
        "dataset_report",
        "preflight",
        "adapter_preflight",
        "model",
        "model_inventory",
        "python",
        "python_overlay",
        "trainer",
        "launcher",
        "select_identities",
        "select_freeze",
        "runtime_home",
        "adapter_output",
        "train_report",
        "log",
        "launch_record",
        "queue",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--idle-memory-mib", type=int, default=1024)
    args = parser.parse_args()

    file_names = {
        "dataset",
        "dataset_report",
        "preflight",
        "adapter_preflight",
        "model_inventory",
        "python",
        "trainer",
        "launcher",
        "select_identities",
        "select_freeze",
    }
    paths = {
        name: Path(getattr(args, name)).resolve()
        for name in vars(args)
        if name != "idle_memory_mib"
    }
    queue_path = paths["queue"]
    if queue_path.exists():
        raise FileExistsError(queue_path)
    for name in file_names:
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    for name in ("model", "python_overlay", "runtime_home"):
        if not paths[name].is_dir():
            raise FileNotFoundError(paths[name])
    for name in ("adapter_output", "train_report", "log", "launch_record"):
        if paths[name].exists():
            raise FileExistsError(f"non-overwriting output already exists: {paths[name]}")

    dataset_sha = sha256_file(paths["dataset"])
    if dataset_sha != EXPECTED_DATASET_SHA:
        raise ValueError(f"unexpected frozen dataset SHA: {dataset_sha}")
    rows = []
    with paths["dataset"].open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    uid_views: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        uid_views[str(row.get("norm_uid") or "")].add(str(row.get("view") or ""))
    expected_views = {"retrieval_hardmix", "sha256_permutation"}
    if (
        len(rows) != 1792
        or len(uid_views) != 896
        or any(views != expected_views for views in uid_views.values())
        or Counter(str(row.get("decision") or "") for row in rows)["MATCH"] != 776
        or any(row.get("gradient_eligible") is not True for row in rows)
    ):
        raise ValueError("frozen typed dataset coverage/role invariant failed")

    dataset_report = json.loads(paths["dataset_report"].read_text(encoding="utf-8"))
    if (
        (dataset_report.get("output") or {}).get("sha256") != dataset_sha
        or int(dataset_report.get("match_rows", -1)) != 388
        or int(dataset_report.get("typed_nonmatch_rows", -1)) != 508
        or int(dataset_report.get("same_family_selected_match_rows", -1)) != 208
        or int((dataset_report.get("candidate_lane_selections") or {}).get("same_frozen_r3_sibling_leaf", -1)) != 516
        or dataset_report.get("fresh_select_uid_overlap") != 0
        or dataset_report.get("fresh_select_source_group_overlap") != 0
        or dataset_report.get("select_labels_or_consensus_read") is not False
    ):
        raise ValueError("dataset report fails supervision/firewall invariants")

    preflight = json.loads(paths["preflight"].read_text(encoding="utf-8"))
    recipe = preflight.get("recipe") or {}
    tokens = (preflight.get("tokenization") or {}).get("tokens") or {}
    if (
        preflight.get("status") != "PASS_EXACT_TEMPLATE_NO_TRUNCATION"
        or (preflight.get("dataset") or {}).get("sha256") != dataset_sha
        or (preflight.get("trainer_script") or {}).get("sha256") != sha256_file(paths["trainer"])
        or int((preflight.get("tokenization") or {}).get("example_count", -1)) != 1792
        or int(tokens.get("full_max", 10**9)) > 4096
        or preflight.get("target_truncations") != 0
        or preflight.get("assistant_prefix_alignment_failures") != 0
        or recipe.get("epochs") != 1
        or recipe.get("per_device_batch_size") != 2
        or recipe.get("gradient_accumulation_steps") != 8
        or recipe.get("learning_rate") != 1e-4
        or recipe.get("seed") != 94137
        or (recipe.get("lora") or {}).get("r") != 16
        or (recipe.get("lora") or {}).get("alpha") != 32
        or (recipe.get("lora") or {}).get("dropout") != 0.05
    ):
        raise ValueError("exact preflight differs from the single frozen recipe")

    adapter_preflight = json.loads(paths["adapter_preflight"].read_text(encoding="utf-8"))
    injection = adapter_preflight.get("adapter_injection") or {}
    if (
        adapter_preflight.get("status") != "PASS_TEXT_LANGUAGE_MODEL_LORA_SCOPE_ONLY"
        or (adapter_preflight.get("dataset") or {}).get("sha256") != dataset_sha
        or (adapter_preflight.get("trainer_script") or {}).get("sha256")
        != sha256_file(paths["trainer"])
        or int(injection.get("adapted_linear_count", -1)) != 410
        or injection.get("all_trainable_parameters_are_lora") is not True
        or int(injection.get("vision_or_audio_trainable_parameters", -1)) != 0
    ):
        raise ValueError("Gemma-4 PEFT injection preflight failed or drifted")

    inventory = json.loads(paths["model_inventory"].read_text(encoding="utf-8"))
    if (
        inventory.get("status") != "FROZEN_CONTENT_HASH_INVENTORY"
        or Path(str(inventory.get("root") or "")).resolve() != paths["model"]
        or inventory.get("content_inventory_sha256") != EXPECTED_MODEL_CONTENT_SHA
        or int(inventory.get("file_count", -1)) != 12
    ):
        raise ValueError("Gemma-4 base-model inventory mismatch")
    overlay = overlay_inventory(paths["python_overlay"])

    select_freeze = json.loads(paths["select_freeze"].read_text(encoding="utf-8"))
    if (
        select_freeze.get("status") != "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES"
        or int(select_freeze.get("selected_count", -1)) != 300
        or ((select_freeze.get("outputs") or {}).get("identities") or {}).get("sha256")
        != sha256_file(paths["select_identities"])
    ):
        raise ValueError("fresh-select identity firewall mismatch")

    command = [
        str(paths["python"]),
        "-u",
        str(paths["trainer"]),
        "--dataset",
        str(paths["dataset"]),
        "--model",
        str(paths["model"]),
        "--model-inventory",
        str(paths["model_inventory"]),
        "--report",
        str(paths["train_report"]),
        "--output",
        str(paths["adapter_output"]),
        "--max-length",
        "4096",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--gradient-accumulation-steps",
        "8",
        "--learning-rate",
        "1e-4",
        "--weight-decay",
        "0",
        "--max-grad-norm",
        "1",
        "--lora-r",
        "16",
        "--lora-alpha",
        "32",
        "--lora-dropout",
        "0.05",
        "--seed",
        "94137",
        "--log-every-steps",
        "5",
    ]
    forbidden = {str(paths["select_identities"]), str(paths["select_freeze"])}
    if forbidden & set(command):
        raise ValueError("select artifact appears in the training command")

    payload = {
        "schema_version": "silver-match-v3-humor-gemma4-typed-lora-queue-v1",
        "status": "FROZEN_AWAITING_SERVER_CAPACITY",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "task": "humor",
        "role": "task_specific_typed_match_adjudicator_pilot",
        "command": command,
        "environment": {
            "HOME": str(paths["runtime_home"]),
            "HF_HOME": str(paths["runtime_home"] / ".cache" / "huggingface"),
            "XDG_CACHE_HOME": str(paths["runtime_home"] / ".cache"),
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONPATH": str(paths["python_overlay"]),
        },
        "bindings": {
            "dataset": ref(paths["dataset"]),
            "dataset_report": ref(paths["dataset_report"]),
            "preflight": ref(paths["preflight"]),
            "adapter_preflight": ref(paths["adapter_preflight"]),
            "trainer": ref(paths["trainer"]),
            "launcher": ref(paths["launcher"]),
            "queue_freezer": ref(Path(__file__)),
            "model_inventory": ref(paths["model_inventory"]),
            "python": ref(paths["python"]),
            "select_identities": ref(paths["select_identities"], training_access="FORBIDDEN"),
            "select_freeze": ref(paths["select_freeze"], training_access="FORBIDDEN"),
        },
        "model": {
            "path": str(paths["model"]),
            "content_inventory_sha256": EXPECTED_MODEL_CONTENT_SHA,
        },
        "python_overlay": overlay,
        "gpu_policy": {
            "gpu_count_gate_applied": False,
            "projected_owner_count_check_applied": False,
            "target_max_memory_used_mib": args.idle_memory_mib,
            "target_required_utilization_percent": 0,
            "stable_idle_polls_required": 2,
            "co_location_forbidden": True,
            "selection": "lowest-index genuinely idle GPU after two stable polls",
        },
        "outputs": {
            "adapter": str(paths["adapter_output"]),
            "training_report": str(paths["train_report"]),
            "log": str(paths["log"]),
            "launch_record": str(paths["launch_record"]),
        },
        "selection_firewall": {
            "training_rows": 896,
            "training_examples_with_order_permutations": 1792,
            "fresh_select_rows": 300,
            "fresh_select_uid_overlap": 0,
            "fresh_select_source_group_overlap": 0,
            "select_labels_or_consensus_consumed": False,
            "select_truth_may_have_existed_at_queue_freeze_but_was_not_read": True,
            "no_select_driven_hyperparameter_or_seed_search": True,
        },
        "promotion_gate_frozen_before_model_result": {
            "evaluation_population": "all 293 exact-consensus fresh-select rows; 7 residual disagreements excluded",
            "compare_against": "same frozen base Gemma-4 under identical candidates, prompt, generation settings, and orders",
            "primary": "exact decision-and-leaf accuracy gain >= 0.03",
            "exact_match_precision": "may not decrease",
            "exact_match_recall": "report paired change with exact 95% interval",
            "typed_decision_macro_f1": "report all six non-noise decision types and MATCH",
            "typed_abstention": "report per-type precision, recall, and confusion matrix",
            "candidate_order": "two frozen orders; exact output stability may not decrease and must be >=0.90",
            "invalid_output_rate": "<=0.01",
            "failed_gate_action": "quarantine adapter; retain frozen base adjudicator",
            "hyperparameter_or_seed_search": False,
        },
        "recipe": recipe,
    }
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"queue": str(queue_path), "sha256": sha256_file(queue_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
