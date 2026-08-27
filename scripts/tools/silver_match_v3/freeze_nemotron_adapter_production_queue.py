#!/usr/bin/env python3
"""Freeze one non-overwriting direct Nemotron-LoRA production retrieval queue."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


def artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def resolve(raw: str | Path, anchor: Path) -> Path:
    value = Path(raw)
    return value if value.is_absolute() else (anchor.parent / value).resolve()


def validate_model_inventory(path: Path, model: Path) -> dict[str, Any]:
    inventory = json.loads(path.read_text(encoding="utf-8"))
    if (
        inventory.get("status") != "FROZEN_CONTENT_HASH_INVENTORY"
        or Path(str(inventory.get("root") or "")).resolve() != model.resolve()
        or int(inventory.get("file_count", -1)) != len(inventory.get("files") or [])
    ):
        raise ValueError("base-model inventory is not valid for selected model")
    failures = []
    for value in inventory["files"]:
        candidate = model / str(value["relative_path"])
        if (
            not candidate.is_file()
            or candidate.stat().st_size != int(value["size_bytes"])
            or sha256_file(candidate) != value["sha256"]
        ):
            failures.append(str(candidate))
    if failures:
        raise ValueError(f"base-model content changed: {failures[:3]}")
    return inventory


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    selection_path = Path(args.selection).resolve()
    model_inventory_path = Path(args.model_inventory).resolve()
    model_path = Path(args.encoder).resolve()
    adapter_path = Path(args.adapter).resolve()
    repo_root = Path(args.repo_root).resolve()
    # Do not resolve this symlink: a venv's ``bin/python`` commonly points at
    # the base interpreter, while its unrevolved argv[0] is what activates the
    # adjacent pyvenv.cfg and the sealed venv site-packages.
    python_path = Path(args.python)
    if not python_path.is_absolute():
        python_path = python_path.absolute()
    output_path = Path(args.output_candidate).resolve()
    audit_path = Path(args.output_audit).resolve()
    log_path = Path(args.output_log).resolve()
    run_record_path = Path(args.output_run_record).resolve()
    for path in (output_path, output_path.with_suffix(output_path.suffix + ".meta.json"), audit_path, log_path, run_record_path):
        if path.exists():
            raise FileExistsError(f"non-overwriting queue target already exists: {path}")
    if not python_path.is_file():
        raise FileNotFoundError(python_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    chosen = selection.get("chosen") or {}
    if (
        selection.get("task") != args.task
        or selection.get("status") != "SELECTED_FOR_PRODUCTION_RETRIEVAL"
        or selection.get("selection_split") != "external_dev_only"
        or selection.get("frozen_external_test_consumed") is not False
        or chosen.get("kind") != "nemotron_lora_adapter"
        or int(chosen.get("candidate_depth", -1)) != args.output_k
        or (chosen.get("external_dev_metrics") or {}).get("promotion_gate", {}).get("passed")
        is not True
    ):
        raise ValueError("selection is not an unconsumed, promoted Nemotron LoRA")
    if args.corpus not in manifest.get("corpora", {}):
        raise KeyError(args.corpus)
    corpus_meta = manifest["corpora"][args.corpus]
    if corpus_meta.get("task") != args.task:
        raise ValueError("corpus is not routed to selected task")
    task_corpora = [
        name
        for name, value in manifest["corpora"].items()
        if value.get("task") == args.task
    ]
    if task_corpora != [args.corpus]:
        raise ValueError(f"queue must cover every task corpus exactly: {task_corpora}")
    corpus_path = resolve(corpus_meta["path"], manifest_path)
    bank_meta = manifest["banks"][args.task]
    bank_path = resolve(bank_meta["path"], manifest_path)
    if sum(1 for _ in read_jsonl(corpus_path)) != int(corpus_meta["count"]):
        raise ValueError("canonical corpus count mismatch")
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if len(bank.get("metrics") or []) != int(bank_meta["count"]):
        raise ValueError("bank count mismatch")
    if args.output_k != 50 or args.output_k > int(bank_meta["count"]):
        raise ValueError("this release is frozen to exact K50")

    selected_adapter = Path(str((chosen.get("adapter") or {}).get("path") or "")).resolve()
    selected_model = Path(str((chosen.get("base_model") or {}).get("path") or "")).resolve()
    selected_inventory_sha = str((chosen.get("base_model") or {}).get("inventory_sha256") or "")
    adapter_files = {
        child.name: sha256_file(child)
        for child in sorted(adapter_path.iterdir())
        if child.is_file()
    }
    if (
        selected_adapter != adapter_path
        or selected_model != model_path
        or (chosen.get("adapter") or {}).get("files") != adapter_files
        or selected_inventory_sha != sha256_file(model_inventory_path)
    ):
        raise ValueError("queue adapter/model differs from selection")
    model_inventory = validate_model_inventory(model_inventory_path, model_path)

    implementation_names = (
        "freeze_nemotron_adapter_production_queue.py",
        "retrieve_nemotron_adapter_direct.py",
        "run_frozen_nemotron_adapter_production.py",
        "audit_candidate_outputs.py",
        "common.py",
        "retrieve.py",
        "train_nemotron_lora.py",
    )
    implementations = {
        name: artifact(repo_root / "scripts/tools/silver_match_v3" / name)
        for name in implementation_names
    }
    bindings = {
        "manifest": artifact(manifest_path),
        "selection": artifact(selection_path),
        "promotion_audit": artifact(
            Path(str((chosen.get("promotion_audit") or {}).get("path") or ""))
        ),
        "bank": artifact(bank_path),
        "canonical_corpus": artifact(corpus_path),
        "model_inventory": artifact(model_inventory_path),
        **{
            f"adapter_{name}": artifact(adapter_path / name)
            for name in sorted(adapter_files)
        },
        **{f"implementation_{name}": value for name, value in implementations.items()},
    }
    expected_versions = {
        package: importlib.metadata.version(package)
        for package in (
            "numpy",
            "peft",
            "safetensors",
            "sentence-transformers",
            "torch",
            "transformers",
        )
    }
    command = [
        str(python_path),
        "-u",
        "-m",
        "scripts.tools.silver_match_v3.retrieve_nemotron_adapter_direct",
        "--manifest",
        str(manifest_path),
        "--task",
        args.task,
        "--corpus",
        args.corpus,
        "--encoder",
        str(model_path),
        "--model-inventory",
        str(model_inventory_path),
        "--adapter",
        str(adapter_path),
        "--selection",
        str(selection_path),
        "--output",
        str(output_path),
        "--output-k",
        str(args.output_k),
        "--device",
        "cuda",
        "--attention",
        args.attention,
        "--max-seq-length",
        str(args.max_seq_length),
        "--query-batch-size",
        str(args.query_batch_size),
        "--encoder-batch-size",
        str(args.encoder_batch_size),
        "--resume",
    ]
    audit_command = [
        str(python_path),
        "-u",
        "-m",
        "scripts.tools.silver_match_v3.audit_candidate_outputs",
        "--manifest",
        str(manifest_path),
        "--corpus",
        args.corpus,
        "--candidates",
        str(output_path),
        "--expected-k",
        str(args.output_k),
        "--output",
        str(audit_path),
    ]
    return {
        "schema_version": "silver-match-v3-frozen-nemotron-production-queue-v1",
        "status": "FROZEN_READY_NOT_LAUNCHED",
        "task": args.task,
        "corpus": args.corpus,
        "expected_rows": int(corpus_meta["count"]),
        "expected_k": args.output_k,
        "bank_metrics": int(bank_meta["count"]),
        "bindings": bindings,
        "model": {
            "path": str(model_path),
            "content_inventory_sha256": model_inventory["content_inventory_sha256"],
            "file_count": model_inventory["file_count"],
            "total_size_bytes": model_inventory["total_size_bytes"],
        },
        "adapter": {"path": str(adapter_path), "files": adapter_files},
        "retrieval_geometry": chosen["retrieval_geometry"],
        "command": command,
        "audit_command": audit_command,
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_VISIBLE_DEVICES": str(args.gpu_index),
            "HF_HOME": args.hf_home,
            "HF_HUB_OFFLINE": "1",
            "HF_MODULES_CACHE": args.hf_modules_cache,
            "HOME": args.home,
            "PYTHONPATH": f"{repo_root / 'vendor'}:{repo_root}",
            "TRANSFORMERS_OFFLINE": "1",
            "XDG_CACHE_HOME": args.xdg_cache_home,
        },
        "runtime": {
            "python": str(python_path),
            "python_version": platform.python_version(),
            "packages": expected_versions,
        },
        "execution": {
            "host_pattern": args.host_pattern,
            "repo_root": str(repo_root),
            "gpu_index": args.gpu_index,
            "gpu_count_gate_applied": False,
            "projected_owner_count_check_applied": False,
        },
        "outputs": {
            "candidate": str(output_path),
            "candidate_meta": str(output_path.with_suffix(output_path.suffix + ".meta.json")),
            "audit": str(audit_path),
            "log": str(log_path),
            "run_record": str(run_record_path),
        },
        "safety": {
            "append_resume_only": True,
            "overwrites_existing_candidate": False,
            "external_labels_opened": False,
            "external_test_consumed": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--model-inventory", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output-candidate", required=True)
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--output-log", required=True)
    parser.add_argument("--output-run-record", required=True)
    parser.add_argument("--output-k", type=int, default=50)
    parser.add_argument("--attention", default="eager", choices=("auto", "eager", "sdpa"))
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--query-batch-size", type=int, default=512)
    parser.add_argument("--encoder-batch-size", type=int, default=64)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--host-pattern", default=r"^skampere3(?:\.stanford\.edu)?$")
    parser.add_argument("--home", required=True)
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--hf-modules-cache", required=True)
    parser.add_argument("--xdg-cache-home", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = freeze(args)
    if not re.fullmatch(plan["execution"]["host_pattern"], platform.node()):
        raise ValueError(f"queue frozen on unexpected host: {platform.node()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": plan["status"],
                "output": artifact(output),
                "expected_rows": plan["expected_rows"],
                "expected_k": plan["expected_k"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
