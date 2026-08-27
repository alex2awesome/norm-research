#!/usr/bin/env python3
"""Freeze a hash-bound, split-isolated Nemotron external-dev evaluation queue."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import read_jsonl, sha256_file


EXPECTED_RUNTIME = {
    "accelerate": "1.12.0",
    "filelock": "3.24.3",
    "fsspec": "2026.2.0",
    "huggingface_hub": "0.36.2",
    "numpy": "2.2.6",
    "peft": "0.18.1",
    "regex": "2026.2.19",
    "safetensors": "0.7.0",
    "scipy": "1.17.1",
    "sentence_transformers": "5.3.0",
    "sklearn": "1.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.9.1+cu128",
    "transformers": "4.57.6",
}


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _parse_binding(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--extra-binding must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip():
        raise ValueError("empty extra binding name")
    return name.strip(), Path(path).resolve()


def _runtime_audit(python: Path) -> dict[str, Any]:
    if Path(sys.executable).resolve() != python.resolve():
        raise ValueError("freezer must run under the exact queued Python executable")
    observed: dict[str, Any] = {}
    for module_name, expected in EXPECTED_RUNTIME.items():
        module = importlib.import_module(module_name)
        version = str(getattr(module, "__version__", ""))
        if version != expected:
            raise ValueError(
                f"runtime mismatch for {module_name}: observed={version}, expected={expected}"
            )
        observed[module_name] = {
            "version": version,
            "path": str(Path(module.__file__).resolve()),
        }
    return {
        "python": str(python.resolve()),
        "python_version": sys.version,
        "expected_versions": EXPECTED_RUNTIME,
        "observed": observed,
        "all_expected_versions_match": True,
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    if args.split != "dev":
        raise ValueError("this freezer is restricted to external dev")
    repo = Path(args.repo).resolve()
    # Keep the requested interpreter entry point in the frozen command.  A
    # venv's ``bin/python`` is normally a symlink to the base interpreter; if
    # we resolve that symlink here, the child no longer discovers pyvenv.cfg
    # and silently loses the frozen environment's site-packages.  The runtime
    # audit still compares resolved executables, so this preserves the venv
    # activation semantics without weakening interpreter identity checks.
    python = Path(args.python).absolute()
    if not python.is_file():
        raise FileNotFoundError(python)
    manifest_path = Path(args.manifest).resolve()
    labels_path = Path(args.labels).resolve()
    adapter = Path(args.adapter).resolve()
    model = Path(args.model).resolve()
    model_inventory_path = Path(args.model_inventory).resolve()
    report_path = Path(args.training_report).resolve()
    evaluator = repo / "scripts/tools/silver_match_v3/evaluate_nemotron_adapter.py"
    runner = repo / "scripts/tools/silver_match_v3/run_frozen_nemotron_external_dev.py"

    runtime_audit = _runtime_audit(python)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank_sha = str(manifest["banks"][args.task]["source_sha256"])
    labels = list(read_jsonl(labels_path))
    if not labels:
        raise ValueError("external-dev artifact is empty")
    foreign = [
        (row.get("task"), row.get("split"), row.get("norm_uid"))
        for row in labels
        if row.get("task") != args.task or row.get("split") != "dev"
    ]
    if foreign:
        raise ValueError(f"external-dev artifact is role mixed: {foreign[:3]}")
    stale = [
        row.get("norm_uid")
        for row in labels
        if row.get("current_bank_source_sha256") != bank_sha
    ]
    if stale:
        raise ValueError(f"external-dev bank provenance mismatch: {stale[:3]}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("task") != args.task or report.get("status") != "PROMOTABLE":
        raise ValueError("training report is not promotable for the requested task")
    generated_adapter = report.get("generated_hashes", {}).get("adapter", {})
    adapter_artifacts = []
    for filename in sorted(generated_adapter):
        artifact = _artifact(adapter / filename)
        if artifact["sha256"] != generated_adapter[filename]:
            raise ValueError(f"adapter hash mismatch: {filename}")
        adapter_artifacts.append(artifact)
    if not adapter_artifacts:
        raise ValueError("training report has no adapter hashes")

    inventory = json.loads(model_inventory_path.read_text(encoding="utf-8"))
    if (
        inventory.get("status") != "FROZEN_CONTENT_HASH_INVENTORY"
        or Path(str(inventory.get("root"))).resolve() != model
        or inventory.get("content_inventory_sha256")
        != "629023c4f3aaf30a29d6de547628d246fc34bd290ea7523d7b4f3052b4e3e172"
    ):
        raise ValueError("Nemotron model inventory is not the exact frozen snapshot")
    for row in inventory.get("files", []):
        artifact = model / str(row["relative_path"])
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(row["size_bytes"])
            or sha256_file(artifact) != row["sha256"]
        ):
            raise ValueError(f"model inventory mismatch: {artifact}")

    bindings: list[dict[str, Any]] = []
    for name, path in (
        ("manifest", manifest_path),
        ("external_dev", labels_path),
        ("training_report", report_path),
        ("model_inventory", model_inventory_path),
        ("evaluator", evaluator),
        ("runner", runner),
        ("queue_freezer", Path(__file__)),
    ):
        bindings.append({"name": name, **_artifact(path)})
    for row in adapter_artifacts:
        bindings.append({"name": f"adapter_{Path(row['path']).name}", **row})
    seen_names = {row["name"] for row in bindings}
    for value in args.extra_binding or []:
        name, path = _parse_binding(value)
        if name in seen_names:
            raise ValueError(f"duplicate binding name: {name}")
        bindings.append({"name": name, **_artifact(path)})
        seen_names.add(name)

    output_path = Path(args.output).resolve()
    log_path = Path(args.log).resolve()
    record_path = Path(args.run_record).resolve()
    for path in (output_path, log_path, record_path):
        if path.exists():
            raise FileExistsError(path)
    command = [
        str(python),
        "-u",
        "-m",
        "scripts.tools.silver_match_v3.evaluate_nemotron_adapter",
        "--manifest",
        str(manifest_path),
        "--labels",
        str(labels_path),
        "--task",
        args.task,
        "--split",
        "dev",
        "--adapter",
        str(adapter),
        "--model",
        str(model),
        "--output",
        str(output_path),
        "--device",
        "cuda",
        "--attention",
        args.attention,
        "--max-seq-length",
        str(args.max_seq_length),
        "--batch-size",
        str(args.batch_size),
        "--min-dev-recall-gain",
        str(args.min_dev_recall_gain),
    ]
    return {
        "schema_version": "silver-match-v3-frozen-nemotron-external-dev-queue-v1",
        "status": "FROZEN_READY",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "repo": str(repo),
        "external_dev": str(labels_path),
        "bindings": bindings,
        "runtime_audit": runtime_audit,
        "external_dev_audit": {
            "rows": len(labels),
            "match_rows": sum(row.get("decision") == "MATCH" for row in labels),
            "foreign_task_or_split_rows": 0,
            "bank_source_sha256": bank_sha,
            "external_test_consumed": False,
        },
        "command": command,
        "environment": {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "HOME": str(repo.parent),
            "HF_HOME": str(repo / "cache/huggingface"),
            "XDG_CACHE_HOME": str(repo / "cache"),
            "HF_MODULES_CACHE": str(repo / "cache/huggingface/modules"),
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": f"{repo / 'vendor'}:{repo}",
        },
        "log": str(log_path),
        "run_record": str(record_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-inventory", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--extra-binding", action="append")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-dev-recall-gain", type=float, default=0.03)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--run-record", required=True)
    parser.add_argument("--queue", required=True)
    args = parser.parse_args()
    queue = Path(args.queue).resolve()
    if queue.exists():
        raise FileExistsError(queue)
    result = freeze(args)
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"queue": str(queue), "sha256": sha256_file(queue), **result}, sort_keys=True))


if __name__ == "__main__":
    main()
