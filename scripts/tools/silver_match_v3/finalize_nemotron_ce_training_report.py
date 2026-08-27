#!/usr/bin/env python3
"""Recover a completed Nemotron CE run that failed only during final reload.

This is deliberately not a trainer or a resume path.  It requires every
declared exposure checkpoint to be complete, validates the immutable input,
split, base-model, trainer, checkpoint, and failure-event ledgers, performs the
same fresh-base adapter/head logit check as training, and atomically creates
only the missing reload verification and training report.  It never updates a
checkpoint, run config, split assignment, or optimizer state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .common import normalize_space, sha256_file
from .run_nemotron_ce import verify_base_manifest
from .train_nemotron_cross_encoder import (
    CLASS_NAMES,
    CLASS_SAMPLING_WEIGHTS,
    HIDDEN_SIZE,
    LORA_TARGETS,
    REPORT_SCHEMA,
    _file_hashes,
    _load_saved_model,
    _load_tokenizer,
    checkpoint_selection_key,
    class_quotas,
    load_pair_examples,
    predict_logits,
)


CHECKPOINT_SCHEMA = "silver-match-v3-nemotron-ce-checkpoint-v1"
BASE_MANIFEST_SCHEMA = "silver-match-v3-nemotron-base-manifest-v1"
RECOVERY_SCHEMA = "silver-match-v3-nemotron-ce-finalization-recovery-v1"
INVENTORY_SCHEMA = "silver-match-v3-nemotron-ce-failed-run-inventory-v1"
RECOVERY_EVAL_BATCH_SIZE = 16
RECOVERY_RELOAD_ATOL = 0.002
RECOVERY_REFERENCE_EXAMPLES = 8
EXPECTED_CHECKPOINT_FILES = {
    "adapter/README.md",
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "checkpoint.json",
    "head.safetensors",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish complete JSON bytes atomically while preserving create-only semantics."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    temp = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{hashlib.sha256(raw).hexdigest()[:12]}"
    )
    try:
        with temp.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temp.exists():
            temp.unlink()


def _validate_base_manifest(
    path: Path, model: Path, *, expected_sha256: str
) -> dict[str, Any]:
    path = path.resolve()
    model = model.resolve()
    contract = verify_base_manifest(model, path, expected_sha256)
    return {
        "manifest": _artifact(path),
        "model": str(model),
        "file_count": contract["file_count"],
        "tree_sha256": contract["tree_sha256"],
    }


def _load_events(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid event JSON at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"event is not an object at {path}:{line_no}")
            rows.append(row)
    if not rows:
        raise ValueError("training event ledger is empty")
    return rows


def _expected_optimizer_updates(config: Mapping[str, Any]) -> dict[int, int]:
    budgets = [int(value) for value in config.get("exposure_budgets") or []]
    world_size = int(config.get("world_size", -1))
    batch_size = int(config.get("batch_size_per_rank", -1))
    accumulation = int(config.get("gradient_accumulation_steps", -1))
    if (
        not budgets
        or budgets != sorted(set(budgets))
        or budgets[0] <= 0
        or world_size <= 0
        or batch_size <= 0
        or accumulation <= 0
        or any(value % world_size for value in budgets)
    ):
        raise ValueError("run config has invalid exposure/update geometry")
    output = {}
    previous = cumulative = 0
    for budget in budgets:
        delta = budget - previous
        local_rows = delta // world_size
        batches = math.ceil(local_rows / batch_size)
        cumulative += math.ceil(batches / accumulation)
        output[budget] = cumulative
        previous = budget
    return output


def _validate_checkpoint(
    path: Path,
    *,
    budget: int,
    config: Mapping[str, Any],
    expected_updates: int,
    expected_cumulative: Mapping[str, int],
    event_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    hashes = _file_hashes(path)
    if set(hashes) != EXPECTED_CHECKPOINT_FILES:
        raise ValueError(f"checkpoint file set differs: {path}")
    metadata_path = path / "checkpoint.json"
    metadata_hash = sha256_file(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    adapter = json.loads((path / "adapter" / "adapter_config.json").read_text())
    lora = config.get("lora") or {}
    if (
        metadata_hash != event_hash
        or metadata.get("schema_version") != CHECKPOINT_SCHEMA
        or int(metadata.get("exposure_budget", -1)) != budget
        or tuple(metadata.get("labels") or ()) != CLASS_NAMES
        or metadata.get("hidden_to_classes") != [HIDDEN_SIZE, len(CLASS_NAMES)]
        or tuple(metadata.get("lora_targets") or ()) != LORA_TARGETS
        or int(metadata.get("optimizer_updates", -1)) != expected_updates
        or metadata.get("cumulative_class_exposures") != dict(expected_cumulative)
        or not isinstance(metadata.get("dev"), Mapping)
        or len(metadata.get("reload_reference") or [])
        != RECOVERY_REFERENCE_EXAMPLES
        or Path(str(adapter.get("base_model_name_or_path") or "")).resolve()
        != Path(str(config.get("model") or "")).resolve()
        or int(adapter.get("r", -1)) != int(lora.get("rank", -2))
        or int(adapter.get("lora_alpha", -1)) != int(lora.get("alpha", -2))
        or float(adapter.get("lora_dropout", -1)) != float(lora.get("dropout", -2))
        or set(adapter.get("target_modules") or []) != set(LORA_TARGETS)
    ):
        raise ValueError(f"checkpoint metadata/config contract differs: {path}")
    record = {
        "path": str(path),
        "exposure_budget": budget,
        "dev": dict(metadata["dev"]),
        "artifact_sha256": hashes,
        "checkpoint_metadata_sha256": metadata_hash,
    }
    return record, metadata


def validate_incomplete_run(
    run_root: Path,
    *,
    trainer: Path,
    trainer_sha256: str,
    base_manifest: Path,
    base_manifest_sha256: str,
    run_config_sha256: str,
    split_assignments_sha256: str,
    events_sha256: str,
) -> dict[str, Any]:
    """Prove training is complete and only final reload/report publication failed."""

    run_root = run_root.resolve()
    trainer = trainer.resolve()
    report_path = run_root / "training_report.json"
    if report_path.exists():
        raise FileExistsError("training report already exists; recovery is not authorized")
    config_path = run_root / "run_config.json"
    split_path = run_root / "split_assignments.jsonl"
    events_path = run_root / "events.jsonl"
    exact_refs = {
        config_path: run_config_sha256,
        split_path: split_assignments_sha256,
        events_path: events_sha256,
        trainer: trainer_sha256,
    }
    for path, expected in exact_refs.items():
        if not expected or sha256_file(path) != expected:
            raise ValueError(f"pinned recovery input SHA-256 differs: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema_version") != REPORT_SCHEMA
        or tuple(config.get("labels") or ()) != CLASS_NAMES
        or config.get("classifier") != [HIDDEN_SIZE, len(CLASS_NAMES)]
        or config.get("sampler_weights") != CLASS_SAMPLING_WEIGHTS
        or tuple(((config.get("lora") or {}).get("targets") or ())) != LORA_TARGETS
        or int((config.get("split_audit") or {}).get("source_group_overlap_count", -1))
        != 0
        or config.get("split_assignments_sha256") != sha256_file(split_path)
    ):
        raise ValueError("immutable run-config/split contract differs")
    pair_hashes = {}
    for role in ("train_pairs", "dev_pairs"):
        refs = config.get(role)
        if not isinstance(refs, Mapping) or not refs:
            raise ValueError(f"run config lacks explicit {role}")
        pair_hashes[role] = {}
        for raw_path, expected in refs.items():
            path = Path(str(raw_path)).resolve()
            if sha256_file(path) != expected:
                raise ValueError(f"immutable {role} hash differs: {path}")
            pair_hashes[role][str(path)] = expected

    events = _load_events(events_path)
    checkpoint_rows = [row for row in events if row.get("event") == "CHECKPOINT_SAVED"]
    checkpoint_events = {int(row["exposure_budget"]): row for row in checkpoint_rows}
    updates = _expected_optimizer_updates(config)
    budgets = list(updates)
    expected_event_sequence = [
        "RUN_STARTED",
        *("CHECKPOINT_SAVED" for _ in budgets),
        "RUN_FAILED",
    ]
    terminal = events[-1]
    terminal_error = normalize_space(terminal.get("error"))
    if (
        [row.get("event") for row in events] != expected_event_sequence
        or int(events[0].get("world_size", -1)) != int(config["world_size"])
        or terminal.get("error_type") != "PermissionError"
        or "/.cache/huggingface/modules/transformers_modules/" not in terminal_error
    ):
        raise ValueError(
            "recovery requires the exact final-reload dynamic-module PermissionError sequence"
        )
    if len(checkpoint_rows) != len(budgets) or set(checkpoint_events) != set(budgets):
        raise ValueError("checkpoint event ledger does not cover every declared budget")

    checkpoints = []
    metadata_by_path = {}
    previous = 0
    cumulative = {name: 0 for name in CLASS_NAMES}
    for budget in budgets:
        delta = budget - previous
        quota = class_quotas(delta)
        cumulative = {name: cumulative[name] + quota[name] for name in CLASS_NAMES}
        checkpoint, metadata = _validate_checkpoint(
            run_root / "checkpoints" / f"exposure-{budget:012d}",
            budget=budget,
            config=config,
            expected_updates=updates[budget],
            expected_cumulative=cumulative,
            event_hash=normalize_space(
                checkpoint_events[budget].get("checkpoint_metadata_sha256")
            ),
        )
        if Path(str(checkpoint_events[budget].get("path") or "")).resolve() != Path(
            checkpoint["path"]
        ):
            raise ValueError("checkpoint event path differs")
        checkpoints.append(checkpoint)
        metadata_by_path[checkpoint["path"]] = metadata
        previous = budget
    selected = max(
        checkpoints,
        key=lambda row: (
            *checkpoint_selection_key(row["dev"]),
            -int(row["exposure_budget"]),
        ),
    )
    model = Path(str(config.get("model") or "")).resolve()
    base_audit = _validate_base_manifest(
        base_manifest, model, expected_sha256=base_manifest_sha256
    )
    return {
        "run_root": run_root,
        "config": config,
        "run_config": _artifact(config_path),
        "split_assignments": _artifact(split_path),
        "events": _artifact(events_path),
        "trainer": _artifact(trainer),
        "base_model": base_audit,
        "pair_hashes": pair_hashes,
        "checkpoints": checkpoints,
        "checkpoint_metadata": metadata_by_path,
        "selected": selected,
        "optimizer_updates": updates[budgets[-1]],
        "warmup_steps": int(
            math.ceil(updates[budgets[-1]] * float(config["warmup_ratio"]))
        ),
        "cumulative_class_exposures": cumulative,
    }


def _trainable_parameter_count(checkpoint: Path) -> int:
    from safetensors import safe_open

    total = 0
    for path in (
        checkpoint / "adapter" / "adapter_model.safetensors",
        checkpoint / "head.safetensors",
    ):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                total += math.prod(handle.get_slice(key).get_shape())
    return total


def _inventory_payload(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": INVENTORY_SCHEMA,
        "status": "FROZEN_POST_FAILURE_PRE_RECOVERY_BYTES",
        "created_at": _now(),
        "host": socket.gethostname(),
        "run_root": str(audit["run_root"]),
        "run_config": audit["run_config"],
        "split_assignments": audit["split_assignments"],
        "events": audit["events"],
        "trainer": audit["trainer"],
        "base_model": audit["base_model"],
        "checkpoints": audit["checkpoints"],
        "selected_checkpoint": audit["selected"],
        "training_report_existed": False,
        "reload_verification_existed": (
            Path(audit["run_root"]) / "reload_verification.json"
        ).exists(),
        "checkpoints_read_only": True,
        "gpu_processes_launched": 0,
    }


def _validate_inventory(
    path: Path,
    *,
    expected_sha256: str,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    path = path.resolve()
    if sha256_file(path) != expected_sha256:
        raise ValueError("post-failure checkpoint inventory SHA-256 differs")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != INVENTORY_SCHEMA
        or payload.get("status") != "FROZEN_POST_FAILURE_PRE_RECOVERY_BYTES"
        or Path(str(payload.get("run_root") or "")).resolve() != audit["run_root"]
        or payload.get("run_config") != audit["run_config"]
        or payload.get("split_assignments") != audit["split_assignments"]
        or payload.get("events") != audit["events"]
        or payload.get("trainer") != audit["trainer"]
        or payload.get("base_model") != audit["base_model"]
        or payload.get("checkpoints") != audit["checkpoints"]
        or payload.get("selected_checkpoint") != audit["selected"]
        or payload.get("training_report_existed") is not False
        or payload.get("checkpoints_read_only") is not True
        or int(payload.get("gpu_processes_launched", -1)) != 0
    ):
        raise ValueError("post-failure checkpoint inventory no longer matches exact bytes")
    return _artifact(path)


def _fresh_reload_report(
    audit: Mapping[str, Any], *, device: torch.device, batch_size: int, atol: float
) -> dict[str, Any]:
    config = audit["config"]
    selected = audit["selected"]
    selected_path = Path(selected["path"])
    metadata = audit["checkpoint_metadata"][str(selected_path)]
    references = metadata["reload_reference"]
    dev_paths = [Path(value) for value in (config.get("dev_pairs") or {})]
    dev_examples = sorted(
        load_pair_examples(dev_paths), key=lambda row: (row.norm_uid, row.metric_id)
    )
    reference_examples = dev_examples[: len(references)]
    observed_ids = [(row.norm_uid, row.metric_id) for row in reference_examples]
    expected_ids = [(row["norm_uid"], row["metric_id"]) for row in references]
    if not references or observed_ids != expected_ids:
        raise ValueError("checkpoint reload-reference identities differ from frozen dev rows")
    tokenizer = _load_tokenizer(str(config["model"]))
    args = argparse.Namespace(model=str(config["model"]), attention=config["attention"])
    reloaded = _load_saved_model(args, selected_path, device)
    observed_logits = predict_logits(
        reloaded,
        reference_examples,
        tokenizer,
        device=device,
        max_length=int(config["max_length"]),
        batch_size=batch_size,
    )
    expected_logits = np.asarray([row["logits"] for row in references], dtype=np.float32)
    maximum_error = float(np.max(np.abs(observed_logits - expected_logits)))
    passed = bool(np.allclose(observed_logits, expected_logits, atol=atol, rtol=0.0))
    report = {
        "status": "PASS" if passed else "FAIL",
        "selected_checkpoint": str(selected_path),
        "examples": len(reference_examples),
        "maximum_absolute_logit_error": maximum_error,
        "absolute_tolerance": atol,
        "adapter_and_head_loaded_into_fresh_base": True,
        "selected_checkpoint_artifact_sha256": selected["artifact_sha256"],
    }
    if not passed:
        raise RuntimeError(
            f"adapter/head reload verification failed: max error {maximum_error}"
        )
    return report


def _validate_existing_reload(
    path: Path, audit: Mapping[str, Any], *, atol: float
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    selected = audit["selected"]
    if (
        report.get("status") != "PASS"
        or Path(str(report.get("selected_checkpoint") or "")).resolve()
        != Path(selected["path"]).resolve()
        or float(report.get("absolute_tolerance", -1)) != atol
        or float(report.get("maximum_absolute_logit_error", math.inf)) > atol
        or report.get("adapter_and_head_loaded_into_fresh_base") is not True
        or report.get("selected_checkpoint_artifact_sha256")
        != selected["artifact_sha256"]
        or int(report.get("examples", -1))
        != len(audit["checkpoint_metadata"][selected["path"]]["reload_reference"])
    ):
        raise ValueError("existing reload verification differs from frozen recovery")
    return report


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.run_root).resolve()
    audit = validate_incomplete_run(
        run_root,
        trainer=Path(args.trainer),
        trainer_sha256=args.trainer_sha256,
        base_manifest=Path(args.base_manifest),
        base_manifest_sha256=args.base_manifest_sha256,
        run_config_sha256=args.run_config_sha256,
        split_assignments_sha256=args.split_assignments_sha256,
        events_sha256=args.events_sha256,
    )
    inventory = _validate_inventory(
        Path(args.checkpoint_inventory),
        expected_sha256=args.checkpoint_inventory_sha256,
        audit=audit,
    )
    reload_path = run_root / "reload_verification.json"
    if reload_path.exists():
        reload_report = _validate_existing_reload(
            reload_path, audit, atol=args.reload_atol
        )
    else:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("fresh-base recovery verification requires CUDA bf16")
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        reload_report = _fresh_reload_report(
            audit,
            device=device,
            batch_size=args.eval_batch_size,
            atol=args.reload_atol,
        )
        _atomic_create_json(reload_path, reload_report)

    config = audit["config"]
    selected = audit["selected"]
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "COMPLETE",
        "created_at": _now(),
        "host": socket.gethostname(),
        "model": config["model"],
        "labels": list(CLASS_NAMES),
        "hidden_to_classes": [HIDDEN_SIZE, len(CLASS_NAMES)],
        "max_sequence_length": int(config["max_length"]),
        "bf16_cuda": True,
        "world_size": int(config["world_size"]),
        "sampler": {
            "weights": CLASS_SAMPLING_WEIGHTS,
            "deterministic": True,
            "cumulative_class_exposures": audit["cumulative_class_exposures"],
            "total_exposures": sum(audit["cumulative_class_exposures"].values()),
        },
        "split_audit": config["split_audit"],
        "separate_learning_rates": {
            "lora": config["lora_learning_rate"],
            "head": config["head_learning_rate"],
        },
        "trainable_parameters": _trainable_parameter_count(Path(selected["path"])),
        "optimizer_updates": audit["optimizer_updates"],
        "warmup_steps": audit["warmup_steps"],
        "checkpoints": audit["checkpoints"],
        "selected_checkpoint": selected,
        "reload_verification": reload_report,
        "input_sha256": {
            **audit["pair_hashes"],
            "trainer": audit["trainer"]["sha256"],
            "run_config": audit["run_config"]["sha256"],
            "split_assignments": audit["split_assignments"]["sha256"],
        },
        "recovery": {
            "schema_version": RECOVERY_SCHEMA,
            "status": "FINALIZED_WITHOUT_RETRAINING_OR_CHECKPOINT_MUTATION",
            "reason": "original run failed during mandatory fresh-base reload/report publication",
            "finalizer": _artifact(Path(__file__)),
            "events": audit["events"],
            "post_failure_checkpoint_inventory": inventory,
            "base_model": audit["base_model"],
            "hf_modules_cache": str(Path(args.hf_modules_cache).resolve()),
            "eval_batch_size": args.eval_batch_size,
            "reload_atol": args.reload_atol,
            "checkpoints_or_run_config_modified": False,
            "optimizer_or_training_steps_executed": 0,
        },
    }
    report_path = run_root / "training_report.json"
    _atomic_create_json(report_path, report)
    return {
        "status": "COMPLETE_RECOVERED_WITHOUT_RETRAINING",
        "training_report": str(report_path),
        "training_report_sha256": sha256_file(report_path),
        "reload_verification": str(reload_path),
        "reload_verification_sha256": sha256_file(reload_path),
        "selected_checkpoint": selected["path"],
        "selected_exposure_budget": selected["exposure_budget"],
        "optimizer_or_training_steps_executed": 0,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--trainer-sha256", required=True)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--base-manifest-sha256", required=True)
    parser.add_argument("--run-config-sha256", required=True)
    parser.add_argument("--split-assignments-sha256", required=True)
    parser.add_argument("--events-sha256", required=True)
    inventory = parser.add_mutually_exclusive_group(required=True)
    inventory.add_argument("--audit-inventory-output")
    inventory.add_argument("--checkpoint-inventory")
    parser.add_argument("--checkpoint-inventory-sha256")
    parser.add_argument("--hf-modules-cache", required=True)
    parser.add_argument("--eval-batch-size", required=True, type=int)
    parser.add_argument("--reload-atol", required=True, type=float)
    args = parser.parse_args(argv)
    for name in (
        "trainer_sha256",
        "base_manifest_sha256",
        "run_config_sha256",
        "split_assignments_sha256",
        "events_sha256",
    ):
        if len(getattr(args, name)) != 64:
            parser.error(f"--{name.replace('_', '-')} must be a SHA-256 digest")
    if bool(args.checkpoint_inventory) != bool(args.checkpoint_inventory_sha256):
        parser.error(
            "--checkpoint-inventory and --checkpoint-inventory-sha256 are paired"
        )
    if args.checkpoint_inventory_sha256 and len(args.checkpoint_inventory_sha256) != 64:
        parser.error("--checkpoint-inventory-sha256 must be a SHA-256 digest")
    cache = Path(args.hf_modules_cache).resolve()
    if not str(cache).startswith("/lfs/"):
        parser.error("--hf-modules-cache must be an absolute /lfs path")
    if args.eval_batch_size < 1 or args.reload_atol < 0:
        parser.error("--eval-batch-size must be positive and --reload-atol non-negative")
    if (
        args.eval_batch_size != RECOVERY_EVAL_BATCH_SIZE
        or args.reload_atol != RECOVERY_RELOAD_ATOL
    ):
        parser.error(
            "recovery is pinned to --eval-batch-size 16 and --reload-atol 0.002"
        )
    cache.mkdir(parents=True, exist_ok=True)
    probe = cache / f".write-probe-{os.getpid()}"
    try:
        probe.touch(exist_ok=False)
    finally:
        if probe.exists():
            probe.unlink()
    os.environ["HF_MODULES_CACHE"] = str(cache)
    os.environ["TRANSFORMERS_CACHE"] = str(cache.parent / "transformers")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return args


def main() -> None:
    args = parse_args()
    if args.audit_inventory_output:
        audit = validate_incomplete_run(
            Path(args.run_root),
            trainer=Path(args.trainer),
            trainer_sha256=args.trainer_sha256,
            base_manifest=Path(args.base_manifest),
            base_manifest_sha256=args.base_manifest_sha256,
            run_config_sha256=args.run_config_sha256,
            split_assignments_sha256=args.split_assignments_sha256,
            events_sha256=args.events_sha256,
        )
        output = Path(args.audit_inventory_output).resolve()
        _atomic_create_json(output, _inventory_payload(audit))
        print(
            json.dumps(
                {
                    "status": "FROZEN_POST_FAILURE_PRE_RECOVERY_BYTES",
                    "checkpoint_inventory": str(output),
                    "checkpoint_inventory_sha256": sha256_file(output),
                    "gpu_processes_launched": 0,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    print(json.dumps(finalize(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
