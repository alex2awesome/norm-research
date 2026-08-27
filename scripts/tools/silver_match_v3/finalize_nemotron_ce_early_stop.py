#!/usr/bin/env python3
"""Finalize a deliberately early-stopped Nemotron CE run without retraining.

The trainer normally publishes ``training_report.json`` only after every
declared exposure budget.  This helper is for the narrower case where a
controller stops a healthy run at a sealed checkpoint because later dev
checkpoints are dominated.  It validates all immutable inputs, checkpoint
events and the controller receipt, selects the best available checkpoint by
the trainer's dev-only rule, and performs the same fresh-base reload check
before publishing a standard completed report.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .common import read_jsonl, sha256_file
from .train_nemotron_cross_encoder import (
    HIDDEN_SIZE,
    REPORT_SCHEMA,
    _file_hashes,
    _load_saved_model,
    _load_tokenizer,
    checkpoint_selection_key,
    load_pair_examples,
    output_class_names,
    predict_logits,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _checkpoint_record(path: Path, expected_event: Mapping[str, Any]) -> dict[str, Any]:
    metadata_path = path / "checkpoint.json"
    metadata = _read(metadata_path)
    budget = int(metadata["exposure_budget"])
    if (
        int(expected_event.get("exposure_budget", -1)) != budget
        or Path(str(expected_event.get("path") or "")).resolve() != path.resolve()
        or expected_event.get("checkpoint_metadata_sha256") != sha256_file(metadata_path)
    ):
        raise ValueError(f"checkpoint event contract differs: {path}")
    return {
        "path": str(path.resolve()),
        "exposure_budget": budget,
        "dev": metadata["dev"],
        "artifact_sha256": _file_hashes(path),
        "checkpoint_metadata_sha256": sha256_file(metadata_path),
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.run_root).resolve()
    config_path = root / "run_config.json"
    split_path = root / "split_assignments.jsonl"
    events_path = root / "events.jsonl"
    receipt_path = root / "TERMINATION_RECEIPT.json"
    output = root / "training_report.json"
    reload_path = root / "reload_verification.json"
    if output.exists() or reload_path.exists():
        raise FileExistsError("refusing to overwrite an existing final report")

    config = _read(config_path)
    receipt = _read(receipt_path)
    if (
        config.get("schema_version") != REPORT_SCHEMA
        or config.get("split_assignments_sha256") != sha256_file(split_path)
        or receipt.get("event") != "RUN_TERMINATED_BY_CONTROLLER"
        or int(receipt.get("sealed_exposure_budget", -1)) != args.sealed_budget
    ):
        raise ValueError("run config, split, or early-stop receipt contract differs")
    if args.expected_receipt_sha256 != sha256_file(receipt_path):
        raise ValueError("termination receipt SHA-256 differs")

    for role in ("train_pairs", "dev_pairs"):
        refs = config.get(role)
        if not isinstance(refs, Mapping) or not refs:
            raise ValueError(f"run config lacks {role}")
        for raw_path, expected in refs.items():
            if sha256_file(Path(str(raw_path)).resolve()) != expected:
                raise ValueError(f"immutable {role} differs: {raw_path}")

    events = list(read_jsonl(events_path))
    if not events or events[0].get("event") != "RUN_STARTED":
        raise ValueError("event ledger lacks RUN_STARTED")
    saved = [row for row in events if row.get("event") == "CHECKPOINT_SAVED"]
    if len(saved) != len({int(row["exposure_budget"]) for row in saved}):
        raise ValueError("duplicate checkpoint exposure in event ledger")
    if any(int(row["exposure_budget"]) > args.sealed_budget for row in saved):
        raise ValueError("checkpoint exists beyond sealed early-stop budget")
    checkpoints = [
        _checkpoint_record(Path(str(row["path"])).resolve(), row)
        for row in saved
    ]
    if not checkpoints or max(row["exposure_budget"] for row in checkpoints) != args.sealed_budget:
        raise ValueError("sealed checkpoint is absent")
    selected = max(
        checkpoints,
        key=lambda row: (*checkpoint_selection_key(row["dev"]), -row["exposure_budget"]),
    )
    if int(selected["exposure_budget"]) != args.selected_budget:
        raise ValueError(
            f"dev-only selection chose {selected['exposure_budget']}, not {args.selected_budget}"
        )
    if Path(str(receipt.get("selected_checkpoint") or "")).resolve() != Path(selected["path"]):
        raise ValueError("receipt-selected checkpoint differs from dev-only selection")

    classification_mode = str(config.get("classification_mode") or "three_way")
    labels = output_class_names(classification_mode)
    selected_path = Path(selected["path"])
    metadata = _read(selected_path / "checkpoint.json")
    references = metadata.get("reload_reference") or []
    dev_paths = [Path(str(value)).resolve() for value in config["dev_pairs"]]
    dev_examples = sorted(
        load_pair_examples(dev_paths), key=lambda row: (row.norm_uid, row.metric_id)
    )
    reference_examples = dev_examples[: len(references)]
    if (
        not references
        or [(row.norm_uid, row.metric_id) for row in reference_examples]
        != [(row["norm_uid"], row["metric_id"]) for row in references]
    ):
        raise ValueError("reload reference identities differ from frozen dev")

    device = torch.device("cuda", args.device)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("fresh-base reload verification requires CUDA bf16")
    torch.cuda.set_device(device)
    model_args = argparse.Namespace(
        model=str(config["model"]),
        attention=str(config.get("attention") or "eager"),
        classification_mode=classification_mode,
    )
    tokenizer = _load_tokenizer(str(config["model"]))
    model = _load_saved_model(model_args, selected_path, device)
    observed = predict_logits(
        model,
        reference_examples,
        tokenizer,
        device=device,
        max_length=int(config["max_length"]),
        batch_size=args.eval_batch_size,
        classification_mode=classification_mode,
    )
    expected = np.asarray([row["logits"] for row in references], dtype=np.float32)
    maximum_error = float(np.max(np.abs(observed - expected)))
    if not np.allclose(observed, expected, atol=args.reload_atol, rtol=0.0):
        raise RuntimeError(f"fresh-base reload differs: max error {maximum_error}")
    reload_report = {
        "status": "PASS",
        "selected_checkpoint": str(selected_path),
        "examples": len(reference_examples),
        "maximum_absolute_logit_error": maximum_error,
        "absolute_tolerance": args.reload_atol,
        "adapter_and_head_loaded_into_fresh_base": True,
        "selected_checkpoint_artifact_sha256": selected["artifact_sha256"],
    }
    _write_new(reload_path, reload_report)

    sealed_metadata = _read(
        root / "checkpoints" / f"exposure-{args.sealed_budget:012d}" / "checkpoint.json"
    )
    optimizer_updates = int(sealed_metadata["optimizer_updates"])
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "COMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "model": config["model"],
        "classification_mode": classification_mode,
        "labels": list(labels),
        "hidden_to_classes": [HIDDEN_SIZE, len(labels)],
        "max_sequence_length": int(config["max_length"]),
        "bf16_cuda": True,
        "world_size": int(config["world_size"]),
        "sampler": {
            "weights": config["sampler_weights"],
            "deterministic": True,
            "cumulative_class_exposures": sealed_metadata["cumulative_class_exposures"],
            "total_exposures": args.sealed_budget,
        },
        "split_audit": config["split_audit"],
        "separate_learning_rates": {
            "lora": config["lora_learning_rate"],
            "head": config["head_learning_rate"],
        },
        "optimizer_updates": optimizer_updates,
        "warmup_steps": int(math.ceil(optimizer_updates * float(config["warmup_ratio"]))),
        "checkpoints": checkpoints,
        "selected_checkpoint": selected,
        "reload_verification": reload_report,
        "input_sha256": {
            "train_pairs": config["train_pairs"],
            "dev_pairs": config["dev_pairs"],
            "trainer": _artifact(Path(args.trainer).resolve())["sha256"],
            "run_config": sha256_file(config_path),
            "split_assignments": sha256_file(split_path),
        },
        "controlled_early_stop": {
            "reason": receipt["reason"],
            "sealed_exposure_budget": args.sealed_budget,
            "selected_exposure_budget": args.selected_budget,
            "termination_receipt": _artifact(receipt_path),
            "events": _artifact(events_path),
            "finalizer": _artifact(Path(__file__).resolve()),
            "training_steps_during_finalization": 0,
            "test_data_opened_during_finalization": False,
        },
    }
    _write_new(output, report)
    return {
        "training_report": str(output),
        "training_report_sha256": sha256_file(output),
        "reload_verification": str(reload_path),
        "reload_verification_sha256": sha256_file(reload_path),
        "selected_checkpoint": str(selected_path),
        "selected_exposure_budget": args.selected_budget,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--sealed-budget", type=int, required=True)
    parser.add_argument("--selected-budget", type=int, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--reload-atol", type=float, default=0.002)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args(argv)
    if args.sealed_budget < args.selected_budget or args.eval_batch_size < 1:
        parser.error("invalid budget or batch size")
    if len(args.expected_receipt_sha256) != 64:
        parser.error("--expected-receipt-sha256 must be SHA-256")
    return args


def main() -> None:
    print(json.dumps(finalize(parse_args()), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
