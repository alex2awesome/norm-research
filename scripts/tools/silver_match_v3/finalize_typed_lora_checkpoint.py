#!/usr/bin/env python3
"""Promote one fsynced typed-LoRA dev checkpoint after a fresh-base reload probe."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from .run_nemotron_ce import verify_base_manifest
from .train_gemma4_typed_lora import (
    DEFAULT_FIELD_LOSS_WEIGHTS,
    DEFAULT_STRUCTURAL_LOSS_WEIGHT,
    directory_ref,
    file_ref,
    read_examples,
    reload_probe,
    resolve_language_model_target_scope,
    sha256_file,
    tokenize_dataset,
    write_json_new,
)


def _verify_checkpoint_adapter(checkpoint: dict, checkpoint_path: Path) -> Path:
    reference = checkpoint.get("adapter") or {}
    source = Path(str(reference.get("path") or "")).resolve()
    expected = {
        str(row["relative_path"]): row for row in reference.get("files") or []
    }
    observed = {
        child.relative_to(source).as_posix(): child
        for child in source.rglob("*")
        if child.is_file()
    }
    if not expected or set(expected) != set(observed):
        raise ValueError("checkpoint adapter file inventory differs")
    for relative, path in observed.items():
        row = expected[relative]
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"checkpoint adapter identity drift: {relative}")
    expected_parent = checkpoint_path.parent / "adapter"
    if source != expected_parent.resolve():
        raise ValueError("checkpoint adapter is outside its fsynced checkpoint directory")
    return source


def _gate(checkpoint: dict, expected_exposure: int) -> dict:
    gate = checkpoint.get("confidence_gate") or {}
    generation = checkpoint.get("generation") or {}
    invalid_rate = 1.0 - float(generation.get("valid_predictions", 0)) / float(
        generation.get("rows", 0)
    )
    if (
        checkpoint.get("cumulative_exposure") != expected_exposure
        or gate.get("precision_wilson_gate_met") is not True
        or float(gate.get("exact_precision", -1)) < 0.90
        or float(gate.get("exact_precision_wilson_95_lower", -1)) < 0.85
        or int(gate.get("predicted_exact_count", -1)) < 100
        or invalid_rate > 0.01
    ):
        raise ValueError("first checkpoint does not pass the frozen early gate")
    return {**gate, "invalid_output_rate": invalid_rate}


def run(args: argparse.Namespace) -> dict:
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    gate = _gate(checkpoint, args.expected_exposure)
    source_adapter = _verify_checkpoint_adapter(checkpoint, checkpoint_path)
    model = Path(args.model).resolve()
    inventory = Path(args.model_inventory).resolve()
    verify_base_manifest(model, inventory, args.model_inventory_sha256)
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    adapter = output_root / "adapter"
    shutil.copytree(source_adapter, adapter)
    selection = {
        "schema_version": "silver-match-v3-gemma4-typed-lora-dev-selection-v2",
        "status": "SELECTED_ON_DEV_ONLY",
        "selection_split": "dev",
        "test_or_blind_data_read": False,
        "chosen_cumulative_exposure": args.expected_exposure,
        "chosen_checkpoint": file_ref(checkpoint_path),
        "chosen_dev_report": {
            key: checkpoint.get(key)
            for key in ("weighted_dev_loss", "generation", "confidence_gate")
        },
        "all_checkpoint_selection_summaries": [
            {
                "cumulative_exposure": args.expected_exposure,
                "weighted_dev_loss": checkpoint.get("weighted_dev_loss"),
                "confidence_gate": checkpoint.get("confidence_gate"),
            }
        ],
        "early_stop_policy": {
            "exact_precision_minimum": 0.90,
            "wilson_95_lower_minimum": 0.85,
            "minimum_predictions": 100,
            "maximum_invalid_rate": 0.01,
            "observed": gate,
        },
    }
    selection_path = adapter / "DEV_SELECTION.json"
    write_json_new(selection_path, selection)

    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dev_examples = read_examples(Path(args.dev_dataset).resolve())
    encoded, _ = tokenize_dataset(
        tokenizer,
        dev_examples[:2],
        args.max_length,
        field_loss_weights=DEFAULT_FIELD_LOSS_WEIGHTS,
        structural_loss_weight=DEFAULT_STRUCTURAL_LOSS_WEIGHT,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    architecture, target_regex, expected_modules = resolve_language_model_target_scope(base)
    saved = PeftConfig.from_pretrained(adapter)
    if saved.target_modules != target_regex:
        raise ValueError("selected adapter target scope differs from exact Llama q/k/v/o")
    selected = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    device = torch.device("cuda", 0)
    selected.to(device)
    if any(parameter.requires_grad for parameter in selected.parameters()):
        raise ValueError("freshly reloaded selected adapter is trainable")
    observed = reload_probe(selected, encoded, tokenizer=tokenizer, device=device)
    expected = [float(value) for value in checkpoint["reload_reference_gold_logits"]]
    differences = [abs(left - right) for left, right in zip(expected, observed)]
    if len(expected) != len(observed) or any(
        not math.isclose(left, right, rel_tol=1e-3, abs_tol=2e-2)
        for left, right in zip(expected, observed)
    ):
        raise ValueError("fresh-base checkpoint reload changed deterministic probe logits")
    report = {
        "schema_version": "silver-match-v3-gemma4-typed-lora-train-report-v2",
        "status": "COMPLETE_DEV_SELECTED_ADAPTER_FRESH_RELOAD_VERIFIED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "dataset": file_ref(Path(args.dataset)),
        "dev_dataset": file_ref(Path(args.dev_dataset)),
        "model": str(model),
        "model_inventory": file_ref(inventory),
        "trainer_script": file_ref(Path(args.trainer)),
        "parameters": {
            "target_architecture": architecture,
            "target_scope_regex": target_regex,
            "expected_qkvo_modules": sorted(expected_modules),
        },
        "steps": {
            "cumulative_exposure_completed": args.expected_exposure,
            "early_stopped_after_fsynced_dev_gate": True,
        },
        "adapter": {
            "directory": str(adapter),
            "config": file_ref(adapter / "adapter_config.json"),
            "weights": file_ref(adapter / "adapter_model.safetensors"),
            "adapter_only": True,
            "inference_reload_verified": True,
            "fresh_base_reload_verified": True,
            "reload_probe_value_count": len(observed),
            "reload_probe_max_absolute_difference": max(differences, default=0.0),
            "content": directory_ref(adapter),
        },
        "selection": {**selection, "artifact": file_ref(selection_path)},
        "source_checkpoint": file_ref(checkpoint_path),
        "test_or_blind_data_read": False,
    }
    report_path = output_root / "TRAINING_REPORT.json"
    write_json_new(report_path, report)
    return {**report, "report": file_ref(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-inventory", required=True)
    parser.add_argument("--model-inventory-sha256", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dev-dataset", required=True)
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-exposure", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
