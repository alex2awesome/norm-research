import argparse
import json
from pathlib import Path

import pytest

import scripts.tools.silver_match_v3.finalize_nemotron_ce_training_report as recovery
from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.run_nemotron_ce import build_base_manifest
from scripts.tools.silver_match_v3.train_nemotron_cross_encoder import (
    CLASS_NAMES,
    CLASS_SAMPLING_WEIGHTS,
    HIDDEN_SIZE,
    LORA_TARGETS,
    REPORT_SCHEMA,
    class_quotas,
)


def _json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _fake_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "run"
    trainer = tmp_path / "train_nemotron_cross_encoder.py"
    trainer.write_text("# frozen trainer\n")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    base_manifest = build_base_manifest(model, tmp_path / "base-manifest.json")
    base_path = tmp_path / "base-manifest.json"
    assert base_manifest["tree_sha256"]

    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    train.write_text('{"role":"train"}\n')
    dev.write_text('{"role":"dev"}\n')
    split = root / "split_assignments.jsonl"
    split.parent.mkdir(parents=True)
    split.write_text('{"split":"train"}\n')
    config = {
        "schema_version": REPORT_SCHEMA,
        "model": str(model),
        "train_pairs": {str(train): sha256_file(train)},
        "dev_pairs": {str(dev): sha256_file(dev)},
        "split_seed": 7,
        "dev_fraction": 0.1,
        "seed": 11,
        "max_length": 128,
        "exposure_budgets": [100, 200],
        "sampler_weights": CLASS_SAMPLING_WEIGHTS,
        "batch_size_per_rank": 5,
        "gradient_accumulation_steps": 2,
        "world_size": 1,
        "lora_learning_rate": 1e-4,
        "head_learning_rate": 1e-3,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "targets": list(LORA_TARGETS),
        },
        "classifier": [HIDDEN_SIZE, len(CLASS_NAMES)],
        "labels": list(CLASS_NAMES),
        "attention": "eager",
        "split_assignments_sha256": sha256_file(split),
        "split_audit": {"source_group_overlap_count": 0},
    }
    _json(root / "run_config.json", config)

    cumulative = {name: 0 for name in CLASS_NAMES}
    events = [{"event": "RUN_STARTED", "world_size": 1}]
    optimizer_updates = 0
    previous = 0
    for budget in config["exposure_budgets"]:
        delta = budget - previous
        optimizer_updates += 10
        quotas = class_quotas(delta)
        cumulative = {name: cumulative[name] + quotas[name] for name in CLASS_NAMES}
        checkpoint = root / "checkpoints" / f"exposure-{budget:012d}"
        adapter = checkpoint / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "README.md").write_text("adapter\n")
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
        _json(
            adapter / "adapter_config.json",
            {
                "base_model_name_or_path": str(model),
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": list(reversed(LORA_TARGETS)),
            },
        )
        (checkpoint / "head.safetensors").write_bytes(b"head")
        dev_report = {
            "precision_wilson_gate_met": budget == 200,
            "exact_f_beta_0_5": budget / 1000,
            "exact_precision_wilson_95_lower": 0.8,
            "exact_precision": 0.9,
            "exact_recall": 0.2,
            "predicted_exact_count": 20,
        }
        metadata = _json(
            checkpoint / "checkpoint.json",
            {
                "schema_version": recovery.CHECKPOINT_SCHEMA,
                "exposure_budget": budget,
                "optimizer_updates": optimizer_updates,
                "cumulative_class_exposures": cumulative,
                "labels": list(CLASS_NAMES),
                "hidden_to_classes": [HIDDEN_SIZE, len(CLASS_NAMES)],
                "lora_targets": list(LORA_TARGETS),
                "dev": dev_report,
                "reload_reference": [
                    {
                        "norm_uid": f"u{index}",
                        "metric_id": f"m{index}",
                        "logits": [1, 0, -1],
                    }
                    for index in range(recovery.RECOVERY_REFERENCE_EXAMPLES)
                ],
            },
        )
        events.append(
            {
                "event": "CHECKPOINT_SAVED",
                "exposure_budget": budget,
                "path": str(checkpoint),
                "checkpoint_metadata_sha256": sha256_file(metadata),
            }
        )
        previous = budget
    events.append(
        {
            "event": "RUN_FAILED",
            "error_type": "PermissionError",
            "error": (
                "[Errno 13] Permission denied: '/afs/u/.cache/huggingface/modules/"
                "transformers_modules/model'"
            ),
        }
    )
    (root / "events.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in events)
    )
    return root, trainer, base_path


def _args(root: Path, trainer: Path, base: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_root=str(root),
        trainer=str(trainer),
        trainer_sha256=sha256_file(trainer),
        base_manifest=str(base),
        base_manifest_sha256=sha256_file(base),
        run_config_sha256=sha256_file(root / "run_config.json"),
        split_assignments_sha256=sha256_file(root / "split_assignments.jsonl"),
        events_sha256=sha256_file(root / "events.jsonl"),
        checkpoint_inventory=None,
        checkpoint_inventory_sha256=None,
        audit_inventory_output=None,
        hf_modules_cache=str(root.parent / "cache"),
        eval_batch_size=4,
        reload_atol=0.002,
    )


def _audit(args: argparse.Namespace) -> dict:
    return recovery.validate_incomplete_run(
        Path(args.run_root),
        trainer=Path(args.trainer),
        trainer_sha256=args.trainer_sha256,
        base_manifest=Path(args.base_manifest),
        base_manifest_sha256=args.base_manifest_sha256,
        run_config_sha256=args.run_config_sha256,
        split_assignments_sha256=args.split_assignments_sha256,
        events_sha256=args.events_sha256,
    )


def test_failed_complete_checkpoint_run_recovers_without_training_mutation(
    tmp_path, monkeypatch
):
    root, trainer, base = _fake_run(tmp_path)
    args = _args(root, trainer, base)
    audit = _audit(args)
    inventory = tmp_path / "checkpoint-inventory.json"
    recovery._atomic_create_json(inventory, recovery._inventory_payload(audit))
    args.checkpoint_inventory = str(inventory)
    args.checkpoint_inventory_sha256 = sha256_file(inventory)
    immutable = {
        path: sha256_file(path)
        for path in (
            root / "run_config.json",
            root / "split_assignments.jsonl",
            root / "events.jsonl",
            *sorted((root / "checkpoints").rglob("*")),
        )
        if path.is_file()
    }

    monkeypatch.setattr(recovery.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(recovery.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(recovery.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(recovery, "_trainable_parameter_count", lambda _path: 12345)

    def fake_reload(current, **_kwargs):
        selected = current["selected"]
        return {
            "status": "PASS",
            "selected_checkpoint": selected["path"],
            "examples": recovery.RECOVERY_REFERENCE_EXAMPLES,
            "maximum_absolute_logit_error": 0.0,
            "absolute_tolerance": args.reload_atol,
            "adapter_and_head_loaded_into_fresh_base": True,
            "selected_checkpoint_artifact_sha256": selected["artifact_sha256"],
        }

    monkeypatch.setattr(recovery, "_fresh_reload_report", fake_reload)
    result = recovery.finalize(args)
    assert result["status"] == "COMPLETE_RECOVERED_WITHOUT_RETRAINING"
    assert result["selected_exposure_budget"] == 200
    assert result["optimizer_or_training_steps_executed"] == 0
    assert all(sha256_file(path) == digest for path, digest in immutable.items())
    report = json.loads((root / "training_report.json").read_text())
    assert report["status"] == "COMPLETE"
    assert report["recovery"]["checkpoints_or_run_config_modified"] is False
    assert report["recovery"]["post_failure_checkpoint_inventory"]["sha256"] == sha256_file(
        inventory
    )


def test_inventory_detects_checkpoint_change_before_gpu_reload(tmp_path):
    root, trainer, base = _fake_run(tmp_path)
    args = _args(root, trainer, base)
    audit = _audit(args)
    inventory = tmp_path / "checkpoint-inventory.json"
    recovery._atomic_create_json(inventory, recovery._inventory_payload(audit))
    args.checkpoint_inventory = str(inventory)
    args.checkpoint_inventory_sha256 = sha256_file(inventory)
    selected_adapter = root / "checkpoints" / "exposure-000000000200" / "adapter" / "README.md"
    selected_adapter.write_text("changed\n")
    with pytest.raises(ValueError, match="inventory no longer matches"):
        recovery.finalize(args)


def test_recovery_refuses_active_or_completed_event_ledger(tmp_path):
    root, trainer, base = _fake_run(tmp_path)
    events = root / "events.jsonl"
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    events.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows[:-1])
    )
    args = _args(root, trainer, base)
    with pytest.raises(ValueError, match="exact final-reload"):
        _audit(args)
