import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file
from scripts.tools.silver_match_v3.watch_humor_ce_pilots_then_test import (
    Recipe,
    _validate_recovery_provenance,
    atomic_freeze_json,
    choose_recipe,
    freeze_score_device_override,
    freeze_test_truth,
    parse_args,
    validate_heldout_inputs,
)


def _dev(f_beta: float, precision: float = 0.92):
    return {
        "precision_wilson_gate_met": True,
        "exact_f_beta_0_5": f_beta,
        "exact_precision_wilson_95_lower": 0.86,
        "exact_precision": precision,
        "exact_recall": 0.60,
        "predicted_exact_count": 120,
    }


def _recipe(name: str, priority: int, dev, **irrelevant):
    return {
        "name": name,
        "tie_priority": priority,
        "selected_checkpoint": {"dev": dev},
        **irrelevant,
    }


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_recipe_selection_uses_dev_only_and_predeclared_tie_break():
    worse_dev = _recipe("r16", 0, _dev(0.55), secret_test_score=0.99)
    better_dev = _recipe("r32", 1, _dev(0.65), secret_test_score=0.01)
    assert choose_recipe([worse_dev, better_dev])["name"] == "r32"

    # Even an arbitrary change to held-out results cannot affect a dev tie.
    tied_a = _recipe("r16", 0, _dev(0.60), secret_test_score=-1000)
    tied_b = _recipe("r32", 1, _dev(0.60), secret_test_score=1000)
    assert choose_recipe([tied_a, tied_b])["name"] == "r16"


def test_atomic_selection_freeze_is_create_only(tmp_path):
    path = tmp_path / "PILOT_SELECTION.json"
    atomic_freeze_json(path, {"winner": "r16", "value": 1})
    first_sha = sha256_file(path)
    with pytest.raises(FileExistsError):
        atomic_freeze_json(path, {"winner": "r32", "value": 2})
    assert sha256_file(path) == first_sha
    assert json.loads(path.read_text())["winner"] == "r16"
    assert not list(tmp_path.glob("*.tmp"))


def test_test_truth_freeze_and_pair_audit_are_test_only(tmp_path):
    canonical = tmp_path / "truth.jsonl"
    rows = [
        {
            "norm_uid": "u-train",
            "metric_id": "a1",
            "acceptable_metric_ids": ["a1"],
            "decision": "MATCH",
            "source_group": "g-train",
            "split": "train",
        },
        {
            "norm_uid": "u-test-1",
            "metric_id": "a2",
            "acceptable_metric_ids": ["a2"],
            "decision": "MATCH",
            "source_group": "g-test-1",
            "split": "test",
        },
        {
            "norm_uid": "u-test-2",
            "metric_id": "a3",
            "acceptable_metric_ids": ["a3"],
            "decision": "MATCH",
            "source_group": "g-test-2",
            "split": "test",
        },
    ]
    _write_jsonl(canonical, rows)
    truth = tmp_path / "truth.test-only.jsonl"
    report = freeze_test_truth(canonical, truth)
    assert report["rows"] == report["unique_norm_uids"] == 2
    assert {row["norm_uid"] for row in read_jsonl(truth)} == {
        "u-test-1",
        "u-test-2",
    }

    pairs = tmp_path / "pairs.test.jsonl"
    _write_jsonl(
        pairs,
        [
            {
                "norm_uid": "u-test-1",
                "metric_id": "a2",
                "source_group": "g-test-1",
                "split": "test",
            },
            {
                "norm_uid": "u-test-1",
                "metric_id": "a9",
                "source_group": "g-test-1",
                "split": "test",
            },
            {
                "norm_uid": "u-test-2",
                "metric_id": "a3",
                "source_group": "g-test-2",
                "split": "test",
            },
        ],
    )
    audit = validate_heldout_inputs(pairs, truth)
    assert audit["test_norm_groups"] == 2
    assert audit["test_pair_rows"] == 3
    assert audit["all_rows_split_test"] is True

    bad = tmp_path / "bad.jsonl"
    _write_jsonl(
        bad,
        [
            {
                "norm_uid": "u-test-1",
                "metric_id": "a2",
                "source_group": "g-test-1",
                "split": "dev",
            }
        ],
    )
    with pytest.raises(ValueError, match="non-test"):
        validate_heldout_inputs(bad, truth)


def test_poll_interval_is_capped_at_thirty_seconds():
    assert parse_args(["--poll-seconds", "30"]).poll_seconds == 30
    with pytest.raises(SystemExit):
        parse_args(["--poll-seconds", "31"])


def test_recovered_complete_report_requires_exact_inventory_and_finalizer(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    finalizer = tmp_path / "finalizer.py"
    finalizer.write_text("# exact recovery code\n")
    events = root / "events.jsonl"
    _write_jsonl(
        events,
        [
            {"event": "RUN_STARTED", "world_size": 1},
            {
                "event": "RUN_FAILED",
                "error_type": "PermissionError",
                "error": (
                    "Permission denied: /afs/u/.cache/huggingface/modules/"
                    "transformers_modules/model"
                ),
            },
        ],
    )
    checkpoints = [
        {
            "path": str(root / "checkpoints" / "exposure-000000050000"),
            "exposure_budget": 50_000,
            "artifact_sha256": {"head.safetensors": "a" * 64},
            "checkpoint_metadata_sha256": "b" * 64,
            "dev": _dev(0.6),
        }
    ]
    run_config = {"sha256": "c" * 64}
    split = {"sha256": "d" * 64}
    trainer = {"sha256": "e" * 64}
    inventory = tmp_path / "inventory.json"
    inventory_payload = {
        "schema_version": "silver-match-v3-nemotron-ce-failed-run-inventory-v1",
        "status": "FROZEN_POST_FAILURE_PRE_RECOVERY_BYTES",
        "run_root": str(root),
        "run_config": run_config,
        "split_assignments": split,
        "events": {"path": str(events), "sha256": sha256_file(events)},
        "trainer": trainer,
        "base_model": {},
        "checkpoints": checkpoints,
        "selected_checkpoint": checkpoints[0],
        "training_report_existed": False,
        "reload_verification_existed": False,
        "checkpoints_read_only": True,
        "gpu_processes_launched": 0,
    }
    inventory.write_text(json.dumps(inventory_payload, sort_keys=True) + "\n")
    report = {
        "selected_checkpoint": checkpoints[0],
        "input_sha256": {
            "run_config": run_config["sha256"],
            "split_assignments": split["sha256"],
            "trainer": trainer["sha256"],
        },
        "recovery": {
            "schema_version": "silver-match-v3-nemotron-ce-finalization-recovery-v1",
            "status": "FINALIZED_WITHOUT_RETRAINING_OR_CHECKPOINT_MUTATION",
            "finalizer": {"path": str(finalizer), "sha256": sha256_file(finalizer)},
            "events": {"path": str(events), "sha256": sha256_file(events)},
            "post_failure_checkpoint_inventory": {
                "path": str(inventory),
                "sha256": sha256_file(inventory),
            },
            "hf_modules_cache": "/lfs/recovery/cache",
            "eval_batch_size": 16,
            "reload_atol": 0.002,
            "checkpoints_or_run_config_modified": False,
            "optimizer_or_training_steps_executed": 0,
        },
    }
    recipe = Recipe("pilot", root, 0, 0)
    audit = _validate_recovery_provenance(recipe, report, checkpoints)
    assert audit["optimizer_or_training_steps_executed"] == 0
    inventory_payload["gpu_processes_launched"] = 1
    inventory.write_text(json.dumps(inventory_payload, sort_keys=True) + "\n")
    report["recovery"]["post_failure_checkpoint_inventory"]["sha256"] = sha256_file(
        inventory
    )
    with pytest.raises(ValueError, match="inventory differs"):
        _validate_recovery_provenance(recipe, report, checkpoints)


def test_score_device_override_changes_only_equivalent_physical_h200(
    tmp_path, monkeypatch
):
    selection_path = tmp_path / "PILOT_SELECTION.json"
    selection = {
        "winner": "r32",
        "winner_gpu": 3,
        "winner_record": {"selected_checkpoint": {"path": "/frozen/checkpoint"}},
    }
    selection_path.write_text(json.dumps(selection, sort_keys=True) + "\n")
    monkeypatch.setattr(
        "scripts.tools.silver_match_v3.watch_humor_ce_pilots_then_test._gpu_name",
        lambda _gpu: "NVIDIA H200",
    )
    path = tmp_path / "GPU_SCORE_OVERRIDE.json"
    frozen = freeze_score_device_override(
        path,
        selection_path=selection_path,
        selection=selection,
        score_gpu=6,
    )
    assert frozen["frozen_recipe_gpu"] == 3
    assert frozen["actual_score_gpu"] == 6
    assert frozen["model_checkpoint_changed"] is False
    assert frozen["thresholds_or_batch_size_changed"] is False
    assert frozen["selection_or_winner_changed"] is False
    assert freeze_score_device_override(
        path,
        selection_path=selection_path,
        selection=selection,
        score_gpu=6,
    ) == frozen
