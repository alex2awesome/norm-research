#!/usr/bin/env python3
"""Persistently select the Humor CE pilot on dev, then score untouched test.

The two recipe identities, order, and sk2 GPU assignments are predeclared.
This watcher accepts only complete, hash-consistent trainer releases with a
passing fresh-base reload.  It chooses between their already-selected
checkpoints using development metrics only, atomically freezes that decision,
and only then opens the held-out test pair/truth files.  Test results can never
feed back into recipe or checkpoint selection.

All release artifacts are create-only.  A restarted watcher validates and
reuses a complete selection, score shard, or evaluation report; partial or
inconsistent artifacts fail closed rather than being overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import normalize_space, read_jsonl, sha256_file
from .run_nemotron_ce import (
    EVAL_SCHEMA,
    SCORE_META_SCHEMA,
    build_base_manifest,
    evaluate,
    score,
    verify_base_manifest,
    verify_checkpoint_contract,
)
from .train_nemotron_cross_encoder import (
    CLASS_NAMES,
    HIDDEN_SIZE,
    LORA_TARGETS,
    REPORT_SCHEMA,
    checkpoint_selection_key,
)


ROOT = Path("/lfs/skampere2/0/alexspan/norm-research-silver-v3/runtime/humor_ce_v2")
MODEL = Path(
    "/lfs/skampere2/0/alexspan/norm-research-silver-v3/models/"
    "llama-embed-nemotron-8b-aa3b43a495a9b280d1bdb716da37c54bb495d630-mirror-v1"
)
RUN_NAMES = (
    "humor_ce_r16_a32_lr1e4_seed20260713_v2",
    "humor_ce_r32_a64_lr5e5_seed20260713_v2",
)
RUN_GPUS = (2, 3)
SELECTION_SCHEMA = "silver-match-v3-humor-ce-pilot-selection-v1"
WATCH_SCHEMA = "silver-match-v3-humor-ce-pilot-watch-v1"
TEST_TRUTH_SCHEMA = "silver-match-v3-humor-ce-test-truth-v1"
RECOVERY_SCHEMA = "silver-match-v3-nemotron-ce-finalization-recovery-v1"
RECOVERY_INVENTORY_SCHEMA = "silver-match-v3-nemotron-ce-failed-run-inventory-v1"
SCORE_DEVICE_OVERRIDE_SCHEMA = "silver-match-v3-humor-ce-score-device-override-v1"


@dataclass(frozen=True)
class Recipe:
    name: str
    root: Path
    gpu: int
    tie_priority: int


def default_recipes(root: Path) -> tuple[Recipe, ...]:
    return tuple(
        Recipe(name, root / "runs" / name, gpu, priority)
        for priority, (name, gpu) in enumerate(zip(RUN_NAMES, RUN_GPUS))
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_freeze_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish JSON exactly once without overwrite semantics."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    raw = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        # Hard-link publication is atomic and fails if the immutable target
        # already exists.  Unlike replace(), it can never overwrite a release.
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def append_event(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": WATCH_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    raw = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _verify_hash(path: Path, expected: Any, label: str) -> str:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != normalize_space(expected):
        raise ValueError(f"{label} SHA256 mismatch: {path}")
    return observed


def _validate_checkpoint_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = Path(str(entry.get("path") or "")).resolve()
    expected = entry.get("artifact_sha256")
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError(f"checkpoint lacks artifact hashes: {checkpoint}")
    observed = _file_hashes(checkpoint)
    if observed != dict(expected):
        raise ValueError(f"checkpoint artifact hashes differ: {checkpoint}")
    metadata_path = checkpoint / "checkpoint.json"
    _verify_hash(
        metadata_path,
        entry.get("checkpoint_metadata_sha256"),
        "checkpoint metadata",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if tuple(metadata.get("labels") or ()) != CLASS_NAMES:
        raise ValueError("checkpoint label order drift")
    if metadata.get("hidden_to_classes") != [HIDDEN_SIZE, len(CLASS_NAMES)]:
        raise ValueError("checkpoint classifier shape drift")
    if tuple(metadata.get("lora_targets") or ()) != LORA_TARGETS:
        raise ValueError("checkpoint LoRA targets drift")
    if metadata.get("dev") != entry.get("dev"):
        raise ValueError("checkpoint dev report differs from training report")
    return {
        "path": str(checkpoint),
        "artifact_sha256": observed,
        "checkpoint_metadata_sha256": sha256_file(metadata_path),
        "exposure_budget": int(entry["exposure_budget"]),
        "dev": dict(entry["dev"]),
    }


def _validate_recovery_provenance(
    recipe: Recipe,
    report: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Require explicit post-failure evidence for recovered COMPLETE reports."""

    events_path = recipe.root / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line]
    terminal = events[-1] if events else {}
    recovery = report.get("recovery")
    if terminal.get("event") != "RUN_FAILED" and recovery is None:
        return None
    finalizer = (recovery or {}).get("finalizer") or {}
    event_ref = (recovery or {}).get("events") or {}
    inventory_ref = (recovery or {}).get("post_failure_checkpoint_inventory") or {}
    if (
        not isinstance(recovery, Mapping)
        or recovery.get("schema_version") != RECOVERY_SCHEMA
        or recovery.get("status")
        != "FINALIZED_WITHOUT_RETRAINING_OR_CHECKPOINT_MUTATION"
        or recovery.get("checkpoints_or_run_config_modified") is not False
        or int(recovery.get("optimizer_or_training_steps_executed", -1)) != 0
        or int(recovery.get("eval_batch_size", -1)) != 16
        or float(recovery.get("reload_atol", -1)) != 0.002
        or not str(recovery.get("hf_modules_cache") or "").startswith("/lfs/")
        or terminal.get("error_type") != "PermissionError"
        or "/.cache/huggingface/modules/transformers_modules/"
        not in normalize_space(terminal.get("error"))
    ):
        raise ValueError("recovered COMPLETE report lacks exact recovery provenance")
    finalizer_path = Path(str(finalizer.get("path") or "")).resolve()
    inventory_path = Path(str(inventory_ref.get("path") or "")).resolve()
    if (
        Path(str(event_ref.get("path") or "")).resolve() != events_path.resolve()
        or _verify_hash(events_path, event_ref.get("sha256"), "recovery events")
        != event_ref.get("sha256")
        or _verify_hash(finalizer_path, finalizer.get("sha256"), "recovery finalizer")
        != finalizer.get("sha256")
        or _verify_hash(
            inventory_path,
            inventory_ref.get("sha256"),
            "post-failure checkpoint inventory",
        )
        != inventory_ref.get("sha256")
    ):
        raise ValueError("recovery artifact binding differs")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    inputs = report.get("input_sha256") or {}
    if (
        inventory.get("schema_version") != RECOVERY_INVENTORY_SCHEMA
        or inventory.get("status") != "FROZEN_POST_FAILURE_PRE_RECOVERY_BYTES"
        or Path(str(inventory.get("run_root") or "")).resolve()
        != recipe.root.resolve()
        or inventory.get("training_report_existed") is not False
        or inventory.get("reload_verification_existed") is not False
        or inventory.get("checkpoints_read_only") is not True
        or int(inventory.get("gpu_processes_launched", -1)) != 0
        or inventory.get("checkpoints") != list(checkpoints)
        or inventory.get("selected_checkpoint") != report.get("selected_checkpoint")
        or (inventory.get("run_config") or {}).get("sha256")
        != inputs.get("run_config")
        or (inventory.get("split_assignments") or {}).get("sha256")
        != inputs.get("split_assignments")
        or (inventory.get("trainer") or {}).get("sha256") != inputs.get("trainer")
        or (inventory.get("events") or {}).get("sha256") != event_ref.get("sha256")
    ):
        raise ValueError("recovery inventory differs from completed training report")
    return {
        "schema_version": RECOVERY_SCHEMA,
        "post_failure_checkpoint_inventory_sha256": inventory_ref["sha256"],
        "finalizer_sha256": finalizer["sha256"],
        "optimizer_or_training_steps_executed": 0,
        "reload_atol": 0.002,
        "eval_batch_size": 16,
    }


def _best_checkpoint(checkpoints: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return max(
        checkpoints,
        key=lambda row: (
            *checkpoint_selection_key(row["dev"]),
            -int(row["exposure_budget"]),
        ),
    )


def validate_complete_run(recipe: Recipe, *, model: Path, trainer_path: Path) -> dict[str, Any]:
    """Fail-closed validation of one complete trainer release and all inputs."""

    report_path = recipe.root / "training_report.json"
    reload_path = recipe.root / "reload_verification.json"
    run_config_path = recipe.root / "run_config.json"
    split_path = recipe.root / "split_assignments.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != REPORT_SCHEMA or report.get("status") != "COMPLETE":
        raise ValueError(f"run is not COMPLETE: {recipe.name}")
    if Path(str(report.get("model") or "")).resolve() != model.resolve():
        raise ValueError(f"base model drift: {recipe.name}")
    if tuple(report.get("labels") or ()) != CLASS_NAMES:
        raise ValueError(f"report labels drift: {recipe.name}")
    if report.get("hidden_to_classes") != [HIDDEN_SIZE, len(CLASS_NAMES)]:
        raise ValueError(f"report classifier drift: {recipe.name}")

    inputs = report.get("input_sha256")
    if not isinstance(inputs, Mapping):
        raise ValueError("training report lacks input hash ledger")
    train_hashes = inputs.get("train_pairs")
    dev_hashes = inputs.get("dev_pairs")
    if not isinstance(train_hashes, Mapping) or not isinstance(dev_hashes, Mapping):
        raise ValueError("training report lacks train/dev hashes")
    for label, values in (("train pair", train_hashes), ("dev pair", dev_hashes)):
        for raw_path, digest in values.items():
            _verify_hash(Path(str(raw_path)), digest, label)
    _verify_hash(run_config_path, inputs.get("run_config"), "run config")
    _verify_hash(split_path, inputs.get("split_assignments"), "split assignments")
    _verify_hash(trainer_path, inputs.get("trainer"), "trainer source")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    if run_config.get("split_assignments_sha256") != sha256_file(split_path):
        raise ValueError("run config split hash drift")
    if run_config.get("split_audit") != report.get("split_audit"):
        raise ValueError("run config/report split audit differs")
    if int(report["split_audit"].get("source_group_overlap_count", -1)) != 0:
        raise ValueError("train/dev source-group leakage")

    checkpoint_entries = report.get("checkpoints")
    if not isinstance(checkpoint_entries, list) or not checkpoint_entries:
        raise ValueError("training report has no checkpoints")
    checkpoints = [_validate_checkpoint_entry(row) for row in checkpoint_entries]
    recovery_audit = _validate_recovery_provenance(recipe, report, checkpoints)
    expected_selected = _best_checkpoint(checkpoints)
    selected = report.get("selected_checkpoint")
    if not isinstance(selected, Mapping):
        raise ValueError("training report lacks selected checkpoint")
    if Path(str(selected.get("path") or "")).resolve() != Path(
        expected_selected["path"]
    ).resolve():
        raise ValueError("trainer selected a non-optimal development checkpoint")

    reload_report = json.loads(reload_path.read_text(encoding="utf-8"))
    if reload_report.get("status") != "PASS":
        raise ValueError(f"fresh-base reload did not pass: {recipe.name}")
    if reload_report != report.get("reload_verification"):
        raise ValueError("reload file differs from embedded training report")
    if Path(str(reload_report.get("selected_checkpoint") or "")).resolve() != Path(
        expected_selected["path"]
    ).resolve():
        raise ValueError("reload verification used a different checkpoint")
    if reload_report.get("selected_checkpoint_artifact_sha256") != expected_selected[
        "artifact_sha256"
    ]:
        raise ValueError("reload verification checkpoint hashes drift")

    report_sha = sha256_file(report_path)
    contract = verify_checkpoint_contract(
        Path(expected_selected["path"]),
        report_path,
        report_sha,
        model=model,
    )
    return {
        "name": recipe.name,
        "root": str(recipe.root.resolve()),
        "gpu": recipe.gpu,
        "tie_priority": recipe.tie_priority,
        "training_report": str(report_path.resolve()),
        "training_report_sha256": report_sha,
        "reload_verification": str(reload_path.resolve()),
        "reload_verification_sha256": sha256_file(reload_path),
        "run_config_sha256": sha256_file(run_config_path),
        "split_assignments_sha256": sha256_file(split_path),
        "train_pairs_sha256": dict(train_hashes),
        "dev_pairs_sha256": dict(dev_hashes),
        "selected_checkpoint": expected_selected,
        "checkpoint_contract": contract,
        "dev_selection_key": list(checkpoint_selection_key(expected_selected["dev"])),
        "split_audit": report["split_audit"],
        "recovery_audit": recovery_audit,
    }


def choose_recipe(validated: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Choose only from development metrics with a predeclared stable tie break."""

    if len(validated) != 2:
        raise ValueError("exactly two predeclared pilot recipes are required")
    priorities = [int(row["tie_priority"]) for row in validated]
    if len(set(priorities)) != len(priorities):
        raise ValueError("recipe tie priorities must be unique")
    return max(
        validated,
        key=lambda row: (
            *checkpoint_selection_key(row["selected_checkpoint"]["dev"]),
            -int(row["tie_priority"]),
        ),
    )


def freeze_test_truth(source: Path, output: Path) -> dict[str, Any]:
    """Mechanically freeze only preassigned test rows, without changing content."""

    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    count = 0
    uids: set[str] = set()
    groups: dict[str, str] = {}
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in read_jsonl(source):
                if normalize_space(row.get("split")) != "test":
                    continue
                uid = normalize_space(row.get("norm_uid"))
                group = normalize_space(row.get("source_group"))
                if not uid or not group or uid in uids:
                    raise ValueError("test truth has missing/duplicate UID or source group")
                uids.add(uid)
                groups[uid] = group
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        if not count:
            raise ValueError("canonical truth has no test rows")
        os.link(temporary, output)
        _fsync_directory(output.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema_version": TEST_TRUTH_SCHEMA,
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "rows": count,
        "unique_norm_uids": len(uids),
        "splits": {"test": count},
    }


def validate_heldout_inputs(test_pairs: Path, test_truth: Path) -> dict[str, Any]:
    """Open held-out data only after PILOT_SELECTION has been frozen."""

    truth: dict[str, str] = {}
    for row in read_jsonl(test_truth):
        if normalize_space(row.get("split")) != "test":
            raise ValueError("test truth contains a non-test row")
        uid = normalize_space(row.get("norm_uid"))
        group = normalize_space(row.get("source_group"))
        if not uid or not group or uid in truth:
            raise ValueError("test truth UID/source contract is invalid")
        truth[uid] = group
    pair_count = 0
    pair_uids: set[str] = set()
    pair_keys: set[tuple[str, str]] = set()
    for row in read_jsonl(test_pairs):
        if normalize_space(row.get("split")) != "test":
            raise ValueError("test pairs contain a non-test row")
        uid = normalize_space(row.get("norm_uid"))
        metric = normalize_space(row.get("metric_id") or row.get("candidate_metric_id"))
        group = normalize_space(row.get("source_group"))
        key = (uid, metric)
        if not all(key) or key in pair_keys:
            raise ValueError("test pairs contain a missing/duplicate pair identity")
        if uid not in truth or truth[uid] != group:
            raise ValueError("test pair/truth source contract differs")
        pair_keys.add(key)
        pair_uids.add(uid)
        pair_count += 1
    if pair_uids != set(truth):
        raise ValueError("test pair candidates do not cover the test truth universe")
    return {
        "test_pairs": str(test_pairs.resolve()),
        "test_pairs_sha256": sha256_file(test_pairs),
        "test_pair_rows": pair_count,
        "test_truth": str(test_truth.resolve()),
        "test_truth_sha256": sha256_file(test_truth),
        "test_norm_groups": len(truth),
        "all_rows_split_test": True,
    }


def _validate_existing_selection(
    path: Path,
    validated: Sequence[Mapping[str, Any]],
    base_manifest: Path,
) -> dict[str, Any]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError("unknown pilot selection schema")
    current = {row["name"]: row["training_report_sha256"] for row in validated}
    frozen = {
        row["name"]: row["training_report_sha256"]
        for row in selection.get("validated_recipes", [])
    }
    if current != frozen:
        raise ValueError("frozen pilot selection input reports have drifted")
    if selection.get("base_manifest_sha256") != sha256_file(base_manifest):
        raise ValueError("frozen pilot selection base manifest has drifted")
    expected = choose_recipe(validated)
    if selection.get("winner") != expected["name"]:
        raise ValueError("frozen winner differs from deterministic dev selection")
    return selection


def _gpu_memory_used(gpu: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def _gpu_name(gpu: int) -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=name",
            "--format=csv,noheader",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip().splitlines()[0]


def freeze_score_device_override(
    path: Path,
    *,
    selection_path: Path,
    selection: Mapping[str, Any],
    score_gpu: int,
) -> dict[str, Any]:
    """Bind a physical H200 relocation without changing any model/eval input."""

    recipe_gpu = int(selection["winner_gpu"])
    recipe_gpu_name = _gpu_name(recipe_gpu)
    score_gpu_name = _gpu_name(score_gpu)
    if recipe_gpu_name != score_gpu_name or "H200" not in recipe_gpu_name:
        raise ValueError("score-device relocation requires equivalent sk2 H200 devices")
    payload = {
        "schema_version": SCORE_DEVICE_OVERRIDE_SCHEMA,
        "status": "FROZEN_PHYSICAL_DEVICE_ONLY_EQUIVALENT_H200_RELOCATION",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": str(selection_path.resolve()),
        "selection_sha256": sha256_file(selection_path),
        "winner": selection["winner"],
        "frozen_recipe_gpu": recipe_gpu,
        "actual_score_gpu": score_gpu,
        "frozen_recipe_gpu_name": recipe_gpu_name,
        "actual_score_gpu_name": score_gpu_name,
        "model_checkpoint_changed": False,
        "training_report_changed": False,
        "heldout_pairs_or_truth_changed": False,
        "thresholds_or_batch_size_changed": False,
        "selection_or_winner_changed": False,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in payload.items() if key != "created_at"}
        observed = {key: value for key, value in existing.items() if key != "created_at"}
        if observed != comparable:
            raise ValueError("existing score-device override differs")
        return existing
    atomic_freeze_json(path, payload)
    return json.loads(path.read_text(encoding="utf-8"))


def _wait_for_gpu(gpu: int, poll_seconds: int, events: Path) -> None:
    while True:
        memory = _gpu_memory_used(gpu)
        if memory <= 8192:
            append_event(events, "WINNER_GPU_READY", gpu=gpu, memory_used_mib=memory)
            return
        append_event(events, "WAITING_FOR_WINNER_GPU", gpu=gpu, memory_used_mib=memory)
        time.sleep(poll_seconds)


def _validate_existing_score(path: Path, selection: Mapping[str, Any]) -> dict[str, Any]:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if path.exists() != meta_path.exists():
        raise ValueError("partial test score artifact exists; refusing recovery overwrite")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCORE_META_SCHEMA:
        raise ValueError("test score metadata schema drift")
    if meta.get("output_sha256") != sha256_file(path):
        raise ValueError("test score hash drift")
    if meta.get("checkpoint_contract") != selection["winner_record"]["checkpoint_contract"]:
        raise ValueError("test score did not use frozen winning checkpoint")
    return meta


def _validate_existing_evaluation(path: Path, score_path: Path, truth: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != EVAL_SCHEMA:
        raise ValueError("test evaluation schema drift")
    if report.get("scores_sha256") != sha256_file(score_path):
        raise ValueError("test evaluation score hash drift")
    if report.get("truth_sha256") != sha256_file(truth):
        raise ValueError("test evaluation truth hash drift")
    if report.get("test_threshold_tuning_performed") is not False:
        raise ValueError("test evaluation reports threshold tuning")
    return report


def watch(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    model = Path(args.model).resolve()
    recipes = default_recipes(root)
    release = root / "runs" / "pilot_test_release_v1"
    events = release / "WATCH_EVENTS.jsonl"
    selection_path = release / "PILOT_SELECTION.json"
    base_manifest = release / "BASE_MODEL_MANIFEST.json"
    trainer_path = root / "code" / "scripts" / "tools" / "silver_match_v3" / "train_nemotron_cross_encoder.py"
    append_event(
        events,
        "WATCH_STARTED",
        host=socket.gethostname(),
        pid=os.getpid(),
        recipes=[row.name for row in recipes],
    )

    validated: list[dict[str, Any]] = []
    while True:
        missing = [row.name for row in recipes if not (row.root / "training_report.json").is_file()]
        if missing:
            append_event(events, "WAITING_FOR_TRAINING_REPORTS", missing=missing)
            time.sleep(args.poll_seconds)
            continue
        # Reports are terminal artifacts.  Once present, any invalidity is a
        # hard failure rather than a reason to poll a corrupt release forever.
        validated = [
            validate_complete_run(row, model=model, trainer_path=trainer_path)
            for row in recipes
        ]
        append_event(events, "BOTH_RUNS_VALIDATED")
        break

    if base_manifest.exists():
        base_manifest_sha = sha256_file(base_manifest)
        base_contract = verify_base_manifest(model, base_manifest, base_manifest_sha)
    else:
        build_base_manifest(model, base_manifest)
        base_manifest_sha = sha256_file(base_manifest)
        base_contract = verify_base_manifest(model, base_manifest, base_manifest_sha)

    if selection_path.exists():
        selection = _validate_existing_selection(selection_path, validated, base_manifest)
        append_event(events, "PILOT_SELECTION_REUSED", winner=selection["winner"])
    else:
        winner = choose_recipe(validated)
        selection = {
            "schema_version": SELECTION_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "selection_data_role": "development_only",
            "test_opened_before_selection": False,
            "selection_key": [
                "precision_wilson_gate_met",
                "exact_f_beta_0_5",
                "exact_precision_wilson_95_lower",
                "exact_precision",
                "exact_recall",
                "predicted_exact_count",
                "predeclared_recipe_tie_priority",
            ],
            "validated_recipes": validated,
            "winner": winner["name"],
            "winner_gpu": winner["gpu"],
            "winner_record": winner,
            "base_manifest": str(base_manifest),
            "base_manifest_sha256": base_manifest_sha,
            "base_contract": base_contract,
        }
        atomic_freeze_json(selection_path, selection)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        append_event(events, "PILOT_SELECTION_FROZEN", winner=selection["winner"])

    # This is the first point at which the watcher opens either held-out file.
    test_pairs = Path(args.test_pairs).resolve()
    test_truth = Path(args.test_truth).resolve()
    heldout = validate_heldout_inputs(test_pairs, test_truth)
    append_event(events, "HELDOUT_INPUTS_VALIDATED", **heldout)

    winner = selection["winner_record"]
    recipe_gpu = int(selection["winner_gpu"])
    gpu = int(args.score_device) if args.score_device is not None else recipe_gpu
    override = None
    if gpu != recipe_gpu:
        override_path = release / "GPU_SCORE_OVERRIDE.json"
        override = freeze_score_device_override(
            override_path,
            selection_path=selection_path,
            selection=selection,
            score_gpu=gpu,
        )
        append_event(
            events,
            "SCORE_DEVICE_OVERRIDE_FROZEN",
            artifact=str(override_path),
            artifact_sha256=sha256_file(override_path),
            frozen_recipe_gpu=recipe_gpu,
            actual_score_gpu=gpu,
        )
    score_path = release / selection["winner"] / "test_scores.jsonl"
    if score_path.exists() or score_path.with_suffix(score_path.suffix + ".meta.json").exists():
        score_report = _validate_existing_score(score_path, selection)
        append_event(events, "TEST_SCORE_REUSED", output=str(score_path))
    else:
        _wait_for_gpu(gpu, args.poll_seconds, events)
        append_event(events, "TEST_SCORE_STARTED", gpu=gpu, output=str(score_path))
        score_args = argparse.Namespace(
            output=str(score_path),
            model=str(model),
            input_pairs=str(test_pairs),
            checkpoint=winner["selected_checkpoint"]["path"],
            training_report=winner["training_report"],
            training_report_sha256=winner["training_report_sha256"],
            base_manifest=str(base_manifest),
            base_manifest_sha256=selection["base_manifest_sha256"],
            max_length=int(winner["checkpoint_contract"]["max_sequence_length"]),
            batch_size=args.batch_size,
            shard_id=0,
            num_shards=1,
            attention="eager",
            device=gpu,
        )
        score_report = score(score_args)
        append_event(
            events,
            "TEST_SCORE_COMPLETE",
            output=str(score_path),
            output_sha256=score_report["output_sha256"],
        )

    evaluation_path = score_path.parent / "TEST_EVALUATION.json"
    if evaluation_path.exists():
        evaluation = _validate_existing_evaluation(evaluation_path, score_path, test_truth)
        append_event(events, "TEST_EVALUATION_REUSED", output=str(evaluation_path))
    else:
        evaluation = evaluate(
            argparse.Namespace(
                scores=str(score_path),
                truth=str(test_truth),
                output=str(evaluation_path),
            )
        )
        append_event(
            events,
            "TEST_EVALUATION_COMPLETE",
            output=str(evaluation_path),
            retained_precision=evaluation["metrics"]["retained_precision"],
            recall_end_to_end=evaluation["metrics"]["recall_end_to_end"],
        )

    final_path = release / "WATCH_COMPLETE.json"
    final = {
        "schema_version": WATCH_SCHEMA,
        "status": "COMPLETE",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "winner": selection["winner"],
        "winner_gpu": recipe_gpu,
        "score_gpu": gpu,
        "score_device_override": (
            {
                "path": str((release / "GPU_SCORE_OVERRIDE.json").resolve()),
                "sha256": sha256_file(release / "GPU_SCORE_OVERRIDE.json"),
                "contract": override,
            }
            if override is not None
            else None
        ),
        "heldout": heldout,
        "score": str(score_path),
        "score_sha256": sha256_file(score_path),
        "score_meta_sha256": sha256_file(
            score_path.with_suffix(score_path.suffix + ".meta.json")
        ),
        "evaluation": str(evaluation_path),
        "evaluation_sha256": sha256_file(evaluation_path),
        "test_threshold_tuning_performed": False,
    }
    if final_path.exists():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in final.items() if key != "created_at"}
        observed = {key: value for key, value in existing.items() if key != "created_at"}
        if comparable != observed:
            raise ValueError("existing WATCH_COMPLETE report differs")
        return existing
    atomic_freeze_json(final_path, final)
    append_event(events, "WATCH_COMPLETE", winner=selection["winner"])
    return final


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument(
        "--test-pairs",
        default=str(
            ROOT
            / "data/existing_truth_compact400k_v2/"
            "existing_truth.compact400k.v2.test.pairs.jsonl"
        ),
    )
    parser.add_argument(
        "--test-truth",
        default=str(
            ROOT
            / "data/existing_truth_compact400k_v2/"
            "truth.canonical.pair-eligible.v2.test-only.jsonl"
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--score-device", type=int)
    parser.add_argument("--prepare-test-truth-source")
    parser.add_argument("--prepare-test-truth-output")
    args = parser.parse_args(argv)
    if bool(args.prepare_test_truth_source) != bool(args.prepare_test_truth_output):
        parser.error("provide both --prepare-test-truth-source and --prepare-test-truth-output")
    if not 1 <= args.poll_seconds <= 30:
        parser.error("--poll-seconds must be in [1, 30]")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.score_device is not None and not 0 <= args.score_device <= 7:
        parser.error("--score-device must be an sk2 GPU index in [0, 7]")
    return args


def main() -> None:
    args = parse_args()
    if args.prepare_test_truth_source:
        result = freeze_test_truth(
            Path(args.prepare_test_truth_source), Path(args.prepare_test_truth_output)
        )
    else:
        result = watch(args)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
