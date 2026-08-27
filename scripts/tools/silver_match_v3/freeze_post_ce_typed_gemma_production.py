#!/usr/bin/env python3
"""Freeze and consolidate task-generic post-CE typed Gemma production.

``freeze`` filters exact diverse candidate unions to every norm routed onward
by the two-seed CE consensus, validates a COMPLETE dev-selected task LoRA, and
emits a direct-batch paired-Gemma command. ``consolidate`` ignores the base arm
and promotes a LoRA decision only when both original and hashed orders are
schema-valid and agree on decision plus leaf. Invalid or disagreeing rows are
retained, explicitly marked for exhaustive rescue, and never silently dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .adjudicate_gemma import CONFIDENCES, DECISIONS, ordered_candidates
from .common import normalize_space, read_jsonl, sha256_file
from .freeze_postinference_analysis_release import _load_task, _validate_ce
from .freeze_task_final_stack_handoff import PROMPT_MANIFEST_SCHEMA
from .run_paired_gemma_lora_batch import (
    EXPECTED_MODEL_CONTENT_SHA256,
    EXPECTED_MODEL_FILE_COUNT,
    SCHEMA as PAIRED_SCHEMA,
)


QUEUE_SCHEMA = "silver-match-v3-post-ce-typed-gemma-production-queue-v1"
CANDIDATE_SCHEMA = "silver-match-v3-post-ce-unresolved-candidates-v1"
CANDIDATE_REPORT_SCHEMA = "silver-match-v3-post-ce-unresolved-candidates-report-v1"
OUTPUT_SCHEMA = "silver-match-v3-post-ce-typed-gemma-consolidated-v1"
REPORT_SCHEMA = "silver-match-v3-post-ce-typed-gemma-consolidation-report-v1"
TRAIN_REPORT_SCHEMA = "silver-match-v3-gemma4-typed-lora-train-report-v2"
TRAIN_COMPLETE_STATUS = "COMPLETE_DEV_SELECTED_ADAPTER_FRESH_RELOAD_VERIFIED"
INFERENCE_META_SCHEMA = "silver-match-v3-paired-gemma4-lora-inference-meta-v1"
INFERENCE_COMPLETE_STATUS = "COMPLETE_TRUTH_BLIND_PAIRED_INFERENCE"
ORDERS = ("original", "hashed")


def _resolve(raw: str | Path, anchor: Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (anchor.parent / path).resolve()


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _directory_ref(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = [
        {
            "relative_path": child.relative_to(path).as_posix(),
            "sha256": sha256_file(child),
            "bytes": child.stat().st_size,
        }
        for child in sorted(value for value in path.rglob("*") if value.is_file())
    ]
    if not files:
        raise ValueError(f"empty directory: {path}")
    canonical = json.dumps(files, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return {
        "path": str(path),
        "content_manifest_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "file_count": len(files),
        "bytes": sum(row["bytes"] for row in files),
        "files": files,
    }


def _verify_ref(ref: Mapping[str, Any], anchor: Path, *, directory: bool = False) -> Path:
    raw = ref.get("path")
    if not raw:
        raise ValueError("artifact reference lacks path")
    path = _resolve(str(raw), anchor)
    if directory:
        observed = _directory_ref(path)
        if (
            ref.get("content_manifest_sha256") != observed["content_manifest_sha256"]
            or int(ref.get("file_count", -1)) != observed["file_count"]
        ):
            raise ValueError(f"directory artifact changed: {path}")
    elif not path.is_file() or ref.get("sha256") != sha256_file(path):
        raise ValueError(f"file artifact changed: {path}")
    return path


def parse_candidate_bindings(values: Sequence[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        corpus, separator, raw = value.partition("=")
        corpus = normalize_space(corpus)
        if not separator or not corpus or not raw or corpus in output:
            raise ValueError(f"invalid/duplicate candidate binding: {value!r}")
        output[corpus] = Path(raw).resolve()
    if not output:
        raise ValueError("at least one candidate union is required")
    return output


def _candidate_meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def _validate_union_meta(
    *,
    path: Path,
    manifest_path: Path,
    task: str,
    corpus: str,
    bank_sha: str,
    expected_count: int,
    minimum_k: int,
) -> tuple[dict[str, Any], Path]:
    path = path.resolve()
    meta_path = _candidate_meta_path(path)
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"candidate union/meta missing: {path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lanes = (meta.get("union") or {}).get("lanes") or []
    complete_lanes = [
        row
        for row in lanes
        if isinstance(row, Mapping) and row.get("kind") == "complete-bank"
    ]
    if (
        meta.get("output_sha256") != sha256_file(path)
        or meta.get("manifest_sha256") != sha256_file(manifest_path)
        or normalize_space(meta.get("task")) != task
        or normalize_space(meta.get("corpus")) != corpus
        or normalize_space(meta.get("bank_source_sha256")) != bank_sha
        or int(meta.get("input_count", -1)) != expected_count
        or int(meta.get("output_k", -1)) < minimum_k
        or len(complete_lanes) < 2
    ):
        raise ValueError(f"candidate union is not an exact diverse production union: {corpus}")
    names = [normalize_space(row.get("name")) for row in lanes]
    if "" in names or len(names) != len(set(names)):
        raise ValueError(f"candidate union has invalid lane identities: {corpus}")
    return meta, meta_path


def _validate_training_stack(
    *,
    task: str,
    training_report_path: Path,
    adapter_path: Path,
    prompt_manifest_path: Path,
    model_path: Path,
    model_inventory_path: Path,
) -> tuple[dict[str, Any], Path]:
    training_report_path = training_report_path.resolve()
    adapter_path = adapter_path.resolve()
    prompt_manifest_path = prompt_manifest_path.resolve()
    model_path = model_path.resolve()
    model_inventory_path = model_inventory_path.resolve()
    report = json.loads(training_report_path.read_text(encoding="utf-8"))
    selection = report.get("selection") or {}
    adapter = report.get("adapter") or {}
    disjoint = report.get("source_disjoint_audit") or {}
    report_inventory = report.get("model_inventory") or {}
    observed_adapter = _directory_ref(adapter_path)
    if (
        report.get("schema_version") != TRAIN_REPORT_SCHEMA
        or report.get("status") != TRAIN_COMPLETE_STATUS
        or Path(str(report.get("model") or "")).resolve() != model_path
        or selection.get("status") != "SELECTED_ON_DEV_ONLY"
        or selection.get("selection_split") != "dev"
        or selection.get("test_or_blind_data_read") is not False
        or adapter.get("adapter_only") is not True
        or adapter.get("inference_reload_verified") is not True
        or adapter.get("fresh_base_reload_verified") is not True
        or Path(str(adapter.get("directory") or "")).resolve() != adapter_path
        or (adapter.get("content") or {}).get("content_manifest_sha256")
        != observed_adapter["content_manifest_sha256"]
        or disjoint.get("status") != "PASS_SOURCE_DISJOINT_HELDOUT_GRADIENT_EXCLUDED"
        or int(disjoint.get("norm_uid_overlap_count", -1)) != 0
        or int(disjoint.get("source_group_overlap_count", -1)) != 0
        or int(disjoint.get("heldout_gradient_eligible_count", -1)) != 0
        or report_inventory.get("sha256") != sha256_file(model_inventory_path)
    ):
        raise ValueError("typed Gemma training report is not a complete dev-selected adapter")
    for name in ("config", "weights"):
        _verify_ref(adapter.get(name) or {}, training_report_path)
    for required in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_path / required).is_file():
            raise FileNotFoundError(adapter_path / required)
    for name, expected_split in (("dataset", "train"), ("dev_dataset", "dev")):
        dataset_path = _verify_ref(report.get(name) or {}, training_report_path)
        rows = list(read_jsonl(dataset_path))
        if not rows or any(
            row.get("task") != task
            or str(row.get("split") or "").lower() != expected_split
            or not isinstance(row.get("gradient_eligible"), bool)
            for row in rows
        ) or (expected_split == "dev" and any(row["gradient_eligible"] for row in rows)) or (
            expected_split == "train"
            and not any(row["gradient_eligible"] for row in rows)
        ):
            raise ValueError(f"typed Gemma {name} is not an exact task/{expected_split} input")

    inventory = json.loads(model_inventory_path.read_text(encoding="utf-8"))
    if (
        inventory.get("status") != "FROZEN_CONTENT_HASH_INVENTORY"
        or Path(str(inventory.get("root") or "")).resolve() != model_path
        or int(inventory.get("file_count", -1)) != EXPECTED_MODEL_FILE_COUNT
        or inventory.get("content_inventory_sha256") != EXPECTED_MODEL_CONTENT_SHA256
    ):
        raise ValueError("base Gemma model inventory differs from the paired runner contract")
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    for required in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        if not (model_path / required).is_file():
            raise FileNotFoundError(model_path / required)

    prompt_manifest = json.loads(prompt_manifest_path.read_text(encoding="utf-8"))
    prompt_ref = prompt_manifest.get("output") or {}
    prompt_path = _verify_ref(prompt_ref, prompt_manifest_path)
    if (
        prompt_manifest.get("schema_version") != PROMPT_MANIFEST_SCHEMA
        or prompt_manifest.get("status") != "FROZEN_TASK_LOCAL_RULES_ONLY_COMPOSITE"
        or prompt_manifest.get("task") != task
        or prompt_manifest.get("truth_examples_included") is not False
        or prompt_manifest.get("truth_labels_votes_or_outcomes_included") is not False
        or prompt_manifest.get("example_uids_included") is not False
    ):
        raise ValueError("prompt manifest is not a task-local truth-free composite")
    return report, prompt_path


def _write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    count = 0
    try:
        with temp.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return count


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def freeze_queue(
    *,
    manifest_path: Path,
    task: str,
    ce_path: Path,
    ce_report_path: Path,
    candidate_paths: Mapping[str, Path],
    training_report_path: Path,
    adapter_path: Path,
    prompt_manifest_path: Path,
    model_path: Path,
    model_inventory_path: Path,
    python_path: Path,
    candidate_output_path: Path,
    candidate_report_path: Path,
    queue_output_path: Path,
    inference_output_root: Path,
    max_candidates: int = 16,
    batch_size: int = 128,
    max_model_len: int = 4096,
    max_tokens: int = 160,
    gpu_memory_utilization: float = 0.88,
    seed: int = 17,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    ce_path = ce_path.resolve()
    ce_report_path = ce_report_path.resolve()
    python_path = python_path.resolve()
    candidate_output_path = candidate_output_path.resolve()
    candidate_report_path = candidate_report_path.resolve()
    queue_output_path = queue_output_path.resolve()
    inference_output_root = inference_output_root.resolve()
    if min(max_candidates, batch_size, max_model_len, max_tokens) < 1:
        raise ValueError("inference sizes must be positive")
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError("gpu_memory_utilization must be in (0,1)")
    if not python_path.is_file():
        raise FileNotFoundError(python_path)
    if any(path.exists() for path in (candidate_output_path, candidate_report_path, queue_output_path)):
        raise FileExistsError("refusing to overwrite post-CE queue artifacts")

    manifest, corpora, canonical, metric_ids, bank_path = _load_task(manifest_path, task)
    bank_sha = str(manifest["banks"][task]["source_sha256"])
    ce = _validate_ce(
        ce_path,
        ce_report_path,
        canonical=canonical,
        task=task,
        metric_ids=metric_ids,
    )
    ce_report_payload = json.loads(ce_report_path.read_text(encoding="utf-8"))
    progressive_ce = ce_report_payload.get("progressive_queue") is True
    if set(candidate_paths) != set(corpora):
        raise ValueError("candidate unions do not cover every task corpus exactly")
    training_report, prompt_path = _validate_training_stack(
        task=task,
        training_report_path=training_report_path,
        adapter_path=adapter_path,
        prompt_manifest_path=prompt_manifest_path,
        model_path=model_path,
        model_inventory_path=model_inventory_path,
    )

    canonical_by_corpus: dict[str, list[dict[str, Any]]] = {corpus: [] for corpus in corpora}
    for row in canonical:
        canonical_by_corpus[str(row["corpus"])].append(row)
    unresolved_rows: list[dict[str, Any]] = []
    corpus_audit: dict[str, Any] = {}
    observed_uids: set[str] = set()
    for corpus in corpora:
        union_path = candidate_paths[corpus].resolve()
        union_meta, union_meta_path = _validate_union_meta(
            path=union_path,
            manifest_path=manifest_path,
            task=task,
            corpus=corpus,
            bank_sha=bank_sha,
            expected_count=len(canonical_by_corpus[corpus]),
            minimum_k=max_candidates,
        )
        corpus_unresolved = 0
        bundles = zip_longest(
            canonical_by_corpus[corpus], read_jsonl(union_path), fillvalue=None
        )
        for canonical_row, candidate_row in bundles:
            if canonical_row is None or candidate_row is None:
                raise ValueError(f"canonical/candidate union length mismatch: {corpus}")
            uid = normalize_space(canonical_row.get("norm_uid"))
            candidate_values = list(candidate_row.get("candidates") or [])
            candidate_ids = [normalize_space(row.get("metric_id")) for row in candidate_values]
            if (
                not uid
                or uid in observed_uids
                or normalize_space(candidate_row.get("norm_uid")) != uid
                or candidate_row.get("task") != task
                or candidate_row.get("corpus") != corpus
                or int(candidate_row.get("row", -1)) != int(canonical_row.get("row", -1))
                or normalize_space(candidate_row.get("bank_source_sha256")) != bank_sha
                or len(candidate_ids) != int(union_meta["output_k"])
                or "" in candidate_ids
                or len(candidate_ids) != len(set(candidate_ids))
                or not set(candidate_ids) <= metric_ids
                or [int(row.get("rank", -1)) for row in candidate_values]
                != list(range(1, len(candidate_values) + 1))
            ):
                raise ValueError(f"candidate union row routing/bank/order mismatch: {corpus}/{uid}")
            observed_uids.add(uid)
            ce_row = ce[uid]
            ce_ids = {
                normalize_space(row.get("metric_id")) for row in ce_row.get("candidates") or []
            }
            if ce_ids:
                expected_ids = set(candidate_ids[: len(ce_ids)])
                if progressive_ce:
                    if not ce_ids <= set(candidate_ids):
                        raise ValueError(
                            f"progressive CE candidates escape diverse union: {uid}"
                        )
                elif ce_ids != expected_ids:
                    raise ValueError(
                        f"CE and diverse-union candidate universes differ: {uid}"
                    )
            if ce_row.get("automatic_match") is True:
                continue
            projected = [dict(row) for row in candidate_values[:max_candidates]]
            unresolved_rows.append(
                {
                    "schema_version": CANDIDATE_SCHEMA,
                    "task": task,
                    "corpus": corpus,
                    "row": int(canonical_row.get("row", -1)),
                    "norm_uid": uid,
                    "bank_source_sha256": bank_sha,
                    "candidates": projected,
                    "ce_routing_category": ce_row["routing_category"],
                    "ce_consensus_sha256": sha256_file(ce_path),
                    "source_candidate_union_sha256": sha256_file(union_path),
                    "source_candidate_union_meta_sha256": sha256_file(union_meta_path),
                }
            )
            corpus_unresolved += 1
        corpus_audit[corpus] = {
            "canonical_count": len(canonical_by_corpus[corpus]),
            "ce_automatic_match_count": sum(
                ce[normalize_space(row["norm_uid"])].get("automatic_match") is True
                for row in canonical_by_corpus[corpus]
            ),
            "ce_routed_count": corpus_unresolved,
            "candidate_union": _artifact(union_path),
            "candidate_union_meta": _artifact(union_meta_path),
            "complete_bank_lane_count": sum(
                row.get("kind") == "complete-bank"
                for row in (union_meta.get("union") or {}).get("lanes") or []
            ),
        }
    expected = {normalize_space(row["norm_uid"]) for row in canonical}
    unresolved = {
        uid for uid, row in ce.items() if row.get("automatic_match") is not True
    }
    output_uids = [row["norm_uid"] for row in unresolved_rows]
    if (
        observed_uids != expected
        or set(output_uids) != unresolved
        or len(output_uids) != len(set(output_uids))
    ):
        raise ValueError("unresolved filtering dropped, duplicated, or added a CE-routed norm")

    created: list[Path] = []
    try:
        _write_jsonl_new(candidate_output_path, unresolved_rows)
        created.append(candidate_output_path)
        routing_counts = Counter(row["routing_category"] for row in ce.values())
        candidate_report = {
            "schema_version": CANDIDATE_REPORT_SCHEMA,
            "status": "FROZEN_EXACT_CE_UNRESOLVED_CANDIDATE_INPUT",
            "release_ready": False,
            "task": task,
            "manifest": _artifact(manifest_path),
            "bank": {**_artifact(bank_path), "source_sha256": bank_sha, "metric_count": len(metric_ids)},
            "ce_consensus": _artifact(ce_path),
            "ce_consensus_report": _artifact(ce_report_path),
            "canonical_count": len(canonical),
            "ce_automatic_match_count": len(canonical) - len(unresolved),
            "ce_routed_count": len(unresolved),
            "candidate_output": {**_artifact(candidate_output_path), "count": len(unresolved)},
            "candidate_depth": max_candidates,
            "ce_routing_category_counts": dict(sorted(routing_counts.items())),
            "corpora": corpus_audit,
            "routing_audit": {
                "all_canonical_norms_seen_once": True,
                "automatic_and_routed_partition_exact": True,
                "every_ce_routed_norm_emitted_once": True,
                "ce_and_union_candidate_universes_identical": True,
                "minimum_complete_bank_lanes_per_corpus": 2,
                "diagnostic_subset_accepted": False,
            },
        }
        _write_json_new(candidate_report_path, candidate_report)
        created.append(candidate_report_path)
        runner_path = Path(__file__).with_name("run_paired_gemma_lora_batch.py").resolve()
        adapter_config = json.loads(
            (adapter_path / "adapter_config.json").read_text(encoding="utf-8")
        )
        max_lora_rank = int(adapter_config.get("r", -1))
        if max_lora_rank < 1:
            raise ValueError("adapter config lacks a positive LoRA rank")
        command = [
            str(python_path),
            "-u",
            "-m",
            "scripts.tools.silver_match_v3.run_paired_gemma_lora_batch",
            "--manifest",
            str(manifest_path),
            "--candidates",
            str(candidate_output_path),
            "--prompt",
            str(prompt_path),
            "--model",
            str(model_path.resolve()),
            "--model-inventory",
            str(model_inventory_path.resolve()),
            "--adapter",
            str(adapter_path.resolve()),
            "--adapter-name",
            f"{task.replace('-', '_')}_typed_production",
            "--adapter-id",
            "1",
            "--output-root",
            str(inference_output_root),
            "--max-candidates",
            str(max_candidates),
            "--batch-size",
            str(batch_size),
            "--max-model-len",
            str(max_model_len),
            "--max-tokens",
            str(max_tokens),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--max-lora-rank",
            str(max_lora_rank),
            "--seed",
            str(seed),
            "--keep-raw",
            "--resume",
        ]
        queue = {
            "schema_version": QUEUE_SCHEMA,
            "status": "FROZEN_AWAITING_DIRECT_BATCH_VLLM_INFERENCE",
            "release_ready": False,
            "task": task,
            "command": command,
            "backend": "direct_batch_vllm_not_openai_server",
            "orders": list(ORDERS),
            "inputs": {
                "manifest": _artifact(manifest_path),
                "bank": {**_artifact(bank_path), "source_sha256": bank_sha},
                "ce_consensus": _artifact(ce_path),
                "ce_consensus_report": _artifact(ce_report_path),
                "unresolved_candidates": _artifact(candidate_output_path),
                "unresolved_candidates_report": _artifact(candidate_report_path),
                "training_report": _artifact(training_report_path),
                "adapter": _directory_ref(adapter_path),
                "prompt_manifest": _artifact(prompt_manifest_path),
                "prompt": _artifact(prompt_path),
                "model_inventory": _artifact(model_inventory_path),
                "model_path": str(model_path.resolve()),
                "python": _artifact(python_path),
            },
            "implementations": {
                "paired_runner": _artifact(runner_path),
                "freezer_consolidator": _artifact(Path(__file__)),
            },
            "training_gate": {
                "status": training_report["status"],
                "selection_split": "dev",
                "test_or_blind_data_read": False,
                "fresh_base_adapter_reload_verified": True,
            },
            "coverage": {
                "canonical_count": len(canonical),
                "ce_automatic_match_count": len(canonical) - len(unresolved),
                "paired_inference_count_per_order": len(unresolved),
                "corpus_count": len(corpora),
            },
            "production_policy": {
                "base_arm_is_production_truth": False,
                "base_arm_used_for_consolidation": False,
                "lora_requires_schema_valid_original_and_hashed_orders": True,
                "lora_requires_exact_decision_and_metric_stability": True,
                "invalid_or_disagreement_requires_exhaustive_rescue": True,
                "every_non_ce_row_requires_exhaustive_rescue_before_final_release": True,
                "no_ce_routed_norm_may_be_dropped": True,
            },
            "expected_outputs": {
                "original": str(inference_output_root / "paired.original.jsonl"),
                "hashed": str(inference_output_root / "paired.hashed.jsonl"),
                "meta": str(inference_output_root / "paired_inference.meta.json"),
            },
        }
        _write_json_new(queue_output_path, queue)
        return queue
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


def _validate_inference_freeze(
    *,
    meta: Mapping[str, Any],
    meta_path: Path,
    queue: Mapping[str, Any],
) -> tuple[Path, str]:
    freeze_path = _verify_ref(meta.get("inference_freeze") or {}, meta_path)
    freeze_sha = sha256_file(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_inputs = freeze.get("inputs") or {}
    queue_inputs = queue.get("inputs") or {}
    prompt_components = freeze_inputs.get("prompt_components") or []
    paired = freeze.get("paired_contract") or {}
    firewall = freeze.get("truth_firewall") or {}
    if (
        freeze.get("schema_version")
        != "silver-match-v3-paired-gemma4-lora-inference-freeze-v1"
        or freeze.get("status") != "FROZEN_BEFORE_PAIRED_MODEL_INFERENCE"
        or freeze.get("task") != queue.get("task")
        or freeze.get("backend") != "direct_batch_vllm_not_openai_server"
        or firewall.get("truth_read") is not False
        or firewall.get("truth_path_argument_exists") is not False
        or firewall.get("scoring_in_separate_process_after_predictions") is not True
        or paired.get("systems") != ["base", "lora"]
        or paired.get("orders") != list(ORDERS)
        or paired.get("same_model_instance") is not True
        or paired.get("same_candidate_set_and_rendered_prompt_within_each_pair")
        is not True
        or paired.get("no_hyperparameter_or_seed_search") is not True
        or (freeze_inputs.get("manifest") or {}).get("sha256")
        != queue_inputs["manifest"]["sha256"]
        or (freeze_inputs.get("candidates") or {}).get("sha256")
        != queue_inputs["unresolved_candidates"]["sha256"]
        or len(prompt_components) != 1
        or prompt_components[0].get("sha256") != queue_inputs["prompt"]["sha256"]
        or (freeze_inputs.get("runner_script") or {}).get("sha256")
        != (queue.get("implementations") or {})["paired_runner"]["sha256"]
        or (freeze_inputs.get("model_inventory") or {}).get("sha256")
        != queue_inputs["model_inventory"]["sha256"]
        or Path(str((freeze_inputs.get("model_identity") or {}).get("path") or "")).resolve()
        != Path(str(queue_inputs["model_path"])).resolve()
        or Path(str((freeze_inputs.get("adapter_identity") or {}).get("path") or "")).resolve()
        != Path(str(queue_inputs["adapter"]["path"])).resolve()
    ):
        raise ValueError("paired inference freeze differs from the exact production queue")
    frozen_adapter_files = {
        name: ref.get("sha256")
        for name, ref in (freeze_inputs.get("adapter_identity") or {}).get("files", {}).items()
    }
    queued_adapter_files = {
        row["relative_path"]: row["sha256"] for row in queue_inputs["adapter"]["files"]
    }
    if frozen_adapter_files != queued_adapter_files:
        raise ValueError("paired inference used a different task LoRA adapter")
    return freeze_path, freeze_sha


def _index_paired(
    path: Path,
    *,
    order: str,
    input_rows: Mapping[str, Mapping[str, Any]],
    task: str,
    bank_sha: str,
    freeze_sha: str,
    model_path: str,
    adapter_path: str,
    prompt_sha: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        uid = normalize_space(row.get("norm_uid"))
        candidate = input_rows.get(uid)
        if not uid or uid in indexed or candidate is None:
            raise ValueError(f"paired {order} output has missing/duplicate/foreign UID: {uid!r}")
        source_cards = list(candidate.get("candidates") or [])
        expected_ids = [
            str(value["metric_id"])
            for value in ordered_candidates(source_cards, order, uid)
        ]
        if (
            row.get("schema_version") != PAIRED_SCHEMA
            or row.get("order_mode") != order
            or row.get("task") != task
            or row.get("corpus") != candidate.get("corpus")
            or int(row.get("row", -1)) != int(candidate.get("row", -1))
            or row.get("candidate_bank_source_sha256") != bank_sha
            or list(row.get("candidate_ids") or []) != expected_ids
            or row.get("inference_freeze_sha256") != freeze_sha
            or Path(str(row.get("model") or "")).resolve() != Path(model_path).resolve()
            or Path(str(row.get("adapter") or "")).resolve()
            != Path(adapter_path).resolve()
            or row.get("prompt_sha256") != prompt_sha
            or row.get("base_item_prompt_sha256")
            != row.get("lora_item_prompt_sha256")
            or not isinstance(row.get("base"), Mapping)
            or not isinstance(row.get("lora"), Mapping)
        ):
            raise ValueError(f"paired {order} row contract failed: {uid}")
        indexed[uid] = row
    if set(indexed) != set(input_rows):
        raise ValueError(f"paired {order} output does not exactly cover unresolved input")
    return indexed


def _valid_lora(payload: Mapping[str, Any], candidate_ids: set[str]) -> bool:
    decision = str(payload.get("decision") or "")
    metric_id = payload.get("metric_id")
    return bool(
        decision in DECISIONS
        and payload.get("confidence") in CONFIDENCES
        and normalize_space(payload.get("reason"))
        and payload.get("parse_error") is None
        and ((decision == "MATCH" and str(metric_id or "") in candidate_ids)
             or (decision != "MATCH" and metric_id is None))
    )


def _minimum_confidence(values: Sequence[str]) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    return min(values, key=order.__getitem__)


def consolidate(
    *,
    queue_path: Path,
    paired_original_path: Path,
    paired_hashed_path: Path,
    inference_meta_path: Path,
    output_path: Path,
    report_output_path: Path,
) -> dict[str, Any]:
    queue_path = queue_path.resolve()
    paired_original_path = paired_original_path.resolve()
    paired_hashed_path = paired_hashed_path.resolve()
    inference_meta_path = inference_meta_path.resolve()
    output_path = output_path.resolve()
    report_output_path = report_output_path.resolve()
    if output_path.exists() or report_output_path.exists():
        raise FileExistsError("refusing to overwrite typed Gemma consolidation")
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    if (
        queue.get("schema_version") != QUEUE_SCHEMA
        or queue.get("release_ready") is not False
        or (queue.get("production_policy") or {}).get("base_arm_used_for_consolidation")
        is not False
    ):
        raise ValueError("unknown or unsafe post-CE typed Gemma queue")
    task = str(queue["task"])
    inputs = queue.get("inputs") or {}
    manifest_path = _verify_ref(inputs["manifest"], queue_path)
    ce_path = _verify_ref(inputs["ce_consensus"], queue_path)
    ce_report_path = _verify_ref(inputs["ce_consensus_report"], queue_path)
    candidate_path = _verify_ref(inputs["unresolved_candidates"], queue_path)
    candidate_report_path = _verify_ref(inputs["unresolved_candidates_report"], queue_path)
    _verify_ref(inputs["training_report"], queue_path)
    _verify_ref(inputs["adapter"], queue_path, directory=True)
    _verify_ref(inputs["prompt_manifest"], queue_path)
    _verify_ref(inputs["prompt"], queue_path)
    _verify_ref(inputs["model_inventory"], queue_path)
    _verify_ref((queue.get("implementations") or {})["paired_runner"], queue_path)
    _verify_ref((queue.get("implementations") or {})["freezer_consolidator"], queue_path)

    manifest, corpora, canonical, metric_ids, _ = _load_task(manifest_path, task)
    bank_sha = str(manifest["banks"][task]["source_sha256"])
    ce = _validate_ce(
        ce_path,
        ce_report_path,
        canonical=canonical,
        task=task,
        metric_ids=metric_ids,
    )
    candidate_report = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    candidate_rows = list(read_jsonl(candidate_path))
    input_rows = {normalize_space(row.get("norm_uid")): row for row in candidate_rows}
    unresolved = {uid for uid, row in ce.items() if row.get("automatic_match") is not True}
    if (
        candidate_report.get("schema_version") != CANDIDATE_REPORT_SCHEMA
        or candidate_report.get("status") != "FROZEN_EXACT_CE_UNRESOLVED_CANDIDATE_INPUT"
        or (candidate_report.get("candidate_output") or {}).get("sha256") != sha256_file(candidate_path)
        or set(input_rows) != unresolved
        or "" in input_rows
        or len(input_rows) != len(candidate_rows)
    ):
        raise ValueError("frozen unresolved input/report no longer matches CE routing")

    meta = json.loads(inference_meta_path.read_text(encoding="utf-8"))
    meta_outputs = meta.get("outputs") or {}
    if (
        meta.get("schema_version") != INFERENCE_META_SCHEMA
        or meta.get("status") != INFERENCE_COMPLETE_STATUS
        or meta.get("task") != task
        or meta.get("truth_read") is not False
        or meta.get("backend") != "direct_batch_vllm_not_openai_server"
        or meta.get("same_loaded_base_model_instance_for_both_arms") is not True
        or set(meta_outputs) != set(ORDERS)
    ):
        raise ValueError("paired inference metadata is incomplete or unsafe")
    for order, path in (("original", paired_original_path), ("hashed", paired_hashed_path)):
        ref = meta_outputs[order]
        if (
            _resolve(str(ref.get("path") or ""), inference_meta_path) != path
            or ref.get("sha256") != sha256_file(path)
            or int(ref.get("count", -1)) != len(unresolved)
        ):
            raise ValueError(f"paired inference meta does not bind {order} output")
    freeze_path, freeze_sha = _validate_inference_freeze(
        meta=meta,
        meta_path=inference_meta_path,
        queue=queue,
    )
    prompt_sha = normalize_space(meta.get("prompt_sha256"))
    if not prompt_sha:
        raise ValueError("paired inference metadata lacks the composite prompt hash")
    original = _index_paired(
        paired_original_path,
        order="original",
        input_rows=input_rows,
        task=task,
        bank_sha=bank_sha,
        freeze_sha=freeze_sha,
        model_path=str(inputs["model_path"]),
        adapter_path=str(inputs["adapter"]["path"]),
        prompt_sha=prompt_sha,
    )
    hashed = _index_paired(
        paired_hashed_path,
        order="hashed",
        input_rows=input_rows,
        task=task,
        bank_sha=bank_sha,
        freeze_sha=freeze_sha,
        model_path=str(inputs["model_path"]),
        adapter_path=str(inputs["adapter"]["path"]),
        prompt_sha=prompt_sha,
    )

    final_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    by_corpus: dict[str, Counter[str]] = {corpus: Counter() for corpus in corpora}
    for canonical_row in canonical:
        uid = normalize_space(canonical_row["norm_uid"])
        corpus = str(canonical_row["corpus"])
        ce_row = ce[uid]
        common = {
            "schema_version": OUTPUT_SCHEMA,
            "norm_uid": uid,
            "task": task,
            "corpus": corpus,
            "row": int(canonical_row.get("row", -1)),
            "candidate_bank_source_sha256": bank_sha,
            "ce_routing_category": ce_row["routing_category"],
            "ce_consensus_sha256": sha256_file(ce_path),
            "base_arm_used_for_production": False,
            "release_ready": False,
        }
        if ce_row.get("automatic_match") is True:
            final = {
                **common,
                "decision": "MATCH",
                "metric_id": ce_row["metric_id"],
                "confidence": "high",
                "reason": "Both CE seeds passed their frozen dev gates and selected this leaf.",
                "production_route": "CE_AUTOMATIC_SAME_LEAF_TWO_GATE",
                "order_stable_lora": None,
                "requires_exhaustive_rescue": False,
                "lora_order_evidence": None,
            }
            counts["ce_automatic_match"] += 1
            by_corpus[corpus]["ce_automatic_match"] += 1
        else:
            left = original[uid]["lora"]
            right = hashed[uid]["lora"]
            candidate_ids = {
                str(value["metric_id"]) for value in input_rows[uid]["candidates"]
            }
            left_valid = _valid_lora(left, candidate_ids)
            right_valid = _valid_lora(right, candidate_ids)
            stable = bool(
                left_valid
                and right_valid
                and (left.get("decision"), left.get("metric_id"))
                == (right.get("decision"), right.get("metric_id"))
            )
            evidence = {
                "original": dict(left),
                "hashed": dict(right),
                "original_schema_valid": left_valid,
                "hashed_schema_valid": right_valid,
                "decision_and_metric_stable": stable,
            }
            if stable:
                decision = str(left["decision"])
                final = {
                    **common,
                    "decision": decision,
                    "metric_id": left.get("metric_id"),
                    "confidence": _minimum_confidence(
                        [str(left["confidence"]), str(right["confidence"])]
                    ),
                    "reason": str(left["reason"]),
                    "production_route": "LORA_SCHEMA_VALID_ORDER_STABLE_PROVISIONAL",
                    "order_stable_lora": True,
                    # Existing production doctrine requires every non-CE row,
                    # including a stable typed decision, to continue through
                    # repeated full-bank rescue before a final release.
                    "requires_exhaustive_rescue": True,
                    "lora_order_evidence": evidence,
                }
                counts["lora_stable_match" if decision == "MATCH" else "lora_stable_typed_abstention"] += 1
                by_corpus[corpus][
                    "lora_stable_match" if decision == "MATCH" else "lora_stable_typed_abstention"
                ] += 1
            else:
                any_invalid = not (left_valid and right_valid)
                final = {
                    **common,
                    "decision": "INVALID_OUTPUT" if any_invalid else "UNSTABLE_MATCH",
                    "metric_id": None,
                    "confidence": "low",
                    "reason": (
                        "At least one LoRA order violated the typed output schema."
                        if any_invalid
                        else "The schema-valid LoRA orders disagreed on decision or metric."
                    ),
                    "production_route": (
                        "LORA_INVALID_PRESERVED_FOR_RESCUE"
                        if any_invalid
                        else "LORA_ORDER_DISAGREEMENT_PRESERVED_FOR_RESCUE"
                    ),
                    "order_stable_lora": False,
                    "requires_exhaustive_rescue": True,
                    "lora_order_evidence": evidence,
                }
                key = "lora_invalid_rescue" if any_invalid else "lora_disagreement_rescue"
                counts[key] += 1
                by_corpus[corpus][key] += 1
        final_rows.append(final)
    if len(final_rows) != len(canonical):
        raise AssertionError("consolidation lost canonical rows")
    rescue_count = counts["lora_invalid_rescue"] + counts["lora_disagreement_rescue"]
    created: list[Path] = []
    try:
        _write_jsonl_new(output_path, final_rows)
        created.append(output_path)
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": "COMPLETE_CONSERVATIVE_POST_CE_TYPED_GEMMA_CONSOLIDATION",
            "release_ready": False,
            "task": task,
            "queue": _artifact(queue_path),
            "manifest": _artifact(manifest_path),
            "ce_consensus": _artifact(ce_path),
            "ce_consensus_report": _artifact(ce_report_path),
            "unresolved_candidates": _artifact(candidate_path),
            "unresolved_candidates_report": _artifact(candidate_report_path),
            "paired_original": _artifact(paired_original_path),
            "paired_hashed": _artifact(paired_hashed_path),
            "paired_inference_meta": _artifact(inference_meta_path),
            "paired_inference_freeze": _artifact(freeze_path),
            "output": {**_artifact(output_path), "count": len(final_rows)},
            "counts": {
                "canonical": len(canonical),
                "ce_automatic_match": counts["ce_automatic_match"],
                "ce_routed_to_lora": len(unresolved),
                "lora_stable_match": counts["lora_stable_match"],
                "lora_stable_typed_abstention": counts["lora_stable_typed_abstention"],
                "lora_invalid_rescue": counts["lora_invalid_rescue"],
                "lora_disagreement_rescue": counts["lora_disagreement_rescue"],
                "order_failure_rescue_required": rescue_count,
                "exhaustive_rescue_required": len(unresolved),
            },
            "by_corpus": {
                corpus: dict(sorted(counter.items())) for corpus, counter in by_corpus.items()
            },
            "routing_audit": {
                "every_canonical_norm_emitted_once": True,
                "every_ce_routed_norm_has_both_orders": True,
                "only_schema_valid_order_stable_lora_decisions_consolidated": True,
                "all_invalid_and_disagreements_preserved_for_rescue": True,
                "all_non_ce_rows_preserved_for_exhaustive_rescue": True,
                "base_arm_used_for_production": False,
                "base_arm_used_for_consolidation": False,
                "output_release_ready": False,
            },
        }
        _write_json_new(report_output_path, report)
        return report
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


def _freeze_cli(args: argparse.Namespace) -> dict[str, Any]:
    return freeze_queue(
        manifest_path=Path(args.manifest),
        task=args.task,
        ce_path=Path(args.ce_consensus),
        ce_report_path=Path(args.ce_report),
        candidate_paths=parse_candidate_bindings(args.candidate),
        training_report_path=Path(args.training_report),
        adapter_path=Path(args.adapter),
        prompt_manifest_path=Path(args.prompt_manifest),
        model_path=Path(args.model),
        model_inventory_path=Path(args.model_inventory),
        python_path=Path(args.python),
        candidate_output_path=Path(args.candidate_output),
        candidate_report_path=Path(args.candidate_report_output),
        queue_output_path=Path(args.queue_output),
        inference_output_root=Path(args.inference_output_root),
        max_candidates=args.max_candidates,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )


def _consolidate_cli(args: argparse.Namespace) -> dict[str, Any]:
    return consolidate(
        queue_path=Path(args.queue),
        paired_original_path=Path(args.paired_original),
        paired_hashed_path=Path(args.paired_hashed),
        inference_meta_path=Path(args.inference_meta),
        output_path=Path(args.output),
        report_output_path=Path(args.report_output),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--manifest", required=True)
    freeze_parser.add_argument("--task", required=True)
    freeze_parser.add_argument("--ce-consensus", required=True)
    freeze_parser.add_argument("--ce-report", required=True)
    freeze_parser.add_argument("--candidate", action="append", required=True)
    freeze_parser.add_argument("--training-report", required=True)
    freeze_parser.add_argument("--adapter", required=True)
    freeze_parser.add_argument("--prompt-manifest", required=True)
    freeze_parser.add_argument("--model", required=True)
    freeze_parser.add_argument("--model-inventory", required=True)
    freeze_parser.add_argument("--python", required=True)
    freeze_parser.add_argument("--candidate-output", required=True)
    freeze_parser.add_argument("--candidate-report-output", required=True)
    freeze_parser.add_argument("--queue-output", required=True)
    freeze_parser.add_argument("--inference-output-root", required=True)
    freeze_parser.add_argument("--max-candidates", type=int, default=16)
    freeze_parser.add_argument("--batch-size", type=int, default=128)
    freeze_parser.add_argument("--max-model-len", type=int, default=4096)
    freeze_parser.add_argument("--max-tokens", type=int, default=160)
    freeze_parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    freeze_parser.add_argument("--seed", type=int, default=17)
    consolidate_parser = subparsers.add_parser("consolidate")
    consolidate_parser.add_argument("--queue", required=True)
    consolidate_parser.add_argument("--paired-original", required=True)
    consolidate_parser.add_argument("--paired-hashed", required=True)
    consolidate_parser.add_argument("--inference-meta", required=True)
    consolidate_parser.add_argument("--output", required=True)
    consolidate_parser.add_argument("--report-output", required=True)
    args = parser.parse_args()
    payload = _freeze_cli(args) if args.command == "freeze" else _consolidate_cli(args)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
