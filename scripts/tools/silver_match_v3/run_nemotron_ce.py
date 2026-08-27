#!/usr/bin/env python3
"""Score and evaluate immutable shards from a Nemotron CE checkpoint.

This runner is intentionally stricter than a typical inference script.  It
loads the base model afresh, verifies a content-addressed base manifest and
every adapter/head file recorded by the training report, and then reuses the
trainer's exact bidirectional tokenization, mask-mean pooling, and dynamic
4096->3 or 4096->2 head. Shards are assigned by ``norm_uid`` so every candidate for a norm stays
together.  Shard artifacts are create-only and can be merged only after an
exact streaming comparison with the original pair input.

Evaluation never searches for thresholds. It applies the gate frozen on the
checkpoint's development set. Three-way checkpoints retain one metric; binary
checkpoints independently retain every candidate above the frozen P(EXACT)
threshold. A gold metric absent from the scored candidates is therefore a
retrieval miss, not an abstention success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TextIO

import numpy as np
import torch

from .common import normalize_space, read_jsonl, sha256_file
from .train_nemotron_cross_encoder import (
    CLASS_NAMES,
    CLASS_TO_ID,
    HIDDEN_SIZE,
    LORA_TARGETS,
    MAX_SEQUENCE_LENGTH,
    PairExample,
    REPORT_SCHEMA,
    _load_saved_model,
    _load_tokenizer,
    bidirectional_collate,
    normalize_class,
    output_class_names,
    wilson_interval,
)


SCORE_SCHEMA = "silver-match-v3-nemotron-ce-scores-v1"
SCORE_META_SCHEMA = "silver-match-v3-nemotron-ce-score-meta-v1"
EVAL_SCHEMA = "silver-match-v3-nemotron-ce-evaluation-v1"
BASE_MANIFEST_SCHEMA = "silver-match-v3-nemotron-base-manifest-v1"
CHECKPOINT_SCHEMA = "silver-match-v3-nemotron-ce-checkpoint-v1"
PAIR_ID_KEYS = ("norm_uid", "metric_id")


def _json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _canonical_json_sha(payload: Any) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _model_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: str(path.relative_to(root)),
    )


def build_base_manifest(model: Path, output: Path) -> dict[str, Any]:
    """Content-lock every file in a local base model tree (create-only)."""

    model = model.resolve()
    output = output.resolve()
    if not model.is_dir():
        raise ValueError("the production base must be a local model directory")
    if output == model or model in output.parents:
        raise ValueError("base manifest output must live outside the model tree")
    files = [
        {
            "path": str(path.relative_to(model)),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _model_files(model)
    ]
    if not files:
        raise ValueError("base model directory is empty")
    payload = {
        "schema_version": BASE_MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model),
        "files": files,
        "tree_sha256": _canonical_json_sha(files),
    }
    _json_exclusive(output, payload)
    return payload


def verify_base_manifest(
    model: Path, manifest_path: Path, expected_manifest_sha256: str
) -> dict[str, Any]:
    """Verify the manifest identity and all current base-model bytes."""

    model = model.resolve()
    manifest_path = manifest_path.resolve()
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("base manifest SHA256 mismatch")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != BASE_MANIFEST_SCHEMA:
        raise ValueError("unknown base manifest schema")
    if Path(str(payload.get("model") or "")).resolve() != model:
        raise ValueError("base manifest was locked for a different model path")
    expected = {
        normalize_space(row.get("path")): row
        for row in payload.get("files", [])
        if isinstance(row, Mapping)
    }
    observed_paths = {
        str(path.relative_to(model)): path for path in _model_files(model)
    }
    if not expected or set(expected) != set(observed_paths):
        raise ValueError("base model file set differs from locked manifest")
    verified = []
    for relative in sorted(expected):
        path = observed_paths[relative]
        row = expected[relative]
        if path.stat().st_size != int(row.get("size", -1)):
            raise ValueError(f"base model size mismatch: {relative}")
        digest = sha256_file(path)
        if digest != row.get("sha256"):
            raise ValueError(f"base model hash mismatch: {relative}")
        verified.append(
            {"path": relative, "size": path.stat().st_size, "sha256": digest}
        )
    tree_sha = _canonical_json_sha(verified)
    if tree_sha != payload.get("tree_sha256"):
        raise ValueError("base model tree hash mismatch")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": expected_manifest_sha256,
        "tree_sha256": tree_sha,
        "file_count": len(verified),
    }


def _checkpoint_record(report: Mapping[str, Any]) -> Mapping[str, Any]:
    selected = report.get("selected_checkpoint")
    if not isinstance(selected, Mapping):
        raise ValueError("training report lacks selected_checkpoint")
    return selected


def verify_checkpoint_contract(
    checkpoint: Path,
    training_report_path: Path,
    expected_training_report_sha256: str,
    *,
    model: Path | None = None,
) -> dict[str, Any]:
    """Verify metadata schema and every selected adapter/head artifact."""

    checkpoint = checkpoint.resolve()
    training_report_path = training_report_path.resolve()
    report_sha = sha256_file(training_report_path)
    if report_sha != expected_training_report_sha256:
        raise ValueError("training report SHA256 mismatch")
    report = json.loads(training_report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != REPORT_SCHEMA or report.get("status") != "COMPLETE":
        raise ValueError("training report is not a completed Nemotron CE report")
    if model is not None:
        expected_model = normalize_space(report.get("model"))
        if not expected_model:
            raise ValueError("training report lacks its base model identity")
        if Path(expected_model).resolve() != model.resolve():
            raise ValueError("inference base differs from the training report model")
    selected = _checkpoint_record(report)
    expected_hashes = selected.get("artifact_sha256")
    if not isinstance(expected_hashes, Mapping) or not expected_hashes:
        raise ValueError("training report lacks selected artifact hashes")
    observed_files = {
        str(path.relative_to(checkpoint)): path
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file()
    }
    if set(observed_files) != set(expected_hashes):
        raise ValueError("checkpoint artifact file set differs from training report")
    for relative, path in observed_files.items():
        if sha256_file(path) != str(expected_hashes[relative]):
            raise ValueError(f"checkpoint artifact hash mismatch: {relative}")

    metadata_path = checkpoint / "checkpoint.json"
    if selected.get("checkpoint_metadata_sha256") != sha256_file(metadata_path):
        raise ValueError("selected checkpoint metadata hash mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unknown checkpoint metadata schema")
    classification_mode = normalize_space(
        metadata.get("classification_mode") or report.get("classification_mode")
    ) or "three_way"
    labels = output_class_names(classification_mode)
    if tuple(metadata.get("labels") or ()) != labels:
        raise ValueError("checkpoint label schema/order mismatch")
    if metadata.get("hidden_to_classes") != [HIDDEN_SIZE, len(labels)]:
        raise ValueError("checkpoint classifier shape contract mismatch")
    if tuple(metadata.get("lora_targets") or ()) != LORA_TARGETS:
        raise ValueError("checkpoint LoRA target contract mismatch")
    if tuple(report.get("labels") or ()) != labels:
        raise ValueError("training report label schema/order mismatch")
    if report.get("hidden_to_classes") != [HIDDEN_SIZE, len(labels)]:
        raise ValueError("training report classifier shape contract mismatch")

    head_path = checkpoint / "head.safetensors"
    adapter_config_path = checkpoint / "adapter" / "adapter_config.json"
    if not head_path.is_file() or not adapter_config_path.is_file():
        raise ValueError("checkpoint is missing head or adapter config")
    if model is not None:
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        adapter_base = normalize_space(adapter_config.get("base_model_name_or_path"))
        if not adapter_base or Path(adapter_base).resolve() != model.resolve():
            raise ValueError("PEFT adapter base differs from the locked inference base")
    dev = metadata.get("dev")
    if not isinstance(dev, Mapping):
        raise ValueError("checkpoint lacks frozen development gate")
    selected_dev = selected.get("dev")
    if not isinstance(selected_dev, Mapping) or dict(selected_dev) != dict(dev):
        raise ValueError("checkpoint development gate differs from training report")
    score_threshold = dev.get("score_threshold")
    margin_threshold = dev.get("top_margin_threshold")
    if not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in (score_threshold, margin_threshold)
    ):
        raise ValueError("checkpoint development gate is invalid")
    max_length = report.get("max_sequence_length")
    if not isinstance(max_length, int) or not 1 <= max_length <= MAX_SEQUENCE_LENGTH:
        raise ValueError("training report has an invalid max_sequence_length")
    return {
        "checkpoint": str(checkpoint),
        "training_report": str(training_report_path),
        "training_report_sha256": report_sha,
        "checkpoint_metadata_sha256": sha256_file(metadata_path),
        "head_sha256": sha256_file(head_path),
        "adapter_tree_sha256": _canonical_json_sha(
            [
                [relative, str(expected_hashes[relative])]
                for relative in sorted(expected_hashes)
                if relative.startswith("adapter/")
            ]
        ),
        "artifact_sha256": dict(sorted(expected_hashes.items())),
        "classification_mode": classification_mode,
        "labels": list(labels),
        "score_threshold": float(score_threshold),
        "top_margin_threshold": float(margin_threshold),
        "max_sequence_length": max_length,
        "threshold_provenance": "checkpoint.dev",
    }


def validate_loaded_head(
    checkpoint: Path, labels: Sequence[str] = CLASS_NAMES
) -> dict[str, list[int]]:
    """Inspect the safetensors header and enforce the dynamic 4096->N head."""

    from safetensors.torch import load_file

    state = load_file(checkpoint / "head.safetensors", device="cpu")
    shapes = {name: list(value.shape) for name, value in state.items()}
    expected = {"weight": [len(labels), HIDDEN_SIZE], "bias": [len(labels)]}
    if shapes != expected:
        raise ValueError(f"classifier head tensor contract mismatch: {shapes}")
    return shapes


def pair_shard(norm_uid: str, num_shards: int) -> int:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    try:
        value = int(norm_uid[:16], 16)
    except ValueError:
        value = int(hashlib.sha256(norm_uid.encode("utf-8")).hexdigest()[:16], 16)
    return value % num_shards


@dataclass(frozen=True)
class ScorePair:
    norm_uid: str
    metric_id: str
    source_group: str
    split: str
    example: PairExample
    gold_relation: str | None


def score_pair_from_row(row: Mapping[str, Any], source: str = "<memory>") -> ScorePair:
    uid = normalize_space(row.get("norm_uid") or row.get("uid"))
    metric_id = normalize_space(row.get("candidate_metric_id") or row.get("metric_id"))
    source_group = normalize_space(row.get("source_group") or row.get("split_group"))
    split = normalize_space(row.get("split"))
    norm_text = normalize_space(
        row.get("norm")
        or row.get("statement")
        or row.get("human_statement")
        or row.get("query")
    )
    evidence = normalize_space(row.get("evidence") or row.get("context"))
    metric_card = normalize_space(row.get("metric_card"))
    if not metric_card:
        metric = row.get("metric")
        if isinstance(metric, Mapping):
            name = normalize_space(metric.get("name") or metric.get("metric_name"))
            description = normalize_space(
                metric.get("description") or metric.get("definition")
            )
            metric_card = name + ((". Definition: " + description) if description else "")
    missing = [
        key
        for key, value in (
            ("norm_uid", uid),
            ("metric_id", metric_id),
            ("source_group", source_group),
            ("split", split),
            ("norm/query", norm_text),
            ("metric_card", metric_card),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"incomplete score pair in {source}: missing {missing}")
    raw_relation = next(
        (
            row[key]
            for key in ("relation", "ce_label", "target", "class_label", "label")
            if row.get(key) is not None
        ),
        None,
    )
    gold_relation = normalize_class(raw_relation) if raw_relation is not None else None
    # The collator requires a label tensor, but labels never enter inference.
    example = PairExample(
        norm_uid=uid,
        source_group=source_group,
        metric_id=metric_id,
        norm_text=norm_text,
        evidence=evidence,
        metric_card=metric_card,
        label=gold_relation or "REJECT",
    )
    return ScorePair(uid, metric_id, source_group, split, example, gold_relation)


def _pair_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        normalize_space(row.get("norm_uid")),
        normalize_space(row.get("metric_id") or row.get("candidate_metric_id")),
    )


def _pair_source_contract(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        normalize_space(row.get("source_group") or row.get("split_group")),
        normalize_space(row.get("split")),
    )


def _iter_score_pairs(path: Path, shard_id: int, num_shards: int) -> Iterator[ScorePair]:
    uid_contract: dict[str, tuple[str, str]] = {}
    current_uid: str | None = None
    current_metrics: set[str] = set()
    for line_no, row in enumerate(read_jsonl(path), 1):
        pair = score_pair_from_row(row, f"{path}:{line_no}")
        contract = uid_contract.setdefault(pair.norm_uid, (pair.source_group, pair.split))
        if contract != (pair.source_group, pair.split):
            raise ValueError(f"norm crosses source_group/split: {pair.norm_uid}")
        if pair.norm_uid != current_uid:
            current_uid = pair.norm_uid
            current_metrics = set()
        if pair.metric_id in current_metrics:
            raise ValueError(f"duplicate input pair: {(pair.norm_uid, pair.metric_id)}")
        current_metrics.add(pair.metric_id)
        if pair_shard(pair.norm_uid, num_shards) == shard_id:
            yield pair


def _batches(values: Iterable[ScorePair], batch_size: int) -> Iterator[list[ScorePair]]:
    batch: list[ScorePair] = []
    for value in values:
        batch.append(value)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_production_model(
    model_path: Path,
    checkpoint: Path,
    attention: str,
    device: torch.device,
    classification_mode: str,
):
    # ``_load_saved_model`` constructs AutoModel from_pretrained before applying
    # PEFT, which is the fresh-base reload contract used by training itself.
    validate_loaded_head(checkpoint, output_class_names(classification_mode))
    args = argparse.Namespace(
        model=str(model_path),
        attention=attention,
        classification_mode=classification_mode,
    )
    model = _load_saved_model(args, checkpoint, device)
    return model


def score(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Nemotron CE production scoring requires CUDA bf16")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("selected CUDA device does not support bfloat16")
    output = Path(args.output).resolve()
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    if output.exists() or meta_path.exists():
        raise FileExistsError(f"refusing to overwrite score shard: {output}")
    model_path = Path(args.model).resolve()
    input_path = Path(args.input_pairs).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    base_contract = verify_base_manifest(
        model_path,
        Path(args.base_manifest),
        args.base_manifest_sha256,
    )
    checkpoint_contract = verify_checkpoint_contract(
        checkpoint,
        Path(args.training_report),
        args.training_report_sha256,
        model=model_path,
    )
    if args.max_length != checkpoint_contract["max_sequence_length"]:
        raise ValueError(
            "inference max_length differs from the training report: "
            f"{args.max_length} != {checkpoint_contract['max_sequence_length']}"
        )
    tokenizer = _load_tokenizer(str(model_path))
    classification_mode = str(checkpoint_contract["classification_mode"])
    labels = output_class_names(classification_mode)
    score_labels = ("REJECT", "EXACT") if classification_mode == "binary" else labels
    model = _load_production_model(
        model_path,
        checkpoint,
        args.attention,
        device,
        classification_mode,
    )
    collate = bidirectional_collate(
        tokenizer,
        max_length=args.max_length,
        classification_mode=classification_mode,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    norm_uids: set[str] = set()
    with output.open("x", encoding="utf-8") as handle:
        with torch.inference_mode():
            for batch in _batches(
                _iter_score_pairs(input_path, args.shard_id, args.num_shards),
                args.batch_size,
            ):
                encoded = collate([row.example for row in batch])
                input_ids = encoded["input_ids"].to(device, non_blocking=True)
                mask = encoded["attention_mask"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids, mask)
                probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                for pair, values in zip(batch, probabilities):
                    predicted = score_labels[int(np.argmax(values))]
                    record: dict[str, Any] = {
                        "schema_version": SCORE_SCHEMA,
                        "norm_uid": pair.norm_uid,
                        "metric_id": pair.metric_id,
                        "source_group": pair.source_group,
                        "split": pair.split,
                        "predicted_relation": predicted,
                        "probabilities": {
                            label: float(values[index])
                            for index, label in enumerate(score_labels)
                        },
                    }
                    if pair.gold_relation is not None:
                        record["gold_relation"] = pair.gold_relation
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    row_count += 1
                    norm_uids.add(pair.norm_uid)
        handle.flush()
        os.fsync(handle.fileno())
    meta = {
        "schema_version": SCORE_META_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_pairs": str(input_path),
        "input_pairs_sha256": sha256_file(input_path),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "row_count": row_count,
        "norm_group_count": len(norm_uids),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "base_contract": base_contract,
        "checkpoint_contract": checkpoint_contract,
        "classification_mode": classification_mode,
        "labels": list(labels),
        "score_labels": list(score_labels),
        "bidirectional_concatenation": True,
        "pooling": "native_attention_mask_mean",
        "max_length": args.max_length,
        "cuda_bf16": True,
        "attention": args.attention,
    }
    _json_exclusive(meta_path, meta)
    return meta


def _open_row_iter(path: Path) -> tuple[TextIO, Iterator[dict[str, Any]]]:
    handle = path.open("r", encoding="utf-8", errors="replace")

    def values() -> Iterator[dict[str, Any]]:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            yield value

    return handle, values()


def _score_meta(path: Path) -> dict[str, Any]:
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCORE_META_SCHEMA:
        raise ValueError(f"unknown score metadata schema: {path}")
    if meta.get("output_sha256") != sha256_file(path):
        raise ValueError(f"score shard hash mismatch: {path}")
    return meta


def _merge_invariants(meta: Mapping[str, Any]) -> dict[str, Any]:
    mutable = {
        "created_at",
        "output",
        "output_sha256",
        "row_count",
        "norm_group_count",
        "shard_id",
    }
    return {key: value for key, value in meta.items() if key not in mutable}


def merge_score_shards(input_paths: Sequence[Path], output: Path) -> dict[str, Any]:
    """Validate shards against source order and merge with constant memory."""

    output = output.resolve()
    meta_out = output.with_suffix(output.suffix + ".meta.json")
    if output.exists() or meta_out.exists():
        raise FileExistsError(f"refusing to overwrite merged scores: {output}")
    if len(input_paths) < 2:
        raise ValueError("provide at least two score shards")
    records: dict[int, tuple[Path, dict[str, Any]]] = {}
    invariant = None
    num_shards = None
    for raw in input_paths:
        path = raw.resolve()
        meta = _score_meta(path)
        shard_id = int(meta.get("shard_id", -1))
        current_total = int(meta.get("num_shards", -1))
        if current_total < 2 or not 0 <= shard_id < current_total:
            raise ValueError("invalid score shard coordinates")
        if shard_id in records:
            raise ValueError(f"duplicate score shard id: {shard_id}")
        if num_shards is None:
            num_shards = current_total
        elif current_total != num_shards:
            raise ValueError("score shards disagree on num_shards")
        current_invariant = _merge_invariants(meta)
        if invariant is None:
            invariant = current_invariant
        elif current_invariant != invariant:
            raise ValueError("score shard runtime metadata differs")
        records[shard_id] = (path, meta)
    assert num_shards is not None and invariant is not None
    if set(records) != set(range(num_shards)):
        raise ValueError("score shard set is incomplete")

    handles: dict[int, TextIO] = {}
    iterators: dict[int, Iterator[dict[str, Any]]] = {}
    for shard_id, (path, _) in records.items():
        handles[shard_id], iterators[shard_id] = _open_row_iter(path)
    source = Path(str(invariant["input_pairs"]))
    if sha256_file(source) != invariant.get("input_pairs_sha256"):
        raise ValueError("source pair input changed since scoring")
    count = 0
    norm_uids: set[str] = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="utf-8") as out:
            for source_row in read_jsonl(source):
                expected = _pair_identity(source_row)
                if not all(expected):
                    raise ValueError("source pair is missing norm_uid/metric_id")
                shard_id = pair_shard(expected[0], num_shards)
                try:
                    scored = next(iterators[shard_id])
                except StopIteration as exc:
                    raise ValueError(f"score shard {shard_id} ends before source") from exc
                if _pair_identity(scored) != expected:
                    raise ValueError(
                        f"score shard/source pair order mismatch: {expected} vs "
                        f"{_pair_identity(scored)}"
                    )
                expected_source = _pair_source_contract(source_row)
                if not all(expected_source) or _pair_source_contract(scored) != expected_source:
                    raise ValueError(
                        f"score shard/source split contract mismatch: {expected}"
                    )
                if scored.get("schema_version") != SCORE_SCHEMA:
                    raise ValueError(f"score row schema mismatch: {expected}")
                _probability_vector(scored)
                out.write(json.dumps(scored, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
                norm_uids.add(expected[0])
            for shard_id, iterator in iterators.items():
                try:
                    extra = next(iterator)
                except StopIteration:
                    continue
                raise ValueError(f"score shard {shard_id} has extra row: {_pair_identity(extra)}")
            out.flush()
            os.fsync(out.fileno())
    except Exception:
        # An incomplete artifact is intentionally left visible and never reused.
        raise
    finally:
        for handle in handles.values():
            handle.close()
    if count != sum(int(meta["row_count"]) for _, meta in records.values()):
        raise ValueError("merged score count differs from shard metadata")
    meta = {
        **invariant,
        "schema_version": SCORE_META_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "row_count": count,
        "norm_group_count": len(norm_uids),
        "shard_id": 0,
        "num_shards": 1,
        "combined_from_num_shards": num_shards,
        "combined_shards": {
            str(shard_id): {
                "path": str(path),
                "sha256": sha256_file(path),
                "meta_sha256": sha256_file(path.with_suffix(path.suffix + ".meta.json")),
            }
            for shard_id, (path, _) in sorted(records.items())
        },
    }
    _json_exclusive(meta_out, meta)
    return meta


@dataclass
class TruthNorm:
    norm_uid: str
    source_group: str
    split: str
    exact_metric_ids: set[str]


def load_truth_universe(path: Path) -> dict[str, TruthNorm]:
    output: dict[str, TruthNorm] = {}
    for line_no, row in enumerate(read_jsonl(path), 1):
        uid = normalize_space(row.get("norm_uid") or row.get("uid"))
        group = normalize_space(row.get("source_group") or row.get("split_group"))
        split = normalize_space(row.get("split"))
        if not uid or not group or not split:
            raise ValueError(f"truth row lacks norm_uid/source_group/split: {path}:{line_no}")
        value = output.setdefault(uid, TruthNorm(uid, group, split, set()))
        if (value.source_group, value.split) != (group, split):
            raise ValueError(f"truth norm crosses source_group/split: {uid}")
        raw_relation = next(
            (
                row[key]
                for key in ("relation", "ce_label", "target", "class_label", "label")
                if row.get(key) is not None
            ),
            None,
        )
        decision = normalize_space(row.get("decision")).upper()
        relation = normalize_class(raw_relation) if raw_relation is not None else None
        metric_id = normalize_space(row.get("candidate_metric_id") or row.get("metric_id"))
        acceptable = row.get("acceptable_metric_ids") or row.get("equivalent_metric_ids") or []
        if isinstance(acceptable, str):
            acceptable = [acceptable]
        if relation == "EXACT" or (relation is None and decision in {"MATCH", "EXACT"}):
            if not metric_id and not acceptable:
                raise ValueError(f"EXACT truth row lacks metric identity: {uid}")
            if metric_id:
                value.exact_metric_ids.add(metric_id)
            value.exact_metric_ids.update(
                normalize_space(item) for item in acceptable if normalize_space(item)
            )
    if not output:
        raise ValueError("truth universe is empty")
    return output


def _f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    beta2 = beta * beta
    denominator = beta2 * precision + recall
    return (1.0 + beta2) * precision * recall / denominator if denominator else 0.0


def _probability_vector(
    row: Mapping[str, Any], labels: Sequence[str] | None = None
) -> np.ndarray:
    raw = row.get("probabilities")
    if not isinstance(raw, Mapping):
        raise ValueError("score row lacks probability mapping")
    if labels is None:
        labels = (
            ("REJECT", "EXACT")
            if set(raw) == {"REJECT", "EXACT"}
            else CLASS_NAMES
        )
    values = np.asarray([float(raw[name]) for name in labels], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("score probabilities are invalid")
    if not math.isclose(float(values.sum()), 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError("score probabilities do not sum to one")
    return values


def _relation_confusion(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matrix = {gold: {pred: 0 for pred in CLASS_NAMES} for gold in CLASS_NAMES}
    support = correct = 0
    for row in rows:
        raw = row.get("gold_relation")
        if raw is None:
            continue
        gold = normalize_class(raw)
        values = _probability_vector(row)
        predicted = CLASS_NAMES[int(np.argmax(values))]
        matrix[gold][predicted] += 1
        support += 1
        correct += int(gold == predicted)
    return {
        "labels": list(CLASS_NAMES),
        "matrix": matrix,
        "support": support,
        "accuracy": correct / support if support else None,
    }


def _audit_splits_sources(
    scored: Sequence[Mapping[str, Any]], truth: Mapping[str, TruthNorm]
) -> dict[str, Any]:
    seen_contract: dict[str, tuple[str, str]] = {}
    extra = set()
    for row in scored:
        uid = normalize_space(row.get("norm_uid"))
        contract = (normalize_space(row.get("source_group")), normalize_space(row.get("split")))
        if not uid or not all(contract):
            raise ValueError("score row lacks norm_uid/source_group/split")
        if seen_contract.setdefault(uid, contract) != contract:
            raise ValueError(f"scored norm crosses source_group/split: {uid}")
        if uid not in truth:
            extra.add(uid)
        elif contract != (truth[uid].source_group, truth[uid].split):
            raise ValueError(f"score/truth source_group or split mismatch: {uid}")
    if extra:
        raise ValueError(f"scores contain norms outside truth universe: {len(extra)}")
    group_splits: dict[str, set[str]] = defaultdict(set)
    for value in truth.values():
        group_splits[value.source_group].add(value.split)
    crossing = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1}
    if crossing:
        raise ValueError(f"truth source groups cross splits: {list(crossing)[:10]}")
    return {
        "complete": True,
        "truth_norm_groups": len(truth),
        "scored_norm_groups": len(seen_contract),
        "missing_scored_norm_groups": len(set(truth) - set(seen_contract)),
        "extra_scored_norm_groups": 0,
        "truth_source_groups": len(group_splits),
        "source_groups_crossing_splits": 0,
        "truth_split_counts": dict(sorted(Counter(v.split for v in truth.values()).items())),
        "scored_split_counts": dict(sorted(Counter(v[1] for v in seen_contract.values()).items())),
    }


def evaluate_rows(
    scored: Sequence[Mapping[str, Any]],
    truth: Mapping[str, TruthNorm],
    *,
    score_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
    """Apply a fixed gate and report group-level plus pair-level performance."""

    audit = _audit_splits_sources(scored, truth)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    pair_ids: set[tuple[str, str]] = set()
    for row in scored:
        identity = _pair_identity(row)
        if not all(identity) or identity in pair_ids:
            raise ValueError(f"missing/duplicate scored pair identity: {identity}")
        pair_ids.add(identity)
        _probability_vector(row)
        groups[identity[0]].append(row)

    gold_exact_groups = sum(bool(value.exact_metric_ids) for value in truth.values())
    candidate_present = top1_correct = retained = retained_correct = 0
    no_candidate_groups = 0
    per_split: dict[str, Counter[str]] = defaultdict(Counter)
    for uid, value in truth.items():
        candidates = groups.get(uid, [])
        split_count = per_split[value.split]
        split_count["groups"] += 1
        split_count["gold_exact"] += int(bool(value.exact_metric_ids))
        candidate_ids = {_pair_identity(row)[1] for row in candidates}
        present = bool(value.exact_metric_ids & candidate_ids)
        candidate_present += int(bool(value.exact_metric_ids) and present)
        if not candidates:
            no_candidate_groups += 1
            continue
        ranked = sorted(
            candidates,
            key=lambda row: (
                -float(_probability_vector(row)[CLASS_TO_ID["EXACT"]]),
                _pair_identity(row)[1],
            ),
        )
        top = ranked[0]
        top_values = _probability_vector(top)
        top_score = float(top_values[CLASS_TO_ID["EXACT"]])
        second = (
            float(_probability_vector(ranked[1])[CLASS_TO_ID["EXACT"]])
            if len(ranked) > 1
            else 0.0
        )
        correct = _pair_identity(top)[1] in value.exact_metric_ids
        top1_correct += int(correct)
        predicts_exact = (
            int(np.argmax(top_values)) == CLASS_TO_ID["EXACT"]
            and top_score >= score_threshold
            and top_score - second >= margin_threshold
        )
        retained += int(predicts_exact)
        retained_correct += int(predicts_exact and correct)
        split_count["candidate_present"] += int(bool(value.exact_metric_ids) and present)
        split_count["top1_correct"] += int(correct)
        split_count["retained"] += int(predicts_exact)
        split_count["retained_correct"] += int(predicts_exact and correct)

    retained_precision = retained_correct / retained if retained else 1.0
    recall_present = retained_correct / candidate_present if candidate_present else 0.0
    recall_e2e = retained_correct / gold_exact_groups if gold_exact_groups else 0.0
    interval = wilson_interval(retained_correct, retained)
    top1_present = top1_correct / candidate_present if candidate_present else 0.0
    top1_e2e = top1_correct / gold_exact_groups if gold_exact_groups else 0.0
    retrieval_recall = candidate_present / gold_exact_groups if gold_exact_groups else 0.0
    split_reports = {}
    for split, count in sorted(per_split.items()):
        precision = count["retained_correct"] / count["retained"] if count["retained"] else 1.0
        recall = count["retained_correct"] / count["gold_exact"] if count["gold_exact"] else 0.0
        split_reports[split] = {
            **dict(count),
            "retained_precision": precision,
            "end_to_end_recall": recall,
            "f_beta_0_5": _f_beta(precision, recall),
        }
    return {
        "thresholds": {
            "score": float(score_threshold),
            "top_candidate_margin": float(margin_threshold),
            "provenance": "checkpoint.dev",
            "tuned_during_evaluation": False,
        },
        "norm_groups": len(truth),
        "scored_norm_groups": len(groups),
        "gold_exact_groups": gold_exact_groups,
        "candidate_present_exact_groups": candidate_present,
        "retrieval_recall": retrieval_recall,
        "groups_with_no_scored_candidates": no_candidate_groups,
        "ungated_top1_exact_count": top1_correct,
        "ungated_top1_exact_precision_over_scored_groups": (
            top1_correct / len(groups) if groups else 0.0
        ),
        "ungated_top1_recall_candidate_present": top1_present,
        "ungated_top1_recall_end_to_end": top1_e2e,
        "retained_count": retained,
        "retained_exact_count": retained_correct,
        "retained_precision": retained_precision,
        "retained_precision_wilson_95": interval,
        "retained_precision_wilson_95_lower": interval[0] if interval else None,
        "recall_candidate_present": recall_present,
        "recall_end_to_end": recall_e2e,
        "f_beta_0_5_candidate_present": _f_beta(retained_precision, recall_present),
        "f_beta_0_5_end_to_end": _f_beta(retained_precision, recall_e2e),
        "abstention_rate": 1.0 - (retained / len(truth)) if truth else 0.0,
        "relation_confusion": _relation_confusion(scored),
        "split_source_audit": audit,
        "by_split": split_reports,
    }


def _f1(precision: float, recall: float) -> float:
    return (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )


def evaluate_binary_rows(
    scored: Sequence[Mapping[str, Any]],
    truth: Mapping[str, TruthNorm],
    *,
    score_threshold: float,
) -> dict[str, Any]:
    """Evaluate frozen-threshold set-valued binary predictions.

    Every candidate above the dev-frozen P(EXACT) threshold is retained. Thus
    norms may abstain, return one metric, or return multiple metrics.
    """

    audit = _audit_splits_sources(scored, truth)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    pair_ids: set[tuple[str, str]] = set()
    for row in scored:
        identity = _pair_identity(row)
        if not all(identity) or identity in pair_ids:
            raise ValueError(f"missing/duplicate scored pair identity: {identity}")
        pair_ids.add(identity)
        raw = row.get("probabilities")
        if not isinstance(raw, Mapping) or set(raw) != {"REJECT", "EXACT"}:
            raise ValueError("binary score probabilities must be EXACT/REJECT")
        _probability_vector(row, ("REJECT", "EXACT"))
        groups[identity[0]].append(row)

    micro_tp = micro_fp = micro_fn = 0
    gold_pair_count = present_gold_pairs = 0
    gold_groups = candidate_any_groups = candidate_full_groups = 0
    predicted_groups = any_correct_groups = full_capture_groups = 0
    exact_set_groups = zero_gold_groups = zero_gold_abstained = 0
    predicted_multi_groups = gold_multi_groups = 0
    macro_precision: list[float] = []
    macro_recall: list[float] = []
    macro_f1: list[float] = []
    macro_positive_precision: list[float] = []
    macro_positive_recall: list[float] = []
    macro_positive_f1: list[float] = []

    for uid, value in truth.items():
        candidates = groups.get(uid, [])
        candidate_ids = {_pair_identity(row)[1] for row in candidates}
        gold = set(value.exact_metric_ids)
        predicted = {
            _pair_identity(row)[1]
            for row in candidates
            if float(row["probabilities"]["EXACT"]) >= score_threshold
        }
        tp = len(predicted & gold)
        fp = len(predicted - gold)
        fn = len(gold - predicted)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        gold_pair_count += len(gold)
        present_gold_pairs += len(gold & candidate_ids)
        gold_groups += int(bool(gold))
        candidate_any_groups += int(bool(gold) and bool(gold & candidate_ids))
        candidate_full_groups += int(bool(gold) and gold <= candidate_ids)
        predicted_groups += int(bool(predicted))
        any_correct_groups += int(bool(predicted & gold))
        full_capture_groups += int(bool(gold) and gold <= predicted)
        exact_set_groups += int(predicted == gold)
        zero_gold_groups += int(not gold)
        zero_gold_abstained += int(not gold and not predicted)
        predicted_multi_groups += int(len(predicted) > 1)
        gold_multi_groups += int(len(gold) > 1)

        precision = tp / len(predicted) if predicted else (1.0 if not gold else 0.0)
        recall = tp / len(gold) if gold else (1.0 if not predicted else 0.0)
        current_f1 = _f1(precision, recall)
        macro_precision.append(precision)
        macro_recall.append(recall)
        macro_f1.append(current_f1)
        if gold:
            macro_positive_precision.append(precision)
            macro_positive_recall.append(recall)
            macro_positive_f1.append(current_f1)

    precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 1.0
    recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    total_groups = len(truth)
    interval = wilson_interval(micro_tp, micro_tp + micro_fp)
    return {
        "classification_mode": "binary",
        "thresholds": {
            "score": float(score_threshold),
            "top_candidate_margin": None,
            "provenance": "checkpoint.dev",
            "tuned_during_evaluation": False,
        },
        "norm_groups": total_groups,
        "scored_norm_groups": len(groups),
        "micro": {
            "tp": micro_tp,
            "fp": micro_fp,
            "fn": micro_fn,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "precision_wilson_95": interval,
        },
        "macro_all_norms": {
            "precision": float(np.mean(macro_precision)),
            "recall": float(np.mean(macro_recall)),
            "f1": float(np.mean(macro_f1)),
        },
        "macro_gold_positive_norms": {
            "precision": float(np.mean(macro_positive_precision)) if macro_positive_precision else 0.0,
            "recall": float(np.mean(macro_positive_recall)) if macro_positive_recall else 0.0,
            "f1": float(np.mean(macro_positive_f1)) if macro_positive_f1 else 0.0,
        },
        "norm_level": {
            "gold_positive_groups": gold_groups,
            "predicted_positive_groups": predicted_groups,
            "any_correct_groups": any_correct_groups,
            "any_correct_precision": (
                any_correct_groups / predicted_groups if predicted_groups else 1.0
            ),
            "any_correct_recall": any_correct_groups / gold_groups if gold_groups else 0.0,
            "full_gold_set_capture": (
                full_capture_groups / gold_groups if gold_groups else 0.0
            ),
            "exact_set_accuracy": exact_set_groups / total_groups if total_groups else 0.0,
        },
        "abstention": {
            "abstained_groups": total_groups - predicted_groups,
            "overall_rate": 1.0 - predicted_groups / total_groups if total_groups else 0.0,
            "zero_gold_groups": zero_gold_groups,
            "zero_gold_abstained": zero_gold_abstained,
            "zero_gold_abstention_rate": (
                zero_gold_abstained / zero_gold_groups if zero_gold_groups else None
            ),
        },
        "multiple_positive": {
            "predicted_groups": predicted_multi_groups,
            "predicted_rate": predicted_multi_groups / total_groups if total_groups else 0.0,
            "gold_groups": gold_multi_groups,
            "gold_rate": gold_multi_groups / total_groups if total_groups else 0.0,
        },
        "retrieval_ceiling": {
            "gold_exact_pairs": gold_pair_count,
            "candidate_present_gold_pairs": present_gold_pairs,
            "pair_recall": present_gold_pairs / gold_pair_count if gold_pair_count else 0.0,
            "gold_positive_groups": gold_groups,
            "any_gold_present_groups": candidate_any_groups,
            "any_gold_present_recall": (
                candidate_any_groups / gold_groups if gold_groups else 0.0
            ),
            "full_gold_set_present_groups": candidate_full_groups,
            "full_gold_set_recall": (
                candidate_full_groups / gold_groups if gold_groups else 0.0
            ),
        },
        "split_source_audit": audit,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    scores_path = Path(args.scores).resolve()
    score_meta = _score_meta(scores_path)
    contract = score_meta.get("checkpoint_contract")
    if not isinstance(contract, Mapping) or contract.get("threshold_provenance") != "checkpoint.dev":
        raise ValueError("score artifact lacks a development-frozen checkpoint gate")
    rows = list(read_jsonl(scores_path))
    truth_path = Path(args.truth).resolve()
    truth = load_truth_universe(truth_path)
    classification_mode = str(contract.get("classification_mode") or "three_way")
    if classification_mode == "binary":
        metrics = evaluate_binary_rows(
            rows,
            truth,
            score_threshold=float(contract["score_threshold"]),
        )
    else:
        metrics = evaluate_rows(
            rows,
            truth,
            score_threshold=float(contract["score_threshold"]),
            margin_threshold=float(contract["top_margin_threshold"]),
        )
    report = {
        "schema_version": EVAL_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scores": str(scores_path),
        "scores_sha256": sha256_file(scores_path),
        "scores_meta_sha256": sha256_file(
            scores_path.with_suffix(scores_path.suffix + ".meta.json")
        ),
        "truth": str(truth_path),
        "truth_sha256": sha256_file(truth_path),
        "checkpoint_contract": contract,
        "metrics": metrics,
        "test_threshold_tuning_performed": False,
    }
    _json_exclusive(Path(args.output).resolve(), report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    lock = sub.add_parser("lock-base", help="content-lock a local base model")
    lock.add_argument("--model", required=True)
    lock.add_argument("--output", required=True)

    scorer = sub.add_parser("score", help="score one immutable deterministic shard")
    scorer.add_argument("--input-pairs", required=True)
    scorer.add_argument("--output", required=True)
    scorer.add_argument("--model", required=True)
    scorer.add_argument("--base-manifest", required=True)
    scorer.add_argument("--base-manifest-sha256", required=True)
    scorer.add_argument("--checkpoint", required=True)
    scorer.add_argument("--training-report", required=True)
    scorer.add_argument("--training-report-sha256", required=True)
    scorer.add_argument("--batch-size", type=int, default=8)
    scorer.add_argument("--max-length", type=int, default=MAX_SEQUENCE_LENGTH)
    scorer.add_argument("--device", type=int, default=0)
    scorer.add_argument("--shard-id", type=int, default=0)
    scorer.add_argument("--num-shards", type=int, default=1)
    scorer.add_argument("--attention", choices=("auto", "eager", "sdpa"), default="eager")

    merger = sub.add_parser("merge", help="validate and merge a complete shard set")
    merger.add_argument("--inputs", nargs="+", required=True)
    merger.add_argument("--output", required=True)

    evaluator = sub.add_parser("evaluate", help="evaluate with checkpoint-dev thresholds")
    evaluator.add_argument("--scores", required=True)
    evaluator.add_argument("--truth", required=True)
    evaluator.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "score":
        if args.batch_size <= 0:
            parser.error("--batch-size must be positive")
        if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
            parser.error("invalid --shard-id/--num-shards")
        for name in ("base_manifest_sha256", "training_report_sha256"):
            value = getattr(args, name)
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
                parser.error(f"--{name.replace('_', '-')} must be a SHA256 hex digest")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "lock-base":
        result = build_base_manifest(Path(args.model), Path(args.output))
    elif args.command == "score":
        result = score(args)
    elif args.command == "merge":
        result = merge_score_shards(
            [Path(value) for value in args.inputs], Path(args.output)
        )
    else:
        result = evaluate(args)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
