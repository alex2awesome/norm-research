import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file
from scripts.tools.silver_match_v3.run_nemotron_ce import (
    BASE_MANIFEST_SCHEMA,
    CHECKPOINT_SCHEMA,
    CLASS_NAMES,
    HIDDEN_SIZE,
    LORA_TARGETS,
    REPORT_SCHEMA,
    SCORE_META_SCHEMA,
    SCORE_SCHEMA,
    TruthNorm,
    _iter_score_pairs,
    build_base_manifest,
    evaluate_binary_rows,
    evaluate_rows,
    merge_score_shards,
    pair_shard,
    score_pair_from_row,
    validate_loaded_head,
    verify_base_manifest,
    verify_checkpoint_contract,
)


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _score(uid, metric, group, split, exact, family, reject, gold=None):
    row = {
        "schema_version": SCORE_SCHEMA,
        "norm_uid": uid,
        "metric_id": metric,
        "source_group": group,
        "split": split,
        "probabilities": {"EXACT": exact, "FAMILY": family, "REJECT": reject},
        "predicted_relation": CLASS_NAMES[max(range(3), key=(exact, family, reject).__getitem__)],
    }
    if gold:
        row["gold_relation"] = gold
    return row


def test_evaluation_counts_missing_exact_candidate_as_retrieval_miss():
    truth = {
        "u1": TruthNorm("u1", "g1", "test", {"m1"}),
        "u2": TruthNorm("u2", "g2", "test", {"m2"}),
        "u3": TruthNorm("u3", "g3", "test", set()),
    }
    rows = [
        _score("u1", "m1", "g1", "test", 0.90, 0.05, 0.05, "EXACT"),
        _score("u1", "mx", "g1", "test", 0.10, 0.10, 0.80, "REJECT"),
        # u2's gold m2 is absent.  A confident wrong retention is an FP and the
        # missing positive remains in the end-to-end recall denominator.
        _score("u2", "m3", "g2", "test", 0.80, 0.10, 0.10, "REJECT"),
        _score("u3", "m9", "g3", "test", 0.10, 0.10, 0.80, "REJECT"),
    ]
    report = evaluate_rows(rows, truth, score_threshold=0.70, margin_threshold=0.10)
    assert report["gold_exact_groups"] == 2
    assert report["candidate_present_exact_groups"] == 1
    assert report["retrieval_recall"] == 0.5
    assert report["retained_count"] == 2
    assert report["retained_exact_count"] == 1
    assert report["retained_precision"] == 0.5
    assert report["recall_candidate_present"] == 1.0
    assert report["recall_end_to_end"] == 0.5
    assert report["abstention_rate"] == pytest.approx(1 / 3)
    confusion = report["relation_confusion"]
    assert confusion["support"] == 4
    assert confusion["matrix"]["EXACT"]["EXACT"] == 1
    assert confusion["matrix"]["REJECT"]["EXACT"] == 1
    assert report["thresholds"]["tuned_during_evaluation"] is False


def test_binary_evaluation_is_set_valued_and_reports_abstention_and_ceiling():
    truth = {
        "u1": TruthNorm("u1", "g1", "test", {"m1", "m2"}),
        "u2": TruthNorm("u2", "g2", "test", {"m3"}),
        "u3": TruthNorm("u3", "g3", "test", set()),
    }

    def binary(uid, metric, group, exact):
        return {
            "schema_version": SCORE_SCHEMA,
            "norm_uid": uid,
            "metric_id": metric,
            "source_group": group,
            "split": "test",
            "probabilities": {"REJECT": 1.0 - exact, "EXACT": exact},
        }

    rows = [
        binary("u1", "m1", "g1", 0.9),
        binary("u1", "m2", "g1", 0.8),
        binary("u1", "mx", "g1", 0.1),
        # u2's gold metric is absent: retrieval ceiling and recall must reflect it.
        binary("u2", "wrong", "g2", 0.7),
        binary("u3", "other", "g3", 0.2),
    ]
    report = evaluate_binary_rows(rows, truth, score_threshold=0.6)
    assert report["micro"] == {
        "tp": 2,
        "fp": 1,
        "fn": 1,
        "precision": pytest.approx(2 / 3),
        "recall": pytest.approx(2 / 3),
        "f1": pytest.approx(2 / 3),
        "precision_wilson_95": report["micro"]["precision_wilson_95"],
    }
    assert report["norm_level"]["full_gold_set_capture"] == pytest.approx(0.5)
    assert report["abstention"]["zero_gold_abstention_rate"] == 1.0
    assert report["multiple_positive"]["predicted_rate"] == pytest.approx(1 / 3)
    assert report["retrieval_ceiling"]["pair_recall"] == pytest.approx(2 / 3)
    assert report["retrieval_ceiling"]["full_gold_set_recall"] == pytest.approx(0.5)


def test_binary_checkpoint_contract_accepts_dynamic_two_class_head(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    checkpoint = tmp_path / "checkpoint"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(model.resolve())}), encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    labels = ["NON_EXACT", "EXACT"]
    save_file(
        {
            "weight": torch.zeros((2, HIDDEN_SIZE)),
            "bias": torch.zeros((2,)),
        },
        checkpoint / "head.safetensors",
    )
    dev = {"score_threshold": 0.42, "top_margin_threshold": 0.0}
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA,
        "classification_mode": "binary",
        "labels": labels,
        "hidden_to_classes": [HIDDEN_SIZE, 2],
        "lora_targets": list(LORA_TARGETS),
        "dev": dev,
    }
    (checkpoint / "checkpoint.json").write_text(json.dumps(metadata), encoding="utf-8")
    hashes = {
        str(path.relative_to(checkpoint)): sha256_file(path)
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file()
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "COMPLETE",
        "model": str(model.resolve()),
        "classification_mode": "binary",
        "labels": labels,
        "hidden_to_classes": [HIDDEN_SIZE, 2],
        "max_sequence_length": 1024,
        "selected_checkpoint": {
            "artifact_sha256": hashes,
            "checkpoint_metadata_sha256": sha256_file(checkpoint / "checkpoint.json"),
            "dev": dev,
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    contract = verify_checkpoint_contract(
        checkpoint, report_path, sha256_file(report_path), model=model
    )
    assert contract["classification_mode"] == "binary"
    assert contract["labels"] == labels
    assert validate_loaded_head(checkpoint, labels) == {
        "weight": [2, HIDDEN_SIZE],
        "bias": [2],
    }


def _checkpoint_fixture(tmp_path: Path, model: Path):
    checkpoint = tmp_path / "checkpoint"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(model.resolve())}), encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    save_file(
        {
            "weight": torch.zeros((len(CLASS_NAMES), HIDDEN_SIZE)),
            "bias": torch.zeros((len(CLASS_NAMES),)),
        },
        checkpoint / "head.safetensors",
    )
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA,
        "labels": list(CLASS_NAMES),
        "hidden_to_classes": [HIDDEN_SIZE, len(CLASS_NAMES)],
        "lora_targets": list(LORA_TARGETS),
        "dev": {"score_threshold": 0.72, "top_margin_threshold": 0.11},
    }
    (checkpoint / "checkpoint.json").write_text(json.dumps(metadata), encoding="utf-8")
    hashes = {
        str(path.relative_to(checkpoint)): sha256_file(path)
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file()
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "COMPLETE",
        "model": str(model.resolve()),
        "labels": list(CLASS_NAMES),
        "hidden_to_classes": [HIDDEN_SIZE, len(CLASS_NAMES)],
        "max_sequence_length": 1024,
        "selected_checkpoint": {
            "artifact_sha256": hashes,
            "checkpoint_metadata_sha256": sha256_file(checkpoint / "checkpoint.json"),
            "dev": metadata["dev"],
        },
    }
    report_path = tmp_path / "training_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return checkpoint, report_path


def test_base_and_checkpoint_reload_contracts_are_content_addressed(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    manifest_path = tmp_path / "base-manifest.json"
    manifest = build_base_manifest(model, manifest_path)
    assert manifest["schema_version"] == BASE_MANIFEST_SCHEMA
    verified_base = verify_base_manifest(
        model, manifest_path, sha256_file(manifest_path)
    )
    assert verified_base["file_count"] == 2

    checkpoint, report_path = _checkpoint_fixture(tmp_path, model)
    contract = verify_checkpoint_contract(
        checkpoint, report_path, sha256_file(report_path), model=model
    )
    assert contract["labels"] == list(CLASS_NAMES)
    assert contract["head_sha256"] == sha256_file(checkpoint / "head.safetensors")
    assert contract["threshold_provenance"] == "checkpoint.dev"
    assert validate_loaded_head(checkpoint) == {
        "weight": [len(CLASS_NAMES), HIDDEN_SIZE],
        "bias": [len(CLASS_NAMES)],
    }

    (checkpoint / "head.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_checkpoint_contract(checkpoint, report_path, sha256_file(report_path))
    (model / "config.json").write_text('{"drift": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="base model size mismatch|base model hash mismatch"):
        verify_base_manifest(model, manifest_path, sha256_file(manifest_path))


def _score_shards(tmp_path: Path, source: Path, rows, num_shards=3):
    paths = []
    invariant = {
        "schema_version": SCORE_META_SCHEMA,
        "created_at": "2026-07-13T00:00:00+00:00",
        "input_pairs": str(source),
        "input_pairs_sha256": sha256_file(source),
        "base_contract": {"tree_sha256": "a" * 64},
        "checkpoint_contract": {
            "checkpoint_metadata_sha256": "b" * 64,
            "threshold_provenance": "checkpoint.dev",
            "score_threshold": 0.7,
            "top_margin_threshold": 0.1,
        },
        "labels": list(CLASS_NAMES),
        "bidirectional_concatenation": True,
        "pooling": "native_attention_mask_mean",
        "max_length": 1024,
        "cuda_bf16": True,
        "attention": "eager",
        "num_shards": num_shards,
    }
    for shard_id in range(num_shards):
        path = tmp_path / f"scores-{shard_id}.jsonl"
        selected = [
            row for row in rows if pair_shard(row["norm_uid"], num_shards) == shard_id
        ]
        _write_jsonl(path, selected)
        meta = {
            **invariant,
            "output": str(path),
            "output_sha256": sha256_file(path),
            "row_count": len(selected),
            "norm_group_count": len({row["norm_uid"] for row in selected}),
            "shard_id": shard_id,
        }
        path.with_suffix(".jsonl.meta.json").write_text(json.dumps(meta), encoding="utf-8")
        paths.append(path)
    return paths


def test_deterministic_pair_shards_merge_in_original_source_order(tmp_path):
    source_rows = []
    score_rows = []
    for index in range(9):
        uid = f"{index:064x}"
        for metric in ("m0", "m1"):
            source_rows.append(
                {
                    "norm_uid": uid,
                    "metric_id": metric,
                    "source_group": f"g{index}",
                    "split": "test",
                }
            )
            score_rows.append(
                _score(uid, metric, f"g{index}", "test", 0.1, 0.2, 0.7)
            )
    source = tmp_path / "pairs.jsonl"
    _write_jsonl(source, source_rows)
    paths = _score_shards(tmp_path, source, score_rows)
    output = tmp_path / "combined.jsonl"
    meta = merge_score_shards(paths, output)
    assert [
        (row["norm_uid"], row["metric_id"]) for row in read_jsonl(output)
    ] == [(row["norm_uid"], row["metric_id"]) for row in source_rows]
    assert meta["row_count"] == len(source_rows)
    assert meta["combined_from_num_shards"] == 3
    assert meta["output_sha256"] == sha256_file(output)


def test_score_pair_preserves_identity_split_and_gold_relation():
    row = {
        "norm_uid": "a" * 64,
        "candidate_metric_id": "a7",
        "source_group": "source:g1",
        "split": "blind",
        "query": "The joke punches down.",
        "evidence": "This is needlessly cruel.",
        "metric_card": "Avoid cruelty. Definition: Do not punch down.",
        "relation": "EXACT",
    }
    parsed = score_pair_from_row(row)
    assert (parsed.norm_uid, parsed.metric_id, parsed.source_group, parsed.split) == (
        "a" * 64,
        "a7",
        "source:g1",
        "blind",
    )
    assert parsed.gold_relation == "EXACT"
    assert pair_shard(parsed.norm_uid, 11) == pair_shard(parsed.norm_uid, 11)


def test_pair_stream_allows_shuffled_norm_groups_without_global_pair_memory(tmp_path):
    path = tmp_path / "shuffled.jsonl"
    rows = []
    for uid, metric in (("a" * 64, "m1"), ("b" * 64, "m2"), ("a" * 64, "m3")):
        rows.append(
            {
                "norm_uid": uid,
                "metric_id": metric,
                "source_group": f"group-{uid[0]}",
                "split": "test",
                "query": f"query {uid}",
                "metric_card": f"card {metric}",
            }
        )
    _write_jsonl(path, rows)
    observed = list(_iter_score_pairs(path, 0, 1))
    assert [(row.norm_uid, row.metric_id) for row in observed] == [
        (row["norm_uid"], row["metric_id"]) for row in rows
    ]
