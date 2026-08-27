import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from scripts.tools.silver_match_v3.aggregate_nemotron_ce_seed_consensus import (
    CONSENSUS_REPORT_SCHEMA,
    CONSENSUS_SCHEMA,
    SEED_MANIFEST_SCHEMA,
    NormUniverse,
    SeedArtifact,
    aggregate_seed_consensus,
    load_seed_manifest,
)
from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file
from scripts.tools.silver_match_v3.run_nemotron_ce import (
    CHECKPOINT_SCHEMA,
    HIDDEN_SIZE,
    LORA_TARGETS,
    REPORT_SCHEMA,
    SCORE_META_SCHEMA,
    SCORE_SCHEMA,
    verify_checkpoint_contract,
)
from scripts.tools.silver_match_v3.train_nemotron_cross_encoder import CLASS_NAMES


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _score(uid, metric, exact, family, reject, *, corpus="standup", split="test"):
    probabilities = {"EXACT": exact, "FAMILY": family, "REJECT": reject}
    predicted = max(CLASS_NAMES, key=lambda label: probabilities[label])
    return {
        "schema_version": SCORE_SCHEMA,
        "norm_uid": uid,
        "metric_id": metric,
        "source_group": f"humor\x1f{corpus}\x1fsource\x1f{uid}",
        "split": split,
        "predicted_relation": predicted,
        "probabilities": probabilities,
    }


def _make_seed(tmp_path: Path, name: str, rows, *, score_gate=0.70, margin_gate=0.10):
    root = tmp_path / name
    checkpoint = root / "checkpoint"
    adapter = checkpoint / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/frozen/nemotron"}), encoding="utf-8"
    )
    (adapter / "adapter_model.safetensors").write_bytes(f"adapter-{name}".encode())
    save_file(
        {
            "weight": torch.zeros((len(CLASS_NAMES), HIDDEN_SIZE)),
            "bias": torch.zeros((len(CLASS_NAMES),)),
        },
        checkpoint / "head.safetensors",
    )
    dev = {
        "score_threshold": score_gate,
        "top_margin_threshold": margin_gate,
    }
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA,
        "labels": list(CLASS_NAMES),
        "hidden_to_classes": [HIDDEN_SIZE, len(CLASS_NAMES)],
        "lora_targets": list(LORA_TARGETS),
        "dev": dev,
    }
    (checkpoint / "checkpoint.json").write_text(json.dumps(metadata), encoding="utf-8")
    artifact_hashes = {
        str(path.relative_to(checkpoint)): sha256_file(path)
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file()
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "COMPLETE",
        "model": "/frozen/nemotron",
        "labels": list(CLASS_NAMES),
        "hidden_to_classes": [HIDDEN_SIZE, len(CLASS_NAMES)],
        "max_sequence_length": 1024,
        "selected_checkpoint": {
            "artifact_sha256": artifact_hashes,
            "checkpoint_metadata_sha256": sha256_file(checkpoint / "checkpoint.json"),
            "dev": dev,
        },
    }
    training_report = root / "training_report.json"
    training_report.write_text(json.dumps(report), encoding="utf-8")
    training_sha = sha256_file(training_report)
    contract = verify_checkpoint_contract(checkpoint, training_report, training_sha)

    scores = root / "scores.merged.jsonl"
    _write_jsonl(scores, rows)
    meta = {
        "schema_version": SCORE_META_SCHEMA,
        "output": str(scores),
        "output_sha256": sha256_file(scores),
        "row_count": len(rows),
        "norm_group_count": len({row["norm_uid"] for row in rows}),
        "num_shards": 1,
        "combined_from_num_shards": 2,
        "labels": list(CLASS_NAMES),
        "checkpoint_contract": contract,
    }
    scores_meta = scores.with_suffix(scores.suffix + ".meta.json")
    scores_meta.write_text(json.dumps(meta), encoding="utf-8")
    return SeedArtifact(
        seed_id=name,
        scores=scores,
        scores_sha256=sha256_file(scores),
        scores_meta=scores_meta,
        scores_meta_sha256=sha256_file(scores_meta),
        checkpoint=checkpoint,
        training_report=training_report,
        training_report_sha256=training_sha,
    )


def _run(tmp_path, rows_a, rows_b, *, universe=None):
    seed_a = _make_seed(tmp_path, "seed-a", rows_a)
    seed_b = _make_seed(tmp_path, "seed-b", rows_b)
    output = tmp_path / "consensus.jsonl"
    report_output = tmp_path / "consensus.report.json"
    report = aggregate_seed_consensus(
        seed_a,
        seed_b,
        output=output,
        report_output=report_output,
        norm_universe=universe,
    )
    return list(read_jsonl(output)), report, seed_a, seed_b


def test_same_leaf_two_seed_consensus_is_the_only_automatic_match(tmp_path):
    rows_a = [
        _score("u1", "m1", 0.91, 0.04, 0.05),
        _score("u1", "m2", 0.09, 0.06, 0.85),
    ]
    rows_b = [
        _score("u1", "m2", 0.08, 0.07, 0.85),  # candidate order may differ
        _score("u1", "m1", 0.88, 0.06, 0.06),
    ]
    output, report, _, _ = _run(tmp_path, rows_a, rows_b)
    assert len(output) == 1
    row = output[0]
    assert row["schema_version"] == CONSENSUS_SCHEMA
    assert row["decision"] == "MATCH"
    assert row["routing_category"] == "MATCH"
    assert row["metric_id"] == "m1"
    assert row["human_abstention_subtype_assigned"] is False
    candidates = {candidate["metric_id"]: candidate for candidate in row["candidates"]}
    assert candidates["m1"]["seed-a"]["probabilities"]["EXACT"] == 0.91
    assert candidates["m1"]["seed-b"]["probabilities"]["EXACT"] == 0.88
    assert report["schema_version"] == CONSENSUS_REPORT_SCHEMA
    assert report["metrics"]["overall"]["automatic_match_rate"] == 1.0
    assert report["metrics"]["by_split"]["test"]["automatic_match_count"] == 1
    assert report["metrics"]["by_corpus"]["standup"]["automatic_match_count"] == 1
    assert report["validation"]["test_threshold_tuning_performed"] is False


def test_different_retained_leaf_routes_seed_disagreement(tmp_path):
    rows_a = [
        _score("u1", "m1", 0.90, 0.05, 0.05),
        _score("u1", "m2", 0.10, 0.05, 0.85),
    ]
    rows_b = [
        _score("u1", "m1", 0.10, 0.05, 0.85),
        _score("u1", "m2", 0.90, 0.05, 0.05),
    ]
    output, report, _, _ = _run(tmp_path, rows_a, rows_b)
    row = output[0]
    assert row["decision"] == "ROUTE_TO_ADJUDICATION"
    assert row["routing_category"] == "SEED_DISAGREEMENT"
    assert row["metric_id"] is None
    assert report["metrics"]["overall"]["seed_disagreement_routing_rate"] == 1.0
    assert report["metrics"]["overall"]["any_seed_disagreement_rate"] == 1.0


def test_same_leaf_below_one_or_both_frozen_gates_is_not_matched(tmp_path):
    rows_a = [
        _score("u1", "m1", 0.69, 0.20, 0.11),
        _score("u1", "m2", 0.50, 0.10, 0.40),
    ]
    rows_b = [
        _score("u1", "m1", 0.90, 0.05, 0.05),
        _score("u1", "m2", 0.10, 0.05, 0.85),
    ]
    output, report, _, _ = _run(tmp_path, rows_a, rows_b)
    assert output[0]["routing_category"] == "BELOW_GATE"
    assert output[0]["automatic_match"] is False
    assert report["metrics"]["overall"]["provisional_abstention_rate"] == 1.0
    assert report["metrics"]["overall"]["gate_pass_disagreement_rate"] == 1.0


def test_family_reject_and_no_candidate_routes_are_distinct(tmp_path):
    rows_a = [
        _score("family", "m1", 0.20, 0.70, 0.10, corpus="c1"),
        _score("reject", "m1", 0.10, 0.10, 0.80, corpus="c2"),
    ]
    rows_b = [
        _score("family", "m1", 0.15, 0.75, 0.10, corpus="c1"),
        _score("reject", "m1", 0.05, 0.05, 0.90, corpus="c2"),
    ]
    universe_path = tmp_path / "universe.jsonl"
    _write_jsonl(
        universe_path,
        [
            {
                "norm_uid": uid,
                "source_group": f"humor\x1f{corpus}\x1fsource\x1f{uid}",
                "split": "test",
                "task": "humor",
                "corpus": corpus,
            }
            for uid, corpus in (("family", "c1"), ("reject", "c2"), ("empty", "c3"))
        ],
    )
    universe = NormUniverse(universe_path, sha256_file(universe_path))
    output, report, _, _ = _run(tmp_path, rows_a, rows_b, universe=universe)
    by_uid = {row["norm_uid"]: row for row in output}
    assert by_uid["family"]["routing_category"] == "FAMILY_SIGNAL"
    assert by_uid["reject"]["routing_category"] == "CE_REJECT_BOTH"
    assert by_uid["empty"]["routing_category"] == "NO_CANDIDATES"
    assert by_uid["empty"]["candidates"] == []
    counts = report["metrics"]["overall"]["routing_category_counts"]
    assert counts["FAMILY_SIGNAL"] == counts["CE_REJECT_BOTH"] == 1
    assert counts["NO_CANDIDATES"] == 1
    assert report["metrics"]["overall"]["ce_reject_both_noise_routing_rate"] == pytest.approx(1 / 3)
    assert report["zero_candidate_norms_observable"] is True


def test_candidate_universe_drift_fails_closed(tmp_path):
    rows_a = [_score("u1", "m1", 0.9, 0.05, 0.05)]
    rows_b = [_score("u1", "m2", 0.9, 0.05, 0.05)]
    seed_a = _make_seed(tmp_path, "seed-a", rows_a)
    seed_b = _make_seed(tmp_path, "seed-b", rows_b)
    with pytest.raises(ValueError, match="candidate universe drift"):
        aggregate_seed_consensus(
            seed_a,
            seed_b,
            output=tmp_path / "out.jsonl",
            report_output=tmp_path / "report.json",
        )


def test_score_and_checkpoint_hash_drift_fail_closed(tmp_path):
    rows = [_score("u1", "m1", 0.9, 0.05, 0.05)]
    seed_a = _make_seed(tmp_path, "seed-a", rows)
    seed_b = _make_seed(tmp_path, "seed-b", rows)
    seed_a.scores.write_text(seed_a.scores.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="score SHA256 mismatch"):
        aggregate_seed_consensus(
            seed_a,
            seed_b,
            output=tmp_path / "out-a.jsonl",
            report_output=tmp_path / "report-a.json",
        )

    seed_a = _make_seed(tmp_path, "seed-a-2", rows)
    seed_b = _make_seed(tmp_path, "seed-b-2", rows)
    (seed_b.checkpoint / "adapter" / "adapter_model.safetensors").write_bytes(b"drift")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        aggregate_seed_consensus(
            seed_a,
            seed_b,
            output=tmp_path / "out-b.jsonl",
            report_output=tmp_path / "report-b.json",
        )


def test_frozen_seed_manifest_is_hash_bound_and_relocatable(tmp_path):
    rows = [_score("u1", "m1", 0.9, 0.05, 0.05)]
    seeds = [_make_seed(tmp_path, name, rows) for name in ("seed-a", "seed-b")]
    manifest = tmp_path / "seed-manifest.json"
    payload = {
        "schema_version": SEED_MANIFEST_SCHEMA,
        "seeds": [
            {
                "seed_id": seed.seed_id,
                "scores": str(seed.scores.relative_to(tmp_path)),
                "scores_sha256": seed.scores_sha256,
                "scores_meta": str(seed.scores_meta.relative_to(tmp_path)),
                "scores_meta_sha256": seed.scores_meta_sha256,
                "checkpoint": str(seed.checkpoint.relative_to(tmp_path)),
                "training_report": str(seed.training_report.relative_to(tmp_path)),
                "training_report_sha256": seed.training_report_sha256,
            }
            for seed in seeds
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    loaded, universe, provenance = load_seed_manifest(manifest, sha256_file(manifest))
    assert [seed.seed_id for seed in loaded] == ["seed-a", "seed-b"]
    assert loaded[0].scores == seeds[0].scores
    assert universe is None
    assert provenance["sha256"] == sha256_file(manifest)

    frozen_sha = sha256_file(manifest)
    manifest.write_text(manifest.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="seed manifest SHA256 mismatch"):
        load_seed_manifest(manifest, frozen_sha)
