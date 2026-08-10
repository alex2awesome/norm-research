from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.tools.silver_match_v3 import finalize_humor_c2_full285_deployment as finalize
from scripts.tools.silver_match_v3 import build_humor_c2_early_manual_audit as early_audit
from scripts.tools.silver_match_v3 import merge_package_humor_full285_ce as merge
from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.run_humor_c2_production_paired_vllm import (
    META_SCHEMA,
    PREDICTION_SCHEMA,
)
from scripts.tools.silver_match_v3.run_nemotron_ce import SCORE_META_SCHEMA, SCORE_SCHEMA


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _score_meta(path: Path, rows: int, norms: int) -> None:
    payload = {
        "schema_version": SCORE_META_SCHEMA,
        "output_sha256": sha256_file(path),
        "row_count": rows,
        "norm_group_count": norms,
        "classification_mode": "binary",
        "score_labels": ["REJECT", "EXACT"],
        "checkpoint_contract": {
            "checkpoint_metadata_sha256": merge.EXPECTED_CHECKPOINT_SHA,
            "threshold_provenance": "checkpoint.dev",
            "score_threshold": merge.EXPECTED_POSITIVE_THRESHOLD,
        },
    }
    path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(payload))


def _score(uid: str, metric: str, probability: float) -> dict:
    return {
        "schema_version": SCORE_SCHEMA,
        "norm_uid": uid,
        "metric_id": metric,
        "source_group": f"group {uid}",
        "split": "production",
        "predicted_relation": "EXACT" if probability >= 0.5 else "REJECT",
        "probabilities": {"EXACT": probability, "REJECT": 1 - probability},
    }


def test_mini_full_surface_and_frozen_finalization(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(merge, "EXPECTED_UIDS", 2)
    monkeypatch.setattr(merge, "EXPECTED_K200", 2)
    monkeypatch.setattr(merge, "EXPECTED_K85", 1)
    monkeypatch.setattr(merge, "EXPECTED_BANK", 3)
    monkeypatch.setattr(merge, "EXPECTED_K200_ROWS", 4)
    monkeypatch.setattr(merge, "EXPECTED_K85_ROWS", 2)
    monkeypatch.setattr(merge, "EXPECTED_FULL_ROWS", 6)
    k200, k85, pairs, bank = (tmp_path / name for name in ("k200.jsonl", "k85.jsonl", "pairs.jsonl", "bank.json"))
    _jsonl(k200, [_score("u1", "a", .999), _score("u1", "b", .1),
                   _score("u2", "b", .999), _score("u2", "c", .1)])
    _jsonl(k85, [_score("u1", "c", .2), _score("u2", "a", .2)])
    _score_meta(k200, 4, 2); _score_meta(k85, 2, 2)
    _jsonl(pairs, [
        {"norm_uid": "u1", "metric_id": "c", "source_group": "group\u001fu1", "split": "production",
         "query": "Task: humor. Human evaluative statement: tighter. Evidence passage: Please make this tighter."},
        {"norm_uid": "u2", "metric_id": "a", "source_group": "group u2", "split": "production",
         "query": "Task: humor. Human evaluative statement: unclear"},
    ])
    bank.write_text(json.dumps({
        "source_sha256": merge.EXPECTED_BANK_SOURCE_SHA,
        "metrics": [{"metric_id": value, "name": f"Metric {value}", "description": f"Definition {value}"}
                    for value in ("a", "b", "c")],
    }))
    root = tmp_path / "merged"
    merge.build(argparse.Namespace(k200_scores=str(k200), k85_scores=str(k85),
                                   k85_pairs=str(pairs), bank=str(bank), output_root=str(root), ce_top=16))
    assert sum(1 for _ in open(root / "scores.full285.jsonl")) == 6
    packages = [json.loads(line) for line in open(root / "candidates.top16-plus-positives.jsonl")]
    assert len(packages) == 2
    assert packages[0]["source_group"] == "group u1"
    assert all(set(row["candidate_metric_ids"]) == {"a", "b", "c"} for row in packages)
    assert sum(1 for _ in open(root / "paired_order.prompts.jsonl")) == 4

    monkeypatch.setattr(finalize, "EXPECTED_UIDS", 2)
    monkeypatch.setattr(finalize, "MINIMUM_SLATE_DEPTH", 3)
    typed_root = tmp_path / "typed"
    typed_root.mkdir()
    original = [
        {"schema_version": PREDICTION_SCHEMA, "norm_uid": "u1", "split": "production", "order_mode": "original",
         "candidate_metric_ids": packages[0]["candidate_metric_ids"], "decision": "MATCH", "metric_id": "a", "confidence": "high", "reason": "exact", "parse_error": None},
        {"schema_version": PREDICTION_SCHEMA, "norm_uid": "u2", "split": "production", "order_mode": "original",
         "candidate_metric_ids": packages[1]["candidate_metric_ids"], "decision": "NO_CANDIDATE_FITS", "metric_id": None, "confidence": "high", "reason": "absent", "parse_error": None},
    ]
    reordered = [{**row, "order_mode": "reordered"} for row in original]
    _jsonl(typed_root / "typed.original.jsonl", original)
    _jsonl(typed_root / "typed.reordered.jsonl", reordered)
    meta = {
        "schema_version": META_SCHEMA,
        "status": "COMPLETE_C2_PRODUCTION_PAIRED_INFERENCE",
        "deployment_claim": finalize.DEPLOYMENT_CLAIM,
        "test_or_blind_rows_read": 0,
        "shard_id": 0,
        "num_shards": 1,
        "outputs": {
            order: {"sha256": sha256_file(typed_root / f"typed.{order}.jsonl")}
            for order in ("original", "reordered")
        },
    }
    (typed_root / "INFERENCE_META.json").write_text(json.dumps(meta))
    output, report = tmp_path / "normalized.jsonl", tmp_path / "finalize.json"
    finalize.build(argparse.Namespace(candidate_package=str(root / "candidates.top16-plus-positives.jsonl"),
                                       typed_root=[str(typed_root)], output=str(output), report_output=str(report)))
    rows = [json.loads(line) for line in open(output)]
    assert rows[0]["decision"] == "MATCH"
    assert rows[0]["verification_status"] == "DEV_FROZEN_DEPLOYMENT_BLIND_P855"
    assert rows[1]["decision"] == "NO_CANDIDATE_FITS"
    assert json.loads(report.read_text())["blind_gate"]["promotion_passed"] is False

    monkeypatch.setattr(early_audit, "EXPECTED_BANK", 3)
    audit_root = tmp_path / "early-audit"
    audit_report = early_audit.build(argparse.Namespace(
        candidate_package=str(root / "candidates.top16-plus-positives.jsonl"),
        prompts=str(root / "paired_order.prompts.jsonl"), bank=str(bank),
        typed_root=str(typed_root), output_root=str(audit_root),
    ))
    audit_rows = [json.loads(line) for line in open(audit_root / "EARLY_MANUAL_AUDIT_PACKET.jsonl")]
    assert audit_report["test_or_blind_or_truth_rows_read"] == 0
    assert audit_report["selected_total"] == 2
    assert {row["audit_stratum"] for row in audit_rows} == {
        "hybrid_accepted_match", "stable_typed_abstention",
    }
    assert all("ce_full285_summary" in row and "gold_relation" not in row for row in audit_rows)
