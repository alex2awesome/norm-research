from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.tools.silver_match_v3.common import sha256_file, write_jsonl
from scripts.tools.silver_match_v3.gate_humor_k200_consensus_dev import (
    FAIL_STATUS,
    PASS_STATUS,
    run_gate,
)
from scripts.tools.silver_match_v3.freeze_humor_consensus_completion_queue import (
    _validate_progressive_policy_gate,
)


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, *, misses: int) -> argparse.Namespace:
    bank_hash = "bank-v1"
    bank = tmp_path / "bank.json"
    _json(
        bank,
        {
            "task": "humor",
            "source_sha256": bank_hash,
            "metrics": [
                {"metric_id": "m0", "name": "zero"},
                {"metric_id": "m1", "name": "one"},
                {"metric_id": "m2", "name": "two"},
            ],
        },
    )
    truth = tmp_path / "truth.all.jsonl"
    rows = []
    for index in range(120):
        rows.append(
            {
                "schema_version": "truth",
                "task": "humor",
                "corpus": "humor_multi",
                "norm_uid": f"u{index:03d}",
                "source_group": f"g{index:03d}",
                "split_group": f"g{index:03d}",
                "split": "dev",
                "collection_role": "dev",
                "decision": "MATCH",
                "metric_id": "m2" if index < misses else "m0",
                "bank_source_sha256": bank_hash,
                "dev_selection_eligible": True,
                "training_eligible": False,
                "blind_evaluation_only": False,
            }
        )
    write_jsonl(truth, rows)
    manifest = tmp_path / "MANIFEST.json"
    _json(
        manifest,
        {
            "schema_version": "silver-match-v3-consensus-training-truth-manifest-v1",
            "status": "COMPLETE_EXACT_CONSENSUS_WITH_FROZEN_SPLITS",
            "task": "humor",
            "source_group_cross_split_count": 0,
            "blind_rows_training_eligible": 0,
            "outputs": {
                "all": {
                    "path": str(truth),
                    "sha256": sha256_file(truth),
                    "count": len(rows),
                },
                "dev": {
                    "path": str(truth),
                    "sha256": sha256_file(truth),
                    "count": len(rows),
                },
            },
        },
    )
    k200 = tmp_path / "humor.k200.jsonl"
    fullbank = tmp_path / "humor.full285.jsonl"
    k200_rows = []
    full_rows = []
    for index in range(120):
        common = {
            "task": "humor",
            "corpus": "humor_multi",
            "norm_uid": f"u{index:03d}",
        }
        k200_rows.append(
            {
                **common,
                "candidates": [
                    {"metric_id": "m0", "rank": 1},
                    {"metric_id": "m1", "rank": 2},
                ],
            }
        )
        full_rows.append(
            {
                **common,
                "candidates": [
                    {"metric_id": "m0", "rank": 1},
                    {"metric_id": "m1", "rank": 2},
                    {"metric_id": "m2", "rank": 3},
                ],
            }
        )
    write_jsonl(k200, k200_rows)
    write_jsonl(fullbank, full_rows)
    old_a = tmp_path / "old-a.jsonl"
    old_b = tmp_path / "old-b.jsonl"
    write_jsonl(old_a, [{"norm_uid": "u000"}])
    write_jsonl(old_b, [{"norm_uid": "u000"}])
    prior = tmp_path / "prior.json"
    _json(
        prior,
        {
            "schema_version": "silver-match-v3-candidate-capture-sequence-v1",
            "selection_split": "dev",
            "test_labels_used_for_selection": False,
            "candidate_inputs": {
                "old_a": {"path": str(old_a), "sha256": sha256_file(old_a)},
                "old_b": {"path": str(old_b), "sha256": sha256_file(old_b)},
            },
            "available_lanes": ["old_a:rank", "old_b:rank"],
            "selected_sequence": ["old_a:rank", "old_b:rank"],
        },
    )
    return argparse.Namespace(
        bank=str(bank),
        consensus_dev_truth=str(truth),
        consensus_manifest=str(manifest),
        k200_candidates=str(k200),
        fullbank_candidates=str(fullbank),
        prior_candidate_bundle=str(prior),
        dev_match_labels=str(tmp_path / "dev.matches.jsonl"),
        capture_report=str(tmp_path / "capture.json"),
        rescue_misses=str(tmp_path / "misses.jsonl"),
        gate_report=str(tmp_path / "gate.json"),
        candidate_bundle_output=str(tmp_path / "bundle.json"),
    )


def test_zero_misses_passes_and_freezes_candidate_bundle(tmp_path: Path) -> None:
    args = _fixture(tmp_path, misses=0)
    report = run_gate(args)
    assert report["status"] == PASS_STATUS
    assert report["gate"]["passed"] is True
    assert report["gate"]["k200_miss_upper_bound_one_sided_95"] < 0.05
    assert Path(args.candidate_bundle_output).is_file()
    bundle = json.loads(Path(args.candidate_bundle_output).read_text())
    assert bundle["candidate_inputs"]["production_k200"]["sha256"] == sha256_file(
        Path(args.k200_candidates)
    )
    validated = _validate_progressive_policy_gate(
        Path(args.gate_report),
        bank_path=Path(args.bank),
        primary_candidate=Path(args.k200_candidates),
        rescue_candidate=Path(args.fullbank_candidates),
        capture_report=Path(args.capture_report),
    )
    assert validated["selected_policy"]["passed_deployment_gate"] is True


def test_failed_gate_routes_all_exact_misses_and_freezes_training_only_bundle(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path, misses=4)
    report = run_gate(args)
    assert report["status"] == FAIL_STATUS
    assert report["gate"]["passed"] is False
    assert report["gate"]["k200_miss_count"] == 4
    assert report["rescue"]["misses_routed"] == 4
    bundle_path = Path(args.candidate_bundle_output)
    assert bundle_path.is_file()
    bundle = json.loads(bundle_path.read_text())
    assert bundle["k200_primary_promoted"] is False
    assert bundle["fullbank_required_for_production"] is True
    assert bundle["status"] == (
        "FROZEN_DIVERSE_BUNDLE_K200_NOT_PROMOTED_FULLBANK_REQUIRED"
    )
    assert sum(1 for _ in Path(args.rescue_misses).open()) == 4
