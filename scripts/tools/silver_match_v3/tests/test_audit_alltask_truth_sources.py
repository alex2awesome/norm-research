from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.audit_alltask_truth_sources import (
    _audit_norm_rows,
    _audit_pair_verdicts,
    _overlap_report,
    _queue,
    _remote_record,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _truth_row(uid: str, group: str, decision: str, metric_id: str | None) -> dict[str, object]:
    return {
        "schema_version": "silver-match-v3-exact-multi-pass-resolved-truth-v1",
        "task": "code-review",
        "corpus": "crse",
        "norm_uid": uid,
        "source_group": group,
        "decision": decision,
        "metric_id": metric_id,
        "current_bank_source_sha256": "b" * 64,
        "label_source": "independent_exact_multi_pass_resolution",
        "training_eligible": False,
        "prompt_gradient_eligible": True,
        "source_predictions": {
            "A": {"annotator": "trusted-a"},
            "B": {"annotator": "trusted-b"},
        },
    }


def test_norm_truth_preserves_prompt_only_training_contract(tmp_path: Path) -> None:
    truth = tmp_path / "truth.jsonl"
    freezer = tmp_path / "FREEZE.json"
    _write_jsonl(
        truth,
        [
            _truth_row("u1", "g1", "MATCH", "a1"),
            _truth_row("u2", "g2", "MATCH_FAMILY_ONLY", None),
        ],
    )
    freezer.write_text("{}\n", encoding="utf-8")
    record, uids, groups, role_rows = _audit_norm_rows(
        source_id="source",
        task="code-review",
        role="optimize",
        classification="TRAIN_ELIGIBLE",
        truth_path=truth,
        expected_count=2,
        bank_hash="b" * 64,
        freezer_path=freezer,
        freezer_schema="test",
    )
    assert uids == {"u1", "u2"}
    assert groups == {"g1", "g2"}
    assert [row["role"] for row in role_rows] == ["train", "train"]
    assert all("decision" not in row and "metric_id" not in row for row in role_rows)
    assert record["label_type_counts"] == {"MATCH": 1, "MATCH_FAMILY_ONLY": 1}
    assert record["supervised_model_training_allowed"] is False
    assert record["allowed_uses"] == ["PROMPT_OPTIMIZATION"]


def test_norm_truth_rejects_metric_on_typed_nonmatch(tmp_path: Path) -> None:
    truth = tmp_path / "truth.jsonl"
    freezer = tmp_path / "FREEZE.json"
    _write_jsonl(truth, [_truth_row("u1", "g1", "NO_CANDIDATE_FITS", "a1")])
    freezer.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="typed nonmatch carries metric_id"):
        _audit_norm_rows(
            source_id="source",
            task="code-review",
            role="optimize",
            classification="TRAIN_ELIGIBLE",
            truth_path=truth,
            expected_count=1,
            bank_hash="b" * 64,
            freezer_path=freezer,
            freezer_schema="test",
        )


def test_pair_truth_slices_train_eval_and_excludes_null_scores(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            {
                "task": "code-review",
                "split": "train",
                "key_a": "a",
                "key_b": "b",
                "score": 2,
            },
            {
                "task": "code-review",
                "split": "train",
                "key_a": "c",
                "key_b": "d",
                "score": None,
            },
            {
                "task": "code-review",
                "split": "eval",
                "key_a": "a",
                "key_b": "c",
                "score": 0,
            },
        ],
    )
    records, pair_sets, rubric_sets = _audit_pair_verdicts(
        {
            "source_id": "pairs",
            "path": str(path.relative_to(tmp_path)),
            "tasks": ["code-review"],
        },
        root=tmp_path,
    )
    by_role = {record["role"]: record for record in records}
    assert by_role["train"]["row_count"] == 2
    assert by_role["train"]["eligible_row_count"] == 1
    assert by_role["train"]["rejected_row_count"] == 1
    assert by_role["eval"]["classification"] == "DEV_ONLY"
    assert pair_sets["code-review/train"].isdisjoint(pair_sets["code-review/eval"])
    assert rubric_sets["code-review/train"] & rubric_sets["code-review/eval"] == {"a", "c"}


def test_overlap_fails_closed_on_cross_role_source_group_reuse() -> None:
    overlap = _overlap_report(
        {
            "train": ("code-review", "TRAIN_ELIGIBLE", {"u1"}, {"shared"}),
            "dev": ("code-review", "DEV_ONLY", {"u2"}, {"shared"}),
        },
        {},
        {},
    )
    assert overlap["status"] == "FAIL_CROSS_ROLE_IDENTITY_LEAKAGE"
    assert overlap["leakage_failure_count"] == 1
    assert overlap["norm_truth_overlaps"][0]["uid_overlap"] == 0
    assert overlap["norm_truth_overlaps"][0]["source_group_overlap"] == 1


def test_queue_never_emits_blind_truth_path_and_binds_production_contract() -> None:
    records = [
        {
            "source_id": "blind",
            "task": "code-review",
            "format": "NORM_TO_METRIC_TRUTH",
            "classification": "BLIND_ONLY",
            "truth": {"sha256": "f" * 64, "path": "/forbidden/blind.jsonl"},
            "row_count": 10,
        }
    ]
    overlap = {"norm_truth_overlaps": [], "pair_truth_overlaps": []}
    queue = _queue("code-review", records, overlap)
    text = json.dumps(queue, sort_keys=True)
    assert "/forbidden/blind.jsonl" not in text
    assert queue["norm_metric_truth"]["blind_truth_paths_emitted"] is False
    assert queue["norm_metric_truth"]["blind_seals"] == [
        {"source_id": "blind", "sha256": "f" * 64, "row_count": 10}
    ]
    production = queue["downstream_ce_production_materialization"]
    assert production["minimum_independent_retrieval_lanes"] == 2
    assert production["requires_exact_manifest_norm_universe"] is True


def test_remote_memory_reference_is_rejected_until_content_audit() -> None:
    record = _remote_record(
        {
            "source_id": "remote",
            "format": "GEMMA4_NORM_TO_METRIC_TRIPLETS",
            "task": "code-review",
            "role": "train",
            "intended_classification": "TRAIN_ELIGIBLE",
            "remote_path": "/lfs/unavailable.jsonl",
            "expected_row_count": 100,
            "evidence": "memory only",
        }
    )
    assert record["classification"] == "REJECT"
    assert record["availability"] == "REMOTE_INACCESSIBLE_NOT_AUDITED"
    assert record["allowed_uses"] == []
