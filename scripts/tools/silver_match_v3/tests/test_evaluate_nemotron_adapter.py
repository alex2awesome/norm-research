import json

import pytest

from scripts.tools.silver_match_v3.evaluate_nemotron_adapter import load_eval
from scripts.tools.silver_match_v3.evaluate_nemotron_adapter import paired_comparison


def test_paired_comparison_counts_boundary_crossings():
    before = {
        "items": [
            {"norm_uid": "a", "exact_rank": 5},
            {"norm_uid": "b", "exact_rank": 60},
            {"norm_uid": "c", "exact_rank": 20},
            {"norm_uid": "d", "exact_rank": 70},
        ]
    }
    after = {
        "items": [
            {"norm_uid": "a", "exact_rank": 7},
            {"norm_uid": "b", "exact_rank": 40},
            {"norm_uid": "c", "exact_rank": 55},
            {"norm_uid": "d", "exact_rank": 80},
        ]
    }
    result = paired_comparison(before, after, (50,))
    assert result["at_50"] == {
        "both_hit": 1,
        "base_only_hit": 1,
        "adapter_only_hit": 1,
        "neither_hit": 1,
        "net_additional_hits": 0,
        "rank_improved": 1,
        "rank_worsened": 3,
        "rank_tied": 0,
    }


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path):
    bank = {
        "metrics": [
            {"metric_id": "m1", "name": "Clarity", "description": "Clear."}
        ]
    }
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    norms_path = tmp_path / "norms.jsonl"
    _write_jsonl(
        norms_path,
        [
            {"norm_uid": "u1", "criterion": "Be clear."},
            {"norm_uid": "u2", "criterion": "Also clear."},
        ],
    )
    manifest = {
        "banks": {
            "task": {
                "path": str(bank_path),
                "source_sha256": "bank-sha",
            }
        },
        "corpora": {
            "corpus": {
                "task": "task",
                "path": str(norms_path),
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_load_eval_accepts_split_isolated_artifact(tmp_path):
    manifest_path = _fixture(tmp_path)
    labels_path = tmp_path / "dev.jsonl"
    _write_jsonl(
        labels_path,
        [
            {
                "task": "task",
                "split": "dev",
                "decision": "MATCH",
                "norm_uid": "u1",
                "metric_id": "m1",
                "current_bank_source_sha256": "bank-sha",
            },
            {
                "task": "task",
                "split": "dev",
                "decision": "NO_CANDIDATE_FITS",
                "norm_uid": "u2",
                "metric_id": None,
                "current_bank_source_sha256": "bank-sha",
            },
        ],
    )

    labels, norms, bank, bank_sha = load_eval(
        manifest_path, labels_path, "task", "dev"
    )

    assert [row["norm_uid"] for row in labels] == ["u1"]
    assert [row["norm_uid"] for row in norms] == ["u1"]
    assert [row["metric_id"] for row in bank] == ["m1"]
    assert bank_sha == "bank-sha"


def test_load_eval_rejects_combined_dev_test_artifact(tmp_path):
    manifest_path = _fixture(tmp_path)
    labels_path = tmp_path / "combined.jsonl"
    base = {
        "task": "task",
        "decision": "MATCH",
        "metric_id": "m1",
        "current_bank_source_sha256": "bank-sha",
    }
    _write_jsonl(
        labels_path,
        [
            {**base, "split": "dev", "norm_uid": "u1"},
            {**base, "split": "test", "norm_uid": "u2"},
        ],
    )

    with pytest.raises(ValueError, match="not task/split-isolated"):
        load_eval(manifest_path, labels_path, "task", "dev")
