import copy
import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.select_cross_encoder_balanced_variants import select
from scripts.tools.silver_match_v3 import select_cross_encoder_variants as legacy


def _make_policy(tmp_path):
    policy = tmp_path / "policy.json"
    value = {
        "schema_version": "silver-match-v3-cross-encoder-alltask-policy-v1",
        "scope": ["t"],
        "balanced_objective_revision": "cross-encoder-metric-balanced-v4",
        "balanced_training": {"schema_version": "balanced-test-v1"},
        "implementation": {
            "select_cross_encoder_variants_sha256": sha256_file(
                Path(legacy.__file__).resolve()
            ),
            "balanced_train_cross_encoder_sha256": "trainer-sha",
        },
        "predeclared_variants": [{"name": "v1"}, {"name": "v2"}, {"name": "v3"}],
    }
    policy.write_text(json.dumps(value), encoding="utf-8")
    policy.with_suffix(".ELIGIBILITY.json").write_text(
        json.dumps(
            {
                "policy_sha256": sha256_file(policy),
                "eligible_primary_tasks": ["t"],
            }
        ),
        encoding="utf-8",
    )
    return policy, value


def _make_report(tmp_path, policy, policy_value, name, low, status):
    model = tmp_path / f"model-{name}"
    model.mkdir()
    weights = model / "weights.bin"
    weights.write_text(name, encoding="utf-8")
    binding = {
        "sha256": sha256_file(policy),
        "variant_name": name,
        "balanced_training": policy_value["balanced_training"],
        "balanced_trainer": {"sha256": "trainer-sha"},
    }
    report = {
        "task": "t",
        "frozen_policy": [policy_value, binding],
        "teacher_split_mode": "explicit_role",
        "source_group_split_audit": {"cross_role_source_group_count": 0},
        "frozen_test_consumed": False,
        "grouped_listwise_evaluation_contract": {
            "blind_status": "SEALED_UNCONSUMED"
        },
        "manifest_sha256": "manifest",
        "bank_source_sha256": "bank",
        "explicit_role_inputs": {"dev.jsonl": {"role": "dev", "sha256": "dev"}},
        "status": status,
        "dev_promotable": status == "DEV_PROMOTABLE_PENDING_BLIND",
        "selected_dev": {
            "exact_match_precision_wilson_95": [low, 1.0],
            "exact_f_beta_0_5": low,
            "exact_match_precision": 0.95,
            "exact_match_recall": 0.5,
        },
        "model_dir": str(model),
        "model_hashes": {"weights.bin": sha256_file(weights)},
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_selects_unchanged_balanced_tuple_reports(tmp_path):
    policy, value = _make_policy(tmp_path)
    reports = [
        _make_report(tmp_path, policy, value, "v1", 0.81, "DEV_PROMOTABLE_PENDING_BLIND"),
        _make_report(tmp_path, policy, value, "v2", 0.85, "DEV_PROMOTABLE_PENDING_BLIND"),
        _make_report(tmp_path, policy, value, "v3", 0.9, "REJECTED_DEV_GATE"),
    ]
    result = select(policy, "t", reports)
    assert result["status"] == "TWO_VARIANT_CE_PROPOSAL_PATH_SELECTED"
    assert [row["name"] for row in result["chosen"]] == ["v2", "v1"]
    assert result["frozen_test_consumed"] is False
    assert result["blind_status"] == "SEALED_UNCONSUMED"
    assert result["compatibility"]["training_reports_rewritten"] is False


def test_rejects_mutated_embedded_balanced_policy(tmp_path):
    policy, value = _make_policy(tmp_path)
    reports = [
        _make_report(tmp_path, policy, value, "v1", 0.81, "REJECTED_DEV_GATE"),
        _make_report(tmp_path, policy, value, "v2", 0.82, "REJECTED_DEV_GATE"),
        _make_report(tmp_path, policy, value, "v3", 0.83, "REJECTED_DEV_GATE"),
    ]
    broken = json.loads(reports[0].read_text(encoding="utf-8"))
    broken["frozen_policy"][0] = copy.deepcopy(broken["frozen_policy"][0])
    broken["frozen_policy"][0]["scope"] = ["different-task"]
    reports[0].write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="embedded balanced policy differs"):
        select(policy, "t", reports)
