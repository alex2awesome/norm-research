import json

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.select_cross_encoder_variants import select


def make_report(tmp_path, policy, name, low, status="DEV_PROMOTABLE_PENDING_BLIND"):
    model = tmp_path / f"model-{name}"
    model.mkdir()
    weights = model / "weights.bin"
    weights.write_text(name)
    report = {
        "task": "t",
        "frozen_policy": {"sha256": sha256_file(policy), "variant_name": name},
        "teacher_split_mode": "explicit_role",
        "source_group_split_audit": {"cross_role_source_group_count": 0},
        "frozen_test_consumed": False,
        "manifest_sha256": "manifest",
        "bank_source_sha256": "bank",
        "explicit_role_inputs": {
            "dev.jsonl": {"role": "dev", "sha256": "dev-sha"}
        },
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
    path.write_text(json.dumps(report))
    return path


def test_selects_top_two_eligible_predeclared_variants(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-cross-encoder-alltask-policy-v1",
                "scope": ["t"],
                "predeclared_variants": [
                    {"name": "v1"},
                    {"name": "v2"},
                    {"name": "v3"},
                ],
            }
        )
    )
    paths = [
        make_report(tmp_path, policy, "v1", 0.81),
        make_report(tmp_path, policy, "v2", 0.85),
        make_report(tmp_path, policy, "v3", 0.9, "REJECTED_DEV_GATE"),
    ]
    result = select(policy, "t", paths)
    assert result["status"] == "TWO_VARIANT_CE_PROPOSAL_PATH_SELECTED"
    assert [row["name"] for row in result["chosen"]] == ["v2", "v1"]


def test_selection_honors_policy_eligibility_registry(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-cross-encoder-alltask-policy-v1",
                "scope": ["t"],
                "predeclared_variants": [],
            }
        )
    )
    policy.with_suffix(".ELIGIBILITY.json").write_text(
        json.dumps(
            {
                "policy_sha256": sha256_file(policy),
                "eligible_primary_tasks": [],
            }
        )
    )
    with pytest.raises(ValueError, match="restricts this task"):
        select(policy, "t", [])
