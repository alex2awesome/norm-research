import json

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.select_cross_encoder_variants import (
    _supported_policy_task as selector_supports,
)
from scripts.tools.silver_match_v3.train_cross_encoder import (
    PRESS_RELEASES_POLICY_V2,
    _supported_policy_task as trainer_supports,
    enforce_press_releases_v2_inputs,
)


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_v2_schema_is_press_releases_only():
    policy = {"schema_version": PRESS_RELEASES_POLICY_V2, "scope": ["press-releases"]}
    assert trainer_supports(policy, "press-releases")
    assert selector_supports(policy, "press-releases")
    assert not trainer_supports(policy, "peer-review")
    assert not selector_supports(policy, "peer-review")


def test_v2_role_and_artifact_boundary(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    optimize = tmp_path / "optimize.jsonl"
    select = tmp_path / "select.jsonl"
    train = tmp_path / "train.jsonl"
    dev = tmp_path / "dev.jsonl"
    train_candidates = tmp_path / "train.candidates.jsonl"
    dev_candidates = tmp_path / "dev.candidates.jsonl"
    _jsonl(optimize, [{"norm_uid": "o1"}])
    _jsonl(select, [{"norm_uid": "s1"}, {"norm_uid": "s2"}])
    _jsonl(train, [{"norm_uid": "old"}, {"norm_uid": "o1"}])
    _jsonl(dev, [{"norm_uid": "s1"}, {"norm_uid": "s2"}])
    _jsonl(train_candidates, [{"norm_uid": "o1", "candidates": []}])
    _jsonl(dev_candidates, [{"norm_uid": "s1", "candidates": []}])
    policy = tmp_path / "policy.json"
    value = {
        "schema_version": PRESS_RELEASES_POLICY_V2,
        "immutable_artifacts": {
            "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
            "bank": {"source_sha256": "bank-sha"},
            "optimize_identity": {"path": str(optimize), "sha256": sha256_file(optimize)},
            "select_identity": {"path": str(select), "sha256": sha256_file(select)},
            "candidate_inputs": [
                {"path": str(train_candidates), "sha256": sha256_file(train_candidates)},
                {"path": str(dev_candidates), "sha256": sha256_file(dev_candidates)},
            ],
        },
    }
    policy.write_text(json.dumps(value), encoding="utf-8")
    audit = enforce_press_releases_v2_inputs(
        policy,
        manifest,
        "bank-sha",
        [],
        {"train": [train], "dev": [dev], "test": []},
        [train_candidates, dev_candidates],
    )
    assert audit["optimize_identity_count"] == 1
    assert audit["select_identity_count"] == 2
    assert audit["test_role_consumed"] is False

    _jsonl(dev, [{"norm_uid": "s1"}])
    with pytest.raises(ValueError, match="must equal"):
        enforce_press_releases_v2_inputs(
            policy,
            manifest,
            "bank-sha",
            [],
            {"train": [train], "dev": [dev], "test": []},
            [train_candidates, dev_candidates],
        )
