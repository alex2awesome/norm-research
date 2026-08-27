from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.freeze_runtime_relocation import (
    relocate_manifest,
    relocate_policy,
)


def _dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_path_only_manifest_and_policy_relocation(tmp_path: Path) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    old_bank, new_bank = old / "banks/t.json", new / "banks/t.json"
    old_norm, new_norm = old / "norms/c.jsonl", new / "norms/c.jsonl"
    for left, right, text in (
        (old_bank, new_bank, '{"metrics": []}\n'),
        (old_norm, new_norm, '{"norm_uid":"u"}\n'),
    ):
        left.parent.mkdir(parents=True, exist_ok=True)
        right.parent.mkdir(parents=True, exist_ok=True)
        left.write_text(text, encoding="utf-8")
        right.write_text(text, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _dump(
        manifest,
        {
            "schema_version": "x",
            "banks": {"t": {"path": str(old_bank), "source_sha256": "source"}},
            "corpora": {"c": {"path": str(old_norm), "task": "t"}},
        },
    )
    mirror_manifest = tmp_path / "manifest.mirror.json"
    record, artifacts = relocate_manifest(manifest, mirror_manifest, old, new)
    assert record["all_artifact_hashes_equal"] is True
    assert len(artifacts) == 2
    assert json.loads(mirror_manifest.read_text())["banks"]["t"]["path"] == str(
        new_bank
    )

    source_model, mirror_model = tmp_path / "source_model", tmp_path / "model"
    source_model.mkdir()
    mirror_model.mkdir()
    (mirror_model / "config.json").write_text("same\n", encoding="utf-8")
    policy = tmp_path / "policy.json"
    _dump(
        policy,
        {
            "base_model": {
                "path": str(source_model),
                "file_sha256": {"config.json": sha256_file(mirror_model / "config.json")},
            },
            "fixed_training": {"epochs": 2},
        },
    )
    eligibility = policy.with_suffix(".ELIGIBILITY.json")
    _dump(eligibility, {"policy_sha256": sha256_file(policy), "eligible_primary_tasks": ["t"]})
    mirror_policy = tmp_path / "policy.mirror.json"
    policy_record, model_artifacts = relocate_policy(policy, mirror_policy, mirror_model)
    assert policy_record["only_changed_field"] == "base_model.path"
    assert len(model_artifacts) == 1
    assert json.loads(mirror_policy.read_text())["fixed_training"] == {"epochs": 2}
    mirror_eligibility = mirror_policy.with_suffix(".ELIGIBILITY.json")
    assert json.loads(mirror_eligibility.read_text())["policy_sha256"] == sha256_file(
        mirror_policy
    )


def test_manifest_relocation_rejects_hash_mismatch(tmp_path: Path) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    (old / "banks").mkdir(parents=True)
    (new / "banks").mkdir(parents=True)
    (old / "banks/t.json").write_text("old", encoding="utf-8")
    (new / "banks/t.json").write_text("changed", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _dump(
        manifest,
        {
            "banks": {"t": {"path": str(old / "banks/t.json")}},
            "corpora": {},
        },
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        relocate_manifest(manifest, tmp_path / "out.json", old, new)


def test_manifest_relocation_accepts_bound_remote_inventory(tmp_path: Path) -> None:
    old, new = tmp_path / "unmounted", tmp_path / "new"
    mirror = new / "banks/t.json"
    mirror.parent.mkdir(parents=True)
    mirror.write_text("same", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    source = old / "banks/t.json"
    _dump(
        manifest,
        {"banks": {"t": {"path": str(source)}}, "corpora": {}},
    )
    record, artifacts = relocate_manifest(
        manifest,
        tmp_path / "out.json",
        old,
        new,
        {
            str(source): {
                "sha256": sha256_file(mirror),
                "size_bytes": mirror.stat().st_size,
            }
        },
    )
    assert record["all_artifact_hashes_equal"] is True
    assert artifacts[0]["source_evidence"] == "hash_bound_source_inventory"
