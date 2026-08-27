import json

import pytest

from scripts.tools.silver_match_v3.freeze_upstream_role_reference import main
from scripts.tools.silver_match_v3.train_nemotron_lora import (
    source_group_key,
    split_source_group,
)


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path):
    norms = [
        {
            "norm_uid": f"u{i}",
            "task": "task",
            "corpus": "c",
            "source_id": f"s{i}",
        }
        for i in range(8)
    ]
    norm_path = tmp_path / "norms.jsonl"
    _jsonl(norm_path, norms)
    bank = tmp_path / "bank.json"
    bank.write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "banks": {"task": {"path": bank.name, "source_sha256": "bank"}},
                "corpora": {"c": {"task": "task", "path": norm_path.name}},
            }
        )
    )
    candidates = tmp_path / "candidates.jsonl"
    _jsonl(
        candidates,
        [
            {
                "norm_uid": row["norm_uid"],
                "task": "task",
                "bank_source_sha256": "bank",
                "candidates": [{"metric_id": f"a{j}"} for j in range(3)],
            }
            for row in norms
        ],
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "task": "task",
                "manifest": str(manifest.resolve()),
                "split_seed": 73129,
                "train_percent": 80,
                "dev_percent": 10,
                "test_percent": 10,
            }
        )
    )
    audit = tmp_path / "audit.jsonl"
    _jsonl(
        audit,
        [
            {
                "norm_uid": row["norm_uid"],
                "task": "task",
                "split": split_source_group(source_group_key(row), 73129, 80, 10),
            }
            for row in norms
        ],
    )
    return manifest, candidates, config, audit


def test_freezes_and_matches_audited_roles(tmp_path, monkeypatch):
    manifest, candidates, config, audit = _fixture(tmp_path)
    output = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "freeze",
            "--manifest",
            str(manifest),
            "--task",
            "task",
            "--candidates",
            str(candidates),
            "--run-config",
            str(config),
            "--audit-reference",
            str(audit),
            "--minimum-k",
            "3",
            "--output-root",
            str(output),
        ],
    )
    main()
    report = json.loads((output / "FREEZE.json").read_text())
    assert report["audit_verification"] == {
        "path": str(audit.resolve()),
        "sha256": report["audit_verification"]["sha256"],
        "overlap": 8,
        "exact_role_matches": 8,
        "mismatches": 0,
    }
    assert sum(report["roles"].values()) == 8


def test_rejects_role_mismatch(tmp_path, monkeypatch):
    manifest, candidates, config, audit = _fixture(tmp_path)
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    rows[0]["split"] = "dev" if rows[0]["split"] != "dev" else "test"
    _jsonl(audit, rows)
    monkeypatch.setattr(
        "sys.argv",
        [
            "freeze",
            "--manifest",
            str(manifest),
            "--task",
            "task",
            "--candidates",
            str(candidates),
            "--run-config",
            str(config),
            "--audit-reference",
            str(audit),
            "--minimum-k",
            "3",
            "--output-root",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(ValueError, match="derived roles disagree"):
        main()
