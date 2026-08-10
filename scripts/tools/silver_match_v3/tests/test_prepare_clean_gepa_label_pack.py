import json

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.prepare_clean_gepa_label_pack import main


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path):
    norms = [
        {
            "schema_version": "v3",
            "norm_uid": f"u{i}",
            "task": "task",
            "corpus": "c",
            "source_id": f"s{i}",
            "row": i,
            "norm": f"norm {i}",
        }
        for i in range(4)
    ]
    norm_path = tmp_path / "norms.jsonl"
    _jsonl(norm_path, norms)
    bank = {
        "task": "task",
        "source_sha256": "bank",
        "metrics": [
            {"metric_id": f"a{i}", "name": f"m{i}", "description": "d"}
            for i in range(4)
        ],
    }
    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps(bank))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "banks": {"task": {"path": bank_path.name, "source_sha256": "bank"}},
                "corpora": {"c": {"task": "task", "path": norm_path.name}},
            }
        )
    )
    identities = tmp_path / "panel" / "identities.jsonl"
    _jsonl(
        identities,
        [
            {
                "schema_version": "identity",
                "norm_uid": row["norm_uid"],
                "task": "task",
                "corpus": "c",
                "source_group": f"c:source:s{i}",
                "upstream_split": "train",
                "gepa_role": "optimize",
            }
            for i, row in enumerate(norms)
        ],
    )
    identity_freeze = tmp_path / "panel" / "FREEZE.json"
    identity_freeze.write_text(
        json.dumps(
            {
                "status": "FROZEN_BEFORE_PREDICTIONS_LABELS_OR_OUTCOMES",
                "task": "task",
                "role": "optimize",
                "required_upstream_split": "train",
                "outputs": {"identities": {"sha256": sha256_file(identities)}},
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
                "candidates": [
                    {"metric_id": f"a{j}", "rank": j + 1} for j in range(4)
                ],
            }
            for row in norms
        ],
    )
    upstream = tmp_path / "upstream.json"
    upstream.write_text(
        json.dumps(
            {
                "status": "FROZEN_AND_AUDIT_VERIFIED",
                "task": "task",
                "inputs": {
                    "candidates": {
                        "path": str(candidates.resolve()),
                        "sha256": sha256_file(candidates),
                    }
                },
            }
        )
    )
    return manifest, identities, identity_freeze, upstream, candidates


def test_prepares_truth_hidden_pack_and_k50_accounting(tmp_path, monkeypatch):
    manifest, identities, identity_freeze, upstream, candidates = _fixture(tmp_path)
    output = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare",
            "--manifest",
            str(manifest),
            "--task",
            "task",
            "--identities",
            str(identities),
            "--identity-freeze",
            str(identity_freeze),
            "--upstream-role-freeze",
            str(upstream),
            "--candidates",
            str(candidates),
            "--candidate-k",
            "3",
            "--chunk-size",
            "2",
            "--output-root",
            str(output),
        ],
    )
    main()
    report = json.loads((output / "validation.json").read_text())
    assert report["truth_hidden"] is True
    assert report["count"] == report["source_groups"] == 4
    assert report["chunk_count"] == 2
    assert report["candidate_k"] == 3
    assert report["usage_contract"]["optimize_may_mutate_prompts"] is True


def test_rejects_label_leak_in_identity_panel(tmp_path, monkeypatch):
    manifest, identities, identity_freeze, upstream, candidates = _fixture(tmp_path)
    rows = [json.loads(line) for line in identities.read_text().splitlines()]
    rows[0]["metric_id"] = "a0"
    _jsonl(identities, rows)
    freeze = json.loads(identity_freeze.read_text())
    freeze["outputs"]["identities"]["sha256"] = sha256_file(identities)
    identity_freeze.write_text(json.dumps(freeze))
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare",
            "--manifest",
            str(manifest),
            "--task",
            "task",
            "--identities",
            str(identities),
            "--identity-freeze",
            str(identity_freeze),
            "--upstream-role-freeze",
            str(upstream),
            "--candidates",
            str(candidates),
            "--candidate-k",
            "3",
            "--output-root",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(ValueError, match="forbidden"):
        main()
