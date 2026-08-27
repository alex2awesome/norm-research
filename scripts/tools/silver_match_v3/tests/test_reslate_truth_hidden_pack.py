from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file, write_jsonl
from scripts.tools.silver_match_v3.reslate_truth_hidden_pack import (
    FORBIDDEN_ITEM_FIELDS,
    reslate,
)


def _ref(path: Path, count: int | None = None) -> dict:
    value = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    if count is not None:
        value["count"] = count
    return value


def _pack(root: Path, *, leak: bool = False) -> Path:
    root.mkdir()
    roles = ["train", "train", "train", "dev", "dev", "blind"]
    items = []
    identities = []
    for index, role in enumerate(roles):
        split = "test" if role == "blind" else role
        item = {
            "schema_version": "test-item-v1",
            "task": "demo",
            "corpus": "demo_corpus",
            "norm_uid": f"u{index}",
            "norm": f"norm {index}",
            "context": f"context {index}",
            "source_group": f"demo\x1fsource\x1fg{index}",
            "split_group": f"demo\x1fsource\x1fg{index}",
            "split": split,
            "collection_role": role,
            "truth_hidden": True,
        }
        if leak and index == 0:
            item["decision"] = "MATCH"
        items.append(item)
        identities.append(
            {
                "schema_version": "test-identity-v1",
                "task": "demo",
                "corpus": "demo_corpus",
                "norm_uid": f"u{index}",
                "source_group": f"demo\x1fsource\x1fg{index}",
                "split": split,
                "collection_role": role,
                "truth_hidden": True,
            }
        )
    bank = {
        "schema_version": "test-bank-v1",
        "task": "demo",
        "source_sha256": "bank-source-sha",
        "metrics": [
            {"metric_id": f"m{index}", "name": f"metric {index}"}
            for index in range(5)
        ],
    }
    items_path, bank_path = root / "items.jsonl", root / "bank.json"
    identities_path = root / "identities.jsonl"
    write_jsonl(items_path, items)
    write_jsonl(identities_path, identities)
    bank_path.write_text(json.dumps(bank, sort_keys=True) + "\n")
    chunk_paths = []
    for start in range(0, len(items), 2):
        path = root / "chunks" / f"part-{start // 2:03d}.jsonl"
        write_jsonl(path, items[start : start + 2])
        chunk_paths.append(path)
    role_refs = {}
    for role in sorted(set(roles)):
        path = root / "identities" / f"{role}.jsonl"
        selected = [row for row in identities if row["collection_role"] == role]
        write_jsonl(path, selected)
        role_refs[role] = _ref(path, len(selected))
    validation = {
        "schema_version": "test-pack-v1",
        "status": "FROZEN_TRUTH_HIDDEN_BEFORE_LABELING",
        "task": "demo",
        "count": len(items),
        "source_groups": len(items),
        "chunk_size": 2,
        "chunk_count": len(chunk_paths),
        "bank_metric_count": len(bank["metrics"]),
        "bank_source_sha256": "bank-source-sha",
        "truth_hidden": True,
        "prior_decisions_proposals_predictions_and_outcomes_hidden": True,
        "outputs": {
            "items": _ref(items_path, len(items)),
            "bank": _ref(bank_path, len(bank["metrics"])),
            "identities": _ref(identities_path, len(identities)),
            "identities_by_role": role_refs,
            "chunks": {str(path.resolve()): sha256_file(path) for path in chunk_paths},
        },
    }
    (root / "validation.json").write_text(json.dumps(validation, sort_keys=True) + "\n")
    return root


def _item_map(path: Path) -> dict[str, dict]:
    return {str(row["norm_uid"]): row for row in read_jsonl(path)}


def test_reslate_is_distinct_exact_and_deterministic(tmp_path: Path) -> None:
    source = _pack(tmp_path / "source")
    first, second = tmp_path / "first", tmp_path / "second"
    result = reslate(source, first, seed=20260714)
    reslate(source, second, seed=20260714)

    assert _item_map(first / "items.jsonl") == _item_map(source / "items.jsonl")
    assert sha256_file(first / "items.jsonl") != sha256_file(source / "items.jsonl")
    assert sha256_file(first / "bank.json") != sha256_file(source / "bank.json")
    assert sha256_file(first / "validation.json") != sha256_file(source / "validation.json")
    assert sha256_file(first / "items.jsonl") == sha256_file(second / "items.jsonl")
    assert sha256_file(first / "bank.json") == sha256_file(second / "bank.json")
    assert result["truth_hidden"] is True
    assert result["reslate"]["same_uid_role_group_payload"] is True

    chunk_rows = [
        row
        for path in sorted((first / "chunks").glob("*.jsonl"))
        for row in read_jsonl(path)
    ]
    assert {row["norm_uid"] for row in chunk_rows} == set(_item_map(source / "items.jsonl"))
    assert len(chunk_rows) == len({row["norm_uid"] for row in chunk_rows})
    assert all(not (FORBIDDEN_ITEM_FIELDS & set(row)) for row in chunk_rows)

    source_identities = _item_map(source / "identities.jsonl")
    first_identities = _item_map(first / "identities.jsonl")
    assert first_identities == source_identities
    assert {
        role: {row["norm_uid"] for row in read_jsonl(first / "identities" / f"{role}.jsonl")}
        for role in ("train", "dev", "blind")
    } == {
        "train": {"u0", "u1", "u2"},
        "dev": {"u3", "u4"},
        "blind": {"u5"},
    }


def test_reslate_is_append_only(tmp_path: Path) -> None:
    source = _pack(tmp_path / "source")
    target = tmp_path / "target"
    reslate(source, target, seed=7)
    with pytest.raises(FileExistsError, match="append-only"):
        reslate(source, target, seed=8)


def test_reslate_rejects_truth_leak_before_creating_output(tmp_path: Path) -> None:
    source = _pack(tmp_path / "source", leak=True)
    target = tmp_path / "target"
    with pytest.raises(ValueError, match="leaks truth/proposals"):
        reslate(source, target, seed=3)
    assert not target.exists()


def test_reslate_rejects_tampered_source_chunk(tmp_path: Path) -> None:
    source = _pack(tmp_path / "source")
    path = source / "chunks" / "part-000.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="chunk hash mismatch"):
        reslate(source, tmp_path / "target", seed=3)
