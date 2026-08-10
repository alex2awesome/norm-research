from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import sha256_file, write_jsonl
from scripts.tools.silver_match_v3.materialize_role_pack_manifest import materialize


def _pack(root: Path, name: str, uid: str, source_id: str) -> Path:
    pack = root / name
    pack.mkdir(parents=True)
    items = pack / "items.jsonl"
    bank = pack / "bank.json"
    write_jsonl(
        items,
        [
            {
                "norm_uid": uid,
                "source_group": f"task:source:{source_id}",
                "source_id": source_id,
                "corpus": "corpus",
                "task": "task",
            }
        ],
    )
    bank.write_text(
        json.dumps(
            {
                "task": "task",
                "source_sha256": "bank-source",
                "metrics": [{"metric_id": "m1", "name": "metric"}],
            }
        )
        + "\n"
    )
    (pack / "validation.json").write_text(
        json.dumps(
            {
                "task": "task",
                "truth_hidden": True,
                "outputs": {
                    "items": {"sha256": sha256_file(items)},
                    "bank": {"sha256": sha256_file(bank)},
                },
            }
        )
        + "\n"
    )
    return pack


def _args(output: Path, packs: list[tuple[str, Path]]) -> argparse.Namespace:
    return argparse.Namespace(
        task="task",
        pack=[f"{name}={path}" for name, path in packs],
        output=str(output),
    )


def test_materializes_disjoint_role_packs(tmp_path: Path) -> None:
    first = _pack(tmp_path, "first", "u1", "s1")
    second = _pack(tmp_path, "second", "u2", "s2")
    output = tmp_path / "manifest.json"
    result = materialize(_args(output, [("train", first), ("dev", second)]))
    payload = json.loads(output.read_text())
    assert result["total_norms"] == 2
    assert payload["cross_pack_source_group_overlap"] == 0
    assert set(payload["corpora"]) == {"train", "dev"}


def test_rejects_cross_pack_source_group_overlap(tmp_path: Path) -> None:
    first = _pack(tmp_path, "first", "u1", "same")
    second = _pack(tmp_path, "second", "u2", "same")
    with pytest.raises(ValueError, match="overlap"):
        materialize(
            _args(tmp_path / "manifest.json", [("train", first), ("dev", second)])
        )


def test_can_materialize_one_corpus_for_strict_trainers(tmp_path: Path) -> None:
    first = _pack(tmp_path, "first", "u1", "s1")
    second = _pack(tmp_path, "second", "u2", "s2")
    args = _args(tmp_path / "manifest.json", [("train", first), ("dev", second)])
    args.merged_norms_output = str(tmp_path / "merged.jsonl")
    materialize(args)
    payload = json.loads(Path(args.output).read_text())
    assert set(payload["corpora"]) == {"corpus"}
    assert payload["corpora"]["corpus"]["count"] == 2
    assert payload["merged_norms"]["sha256"] == sha256_file(
        Path(args.merged_norms_output)
    )
