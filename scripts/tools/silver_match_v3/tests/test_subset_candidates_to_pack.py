from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file, write_jsonl
from scripts.tools.silver_match_v3.subset_candidates_to_pack import main


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    pack = tmp_path / "pack"
    items = [
        {"norm_uid": f"{index:064x}", "task": "math-stackexchange"}
        for index in (2, 0)
    ]
    items_path = pack / "items.jsonl"
    write_jsonl(items_path, items)
    (pack / "validation.json").write_text(
        json.dumps(
            {
                "task": "math-stackexchange",
                "bank_source_sha256": "bank-sha",
                "outputs": {"items": {"sha256": sha256_file(items_path)}},
            }
        )
    )
    candidates = tmp_path / "candidates.jsonl"
    write_jsonl(
        candidates,
        [
            {
                "norm_uid": f"{index:064x}",
                "task": "math-stackexchange",
                "bank_source_sha256": "bank-sha",
                "candidates": [
                    {"metric_id": f"m{metric}", "rank": metric + 10}
                    for metric in range(4)
                ],
            }
            for index in range(3)
        ],
    )
    candidates.with_suffix(".jsonl.meta.json").write_text(
        json.dumps(
            {
                "output": str(candidates),
                "sha256": sha256_file(candidates),
            }
        )
    )
    return pack, candidates


def test_candidate_subset_preserves_pack_order_and_rewrites_topk_ranks(
    tmp_path: Path, monkeypatch
) -> None:
    pack, candidates = _fixtures(tmp_path)
    output = tmp_path / "selected.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "subset_candidates_to_pack",
            "--candidates",
            str(candidates),
            "--pack-root",
            str(pack),
            "--output-k",
            "3",
            "--output",
            str(output),
        ],
    )
    main()
    rows = list(read_jsonl(output))
    assert [row["norm_uid"] for row in rows] == [f"{2:064x}", f"{0:064x}"]
    assert [value["rank"] for value in rows[0]["candidates"]] == [1, 2, 3]
    assert json.loads(output.with_suffix(".jsonl.meta.json").read_text())["output_k"] == 3


def test_candidate_subset_fails_when_requested_uid_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    pack, candidates = _fixtures(tmp_path)
    rows = [row for row in read_jsonl(candidates) if row["norm_uid"] != f"{2:064x}"]
    missing = tmp_path / "missing.jsonl"
    write_jsonl(missing, rows)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "subset_candidates_to_pack",
            "--candidates",
            str(missing),
            "--candidate-meta",
            str(tmp_path / "absent-meta.json"),
            "--pack-root",
            str(pack),
            "--output-k",
            "3",
            "--output",
            str(tmp_path / "bad.jsonl"),
        ],
    )
    with pytest.raises(ValueError, match="do not cover pack"):
        main()
