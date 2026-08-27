import json
import sys

import pytest

from scripts.tools.silver_match_v3.combine_jsonl import main


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_combine_records_input_hashes_counts_and_unique_keys(tmp_path, monkeypatch):
    left, right, output = tmp_path / "left.jsonl", tmp_path / "right.jsonl", tmp_path / "out.jsonl"
    _write(left, [{"norm_uid": "a"}])
    _write(right, [{"norm_uid": "b"}])
    monkeypatch.setattr(
        sys,
        "argv",
        ["combine", "--inputs", str(left), str(right), "--output", str(output)],
    )
    main()
    meta = json.loads(output.with_suffix(".jsonl.meta.json").read_text())
    assert meta["count"] == 2
    assert sum(row["count"] for row in meta["inputs"].values()) == 2
    assert len(meta["sha256"]) == 64


def test_combine_rejects_duplicate_uid(tmp_path, monkeypatch):
    left, right, output = tmp_path / "left.jsonl", tmp_path / "right.jsonl", tmp_path / "out.jsonl"
    _write(left, [{"norm_uid": "a"}])
    _write(right, [{"norm_uid": "a"}])
    monkeypatch.setattr(
        sys,
        "argv",
        ["combine", "--inputs", str(left), str(right), "--output", str(output)],
    )
    with pytest.raises(ValueError, match="duplicate norm_uid"):
        main()
    assert not output.exists()
