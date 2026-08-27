import json

import pytest

from scripts.tools.silver_match_v3.watch_exact_truth_consensus import (
    _all_or_none,
    _parse_named_roots,
    raw_pack_progress,
)


def test_raw_progress_requires_exact_named_chunk_coverage(tmp_path):
    pack = tmp_path / "pack"
    (pack / "chunks").mkdir(parents=True)
    (pack / "raw_labels").mkdir()
    for index in range(3):
        (pack / "chunks" / f"part-{index:03d}.jsonl").write_text("{}\n")
    (pack / "raw_labels" / "part-000.json").write_text("{}\n")
    (pack / "raw_labels" / "unrelated.json").write_text("{}\n")
    assert raw_pack_progress(pack) == {"expected": 3, "present": 1, "complete": False}
    for index in (1, 2):
        (pack / "raw_labels" / f"part-{index:03d}.json").write_text("{}\n")
    assert raw_pack_progress(pack)["complete"] is True


def test_named_initial_passes_are_exactly_two_and_distinct(tmp_path):
    parsed = _parse_named_roots([f"a={tmp_path / 'a'}", f"b={tmp_path / 'b'}"])
    assert [name for name, _ in parsed] == ["a", "b"]
    with pytest.raises(ValueError, match="exactly two"):
        _parse_named_roots([f"a={tmp_path / 'a'}"])
    with pytest.raises(ValueError, match="duplicate"):
        _parse_named_roots([f"a={tmp_path / 'a'}", f"a={tmp_path / 'b'}"])


def test_append_only_artifacts_fail_on_partial_state(tmp_path):
    paths = [tmp_path / name for name in ("a", "b", "c")]
    assert _all_or_none(paths, "demo") is False
    paths[0].write_text(json.dumps({}))
    with pytest.raises(RuntimeError, match="partial append-only"):
        _all_or_none(paths, "demo")
    for path in paths[1:]:
        path.write_text(json.dumps({}))
    assert _all_or_none(paths, "demo") is True
