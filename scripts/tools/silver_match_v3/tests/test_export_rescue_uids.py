import json

import pytest

from scripts.tools.silver_match_v3.export_rescue_uids import export_uids


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_export_defaults_to_every_nonmatch_and_preserves_order(tmp_path):
    first = _write(
        tmp_path / "a.jsonl",
        [
            {"norm_uid": "a", "decision": "MATCH"},
            {"norm_uid": "b", "decision": "NOISE"},
        ],
    )
    second = _write(
        tmp_path / "b.jsonl",
        [{"norm_uid": "c", "decision": "UNSTABLE_MATCH"}],
    )
    output = tmp_path / "rescue.uids.txt"
    report = export_uids(
        input_paths=[first, second], output_path=output, include_decisions=None
    )
    assert output.read_text().splitlines() == ["b", "c"]
    assert report["selected_count"] == 2
    assert report["decision_counts"]["MATCH"] == 1


def test_export_rejects_duplicate_uid_across_corpora(tmp_path):
    first = _write(tmp_path / "a.jsonl", [{"norm_uid": "x", "decision": "NOISE"}])
    second = _write(tmp_path / "b.jsonl", [{"norm_uid": "x", "decision": "NOISE"}])
    with pytest.raises(ValueError, match="duplicate"):
        export_uids(
            input_paths=[first, second],
            output_path=tmp_path / "uids.txt",
            include_decisions=None,
        )
