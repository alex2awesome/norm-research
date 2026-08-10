import json

import pytest

from scripts.tools.silver_match_v3.merge_task_candidate_artifacts import main


def _manifest(path):
    path.write_text(
        json.dumps(
            {
                "banks": {"demo": {"source_sha256": "bank"}},
                "corpora": {
                    "left": {"task": "demo", "path": "unused-left.jsonl"},
                    "right": {"task": "demo", "path": "unused-right.jsonl"},
                },
            }
        )
    )


def _candidate(path, uid, corpus, *, bank="bank"):
    row = {
        "schema_version": "silver-match-v3.0",
        "norm_uid": uid,
        "task": "demo",
        "corpus": corpus,
        "row": 0,
        "bank_source_sha256": bank,
        "candidates": [{"metric_id": "m0"}, {"metric_id": "m1"}],
    }
    raw = json.dumps(row, separators=(",", ":")) + "\n"
    path.write_text(raw)
    return raw


def test_merges_sources_byte_for_byte_and_freezes_lineage(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    _manifest(manifest)
    left, right = tmp_path / "left.jsonl", tmp_path / "right.jsonl"
    left_raw = _candidate(left, "u0", "left")
    right_raw = _candidate(right, "u1", "right")
    output, report = tmp_path / "merged.jsonl", tmp_path / "merge.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "merge",
            "--manifest",
            str(manifest),
            "--task",
            "demo",
            "--input",
            str(left),
            "--input",
            str(right),
            "--minimum-k",
            "2",
            "--require-all-task-corpora",
            "--output",
            str(output),
            "--report",
            str(report),
        ],
    )
    main()
    assert output.read_text() == left_raw + right_raw
    frozen = json.loads(report.read_text())
    assert frozen["status"] == "FROZEN_VALIDATED_BYTE_PRESERVING_MERGE"
    assert frozen["row_count"] == 2


def test_rejects_duplicate_uid_across_sources(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    _manifest(manifest)
    left, right = tmp_path / "left.jsonl", tmp_path / "right.jsonl"
    _candidate(left, "same", "left")
    _candidate(right, "same", "right")
    monkeypatch.setattr(
        "sys.argv",
        [
            "merge",
            "--manifest",
            str(manifest),
            "--task",
            "demo",
            "--input",
            str(left),
            "--input",
            str(right),
            "--minimum-k",
            "2",
            "--output",
            str(tmp_path / "merged.jsonl"),
            "--report",
            str(tmp_path / "report.json"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate candidate UID"):
        main()
