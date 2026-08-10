import json

import pytest

from scripts.tools.silver_match_v3.build_gepa_exclusion_union import main


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_builds_identity_union_and_hashes_sealed_without_parsing(tmp_path, monkeypatch):
    norms = tmp_path / "norms.jsonl"
    _jsonl(norms, [
        {"norm_uid": "u1", "task": "task", "corpus": "c", "source_id": "s1"},
        {"norm_uid": "u2", "task": "task", "corpus": "c", "source_id": "s2"},
    ])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"corpora": {"c": {"task": "task", "path": norms.name}}}))
    panel = tmp_path / "panel.jsonl"
    _jsonl(panel, [{
        "norm_uid": "u1", "source_group": "legacy\u001fsource\u001fs1",
        "split_group": "c:source:s1", "metric_id": "hidden",
    }])
    uids = tmp_path / "uids.txt"
    uids.write_text("u2\n")
    sealed = tmp_path / "sealed.json"
    sealed.write_text("this is deliberately not JSON")
    output = tmp_path / "out"
    monkeypatch.setattr("sys.argv", [
        "union", "--manifest", str(manifest), "--task", "task",
        "--panel", f"manual_calibration::{panel}",
        "--uid-file", f"retriever_teacher::{uids}",
        "--hash-only", f"sealed_test::{sealed}",
        "--required-category", "manual_calibration",
        "--required-category", "retriever_teacher",
        "--required-category", "sealed_test",
        "--output-root", str(output),
    ])
    main()
    report = json.loads((output / "EXCLUSION_INVENTORY.json").read_text())
    assert report["identity_union"]["uids"] == report["identity_union"]["source_groups"] == 2
    assert report["sources"][str(sealed.resolve())]["structured_content_parsed"] is False
    assert report["sources"][str(panel.resolve())]["supplied_source_group_mismatch_count"] == 0
    assert report["all_required_categories_present"] is True


def test_missing_required_category_fails(tmp_path, monkeypatch):
    norms = tmp_path / "norms.jsonl"
    _jsonl(norms, [{"norm_uid": "u1", "task": "task", "corpus": "c", "source_id": "s1"}])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"corpora": {"c": {"task": "task", "path": norms.name}}}))
    panel = tmp_path / "panel.jsonl"
    _jsonl(panel, [{"norm_uid": "u1"}])
    monkeypatch.setattr("sys.argv", [
        "union", "--manifest", str(manifest), "--task", "task",
        "--panel", f"manual::{panel}", "--required-category", "blind_audit",
        "--output-root", str(tmp_path / "out"),
    ])
    with pytest.raises(ValueError, match="required exclusion categories absent"):
        main()
