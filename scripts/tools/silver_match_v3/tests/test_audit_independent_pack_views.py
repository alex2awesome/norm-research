import json
import shutil

import pytest

from scripts.tools.silver_match_v3.audit_independent_pack_views import main
from scripts.tools.silver_match_v3.common import sha256_file


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _view(tmp_path, name, seed, uid_order, metric_order):
    root = tmp_path / name
    items = root / "items.jsonl"
    bank = root / "bank.json"
    chunks = root / "chunks" / "part-000.jsonl"
    rows = [
        {
            "norm_uid": uid,
            "task": "task",
            "corpus": "c",
            "row": int(uid[1:]),
            "norm": f"norm {uid}",
        }
        for uid in uid_order
    ]
    _jsonl(items, rows)
    _jsonl(chunks, rows)
    bank.parent.mkdir(parents=True, exist_ok=True)
    bank.write_text(
        json.dumps(
            {
                "task": "task",
                "source_sha256": "bank",
                "metrics": [{"metric_id": metric} for metric in metric_order],
            }
        )
    )
    validation = {
        "task": "task",
        "bank_source_sha256": "bank",
        "seed": seed,
        "source_pack": {"path": "source", "validation_sha256": "source-hash"},
        "outputs": {
            "items": {"sha256": sha256_file(items)},
            "bank": {"sha256": sha256_file(bank)},
            "chunks": {str(chunks.resolve()): sha256_file(chunks)},
        },
    }
    (root / "validation.json").write_text(json.dumps(validation))
    return root


def test_seals_two_distinct_truth_hidden_views(tmp_path, monkeypatch):
    a = _view(tmp_path, "a", 1, ["u0", "u1", "u2"], ["a0", "a1", "a2"])
    b = _view(tmp_path, "b", 2, ["u2", "u0", "u1"], ["a1", "a2", "a0"])
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        "sys.argv",
        ["audit", "--pass-a", str(a), "--pass-b", str(b), "--output", str(output)],
    )
    main()
    report = json.loads(output.read_text())
    assert report["status"] == "FROZEN_MUTUALLY_PREDICTION_HIDDEN_BEFORE_LABELING"
    assert report["same_canonical_item_content_by_uid"] is True
    assert report["pass_predictions_mutually_visible"] is False


def test_rejects_label_field(tmp_path, monkeypatch):
    a = _view(tmp_path, "a", 1, ["u0", "u1", "u2"], ["a0", "a1", "a2"])
    b = _view(tmp_path, "b", 2, ["u2", "u0", "u1"], ["a1", "a2", "a0"])
    rows = [json.loads(line) for line in (b / "items.jsonl").read_text().splitlines()]
    rows[0]["decision"] = "MATCH"
    _jsonl(b / "items.jsonl", rows)
    _jsonl(b / "chunks" / "part-000.jsonl", rows)
    validation = json.loads((b / "validation.json").read_text())
    validation["outputs"]["items"]["sha256"] = sha256_file(b / "items.jsonl")
    validation["outputs"]["chunks"][str((b / "chunks" / "part-000.jsonl").resolve())] = (
        sha256_file(b / "chunks" / "part-000.jsonl")
    )
    (b / "validation.json").write_text(json.dumps(validation))
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit",
            "--pass-a",
            str(a),
            "--pass-b",
            str(b),
            "--output",
            str(tmp_path / "audit.json"),
        ],
    )
    with pytest.raises(ValueError, match="forbidden"):
        main()


def test_accepts_byte_identical_views_relocated_into_isolated_workspaces(
    tmp_path, monkeypatch
):
    source_a = _view(
        tmp_path, "source-a", 1, ["u0", "u1", "u2"], ["a0", "a1", "a2"]
    )
    source_b = _view(
        tmp_path, "source-b", 2, ["u2", "u0", "u1"], ["a1", "a2", "a0"]
    )
    staged_a = tmp_path / "workspace-a" / "pack"
    staged_b = tmp_path / "workspace-b" / "pack"
    shutil.copytree(source_a, staged_a)
    shutil.copytree(source_b, staged_b)
    output = tmp_path / "relocated-audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit",
            "--pass-a",
            str(staged_a),
            "--pass-b",
            str(staged_b),
            "--output",
            str(output),
        ],
    )
    main()
    assert json.loads(output.read_text())["same_frozen_source_pack"] is True
