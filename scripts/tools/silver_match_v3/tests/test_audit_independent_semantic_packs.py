import json

import pytest

from scripts.tools.silver_match_v3.audit_independent_semantic_packs import main
from scripts.tools.silver_match_v3.common import sha256_file


def _write_pack(root, *, item_order, metric_order):
    root.mkdir()
    items = [
        {
            "norm_uid": uid,
            "source_group": f"c:source:{uid}",
            "manual_decision": None,
            "manual_metric_id": None,
            "manual_confidence": None,
            "manual_reason": None,
            "auditor": None,
        }
        for uid in item_order
    ]
    items_path = root / "items.jsonl"
    items_path.write_text("".join(json.dumps(row) + "\n" for row in items))
    bank_path = root / "bank.json"
    bank_path.write_text(
        json.dumps({"metrics": [{"metric_id": metric_id} for metric_id in metric_order]})
    )
    validation = {
        "truth_hidden": True,
        "adjudicator_outputs_read": False,
        "label_pass_outputs_read": False,
        "task": "t",
        "count": len(items),
        "bank_metric_count": len(metric_order),
        "bank_source_sha256": "bank",
        "outputs": {
            "items": {"sha256": sha256_file(items_path)},
            "bank": {"sha256": sha256_file(bank_path)},
        },
    }
    (root / "validation.json").write_text(json.dumps(validation))


def test_accepts_independent_truth_hidden_views(tmp_path, monkeypatch):
    left, right = tmp_path / "left", tmp_path / "right"
    _write_pack(left, item_order=["u1", "u2"], metric_order=["a1", "a2"])
    _write_pack(right, item_order=["u2", "u1"], metric_order=["a2", "a1"])
    output = tmp_path / "audit.json"
    monkeypatch.setattr(
        "sys.argv",
        ["audit", "--left-root", str(left), "--right-root", str(right), "--output", str(output)],
    )
    main()
    assert json.loads(output.read_text())["complete"] is True


def test_rejects_exposed_prediction(tmp_path, monkeypatch):
    left, right = tmp_path / "left", tmp_path / "right"
    _write_pack(left, item_order=["u1", "u2"], metric_order=["a1", "a2"])
    _write_pack(right, item_order=["u2", "u1"], metric_order=["a2", "a1"])
    rows = [json.loads(line) for line in (right / "items.jsonl").read_text().splitlines()]
    rows[0]["decision"] = "MATCH"
    (right / "items.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    validation = json.loads((right / "validation.json").read_text())
    validation["outputs"]["items"]["sha256"] = sha256_file(right / "items.jsonl")
    (right / "validation.json").write_text(json.dumps(validation))
    monkeypatch.setattr(
        "sys.argv",
        ["audit", "--left-root", str(left), "--right-root", str(right), "--output", str(tmp_path / "audit.json")],
    )
    with pytest.raises(ValueError, match="prediction exposed"):
        main()
