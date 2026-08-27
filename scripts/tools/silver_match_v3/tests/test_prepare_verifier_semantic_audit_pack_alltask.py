import json

from scripts.tools.silver_match_v3.prepare_verifier_semantic_audit_pack import main


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_renders_non_humor_task_without_truth_or_predictions(tmp_path, monkeypatch):
    norms = tmp_path / "norms.jsonl"
    _write_jsonl(
        norms,
        [
            {
                "norm_uid": "u1",
                "task": "press-releases",
                "corpus": "press_releases",
                "row": 1,
                "source_id": "s1",
                "norm": "The headline should name the announcement.",
                "context": "The headline is vague.",
            },
            {
                "norm_uid": "u2",
                "task": "press-releases",
                "corpus": "press_releases",
                "row": 2,
                "source_id": "s2",
                "norm": "Use a concrete quotation.",
                "context": "This quote says nothing.",
            },
        ],
    )
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {
                "source_sha256": "bank-source",
                "metrics": [
                    {"metric_id": "a1", "name": "Headline", "description": "Specificity"},
                    {"metric_id": "a2", "name": "Quotation", "description": "Substance"},
                ],
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "corpora": {
                    "press_releases": {
                        "task": "press-releases",
                        "path": norms.name,
                    }
                },
                "banks": {
                    "press-releases": {
                        "path": bank.name,
                        "source_sha256": "bank-source",
                    }
                },
            }
        )
    )
    items = tmp_path / "items.jsonl"
    _write_jsonl(items, [{"norm_uid": "u1", "source_group": "press_releases:source:s1"}])
    forbidden = tmp_path / "forbidden.jsonl"
    _write_jsonl(forbidden, [{"norm_uid": "u2", "source_group": "press_releases:source:s2"}])
    output = tmp_path / "pack"
    monkeypatch.setattr(
        "sys.argv",
        [
            "pack",
            "--manifest",
            str(manifest),
            "--task",
            "press-releases",
            "--items",
            str(items),
            "--forbidden-items",
            str(forbidden),
            "--output-root",
            str(output),
        ],
    )
    main()
    rendered = json.loads((output / "items.jsonl").read_text())
    report = json.loads((output / "validation.json").read_text())
    assert rendered["task"] == report["task"] == "press-releases"
    assert rendered["manual_decision"] is None
    assert report["truth_hidden"] is True
    assert report["adjudicator_outputs_read"] is False


def test_repeated_forbidden_sets_are_all_enforced(tmp_path, monkeypatch):
    norms = tmp_path / "norms.jsonl"
    _write_jsonl(
        norms,
        [
            {
                "norm_uid": f"u{i}",
                "task": "press-releases",
                "corpus": "press_releases",
                "row": i,
                "source_id": f"s{i}",
                "norm": f"Norm {i}",
            }
            for i in range(1, 4)
        ],
    )
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {
                "source_sha256": "bank-source",
                "metrics": [{"metric_id": "a1", "name": "Metric"}],
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "corpora": {
                    "press_releases": {
                        "task": "press-releases",
                        "path": norms.name,
                    }
                },
                "banks": {
                    "press-releases": {
                        "path": bank.name,
                        "source_sha256": "bank-source",
                    }
                },
            }
        )
    )
    items = tmp_path / "items.jsonl"
    _write_jsonl(items, [{"norm_uid": "u1", "source_group": "press_releases:source:s1"}])
    forbidden_a = tmp_path / "forbidden-a.jsonl"
    forbidden_b = tmp_path / "forbidden-b.jsonl"
    _write_jsonl(forbidden_a, [{"norm_uid": "u2", "source_group": "press_releases:source:s2"}])
    _write_jsonl(forbidden_b, [{"norm_uid": "u3", "source_group": "press_releases:source:s3"}])
    output = tmp_path / "pack"
    monkeypatch.setattr(
        "sys.argv",
        [
            "pack",
            "--manifest",
            str(manifest),
            "--task",
            "press-releases",
            "--items",
            str(items),
            "--forbidden-items",
            str(forbidden_a),
            "--forbidden-items",
            str(forbidden_b),
            "--output-root",
            str(output),
        ],
    )
    main()
    report = json.loads((output / "validation.json").read_text())
    assert report["permanent_blind_rows_excluded"] == 2
    assert len(report["input_hashes"]["forbidden_items"]) == 2
