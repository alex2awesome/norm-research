import json

from scripts.tools.silver_match_v3.validate_manifest import validate


def test_validate_small_manifest(tmp_path):
    bank = tmp_path / "bank.json"
    norm = tmp_path / "norm.jsonl"
    bank.write_text(
        json.dumps({"task": "t", "metrics": [{"metric_id": "a0"}]}), encoding="utf-8"
    )
    uid = "a" * 64
    norm.write_text(
        json.dumps(
            {
                "norm_uid": uid,
                "corpus": "c",
                "task": "t",
                "source_id": "s",
                "norm": "be clear",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "3",
                "total_norms": 1,
                "total_corpora": 1,
                "total_tasks": 1,
                "aliases": {},
                "banks": {"t": {"path": str(bank), "count": 1}},
                "corpora": {
                    "c": {
                        "path": str(norm),
                        "task": "t",
                        "count": 1,
                        "coverage_complete": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    report, lock = validate(manifest)
    assert report["status"] == "VALID"
    assert report["task_norm_counts"] == {"t": 1}
    assert lock["norms"]["c"]["sha256"]
