import argparse
import json

from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file
from scripts.tools.silver_match_v3.merge_humor_truth_production_overlay import build


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_overlay_preserves_canonical_order_and_truth_authority(tmp_path):
    canonical_path = tmp_path / "norms.jsonl"
    bank_path = tmp_path / "bank.json"
    truth_path = tmp_path / "truth.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    manifest_path = tmp_path / "manifest.json"
    canonical = [
        {"norm_uid": uid, "task": "humor", "corpus": "humor_multi", "row": index}
        for index, uid in enumerate(("u3", "u1", "u2"))
    ]
    _jsonl(canonical_path, canonical)
    bank_sha = "b" * 64
    bank_path.write_text(
        json.dumps({"source_sha256": bank_sha, "metrics": [{"metric_id": "a1"}]}),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "corpora": {
                    "humor_multi": {
                        "task": "humor",
                        "path": str(canonical_path),
                        "count": 3,
                    }
                },
                "banks": {
                    "humor": {
                        "path": str(bank_path),
                        "count": 1,
                        "source_sha256": bank_sha,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    _jsonl(
        truth_path,
        [
            {
                "norm_uid": "u1",
                "decision": "MATCH",
                "metric_id": "a1",
                "confidence": "high",
            }
        ],
    )
    _jsonl(
        predictions_path,
        [
            {
                "norm_uid": "u2",
                "decision": "NO_CANDIDATE_FITS",
                "metric_id": None,
                "confidence": "medium",
            },
            {
                "norm_uid": "u3",
                "decision": "NOISE",
                "metric_id": None,
                "confidence": "low",
            },
        ],
    )
    output = tmp_path / "final.jsonl"
    audit = tmp_path / "audit.json"
    report = tmp_path / "report.json"
    result = build(
        argparse.Namespace(
            manifest=str(manifest_path),
            manifest_sha256=sha256_file(manifest_path),
            truth=str(truth_path),
            truth_sha256=sha256_file(truth_path),
            predictions=str(predictions_path),
            predictions_sha256=sha256_file(predictions_path),
            output=str(output),
            audit_output=str(audit),
            report_output=str(report),
            task="humor",
            corpus="humor_multi",
            expected_canonical=3,
            expected_truth=1,
            expected_predictions=2,
        )
    )
    rows = list(read_jsonl(output))
    assert [row["norm_uid"] for row in rows] == ["u3", "u1", "u2"]
    assert rows[1]["authoritative_truth_overlay"] is True
    assert rows[1]["metric_id"] == "a1"
    assert result["partition_audit"]["union_equals_canonical"] is True
    assert result["final_audit"]["audited_rows"] == 3
