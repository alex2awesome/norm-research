import json

import pytest

from scripts.tools.silver_match_v3.audit_candidate_outputs import audit_candidates
from scripts.tools.silver_match_v3.common import sha256_file, write_jsonl


def fixture(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {
                "metrics": [
                    {"metric_id": "m1", "name": "one", "description": "one"},
                    {"metric_id": "m2", "name": "two", "description": "two"},
                ]
            }
        )
    )
    norms = tmp_path / "norms.jsonl"
    write_jsonl(
        norms,
        [
            {"norm_uid": "0" * 64, "row": 0},
            {"norm_uid": "1" * 64, "row": 1},
        ],
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "corpora": {"c": {"task": "t", "count": 2, "path": str(norms)}},
                "banks": {
                    "t": {"count": 2, "path": str(bank), "source_sha256": "bank-sha"}
                },
            }
        )
    )
    candidates = tmp_path / "candidates.jsonl"
    rows = []
    for index, uid in enumerate(("0" * 64, "1" * 64)):
        rows.append(
            {
                "norm_uid": uid,
                "corpus": "c",
                "task": "t",
                "row": index,
                "bank_source_sha256": "bank-sha",
                "candidates": [
                    {"metric_id": "m1", "rank": 1},
                    {"metric_id": "m2", "rank": 2},
                ],
            }
        )
    write_jsonl(candidates, rows)
    meta = candidates.with_suffix(".jsonl.meta.json")
    meta.write_text(
        json.dumps(
            {
                "output_sha256": sha256_file(candidates),
                "corpus": "c",
                "task": "t",
                "bank_source_sha256": "bank-sha",
                "output_k": 2,
                "fusion_weights": None,
                "fusion_weights_sha256": None,
                "adapter": None,
                "encoder": "/models/frozen-encoder",
                "query_format": "nemotron",
                "query_views": "evidence+statement",
                "dense_query_instruction": True,
            }
        )
    )
    return manifest, candidates, rows, meta


def test_candidate_audit_proves_exact_coverage_and_bank_membership(tmp_path):
    manifest, candidates, _, _ = fixture(tmp_path)
    report = audit_candidates(
        manifest_path=manifest,
        corpus="c",
        candidate_paths=[candidates],
        expected_k=2,
    )
    assert report["complete"] is True
    assert report["observed_count"] == 2
    assert report["candidate_count_distribution"] == {"2": 2}


def test_candidate_audit_rejects_duplicate_metric_ids(tmp_path):
    manifest, candidates, rows, meta = fixture(tmp_path)
    rows[0]["candidates"][1]["metric_id"] = "m1"
    write_jsonl(candidates, rows)
    payload = json.loads(meta.read_text())
    payload["output_sha256"] = sha256_file(candidates)
    meta.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate/out-of-bank"):
        audit_candidates(
            manifest_path=manifest,
            corpus="c",
            candidate_paths=[candidates],
            expected_k=2,
        )
