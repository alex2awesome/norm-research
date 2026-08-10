import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.audit_task_candidate_coverage import (
    audit_task_candidate_coverage,
)
from scripts.tools.silver_match_v3.common import sha256_file


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _fixture(tmp_path: Path):
    bank = _write(tmp_path / "bank.json", {"metrics": [{"metric_id": "m0"}]})
    norms = tmp_path / "norms.jsonl"
    norms.write_text(json.dumps({"norm_uid": "u0", "row": 0}) + "\n")
    manifest = _write(
        tmp_path / "manifest.json",
        {
            "banks": {
                "legal": {
                    "count": 1,
                    "path": str(bank),
                    "source_sha256": "bank-source",
                }
            },
            "corpora": {
                "legal_corpus": {
                    "task": "legal",
                    "count": 1,
                    "path": str(norms),
                }
            },
        },
    )
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_text(json.dumps({"norm_uid": "u0"}) + "\n")
    meta = _write(tmp_path / "candidate.jsonl.meta.json", {"count": 1})
    audit = _write(
        tmp_path / "candidate.audit.json",
        {
            "complete": True,
            "task": "legal",
            "corpus": "legal_corpus",
            "expected_count": 1,
            "observed_count": 1,
            "expected_k": 1,
            "materialized_k": 1,
            "manifest_sha256": sha256_file(manifest),
            "bank_source_sha256": "bank-source",
            "candidate_inputs": {
                str(candidate): {
                    "count": 1,
                    "sha256": sha256_file(candidate),
                    "meta": str(meta),
                    "meta_sha256": sha256_file(meta),
                }
            },
        },
    )
    return manifest, audit, candidate


def test_seals_exact_task_coverage(tmp_path):
    manifest, audit, _candidate = _fixture(tmp_path)
    result = audit_task_candidate_coverage(
        manifest_path=manifest, task="legal", candidate_audits=[audit]
    )
    assert result["complete"] is True
    assert result["covered_count"] == 1
    assert list(result["corpora"]) == ["legal_corpus"]


def test_rechecks_candidate_hashes(tmp_path):
    manifest, audit, candidate = _fixture(tmp_path)
    candidate.write_text("{}\n")
    with pytest.raises(ValueError, match="candidate artifact differs"):
        audit_task_candidate_coverage(
            manifest_path=manifest, task="legal", candidate_audits=[audit]
        )
