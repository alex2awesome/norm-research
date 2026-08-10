import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.audit_alltask_release_coverage import (
    audit_alltask_coverage,
)
from scripts.tools.silver_match_v3.common import sha256_file


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _fixture(tmp_path: Path):
    bank = _write_json(
        tmp_path / "bank.json",
        {"metrics": [{"metric_id": "a0", "name": "criterion"}]},
    )
    norm = tmp_path / "norms.jsonl"
    norm.write_text(
        json.dumps(
            {
                "norm_uid": "u0",
                "row": 0,
                "corpus": "corpus",
                "task": "task",
                "polarity": "positive",
                "kind": "praise",
                "extraction_valid": True,
            }
        )
        + "\n"
    )
    manifest = _write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": "silver-match-v3.0",
            "total_norms": 1,
            "total_corpora": 1,
            "total_tasks": 1,
            "corpora": {
                "corpus": {
                    "task": "task",
                    "count": 1,
                    "path": str(norm),
                }
            },
            "banks": {
                "task": {
                    "count": 1,
                    "path": str(bank),
                    "source_sha256": "bank-source",
                }
            },
        },
    )
    lock = _write_json(
        tmp_path / "artifact_lock.json",
        {
            "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
            "norms": {
                "corpus": {"path": str(norm), "count": 1, "sha256": sha256_file(norm)}
            },
        },
    )
    selection = _write_json(
        tmp_path / "selection.json",
        {
            "task": "task",
            "selection_split": "external_dev_only",
            "chosen": {
                "kind": "nemotron_base",
                "name": "base",
                "fusion_report": str(tmp_path / "fusion.json"),
                "fusion_report_sha256": "fusion-sha",
            },
        },
    )
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_text(
        json.dumps({"norm_uid": "u0", "candidates": [{"metric_id": "a0"}]}) + "\n"
    )
    candidate_meta = _write_json(tmp_path / "candidate.jsonl.meta.json", {"count": 1})
    candidate_audit = _write_json(
        tmp_path / "candidate.audit.json",
        {
            "schema_version": "silver-match-v3-production-candidate-audit-v1",
            "complete": True,
            "task": "task",
            "corpus": "corpus",
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
                    "meta": str(candidate_meta),
                    "meta_sha256": sha256_file(candidate_meta),
                }
            },
        },
    )
    final = tmp_path / "final.jsonl"
    final.write_text(
        json.dumps(
            {
                "norm_uid": "u0",
                "row": 0,
                "corpus": "corpus",
                "task": "task",
                "bank_source_sha256": "bank-source",
                "decision": "MATCH",
                "metric_id": "a0",
                "confidence": "high",
                "verification_status": "verified_exact_match",
            }
        )
        + "\n"
    )
    blind = _write_json(
        tmp_path / "blind.json",
        {"task": "task", "status": "PASS", "production_final_blind_audit": True},
    )
    return manifest, lock, selection, candidate_audit, final, blind


def test_alltask_coverage_requires_every_gate(tmp_path: Path, monkeypatch):
    manifest, lock, selection, candidate_audit, final, blind = _fixture(tmp_path)
    monkeypatch.setattr(
        "scripts.tools.silver_match_v3.audit_alltask_release_coverage.verify_task_final_risk_release",
        lambda *args, **kwargs: {
            "task": "task",
            "status": "PASS",
            "complete": True,
            "production_final_blind_audit": True,
            "gates": {"strict_recomputed_test_gate": True},
            "match_audit": {"statistics": {"audited_rows": 60}},
            "abstention_audit": {"statistics": {"audited_rows": 60}},
        },
    )
    complete = audit_alltask_coverage(
        manifest_path=manifest,
        artifact_lock_path=lock,
        selection_bindings={"task": selection},
        candidate_audits=[candidate_audit],
        final_bindings={"corpus": final},
        blind_audit_bindings={"task": blind},
    )
    assert complete["complete"] is True
    assert complete["candidate_retrieval"]["covered_count"] == 1

    incomplete = audit_alltask_coverage(
        manifest_path=manifest,
        artifact_lock_path=lock,
        selection_bindings={"task": selection},
        candidate_audits=[],
        final_bindings={},
        blind_audit_bindings={},
    )
    assert incomplete["complete"] is False
    assert incomplete["candidate_retrieval"]["missing_corpora"] == ["corpus"]
    assert incomplete["canonical_final_outputs"]["missing_corpora"] == ["corpus"]
    assert incomplete["production_final_blind_audits"]["missing_tasks"] == ["task"]


def test_arbitrary_blind_pass_marker_is_rejected(tmp_path: Path):
    manifest, lock, selection, candidate_audit, final, blind = _fixture(tmp_path)
    with pytest.raises(ValueError, match="unsupported task final-risk schema"):
        audit_alltask_coverage(
            manifest_path=manifest,
            artifact_lock_path=lock,
            selection_bindings={"task": selection},
            candidate_audits=[candidate_audit],
            final_bindings={"corpus": final},
            blind_audit_bindings={"task": blind},
        )


def test_candidate_hash_drift_fails_closed(tmp_path: Path):
    manifest, lock, selection, candidate_audit, final, blind = _fixture(tmp_path)
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_text("{}\n")
    with pytest.raises(ValueError, match="candidate artifact differs"):
        audit_alltask_coverage(
            manifest_path=manifest,
            artifact_lock_path=lock,
            selection_bindings={"task": selection},
            candidate_audits=[candidate_audit],
            final_bindings={"corpus": final},
            blind_audit_bindings={"task": blind},
        )
