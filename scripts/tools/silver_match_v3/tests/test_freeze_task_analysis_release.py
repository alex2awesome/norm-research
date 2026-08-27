import json

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.freeze_task_analysis_release import freeze_release


def _dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path):
    norms = tmp_path / "norms.jsonl"
    norms.write_text('{"norm_uid":"a"}\n', encoding="utf-8")
    final = tmp_path / "final.jsonl"
    final.write_text('{"norm_uid":"a"}\n', encoding="utf-8")
    manifest = _dump(
        tmp_path / "manifest.json",
        {
            "banks": {"task": {"source_sha256": "b" * 64}},
            "corpora": {"corpus": {"task": "task", "count": 1, "path": str(norms)}},
        },
    )
    plan = _dump(
        tmp_path / "plan.json",
        {
            "status": "FROZEN_READY_FOR_UNLABELED_PRODUCTION",
            "task": "task",
            "manifest": {"sha256": sha256_file(manifest)},
            "bank_source_sha256": "b" * 64,
        },
    )
    final_audit = _dump(
        tmp_path / "final-audit.json",
        {
            "schema_version": "silver-match-v3-final-audit-v1",
            "complete": True,
            "manifest_sha256": sha256_file(manifest),
            "audited_rows": 1,
            "scope": {"tasks": ["task"], "corpora": None},
            "by_corpus": {"corpus": {}},
            "input_hashes": {str(final): sha256_file(final)},
        },
    )
    exclusions = tmp_path / "exclusions.jsonl"
    exclusions.write_text('{"norm_uid":"a"}\n', encoding="utf-8")
    risk = _dump(
        tmp_path / "risk.json",
        {
            "schema_version": "silver-match-v3-false-abstention-audit-v1",
            "prediction_inputs": {str(final): sha256_file(final)},
            "analysis_exclusions": {
                "inputs": {str(exclusions): sha256_file(exclusions)},
                "count": 1,
            },
            "by_task": {
                "task": {
                    "audited_rows": 100,
                    "claim_supported": False,
                    "predicted_match_precision_claim_supported": True,
                }
            },
        },
    )
    return dict(
        manifest_path=manifest,
        task="task",
        plan_path=plan,
        final_audit_path=final_audit,
        final_paths=[final],
        blind_risk_audit_path=risk,
        analysis_exclusion_paths=[exclusions],
    )


def test_freeze_task_release_binds_final_and_blind_audits(tmp_path):
    release = freeze_release(**_fixture(tmp_path))
    assert release["status"] == "TASK_FROZEN_ANALYSIS_READY"
    assert release["precision_claim_supported"] is True
    assert release["false_abstention_claim_supported"] is False
    assert release["analysis_exclusions"]["count"] == 1
    assert release["analysis_firewall"]["may_tune_this_or_other_task_matchers_from_results"] is False


def test_freeze_task_release_rejects_changed_final(tmp_path):
    kwargs = _fixture(tmp_path)
    kwargs["final_paths"][0].write_text('{"norm_uid":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="final files differ"):
        freeze_release(**kwargs)
