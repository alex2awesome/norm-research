import json

import pytest

from scripts.tools.silver_match_v3.common import sha256_file
from scripts.tools.silver_match_v3.record_base_retention_selection import build_selection


def _evidence(tmp_path):
    task = "legal-outcome-prediction"
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"metrics": [{"metric_id": "m1"}]}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "banks": {
                    task: {
                        "count": 1,
                        "path": str(bank),
                        "source_sha256": "bank-source",
                    }
                }
            }
        )
    )
    exact = {"recall_at_50": 0.9, "recall_at_80": 1.0}
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "task": task,
                "split": "dev",
                "selection_role": "promotion_dev",
                "before": {"exact": exact},
                "after": {"exact": exact},
                "promotion_gate": {
                    "passed": False,
                    "minimum_gain": 0.03,
                    "actual_gain": 0.0,
                    "secondary_passed": True,
                },
            }
        )
    )
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "schema_version": "silver-match-v3-nemotron-external-dev-decision-v1",
                "task": task,
                "status": "FROZEN_REJECT_RETAIN_BASE",
                "decision": "REJECT_SELECTED_ADAPTER_RETAIN_FROZEN_BASE",
                "selected_variant": "retry-a",
                "external_dev_gate": {
                    "passed": False,
                    "minimum_exact_recall_at_50_gain": 0.03,
                    "actual_exact_recall_at_50_gain": 0.0,
                    "recall_at_80_non_decrease_passed": True,
                    "before": exact,
                    "after": exact,
                },
                "bindings": {"dev_report": {"sha256": sha256_file(report)}},
                "external_test": {
                    "status": "SEALED_UNCONSUMED",
                    "consumed_during_training": False,
                    "consumed_during_internal_selection": False,
                    "consumed_during_external_dev": False,
                },
            }
        )
    )
    fusion = tmp_path / "fusion.json"
    fusion.write_text(
        json.dumps(
            {
                "task": task,
                "selection_split": "dev",
                "split_counts": {"dev": 30},
                "bank_size": 1,
                "selected": {"component_weights": {"dense_rank": 1.0}},
                "metrics": {"dev": exact},
                "candidate_inputs": {"candidates": "sha"},
                "label_inputs": {"labels": "sha"},
            }
        )
    )
    return task, decision, report, fusion, manifest


def test_records_portable_base_retention_selection(tmp_path):
    task, decision, report, fusion, manifest = _evidence(tmp_path)
    result = build_selection(
        task=task,
        decision_path=decision,
        report_path=report,
        fusion_path=fusion,
        manifest_path=manifest,
    )
    assert result["chosen"]["kind"] == "nemotron_base"
    assert result["selection_split"] == "external_dev_only"
    assert result["frozen_test_consumed"] is False
    assert result["base_retention_evidence"]["external_dev_report"]["sha256"] == sha256_file(report)
    assert result["canonical_release_bindings"]["bank"]["count"] == 1


def test_fails_closed_if_external_test_was_consumed(tmp_path):
    task, decision, report, fusion, manifest = _evidence(tmp_path)
    payload = json.loads(decision.read_text())
    payload["external_test"]["consumed_during_external_dev"] = True
    decision.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="external test"):
        build_selection(
            task=task,
            decision_path=decision,
            report_path=report,
            fusion_path=fusion,
            manifest_path=manifest,
        )


def test_fails_closed_if_report_bytes_differ(tmp_path):
    task, decision, report, fusion, manifest = _evidence(tmp_path)
    report.write_text(report.read_text() + "\n")
    with pytest.raises(ValueError, match="does not match"):
        build_selection(
            task=task,
            decision_path=decision,
            report_path=report,
            fusion_path=fusion,
            manifest_path=manifest,
        )
