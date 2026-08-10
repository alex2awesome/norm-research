import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.audit_final_outputs import (
    DECISION_ORDER,
    audit_outputs,
)


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path: Path):
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"metrics": [{"metric_id": "a1"}, {"metric_id": "a2"}]}))
    expected = tmp_path / "norms.jsonl"
    _write_jsonl(expected, [
        {"norm_uid": "u1", "corpus": "c", "task": "t", "row": 4, "polarity": "positive", "kind": "critique", "extraction_valid": 1},
        {"norm_uid": "u2", "corpus": "c", "task": "t", "row": 9, "polarity": "negative", "kind": "suggestion", "extraction_valid": 0},
    ])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "banks": {"t": {"path": str(bank), "source_sha256": "bank-sha"}},
        "corpora": {"c": {"path": str(expected), "task": "t", "count": 2}},
    }))
    final = tmp_path / "final.jsonl"
    base = {"corpus": "c", "task": "t", "confidence": "high", "bank_source_sha256": "bank-sha"}
    _write_jsonl(final, [
        {**base, "norm_uid": "u1", "row": 4, "decision": "MATCH", "metric_id": "a1", "verification_status": "verified_exact_match"},
        {**base, "norm_uid": "u2", "row": 9, "decision": "NOISE", "metric_id": None, "verification_status": "primary_typed_abstention"},
    ])
    return manifest, final


def test_audit_reports_every_rate_and_exact_coverage(tmp_path):
    manifest, final = _fixture(tmp_path)
    report = audit_outputs(manifest, [final])
    assert report["complete"] is True
    assert report["audited_rows"] == 2
    assert tuple(report["overall"]["decision_counts"]) == DECISION_ORDER
    assert report["overall"]["decision_counts"]["MATCH"] == 1
    assert report["overall"]["decision_counts"]["NOISE"] == 1
    assert report["overall"]["decision_counts"]["CONTEXT_NEEDED"] == 0
    assert report["overall"]["rollup_rates"]["verified_exact_match"] == 0.5
    assert report["overall"]["rollup_rates"]["noise"] == 0.5
    assert report["by_task"]["t"]["verification_status_counts"] == {
        "primary_typed_abstention": 1,
        "verified_exact_match": 1,
    }
    assert report["by_corpus"]["c"]["verification_status_counts"] == {
        "primary_typed_abstention": 1,
        "verified_exact_match": 1,
    }
    assert report["by_task"]["t"]["matched_metric_coverage"]["rate"] == 0.5
    assert report["macro_over_tasks"]["decision_rate_macro_mean"]["MATCH"] == 0.5
    assert report["macro_over_corpora"]["decision_rate_range"]["NOISE"] == {"min": 0.5, "max": 0.5}
    assert report["by_norm_stratum"]["extraction_valid:0"]["decision_rates"]["NOISE"] == 1.0
    assert report["by_norm_stratum"]["polarity:positive"]["decision_rates"]["MATCH"] == 1.0


def test_audit_rejects_wrong_order_even_when_counts_match(tmp_path):
    manifest, final = _fixture(tmp_path)
    rows = [json.loads(line) for line in final.read_text().splitlines()]
    _write_jsonl(final, list(reversed(rows)))
    with pytest.raises(ValueError, match="UID/order mismatch"):
        audit_outputs(manifest, [final])


def test_audit_rejects_metric_on_abstention(tmp_path):
    manifest, final = _fixture(tmp_path)
    rows = [json.loads(line) for line in final.read_text().splitlines()]
    rows[1]["metric_id"] = "a2"
    _write_jsonl(final, rows)
    with pytest.raises(ValueError, match="non-MATCH decision carries"):
        audit_outputs(manifest, [final])


def test_audit_requires_every_manifest_corpus(tmp_path):
    manifest, final = _fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["corpora"]["missing"] = payload["corpora"]["c"] | {"count": 0}
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing final corpora"):
        audit_outputs(manifest, [final])


def test_audit_can_freeze_one_complete_task_before_global_completion(tmp_path):
    manifest, final = _fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    other_bank = tmp_path / "other-bank.json"
    other_bank.write_text(json.dumps({"metrics": [{"metric_id": "b1"}]}))
    other_norms = tmp_path / "other.jsonl"
    _write_jsonl(
        other_norms,
        [{"norm_uid": "v1", "corpus": "d", "task": "other", "row": 0}],
    )
    payload["banks"]["other"] = {
        "path": str(other_bank),
        "source_sha256": "other-sha",
    }
    payload["corpora"]["d"] = {
        "path": str(other_norms),
        "task": "other",
        "count": 1,
    }
    manifest.write_text(json.dumps(payload))
    report = audit_outputs(manifest, [final], tasks={"t"})
    assert report["complete"] is True
    assert report["corpora_audited"] == 1
    assert report["scope"]["tasks"] == ["t"]
    assert report["global_complete"] is False
