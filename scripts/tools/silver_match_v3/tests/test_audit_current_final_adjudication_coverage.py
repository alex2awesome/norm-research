import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.audit_current_final_adjudication_coverage import (
    audit_current_coverage,
)
from scripts.tools.silver_match_v3.common import sha256_file


def _json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return path


def _component(path: Path, rows: list[dict]) -> dict:
    _jsonl(path, rows)
    report = _json(
        Path(str(path) + ".report.json"), {"output_sha256": sha256_file(path)}
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "report": {"path": str(report), "sha256": sha256_file(report)},
    }


def _fixture(tmp_path: Path):
    task = "partial"
    bank_sha = "bank-source"
    bank = _json(tmp_path / "bank.json", {"metrics": [{"metric_id": "m0"}]})
    canonical_rows = [
        {"norm_uid": uid, "row": number, "task": task, "corpus": "c"}
        for number, uid in enumerate(
            ("match", "rescue", "no_criterion", "noise", "unresolved")
        )
    ]
    norms = _jsonl(tmp_path / "norms.jsonl", canonical_rows)
    other_norms = _jsonl(
        tmp_path / "other.jsonl",
        [{"norm_uid": "other", "row": 0, "task": "other", "corpus": "other_c"}],
    )
    manifest = _json(
        tmp_path / "manifest.json",
        {
            "total_norms": 6,
            "banks": {
                task: {"count": 1, "path": str(bank), "source_sha256": bank_sha},
                "other": {"count": 1, "path": str(bank), "source_sha256": bank_sha},
            },
            "corpora": {
                "c": {"task": task, "count": 5, "path": str(norms)},
                "other_c": {"task": "other", "count": 1, "path": str(other_norms)},
            },
        },
    )
    primary_rows = []
    for row in canonical_rows:
        current = {**row, "bank_source_sha256": bank_sha, "metric_id": None}
        current["decision"] = "MATCH" if row["norm_uid"] == "match" else "UNSTABLE_MATCH"
        if current["decision"] == "MATCH":
            current["metric_id"] = "m0"
        primary_rows.append(current)
    primary = _component(tmp_path / "primary.jsonl", primary_rows)
    primary["corpus"] = "c"
    rescue = _component(
        tmp_path / "rescue.jsonl",
        [
            {
                **canonical_rows[1],
                "candidate_bank_source_sha256": bank_sha,
                "strict_two_order_acceptance": True,
                "decision": "CONFIRM_MATCH",
                "metric_id": "m0",
            }
        ],
    )
    typed = _component(
        tmp_path / "typed.jsonl",
        [
            {
                **canonical_rows[2],
                "bank_source_sha256": bank_sha,
                "strict_two_order_abstention": True,
                "confirmed_decision": "NO_EXPLICIT_CRITERION",
            },
            {
                **canonical_rows[3],
                "bank_source_sha256": bank_sha,
                "strict_two_order_abstention": True,
                "confirmed_decision": "NOISE",
            },
        ],
    )
    unresolved = _component(
        tmp_path / "unresolved.jsonl",
        [
            {
                **canonical_rows[4],
                "bank_source_sha256": bank_sha,
                "unresolved_reason": "pending",
            }
        ],
    )
    spec = _json(
        tmp_path / "spec.json",
        {
            "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
            "tasks": {
                task: {
                    "mode": "partial_component_partition_v1",
                    "primary_finals": [primary],
                    "rescue_matches": rescue,
                    "typed_abstentions": typed,
                    "unresolved": unresolved,
                    "remaining_action": "finish",
                },
                "other": {
                    "mode": "no_canonical_final",
                    "remaining_action": "run production",
                },
            },
        },
    )
    return spec, rescue


def test_audits_every_corpus_without_promoting_partial_components(tmp_path):
    spec, _rescue = _fixture(tmp_path)
    report = audit_current_coverage(spec)
    assert report["inventory_complete"] is True
    assert report["release_complete"] is False
    assert report["summary"]["resolved_count"] == 4
    assert report["summary"]["match_count"] == 2
    assert report["summary"]["typed_nonmatch_count"] == 1
    assert report["summary"]["no_explicit_criterion_count"] == 1
    assert report["summary"]["noise_count"] == 1
    assert report["summary"]["typed_decision_counts"] == {
        "NOISE": 1,
        "NO_EXPLICIT_CRITERION": 1,
    }
    assert report["summary"]["unresolved_count"] == 2
    assert report["corpora"]["c"]["canonical_final_ready"] is False


def test_component_hash_drift_fails_closed(tmp_path):
    spec, rescue = _fixture(tmp_path)
    Path(rescue["path"]).write_text("{}\n")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        audit_current_coverage(spec)


def test_append_only_wrapper_binds_superseded_ledger(tmp_path):
    base_spec, _rescue = _fixture(tmp_path)
    predecessor = _json(tmp_path / "v1.json", {"schema_version": "v1"})
    wrapper = _json(
        tmp_path / "v2-policy.json",
        {
            "base_spec": {
                "path": str(base_spec),
                "sha256": sha256_file(base_spec),
            },
            "supersedes": {
                "path": str(predecessor),
                "sha256": sha256_file(predecessor),
                "status": "SUPERSEDED_TERMINOLOGY_ERROR",
                "reason": "NO_EXPLICIT_CRITERION was incorrectly called noise",
            },
        },
    )
    report = audit_current_coverage(wrapper)
    assert report["base_spec"]["sha256"] == sha256_file(base_spec)
    assert report["supersedes"]["status"] == "SUPERSEDED_TERMINOLOGY_ERROR"
    assert report["summary"]["no_explicit_criterion_count"] == 1
    assert report["summary"]["noise_count"] == 1
