from __future__ import annotations

import sys
from pathlib import Path

from scripts.tools.silver_match_v3.common import read_jsonl, write_jsonl
from scripts.tools.silver_match_v3.finalize_independent_relabels import main as finalize
from scripts.tools.silver_match_v3.subset_jsonl_by_reference import main as subset


def _label(uid: str, metric: str = "m0") -> dict:
    return {
        "norm_uid": uid,
        "task": "math-stackexchange",
        "decision": "MATCH",
        "metric_id": metric,
        "confidence": "high",
        "current_bank_source_sha256": "bank-sha",
    }


def test_two_pass_consensus_can_be_locked_pending_blind_audit(
    tmp_path: Path, monkeypatch
) -> None:
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    write_jsonl(first, [_label("u0"), _label("u1")])
    write_jsonl(second, [_label("u0"), _label("u1", "m1")])
    output, report = tmp_path / "consensus.jsonl", tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_independent_relabels",
            "--first",
            str(first),
            "--second",
            str(second),
            "--output",
            str(output),
            "--report",
            str(report),
            "--policy",
            "both_high",
            "--pending-blind-audit",
        ],
    )
    finalize()
    rows = list(read_jsonl(output))
    assert len(rows) == 1
    assert rows[0]["training_eligible"] is False
    assert rows[0]["training_blocked_pending_blind_audit"] is True


def test_reference_subset_preserves_reference_order(tmp_path: Path, monkeypatch) -> None:
    source, reference = tmp_path / "source.jsonl", tmp_path / "reference.jsonl"
    write_jsonl(source, [{"norm_uid": "u0"}, {"norm_uid": "u1"}, {"norm_uid": "u2"}])
    write_jsonl(reference, [{"norm_uid": "u2"}, {"norm_uid": "u0"}])
    output = tmp_path / "subset.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "subset_jsonl_by_reference",
            "--input",
            str(source),
            "--reference",
            str(reference),
            "--output",
            str(output),
        ],
    )
    subset()
    assert [row["norm_uid"] for row in read_jsonl(output)] == ["u2", "u0"]
