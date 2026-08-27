from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.tools.silver_match_v3.common import read_jsonl, sha256_file, write_jsonl
from scripts.tools.silver_match_v3.finalize_three_pass_consensus import (
    _stratified_sample,
    main as finalize,
)
from scripts.tools.silver_match_v3.promote_three_pass_consensus import main as promote


def _row(uid: str, stratum: str, metric_id: str) -> dict:
    return {
        "norm_uid": uid,
        "task": "peer-review",
        "split_group": f"paper:{uid}",
        "boundary_stratum": stratum,
        "decision": "MATCH",
        "metric_id": metric_id,
        "current_bank_source_sha256": "bank-sha",
        "training_eligible": False,
    }


def test_stratified_sample_is_deterministic_balanced_and_leaf_diverse() -> None:
    rows = [
        _row(f"u-{stratum}-{index}", stratum, f"m{index % 9}")
        for stratum in ("evidence", "methods", "novelty", "presentation")
        for index in range(30)
    ]
    first = _stratified_sample(rows, audit_size=80, seed=107)
    second = _stratified_sample(list(reversed(rows)), audit_size=80, seed=107)
    assert [row["norm_uid"] for row in first] == [row["norm_uid"] for row in second]
    assert len({row["norm_uid"] for row in first}) == 80
    assert {
        stratum: sum(row["boundary_stratum"] == stratum for row in first)
        for stratum in ("evidence", "methods", "novelty", "presentation")
    } == {"evidence": 20, "methods": 20, "novelty": 20, "presentation": 20}
    assert len({row["metric_id"] for row in first}) == 9


def test_finalize_builds_disjoint_truth_hidden_audit_pack(
    tmp_path: Path, monkeypatch
) -> None:
    pack = tmp_path / "pack"
    items = [
        {
            "schema_version": "silver-match-v3.0",
            "norm_uid": f"u-{index}",
            "corpus": "reviews",
            "task": "peer-review",
            "row": index,
            "split_group": f"paper:{index}",
            "split": "train",
            "boundary_stratum": f"s{index % 4}",
            "norm": f"norm {index}",
            "context": f"context {index}",
        }
        for index in range(70)
    ]
    bank = {
        "task": "peer-review",
        "source_sha256": "bank-sha",
        "metrics": [
            {"metric_id": f"m{index}", "name": f"Metric {index}"} for index in range(7)
        ],
    }
    items_path, bank_path = pack / "items.jsonl", pack / "bank.json"
    write_jsonl(items_path, items)
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.write_text(json.dumps(bank))
    validation = {
        "task": "peer-review",
        "bank_source_sha256": "bank-sha",
        "outputs": {
            "items": {"sha256": sha256_file(items_path)},
            "bank": {"sha256": sha256_file(bank_path)},
        },
    }
    (pack / "validation.json").write_text(json.dumps(validation))
    pass_paths = []
    for pass_index in range(3):
        path = tmp_path / f"pass-{pass_index}.jsonl"
        write_jsonl(
            path,
            [
                {
                    **_row(item["norm_uid"], item["boundary_stratum"], f"m{index % 7}"),
                    "schema_version": "silver-match-v3.0",
                    "corpus": "reviews",
                    "row": index,
                    "split_group": item["split_group"],
                    "split": "train",
                    "confidence": "high",
                    "reason": f"independent reason {pass_index}",
                }
                for index, item in enumerate(items)
            ],
        )
        pass_paths.append(path)
    output_root = tmp_path / "finalized"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "finalize_three_pass_consensus",
            "--pack-root",
            str(pack),
            "--first",
            str(pass_paths[0]),
            "--second",
            str(pass_paths[1]),
            "--third",
            str(pass_paths[2]),
            "--output-root",
            str(output_root),
            "--audit-size",
            "60",
        ],
    )
    finalize()
    audit = list(read_jsonl(output_root / "blind_audit.proposals.hidden.jsonl"))
    train = list(read_jsonl(output_root / "training_candidates.pending_audit.jsonl"))
    audit_pack = list(read_jsonl(output_root / "blind_audit_pack" / "items.jsonl"))
    report = json.loads((output_root / "consensus.report.json").read_text())
    assert len(audit) == len(audit_pack) == 60
    assert len(train) == 10
    assert all(row["audit_design_weight"] > 0 for row in audit)
    assert not ({row["norm_uid"] for row in audit} & {row["norm_uid"] for row in train})
    assert not (
        {row["split_group"] for row in audit} & {row["split_group"] for row in train}
    )
    assert report["audit_permanently_excluded_from_gradients"] is True
    assert (
        json.loads((output_root / "blind_audit_pack" / "validation.json").read_text())[
            "truth_hidden"
        ]
        is True
    )


def test_promotion_uses_audit_as_gate_but_never_as_training_data(
    tmp_path: Path, monkeypatch
) -> None:
    train = [_row(f"train-{index}", "methods", f"m{index % 4}") for index in range(10)]
    proposals = [
        _row(f"audit-{index}", "evidence", f"m{index % 6}") for index in range(60)
    ]
    audit = []
    for index, proposal in enumerate(proposals):
        audit.append(
            {
                **proposal,
                "decision": "MATCH" if index < 57 else "BANK_GAP_RELATED",
                "metric_id": proposal["metric_id"] if index < 57 else None,
                "confidence": "medium" if index == 0 else "high",
            }
        )
    inputs = {
        "training": tmp_path / "training.jsonl",
        "proposals": tmp_path / "proposals.jsonl",
        "audit": tmp_path / "audit.jsonl",
    }
    write_jsonl(inputs["training"], train)
    write_jsonl(inputs["proposals"], proposals)
    write_jsonl(inputs["audit"], audit)
    output, report = tmp_path / "promoted.jsonl", tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_three_pass_consensus",
            "--training-candidates",
            str(inputs["training"]),
            "--audit-proposals",
            str(inputs["proposals"]),
            "--audit-labels",
            str(inputs["audit"]),
            "--output",
            str(output),
            "--report",
            str(report),
        ],
    )
    promote()
    promoted = list(read_jsonl(output))
    result = json.loads(report.read_text())
    assert result["promotion_cleared"] is True
    assert result["audit_support"] == 60
    assert result["audit_exact"] == 57
    assert result["audit_high_confidence_exact"] == 56
    assert result["audit_items_promoted"] == 0
    assert {row["norm_uid"] for row in promoted} == {row["norm_uid"] for row in train}
    assert not (
        {row["norm_uid"] for row in promoted} & {row["norm_uid"] for row in audit}
    )
    assert all(row["training_eligible"] for row in promoted)


def test_promotion_fails_closed_when_wilson_gate_is_missed(
    tmp_path: Path, monkeypatch
) -> None:
    train = [_row(f"train-{index}", "methods", "m0") for index in range(3)]
    proposals = [_row(f"audit-{index}", "evidence", "m0") for index in range(60)]
    audit = [
        {
            **proposal,
            "decision": "MATCH" if index < 54 else "NO_CRITERION",
            "metric_id": "m0" if index < 54 else None,
            "confidence": "high",
        }
        for index, proposal in enumerate(proposals)
    ]
    training_path, proposal_path, audit_path = (
        tmp_path / "training.jsonl",
        tmp_path / "proposals.jsonl",
        tmp_path / "audit.jsonl",
    )
    write_jsonl(training_path, train)
    write_jsonl(proposal_path, proposals)
    write_jsonl(audit_path, audit)
    output, report = tmp_path / "promoted.jsonl", tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_three_pass_consensus",
            "--training-candidates",
            str(training_path),
            "--audit-proposals",
            str(proposal_path),
            "--audit-labels",
            str(audit_path),
            "--output",
            str(output),
            "--report",
            str(report),
        ],
    )
    promote()
    assert list(read_jsonl(output)) == []
    result = json.loads(report.read_text())
    assert result["audit_exact_rate"] == 0.9
    assert result["audit_exact_wilson_95"][0] < 0.8
    assert result["promotion_cleared"] is False


def test_promotion_fails_closed_on_design_weighted_precision(
    tmp_path: Path, monkeypatch
) -> None:
    train = [_row(f"train-{index}", "methods", "m0") for index in range(3)]
    proposals = [
        {
            **_row(f"audit-{index}", "evidence", "m0"),
            "audit_design_weight": 10.0 if index >= 57 else 1.0,
        }
        for index in range(60)
    ]
    audit = [
        {
            **proposal,
            "decision": "MATCH" if index < 57 else "NO_CANDIDATE_FITS",
            "metric_id": "m0" if index < 57 else None,
            "confidence": "high",
        }
        for index, proposal in enumerate(proposals)
    ]
    training_path, proposal_path, audit_path = (
        tmp_path / "training.jsonl",
        tmp_path / "proposals.jsonl",
        tmp_path / "audit.jsonl",
    )
    write_jsonl(training_path, train)
    write_jsonl(proposal_path, proposals)
    write_jsonl(audit_path, audit)
    output, report = tmp_path / "promoted.jsonl", tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_three_pass_consensus",
            "--training-candidates",
            str(training_path),
            "--audit-proposals",
            str(proposal_path),
            "--audit-labels",
            str(audit_path),
            "--output",
            str(output),
            "--report",
            str(report),
        ],
    )
    promote()
    result = json.loads(report.read_text())
    assert result["audit_exact_rate"] == 0.95
    assert result["audit_design_weighted_exact_rate"] < 0.7
    assert result["promotion_cleared"] is False
    assert list(read_jsonl(output)) == []
