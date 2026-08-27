import argparse
import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.build_humor_ce_collection_inventory import (
    BANK_SOURCE_SHA256,
    build,
    canonical_source_group,
    priority_strata,
)


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_priority_order_and_group_canonicalization() -> None:
    row = {"task": "humor", "corpus": "humor_multi", "source_id": "s1", "norm_uid": "u1"}
    assert canonical_source_group(row) == "humor:humor_multi:source:s1"
    assert priority_strata(
        {"decision": "BANK_GAP", "confidence": "low"}, ["a1", "a2"], {"a2"}
    ) == [
        "legacy_nonmatch_re_adjudication",
        "sonnet_low_or_null_confidence",
        "retriever_top1_disagreement",
        "uncovered_leaf_proxy",
    ]


def test_build_inventory_binds_inputs(tmp_path: Path) -> None:
    norms = tmp_path / "norms.jsonl"
    sonnet = tmp_path / "sonnet.jsonl"
    c1, c2 = tmp_path / "c1.jsonl", tmp_path / "c2.jsonl"
    norm_rows = [
        {
            "schema_version": "silver-match-v3.0",
            "task": "humor",
            "corpus": "humor_multi",
            "source_id": f"s{i}",
            "norm_uid": f"u{i}",
        }
        for i in range(2)
    ]
    _write(norms, norm_rows)
    _write(
        sonnet,
        [
            {
                "task": "humor",
                "corpus": "humor_multi",
                "norm_uid": "u0",
                "decision": "NOISE",
                "confidence": None,
                "current_bank_source_sha256": BANK_SOURCE_SHA256,
            }
        ],
    )
    _write(
        c1,
        [
            {
                "norm_uid": f"u{i}",
                "bank_source_sha256": BANK_SOURCE_SHA256,
                "candidates": [{"metric_id": "a1"}],
            }
            for i in range(2)
        ],
    )
    _write(
        c2,
        [
            {
                "norm_uid": f"u{i}",
                "bank_source_sha256": BANK_SOURCE_SHA256,
                "candidates": [{"metric_id": "a2" if i == 0 else "a1"}],
            }
            for i in range(2)
        ],
    )
    rows, report = build(
        argparse.Namespace(
            norms=str(norms),
            sonnet=str(sonnet),
            candidates=[str(c1), str(c2)],
            uncovered_metric_id=["a2"],
            expected_norms=2,
        )
    )
    assert len(rows) == 2
    assert rows[0]["primary_priority_stratum"] == "legacy_nonmatch_re_adjudication"
    assert rows[1]["primary_priority_stratum"] == "natural_background"
    assert report["status"] == "READY_FOR_ROLE_FREEZE_NOT_LABEL_TRUTH"


def test_candidate_bank_drift_fails(tmp_path: Path) -> None:
    norms = tmp_path / "norms.jsonl"
    sonnet = tmp_path / "sonnet.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    _write(
        norms,
        [{"task": "humor", "corpus": "humor_multi", "source_id": "s", "norm_uid": "u"}],
    )
    _write(sonnet, [])
    _write(
        candidates,
        [{"norm_uid": "u", "bank_source_sha256": "wrong", "candidates": [{"metric_id": "a1"}]}],
    )
    with pytest.raises(ValueError, match="bank drift"):
        build(
            argparse.Namespace(
                norms=str(norms),
                sonnet=str(sonnet),
                candidates=[str(candidates)],
                uncovered_metric_id=[],
                expected_norms=1,
            )
        )
