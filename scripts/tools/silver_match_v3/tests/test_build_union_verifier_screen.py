from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.tools.silver_match_v3.build_union_verifier_screen import build
from scripts.tools.silver_match_v3.common import read_jsonl, write_jsonl


def _verification(path: Path, accepted: set[str], order: str) -> None:
    rows = []
    for uid in ("u1", "u2"):
        rows.append(
            {
                "norm_uid": uid,
                "task": "press-releases",
                "order_mode": order,
                "primary_metric_id": "a1",
                "candidate_bank_source_sha256": "bank",
                "decision": "CONFIRM_MATCH" if uid in accepted else "REJECT_MATCH",
                "metric_id": "a1" if uid in accepted else None,
                "confidence": "high",
                "parse_error": None,
            }
        )
    write_jsonl(path, rows)


def test_union_keeps_rows_confirmed_by_either_strict_pair(tmp_path: Path) -> None:
    primary = tmp_path / "primary.jsonl"
    write_jsonl(
        primary,
        [
            {
                "norm_uid": uid,
                "task": "press-releases",
                "decision": "MATCH",
                "metric_id": "a1",
                "candidate_bank_source_sha256": "bank",
            }
            for uid in ("u1", "u2")
        ],
    )
    paths = []
    for name, accepted in (("left", {"u1"}), ("right", {"u2"})):
        original, hashed = tmp_path / f"{name}.original.jsonl", tmp_path / f"{name}.hashed.jsonl"
        _verification(original, accepted, "original")
        _verification(hashed, accepted, "hashed")
        paths.append((name, original, hashed))
    report = build(
        task="press-releases",
        primary_path=primary,
        variants=paths,
        output_root=tmp_path / "screen",
    )
    assert report["selected_count"] == 2
    assert {row["norm_uid"] for row in read_jsonl(tmp_path / "screen/screened_primary.jsonl")} == {"u1", "u2"}
    assert json.loads((tmp_path / "screen/SCREEN_FREEZE.json").read_text())["contracts"]["truth_or_targets_read"] is False


def test_union_rejects_incomplete_variant_coverage(tmp_path: Path) -> None:
    primary = tmp_path / "primary.jsonl"
    write_jsonl(primary, [{"norm_uid": "u1", "task": "press-releases", "decision": "MATCH", "metric_id": "a1"}])
    original, hashed = tmp_path / "original.jsonl", tmp_path / "hashed.jsonl"
    write_jsonl(original, [])
    write_jsonl(hashed, [])
    with pytest.raises(ValueError):
        build(
            task="press-releases",
            primary_path=primary,
            variants=[("a", original, hashed), ("b", original, hashed)],
            output_root=tmp_path / "screen",
        )
