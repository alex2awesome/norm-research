from __future__ import annotations

import json
from pathlib import Path

import pytest

from methods.metric_seam.pilot import code_depth_retrospective as depth


def test_full_active_panel_point_estimates_without_resampling() -> None:
    result = depth.evaluate(run_inference=False)
    assert result["summary"] == {
        "active_criteria": 18,
        "criteria_with_deep_program": 18,
        "criteria_with_train_selected_shallow_comparator": 15,
        "inferentially_eligible": 4,
        "bh_family_size": 0,
        "multiplicity_controlled_improvements": 0,
        "multiplicity_controlled_improvement_ids": [],
    }
    rows = {row["criterion_id"]: row for row in result["criteria"]}
    a104 = rows["a104"]
    assert a104["train_shallow_selection"]["selected"] == "a104_v0_keyword"
    comparison = a104["heldout_comparison"]
    assert comparison["n_paired"] == 97
    assert comparison["rho_deep"] == pytest.approx(0.649794037210992)
    assert comparison["rho_shallow"] == pytest.approx(0.5089068945741408)
    assert comparison["inferential_eligible"] is True

    assert rows["a407"]["status"] == "shallow_comparator_unavailable"
    assert rows["a155"]["heldout_comparison"]["inferential_eligible"] is False
    assert "paired_support_below_minimum" in rows["a155"]["heldout_comparison"][
        "ineligibility_reasons"
    ]


def test_item_projection_does_not_require_or_read_outcome(tmp_path: Path) -> None:
    path = tmp_path / "items.json"
    path.write_text(
        json.dumps(
            [
                {"datapoint_id": "b", "judgement": object().__class__.__name__},
                {"datapoint_id": "a", "judgement": {"arbitrary": "ignored"}},
            ]
        ),
        encoding="utf-8",
    )
    assert depth._item_ids_without_outcomes(path) == ["b", "a"]


def test_duplicate_reference_rows_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "reference.jsonl"
    row = {
        "aspect_id": "a1",
        "datapoint_id": "x",
        "channel": "pass1",
        "score": 4,
    }
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(depth.RetrospectiveError, match="duplicate reference row"):
        depth._load_reference_rows(path)
