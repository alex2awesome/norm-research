from __future__ import annotations

import json

import pytest

from methods.metric_seam.pilot import evaluate_code_review_a407_historical_v1 as eval_a407


def test_two_pass_pooling_uses_numeric_intersection_only() -> None:
    rows = [
        {
            "aspect_id": "a407",
            "channel": "pass1",
            "datapoint_id": "a",
            "score": 8,
        },
        {
            "aspect_id": "a407",
            "channel": "pass2",
            "datapoint_id": "a",
            "score": 6,
        },
        {
            "aspect_id": "a407",
            "channel": "pass1",
            "datapoint_id": "b",
            "score": "NA",
        },
        {
            "aspect_id": "a407",
            "channel": "pass2",
            "datapoint_id": "b",
            "score": 5,
        },
        {
            "aspect_id": "other",
            "channel": "pass1",
            "datapoint_id": "a",
            "score": 0,
        },
    ]

    reference = eval_a407.load_two_pass_prompt_reference(rows)

    assert reference["pass1"] == {"a": 8}
    assert reference["pass2"] == {"a": 6, "b": 5}
    assert reference["composite"] == {"a": 0.7}


def test_outcome_label_target_is_explicitly_refused() -> None:
    eval_a407.require_prompt_reference_target(eval_a407.REFERENCE_TARGET)

    for target in ("judgement", "items.json.judgement", "merge_outcome"):
        with pytest.raises(eval_a407.OutcomeLabelTargetError):
            eval_a407.require_prompt_reference_target(target)


def test_seeded_mapping_excludes_only_raw_sanitizer_change() -> None:
    items = [
        {"datapoint_id": f"d{index}", "ctext": f"row-{index}", "judgement": 99}
        for index in range(5)
    ]
    heldout_ids = eval_a407.deterministic_heldout_ids(
        [row["datapoint_id"] for row in items], train_count=2, split_seed=7
    )
    changed_id = heldout_ids[1]
    for row in items:
        if row["datapoint_id"] == changed_id:
            row["ctext"] = "TOKEN=secret"

    def projector(text: str) -> str:
        return text.replace("TOKEN=secret", "TOKEN=[redacted]")

    by_id = {row["datapoint_id"]: row for row in items}
    bundle = [
        {
            "item_key": f"heldout_{index:04d}",
            "ctext": projector(by_id[identifier]["ctext"]),
        }
        for index, identifier in enumerate(heldout_ids, 1)
    ]

    mapping = eval_a407.reconstruct_alias_mapping(
        items,
        bundle,
        train_count=2,
        split_seed=7,
        projector=projector,
    )

    assert len(mapping) == 3
    assert sum(row["sanitized_exact"] for row in mapping) == 3
    assert sum(row["raw_exact"] for row in mapping) == 2
    changed = [row for row in mapping if not row["raw_exact"]]
    assert [row["datapoint_id"] for row in changed] == [changed_id]


def test_neutral_candidate_score_remains_noncoverage() -> None:
    candidate = {"score": 0.5}
    noncovered = {
        "declaration_coverage": False,
        "structural_partial_aggregate": None,
    }
    covered = {
        "declaration_coverage": True,
        "structural_partial_aggregate": 0.73,
    }

    assert eval_a407.code_structural_value(candidate, noncovered) is None
    assert eval_a407.code_structural_value(candidate, covered) == 0.73

    with pytest.raises(ValueError):
        eval_a407.code_structural_value(
            candidate,
            {
                "declaration_coverage": False,
                "structural_partial_aggregate": 0.5,
            },
        )


def test_positive_event_and_absence_licensing_are_distinct() -> None:
    assert (
        eval_a407.license_event_claim(2, strict_complete=False)
        == "positive_event_witness"
    )
    assert (
        eval_a407.license_event_claim(0, strict_complete=False)
        == "no_event_detected_unlicensed"
    )
    assert (
        eval_a407.license_event_claim(0, strict_complete=True)
        == "negative_support"
    )
    assert (
        eval_a407.license_event_claim(1, strict_complete=True)
        == "positive_event_witness"
    )


def test_metric_preconditions_leave_undefined_correlations_null() -> None:
    too_small = eval_a407.comparison_metrics([0.0, 1.0], [0.0, 1.0])
    assert too_small["spearman"] is None
    assert too_small["pearson"] is None
    assert too_small["mean_absolute_difference"] == 0.0

    constant = eval_a407.comparison_metrics([0.5, 0.5, 0.5], [0.0, 0.5, 1.0])
    assert constant["spearman"] is None
    assert constant["pearson"] is None

    varying = eval_a407.comparison_metrics([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    assert varying["spearman"] == -1.0
    assert varying["pearson"] == -1.0


def test_emit_and_check_reproduce_canonical_artifacts(tmp_path) -> None:
    emitted = tmp_path / "a407-eval"

    eval_a407.emit_artifacts(emitted)
    eval_a407.check_artifacts(emitted)

    assert (emitted / "evaluation.json").read_bytes() == (
        eval_a407.CANONICAL_OUTPUT / "evaluation.json"
    ).read_bytes()
    assert (emitted / "REPORT.md").read_bytes() == (
        eval_a407.CANONICAL_OUTPUT / "REPORT.md"
    ).read_bytes()
    evaluation = json.loads((emitted / "evaluation.json").read_text())
    assert evaluation["primary_code_vs_historical_composite"]["available_pair_count"] == 74
    assert evaluation["bindings_and_seal_order"]["historical_item_source"][
        "items_judgement_used_as_reference"
    ] is False
