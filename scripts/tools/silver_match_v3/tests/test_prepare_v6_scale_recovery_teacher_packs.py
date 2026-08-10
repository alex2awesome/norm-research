from __future__ import annotations

import copy

import pytest

from scripts.tools.silver_match_v3.prepare_v6_scale_recovery_teacher_packs import (
    _teacher_schema,
    _visible_item,
    rank_candidates,
    validate_teacher_label,
)


def _row(
    index: int,
    *,
    reason: str = "rubric_outside_current_bank_coverage",
    stratum: str | None = None,
    group: str | None = None,
) -> dict[str, object]:
    return {
        "task": "peer-review",
        "norm_uid": f"{index:064x}",
        "query": f"Criterion {index}",
        "source_group": group or f"document-{index % 3}",
        "recovery_reason": reason,
        "supporting_pair_count": index + 1,
        "hidden_balance_metric_ids": [] if stratum is None else [stratum],
    }


def test_rank_candidates_is_deterministic_complete_and_coverage_first() -> None:
    rows = [
        _row(0, stratum="a0"),
        _row(1, stratum="a1"),
        _row(2, stratum="a2"),
        _row(3, stratum=None),
        _row(4, stratum="a0"),
        _row(5, reason="ambiguous_rubric_to_current_bank_mapping", stratum="a3"),
        _row(6, reason="ambiguous_rubric_to_current_bank_mapping", stratum="a4"),
    ]
    first = rank_candidates(copy.deepcopy(rows), task="peer-review")
    second = rank_candidates(copy.deepcopy(list(reversed(rows))), task="peer-review")
    assert [row["norm_uid"] for row in first] == [row["norm_uid"] for row in second]
    assert {row["norm_uid"] for row in first} == {row["norm_uid"] for row in rows}
    assert [row["budget_rank"] for row in first] == list(range(1, len(rows) + 1))
    assert len({row["balance_stratum"] for row in first[:5]} - {"__UNSTRATIFIED__"}) >= 4
    assert {row["recovery_reason"] for row in first[:5]} == {
        "rubric_outside_current_bank_coverage",
        "ambiguous_rubric_to_current_bank_mapping",
    }


def test_teacher_visible_item_hides_every_sampling_and_historical_hint() -> None:
    private = {
        **_row(1, stratum="a7"),
        "rubric_key": "peer-review::raw::document::1",
        "supporting_v6_score_counts": {"2": 4},
        "balance_stratum": "a7",
        "budget_rank": 1,
    }
    visible = _visible_item(private, bank_hash="f" * 64, bank_count=88)
    assert set(visible) == {
        "schema_version",
        "task",
        "norm_uid",
        "query",
        "current_bank_source_sha256",
        "full_bank_metric_count",
        "full_bank_required",
        "truth_hidden",
    }
    assert not (
        {
            "rubric_key",
            "source_group",
            "recovery_reason",
            "supporting_pair_count",
            "supporting_v6_score_counts",
            "hidden_balance_metric_ids",
            "balance_stratum",
            "budget_rank",
        }
        & set(visible)
    )


def test_teacher_label_validator_accepts_exact_family_and_typed_abstention() -> None:
    valid_ids = {f"a{index}" for index in range(8)}
    base = {
        "norm_uid": "0" * 64,
        "confidence": "high",
        "reason": "The criterion directly distinguishes the selected construct.",
    }
    validate_teacher_label(
        {
            **base,
            "decision": "EXACT",
            "primary_metric_id": "a0",
            "pair_labels": [
                {"metric_id": "a0", "relation": "EXACT"},
                {"metric_id": "a1", "relation": "FAMILY"},
                {"metric_id": "a2", "relation": "REJECT"},
                {"metric_id": "a3", "relation": "REJECT"},
            ],
        },
        valid_metric_ids=valid_ids,
    )
    validate_teacher_label(
        {
            **base,
            "decision": "FAMILY",
            "primary_metric_id": None,
            "pair_labels": [
                {"metric_id": "a1", "relation": "FAMILY"},
                {"metric_id": "a2", "relation": "FAMILY"},
                {"metric_id": "a3", "relation": "REJECT"},
                {"metric_id": "a4", "relation": "REJECT"},
            ],
        },
        valid_metric_ids=valid_ids,
    )
    validate_teacher_label(
        {
            **base,
            "decision": "NO_CANDIDATE_FITS",
            "primary_metric_id": None,
            "pair_labels": [{"metric_id": "a6", "relation": "REJECT"}],
        },
        valid_metric_ids=valid_ids,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda label: label.update(primary_metric_id=None),
        lambda label: label["pair_labels"].pop(),
        lambda label: label["pair_labels"].append(
            {"metric_id": "a0", "relation": "REJECT"}
        ),
        lambda label: label["pair_labels"].append(
            {"metric_id": "a99", "relation": "REJECT"}
        ),
    ],
)
def test_teacher_label_validator_fails_closed(mutation) -> None:
    label = {
        "norm_uid": "0" * 64,
        "decision": "EXACT",
        "primary_metric_id": "a0",
        "pair_labels": [
            {"metric_id": "a0", "relation": "EXACT"},
            {"metric_id": "a1", "relation": "REJECT"},
            {"metric_id": "a2", "relation": "REJECT"},
        ],
        "confidence": "high",
        "reason": "The exact construct wins over the hard negative alternatives.",
    }
    mutation(label)
    with pytest.raises(ValueError):
        validate_teacher_label(label, valid_metric_ids={"a0", "a1", "a2"})


def test_schema_chunk_limit_is_bound_to_requested_chunk_size() -> None:
    assert _teacher_schema(7)["properties"]["labels"]["maxItems"] == 7

