from __future__ import annotations

from copy import deepcopy

import pytest

from methods.metric_seam.family_scale.decomposition_stability import (
    DecompositionSchemaError,
    REPORT_SCHEMA,
    SCHEMA,
    load_submission,
)


METRIC = {
    "name": "Novelty and public-disclosure bars",
    "text": (
        "A claimed invention must be new over a single enabling prior-art "
        "disclosure; public disclosures can defeat novelty."
    ),
}


def _relation(op: str, witness: str, relation: str) -> dict[str, str]:
    return {"op_class": op, "witness_kind": witness, "relation": relation}


COMMON = _relation(
    "computation",
    "Critical Date Comparison",
    "Determine whether the disclosure precedes the critical date.",
)
A_ONLY = _relation(
    "evidence", "claim limitation set", "extract every required claim limitation"
)
B_ONLY = _relation(
    "evidence", "public availability record", "establish public availability date"
)
C_ONLY = _relation(
    "individuation", "single prior disclosure", "bind the anticipation reference"
)
AB = _relation(
    "computation", "enablement determination", "test whether the disclosure enables"
)
BC = _relation(
    "computation", "limitation mapping", "map every limitation to one disclosure"
)


def _submission(fleets: list[dict]) -> dict:
    return {"schema": SCHEMA, "metric": deepcopy(METRIC), "fleets": fleets}


def _fleet(fleet_id: str, relations: list[dict]) -> dict:
    return {"fleet_id": fleet_id, "relations": relations}


def test_two_fleet_freeze_and_exact_stability_are_deterministic() -> None:
    submission = _submission(
        [_fleet("Fleet A", [COMMON, A_ONLY]), _fleet("fleet b", [COMMON, B_ONLY])]
    )
    first = load_submission(submission)
    second = load_submission(deepcopy(submission))
    assert first == second
    assert first["schema"] == REPORT_SCHEMA
    assert first["input_scope"] == "metric_text_only"
    assert first["freeze"]["corpus_accessed"] is False
    assert first["freeze"]["scores_or_labels_accessed"] is False
    stability = first["stability"]
    assert stability["observed_relation_union"] == 3
    assert stability["all_fleet_exact_intersection"] == 1
    assert stability["capture_frequency"] == {"1": 2, "2": 1}
    pair = stability["pairwise"][0]
    assert pair["exact_jaccard"]["numerator"] == 1
    assert pair["exact_jaccard"]["denominator"] == 3
    assert stability["capture_recapture"]["method"] == "two_sample_chapman_descriptive"


def test_three_fleet_capture_frequency_and_chao2() -> None:
    report = load_submission(
        _submission(
            [
                _fleet("a", [COMMON, AB, A_ONLY]),
                _fleet("b", [COMMON, AB, BC, B_ONLY]),
                _fleet("c", [COMMON, BC, C_ONLY]),
            ]
        )
    )
    stability = report["stability"]
    assert stability["observed_relation_union"] == 6
    assert stability["capture_frequency"] == {"1": 3, "2": 2, "3": 1}
    assert stability["recaptured_relation_count"] == 3
    estimate = stability["capture_recapture"]
    assert estimate["identified"] is True
    assert estimate["method"] == "three_fleet_incidence_chao2_descriptive"
    assert estimate["estimate"]["numerator"] == 15
    assert estimate["estimate"]["denominator"] == 2


def test_phrase_normalization_matches_case_spacing_hyphen_and_punctuation() -> None:
    variant = _relation(
        "computation",
        "  CRITICAL   date comparison ",
        "Determine whether the disclosure precedes the critical date",
    )
    report = load_submission(
        _submission([_fleet("a", [COMMON]), _fleet("b", [variant])])
    )
    assert report["stability"]["observed_relation_union"] == 1
    assert report["stability"]["all_fleet_exact_intersection"] == 1


@pytest.mark.parametrize("count", [1, 4])
def test_requires_two_or_three_fleets(count: int) -> None:
    fleets = [_fleet(f"f{index}", [COMMON]) for index in range(count)]
    with pytest.raises(DecompositionSchemaError, match="two or three"):
        load_submission(_submission(fleets))


def test_rejects_duplicate_relations_and_fleet_ids() -> None:
    with pytest.raises(DecompositionSchemaError, match="duplicate after"):
        load_submission(
            _submission([_fleet("a", [COMMON, deepcopy(COMMON)]), _fleet("b", [B_ONLY])])
        )
    with pytest.raises(DecompositionSchemaError, match="fleet_id values"):
        load_submission(
            _submission([_fleet("same", [COMMON]), _fleet(" SAME ", [B_ONLY])])
        )


def test_closed_world_schema_rejects_corpus_or_score_fields() -> None:
    submission = _submission([_fleet("a", [COMMON]), _fleet("b", [B_ONLY])])
    submission["corpus_path"] = "sealed/items.json"
    with pytest.raises(DecompositionSchemaError, match="extra=.*corpus_path"):
        load_submission(submission)

    submission = _submission([_fleet("a", [COMMON]), _fleet("b", [B_ONLY])])
    submission["metric"]["scores"] = [1, 0]
    with pytest.raises(DecompositionSchemaError, match="extra=.*scores"):
        load_submission(submission)


def test_relation_schema_is_exact_and_op_class_is_closed() -> None:
    bad = deepcopy(COMMON)
    bad["confidence"] = 1
    with pytest.raises(DecompositionSchemaError, match="confidence"):
        load_submission(_submission([_fleet("a", [bad]), _fleet("b", [B_ONLY])]))

    bad = deepcopy(COMMON)
    bad["op_class"] = "judgment"
    with pytest.raises(DecompositionSchemaError, match="expected one of"):
        load_submission(_submission([_fleet("a", [bad]), _fleet("b", [B_ONLY])]))
