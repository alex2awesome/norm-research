from __future__ import annotations

import copy

import pytest

from methods.metric_seam.hierarchy_items import (
    build_task_items,
    validate_task_items,
)


def test_source_only_projection_is_deterministic_disjoint_and_label_free():
    texts = [f"document {index} " + ("x" * 5000) for index in range(100)]
    source = {"kind": "test", "path": "fixture", "text_column": "text"}
    first = build_task_items(
        "math-stackexchange", n=80, source_texts=texts, source_record=source
    )
    second = build_task_items(
        "math-stackexchange", n=80, source_texts=list(reversed(texts)), source_record=source
    )
    assert first == second
    manifest, train, heldout = first
    validate_task_items(manifest, train, heldout)
    assert len(train) == len(heldout) == 40
    assert all(set(row) == {"item_key", "ctext"} for row in train + heldout)
    assert manifest["selection"]["outcome_or_reference_values_used"] is False
    assert manifest["policy"]["external_supervision_used"] is False
    assert max(map(lambda row: len(row["ctext"]), train + heldout)) <= \
        manifest["representation"]["max_chars"]


def test_projection_deduplicates_after_prompt_length_truncation():
    common = "a" * 4000
    texts = [common + str(index) for index in range(100)]
    with pytest.raises(ValueError, match="only 1 unique projected"):
        build_task_items("math-stackexchange", n=80, source_texts=texts)


def test_validator_rejects_outcome_field_or_split_overlap():
    manifest, train, heldout = build_task_items(
        "humor", n=40, source_texts=[f"joke {index}" for index in range(60)]
    )
    leaked = copy.deepcopy(train)
    leaked[0]["judgement"] = 1
    with pytest.raises(ValueError, match="outside item_key/ctext"):
        validate_task_items(manifest, leaked, heldout)

    duplicate = copy.deepcopy(heldout)
    duplicate[0]["ctext"] = train[0]["ctext"]
    with pytest.raises(ValueError, match="overlap"):
        validate_task_items(manifest, train, duplicate)
