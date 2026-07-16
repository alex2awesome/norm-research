#!/usr/bin/env python3

from __future__ import annotations

import json

from methods.metric_seam.family_scale.response_normalization import normalize_alignment_raw


def test_clean_fenced_response_passes_through() -> None:
    raw = '```json\n{"clusters": [], "unmatched_unit_ids": ["u_a"]}\n```'
    normalized = normalize_alignment_raw(raw)
    parsed = json.loads(normalized)
    assert parsed == {"clusters": [], "unmatched_unit_ids": ["u_a"]}


def test_stray_trailing_brace_repaired() -> None:
    raw = '{"clusters": [], "unmatched_unit_ids": ["u_a"]}\n}'
    normalized = normalize_alignment_raw(raw)
    parsed = json.loads(normalized)
    assert parsed == {"clusters": [], "unmatched_unit_ids": ["u_a"]}


def test_singleton_cluster_moved_to_unmatched() -> None:
    raw = json.dumps({
        "clusters": [
            {"cluster_id": "c1", "unit_ids": ["u_x"]},
            {"cluster_id": "c2", "unit_ids": ["u_y", "u_z"]},
            {"cluster_id": "c3", "unit_ids": ["u_a", "u_b", "u_c"]},
        ],
        "unmatched_unit_ids": ["u_d"],
    })
    normalized = normalize_alignment_raw(raw)
    parsed = json.loads(normalized)
    assert {"cluster_id": "c1", "unit_ids": ["u_x"]} not in parsed["clusters"]
    assert len(parsed["clusters"]) == 2
    assert "u_x" in parsed["unmatched_unit_ids"]
    assert "u_d" in parsed["unmatched_unit_ids"]


def test_unrepairable_trailing_content_unchanged() -> None:
    raw = '{"clusters": [], "unmatched_unit_ids": ["u_a"]}\n}extra data'
    normalized = normalize_alignment_raw(raw)
    assert normalized == raw


def test_non_object_shaped_json_unchanged() -> None:
    raw = '["not", "an", "object"]'
    normalized = normalize_alignment_raw(raw)
    assert normalized == raw

    raw_missing = '{"clusters": []}'
    normalized_missing = normalize_alignment_raw(raw_missing)
    assert normalized_missing == raw_missing


def test_never_raises_on_adversarial_input() -> None:
    adversarial_inputs = [
        "",
        "   ",
        "not json at all",
        "{",
        "{{{{",
        "[]][",
        "null",
    ]
    for raw in adversarial_inputs:
        normalized = normalize_alignment_raw(raw)
        assert isinstance(normalized, str)
