from __future__ import annotations

import json

from methods.metric_seam import cross_audit_code_review_unused_programs_v1 as audit


def test_cross_audit_is_additive_relation_local_and_depth_corrected() -> None:
    result = audit.build()
    assert result["summary"] == {
        "n_proposals_audited": 9,
        "n_accepted_partial_relation_local": 9,
        "n_rejected": 0,
        "n_unique_programs": 6,
        "accepted_by_level": {"R1": 3, "R2": 4, "R3": 2},
        "accepted_by_audited_depth": {"2": 4, "4": 5},
        "depth_corrections": 5,
        "canonical_corrected_static_unchanged": 50,
        "additive_static_union_if_adopted": 59,
        "additive_static_union_by_level_if_adopted": {"R1": 17, "R2": 19, "R3": 23},
    }
    assert all(row["verdict"] == "accepted_partial_relation_local" for row in result["rows"])
    assert all(row["whole_construct_exact"] is False for row in result["rows"])
    assert set(result["sealed_inputs"].values()) == {False}


def test_external_tool_programs_receive_depth_four() -> None:
    result = audit.build()
    by_aspect: dict[str, set[int]] = {}
    for row in result["rows"]:
        by_aspect.setdefault(row["candidate_aspect_id"], set()).add(row["audited_depth"])
    assert by_aspect["a72"] == {4}
    assert by_aspect["a181"] == {4}
    assert by_aspect["a25"] == {4}
    assert by_aspect["a35"] == {2}
    assert by_aspect["a400"] == {2}
    assert by_aspect["a309"] == {2}


def test_checked_cross_audit_rebuilds_exactly() -> None:
    checked = json.loads(audit.OUT.read_text(encoding="utf-8"))
    assert checked == audit.build()
